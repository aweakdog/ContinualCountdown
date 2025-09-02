# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import List, Dict

import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
import ray
from verl.utils.redo_utils.gradient_analyzer import GradientAnalyzer
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad

# FSDP DEBUG INFO
import torch
import inspect
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
print("[DEBUG] PyTorch version:", torch.__version__)
print("[DEBUG] FSDP class:", FSDP)
print("[DEBUG] FSDP module:", FSDP.__module__)
print("[DEBUG] FSDP doc:", FSDP.__doc__[:200])
print("[DEBUG] FSDP signature:", inspect.signature(FSDP.__init__))
try:
    print("[DEBUG] FSDP source:", FSDP.__module__, FSDP.__init__.__code__.co_filename)
except Exception as e:
    print(f"[DEBUG] Could not get FSDP source: {e}")
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

# Note: Singleton implementation for analyzers now uses Ray's named actors
# to ensure a single instance across all distributed processes.


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.actor_update_step = 0
        self.critic_update_step = 0
        self.config = config
        self.role = role
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        
        # Set local_rank after distributed is initialized
        self.local_rank = torch.distributed.get_rank()
        print(f"[DEBUG] Worker initialized with local_rank: {self.local_rank}")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            # <<< Cascade: Collect original parameter shapes before FSDP wrapping >>>
            # Only rank 0 needs parameter shapes for C_K reset analysis
            self.original_param_shapes = {}
            if self.rank == 0:
                print("[INFO] Collecting original parameter shapes before FSDP wrapping...")
                for fqn, param in actor_module.named_parameters():
                    self.original_param_shapes[fqn] = param.shape
                print(f"[INFO] Collected {len(self.original_param_shapes)} original parameter shapes. Example: {list(self.original_param_shapes.items())[0] if self.original_param_shapes else 'N/A'}")
            # <<< End Cascade modification >>>

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # DEBUG: Print model module tree before FSDP wrapping
        print("[DEBUG] Model module tree BEFORE FSDP wrapping:")
        for name, module in actor_module.named_modules():
            print(f"  {name}: {type(module)}")

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=True,  # Enable original param views for robust FSDP analysis
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # DEBUG: Print FSDP-wrapped module structure
        print("[DEBUG] Model module tree AFTER FSDP wrapping:")
        print(f"Top-level type: {type(actor_module_fsdp)}")
        for name, module in actor_module_fsdp.named_modules():
            print(f"  {name}: {type(module)}")

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            
            # Initialize Gradient Analyzer only on rank 0 to avoid multiple instances
            # Check if this is the first worker (rank 0) to avoid resource conflicts
            is_rank_0 = getattr(self, 'local_rank', 0) == 0
            print(f"[DEBUG] Worker rank check: local_rank={getattr(self, 'local_rank', 'unknown')}, is_rank_0={is_rank_0}")
            
            self.grad_analyzer = None
            
            # Debug: Check configuration values
            # Default to True to enable gradient analysis by default
            fsdp_grad_metric_enabled = self.config.actor.get("fsdp_grad_metric_enabled", True)
            root_fsdp_grad_metric = self.config.get("fsdp_grad_metric_enabled", True)
            
            print(f"[DEBUG] Gradient Analyzer config check:")
            print(f"[DEBUG]   - actor.fsdp_grad_metric_enabled: {fsdp_grad_metric_enabled}")
            print(f"[DEBUG]   - root.fsdp_grad_metric_enabled: {root_fsdp_grad_metric}")
            print(f"[DEBUG]   - self._is_actor: {self._is_actor}")
            print(f"[DEBUG]   - Available actor config keys: {list(self.config.actor.keys()) if hasattr(self.config, 'actor') else 'No actor config'}")

            if fsdp_grad_metric_enabled or root_fsdp_grad_metric:
                if is_rank_0:
                    try:
                        # Rank 0 creates the actor.
                        print("[INFO] Rank 0 creating GradientAnalyzer singleton (named actor)...")
                        from verl.utils.redo_utils.gradient_analyzer import GradientAnalyzer
                        self.grad_analyzer = GradientAnalyzer.options(
                            name="global_gradient_analyzer",
                            num_gpus=1,  # 使用所有空闲GPU (4-7)
                            num_cpus=16   # 增加CPU资源用于并行处理
                        ).remote()
                        print(f"[INFO] GradientAnalyzer singleton created: {self.grad_analyzer}")
                        print(f"[DEBUG] GradientAnalyzer type: {type(self.grad_analyzer)}")
                    except Exception as e:
                        print(f"[ERROR] Failed to initialize GradientAnalyzer: {e}")
                        self.grad_analyzer = None
                else:
                    # Other ranks wait a bit and then get the actor.
                    print(f"[INFO] Rank {getattr(self, 'local_rank', 'unknown')} waiting for GradientAnalyzer to be created...")
                    import time
                    time.sleep(10)  # Increase wait time to 10 seconds
                    
                    # Retry logic for getting the actor
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            self.grad_analyzer = ray.get_actor("global_gradient_analyzer")
                            print(f"[INFO] Rank {getattr(self, 'local_rank', 'unknown')} got GradientAnalyzer handle: {self.grad_analyzer}")
                            print(f"[DEBUG] GradientAnalyzer type: {type(self.grad_analyzer)}")
                            break
                        except Exception as e:
                            print(f"[ERROR] Rank {getattr(self, 'local_rank', 'unknown')} failed to get GradientAnalyzer (attempt {attempt + 1}/{max_retries}): {e}")
                            if attempt < max_retries - 1:
                                time.sleep(5)  # Wait before retry
                            else:
                                self.grad_analyzer = None
            else:
                print(f"[INFO] Gradient analysis disabled - fsdp_grad_metric_enabled={fsdp_grad_metric_enabled}, root_fsdp_grad_metric={root_fsdp_grad_metric}")
                self.grad_analyzer = None

            self.fisher_info_analyzer = None
            if self.config.actor.get("fisher_analysis_enabled", True):
                if self.config.actor.get('fsdp_component_analysis', {}).get('run_fisher_info_analysis', True):
                    if is_rank_0:
                        try:
                            # Rank 0 creates the actor.
                            print("[INFO] Rank 0 creating FisherInfoAnalyzer singleton (named actor)...")
                            from verl.utils.redo_utils.fisher_info_analyzer import FisherInfoAnalyzer
                            self.fisher_info_analyzer = FisherInfoAnalyzer.options(
                                name="global_fisher_info_analyzer",
                                num_gpus=3,  # 使用所有空闲GPU (4-7)
                                num_cpus=16   # 增加CPU资源用于并行处理
                            ).remote(self.config)
                            print(f"[INFO] FisherInfoAnalyzer singleton created: {self.fisher_info_analyzer}")
                        except Exception as e:
                            print(f"[ERROR] Failed to initialize FisherInfoAnalyzer: {e}")
                            self.fisher_info_analyzer = None
                    else:
                        # Other ranks wait a bit and then get the actor.
                        print(f"[INFO] Rank {getattr(self, 'local_rank', 'unknown')} waiting for FisherInfoAnalyzer to be created...")
                        import time
                        time.sleep(10)  # Give rank 0 time to create the actor
                        
                        # Retry logic for getting the Fisher analyzer
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                self.fisher_info_analyzer = ray.get_actor("global_fisher_info_analyzer")
                                print(f"[INFO] Rank {getattr(self, 'local_rank', 'unknown')} got FisherInfoAnalyzer handle: {self.fisher_info_analyzer}")
                                break
                            except Exception as e:
                                print(f"[ERROR] Rank {getattr(self, 'local_rank', 'unknown')} failed to get FisherInfoAnalyzer (attempt {attempt + 1}/{max_retries}): {e}")
                                if attempt < max_retries - 1:
                                    time.sleep(5)  # Wait before retry
                                else:
                                    self.fisher_info_analyzer = None

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                original_param_shapes=self.original_param_shapes, 
                grad_analyzer=self.grad_analyzer,
                fisher_info_analyzer=self.fisher_info_analyzer,
            )
            # (Re-)initialize verl-compatible analyzer and redo for actor after checkpoint/model load
            self.actor_redo_enabled = getattr(self.config, 'redo_enabled', True)
            self.actor_redo_tau = getattr(self.config, 'redo_tau', 0.1)
            self.actor_redo_reset_freq = getattr(self.config, 'redo_reset_freq', 1000)
            self.actor_redo_metric_freq = getattr(self.config, 'redo_metric_freq', 1)
            #if self.actor_redo_enabled:
            #    self.actor_gradredo = VerlGradientReDo(self.actor_module_fsdp, tau=self.actor_redo_tau, frequency=self.actor_redo_reset_freq, optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, 
                                                   actor_module=self.ref_module_fsdp,
                                                   original_param_shapes=self.original_param_shapes) # <<< Cascade: Pass original_param_shapes

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Periodically print plasticity metrics and do redo
        self.actor_update_step += 1
        metrics = {}
        # Print metrics & perform neuron reset only if enabled
        # Print metrics & perform neuron reset only if enabled
        #if getattr(self, 'actor_redo_enabled', True):
        #    if self.actor_update_step % self.actor_redo_reset_freq == 0:
        #        print(f"[Actor] Step {self.actor_update_step}: Performing Gradient-based neuron reset (tau={self.actor_redo_tau})")
        #        self.actor_gradredo.step()
        data = data.to('cuda')

        # ... (rest of the function)

        # After backward pass (i.e., after self.actor.update_policy)
        # with self.ulysses_sharding_manager:
        #     data = self.ulysses_sharding_manager.preprocess_data(data=data)
        #     metrics = {}
        #     with Timer(name='update_policy', logger=None) as timer:
        #         metrics.update(self.actor.update_policy(data=data))
        #     delta_time = timer.last
        #     global_num_tokens = data.meta_info['global_token_num']
        #     estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        #     metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        #
        #     self.actor_lr_scheduler.step()
        #     lr = self.actor_lr_scheduler.get_last_lr()[0]
        #     metrics['actor/lr'] = lr
        #
        #  # logic matches analyzer
        #
        #     log_gpu_memory_usage('After update policy', logger=logger)
        #
        #     # TODO: here, we should return all metrics
        #     output = DataProto(meta_info={'metrics': metrics})

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            #metrics = {}
            # FSDP-safe gradient analysis metrics (if any) should already be in 'metrics' dict
            with Timer(name='update_policy', logger=None) as timer:
                metrics.update(self.actor.update_policy(data=data))
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output = self.ulysses_sharding_manager.preprocess_data(output)
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def extract_layers_for_reset(self, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Extract specific transformer layers from reference model for layer reset.
        Memory-efficient: RefPolicy stays on GPU, only transfers specific layers to CPU.
        
        Args:
            layer_indices: List of layer indices to extract
            
        Returns:
            Dictionary containing state dict for specified layers (on CPU)
        """
        print(f"[LAYER_RESET_DEBUG] RefWorker extracting layers {layer_indices}")
        
        # Import required FSDP types
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        
        if not self._is_ref:
            raise RuntimeError("extract_layers_for_reset can only be called on reference worker")
        
        # Ensure reference model is loaded (should already be on GPU for PPO)
        was_offloaded = False
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                   device_id=torch.cuda.current_device(),
                                   load_grad=False)
            was_offloaded = True
        
        try:
            # Get transformer layers from reference model directly
            # Handle FSDP wrapped models
            if isinstance(self.ref_module_fsdp, FSDP):
                base_model = self.ref_module_fsdp._fsdp_wrapped_module
            else:
                base_model = self.ref_module_fsdp
            
            # Find transformer layers using common patterns
            layer_patterns = ['model.layers', 'transformer.h', 'transformer.layers', 'layers']
            transformer_layers = None
            
            for pattern in layer_patterns:
                try:
                    layers = base_model
                    for attr in pattern.split('.'):
                        layers = getattr(layers, attr)
                    if isinstance(layers, (list, torch.nn.ModuleList)):
                        transformer_layers = list(layers)
                        break
                except AttributeError:
                    continue
            
            if transformer_layers is None:
                raise ValueError("Could not identify transformer layers in the reference model")
            
            # Get the layer container path
            layer_patterns = ['model.layers', 'transformer.h', 'transformer.layers', 'layers']
            layer_container_path = None
            
            # Handle FSDP wrapped models
            if isinstance(self.ref_module_fsdp, FSDP):
                base_model = self.ref_module_fsdp._fsdp_wrapped_module
            else:
                base_model = self.ref_module_fsdp
            
            for pattern in layer_patterns:
                try:
                    container = base_model
                    for attr in pattern.split('.'):
                        container = getattr(container, attr)
                    if isinstance(container, (list, torch.nn.ModuleList)) and len(container) == len(transformer_layers):
                        layer_container_path = pattern
                        break
                except AttributeError:
                    continue
            
            if layer_container_path is None:
                raise RuntimeError("Could not find transformer layers container")
            
            # Memory-efficient extraction: only get specific layers and transfer to CPU
            layer_state_dict = {}
            
            # Extract state dict layer by layer to minimize GPU memory usage
            for layer_idx in layer_indices:
                print(f"[LAYER_RESET_DEBUG] Extracting layer {layer_idx}...")
                
                # Get the specific layer
                target_layer = transformer_layers[layer_idx]
                
                # Extract parameters for this layer and immediately transfer to CPU
                layer_prefix = f"{layer_container_path}.{layer_idx}."
                
                # Use FSDP state dict for this specific layer
                with FSDP.state_dict_type(self.ref_module_fsdp, StateDictType.FULL_STATE_DICT,
                                         FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                    # Get full state dict but only extract what we need
                    full_state_dict = self.ref_module_fsdp.state_dict()
                    
                    # Extract only this layer's parameters and transfer to CPU immediately
                    layer_params_found = 0
                    for param_name, param_tensor in full_state_dict.items():
                        if param_name.startswith(layer_prefix):
                            # Transfer to CPU to save GPU memory
                            layer_state_dict[param_name] = param_tensor.detach().cpu().clone()
                            layer_params_found += 1
                            print(f"[LAYER_RESET_DEBUG] Extracted parameter: {param_name} (shape: {param_tensor.shape})")
                    
                    print(f"[LAYER_RESET_DEBUG] Layer {layer_idx}: found {layer_params_found} parameters with prefix '{layer_prefix}'")
                    
                    # Clear the full state dict to free GPU memory
                    del full_state_dict
                
                # Force garbage collection to free GPU memory
                torch.cuda.empty_cache()
                print(f"[LAYER_RESET_DEBUG] Layer {layer_idx} extracted and transferred to CPU")
            
            print(f"[LAYER_RESET_DEBUG] RefWorker extracted {len(layer_state_dict)} parameters for layers {layer_indices}")
            return layer_state_dict
            
        finally:
            # Only offload if it was originally offloaded (preserve original state)
            if was_offloaded and self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=False)
            # Clean up GPU memory
            torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_learning_rate(self):
        """Reset learning rates to initial values while keeping optimizer state"""
        if self.actor_optimizer is not None and self.actor_lr_scheduler is not None:
            # Reset scheduler's internal state
            self.actor_lr_scheduler.last_epoch = -1
            # Update learning rate
            self.actor_lr_scheduler.step()

    # NOTE: Old LayerResetManager methods (reset_layers, reset_layers_with_ref_dict, get_transformer_layer_count)
    # have been removed and replaced with CKBasedResetManager integration in CriticWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        self.actor_update_step = 0
        self.critic_update_step = 0
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)

        # Initialize Ray shared state manager for reset synchronization
        try:
            from verl.utils.redo_utils.shared_reset_state import SharedResetStateManager
            self._shared_reset_manager = SharedResetStateManager()
            print(f"[CriticWorker] Using Ray shared state for reset synchronization")
        except Exception as e:
            print(f"[CriticWorker] Failed to initialize Ray shared state: {e}")
            raise RuntimeError(f"Ray shared state synchronization is required but failed to initialize: {e}")

        # Initialize C_K-based reset manager for critic
        self.ck_reset_manager = None
        # Check both general CK_RESET_ENABLE and specific CK_RESET_CRITIC_ENABLE
        ck_reset_enabled = hasattr(self.config, 'ck_reset') and self.config.ck_reset.get('enable', False)
        critic_reset_enabled = os.environ.get('CK_RESET_CRITIC_ENABLE', 'true').lower() == 'true'
        
        if ck_reset_enabled and critic_reset_enabled:
            from verl.utils.redo_utils.ck_based_reset_manager import create_ck_based_reset_manager
            self.ck_reset_manager = create_ck_based_reset_manager(self.config.ck_reset)
            print(f"[CriticWorker] Initialized C_K-based reset manager with strategy: {self.config.ck_reset.get('reset_strategy', 'ck_guided')}")
            print(f"[CriticWorker] Critic-specific reset enabled for countdown task group transitions")
        else:
            if not ck_reset_enabled:
                print(f"[CriticWorker] C_K-based reset manager disabled (CK_RESET_ENABLE=false)")
            elif not critic_reset_enabled:
                print(f"[CriticWorker] Critic reset disabled (CK_RESET_CRITIC_ENABLE=false)")
            else:
                print(f"[CriticWorker] C_K-based reset manager disabled")

    def _load_actor_reset_layers(self, current_step):
        """Load actor's selected reset layers for synchronization."""
        try:
            selected_layers = self._shared_reset_manager.load_actor_reset_layers(current_step)
            if selected_layers:
                print(f"[CRITIC_RESET_SYNC] Loaded actor reset layers {selected_layers} for step {current_step} via Ray")
            else:
                print(f"[CRITIC_RESET_SYNC] No actor reset layers found for step {current_step} via Ray")
            return selected_layers
        except Exception as e:
            print(f"[CRITIC_RESET_SYNC] Failed to load actor reset layers via Ray: {e}")
            raise RuntimeError(f"Failed to load actor reset layers: {e}")

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=True,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)
        # Initialize verl-compatible analyzer and redo for critic
        #self.critic_redo_enabled = getattr(self.config, 'redo_enabled', True)
        #self.critic_redo_tau = getattr(self.config, 'redo_tau', 0.1)
        #self.critic_redo_reset_freq = getattr(self.config, 'redo_reset_freq', 1000)
        #self.critic_redo_metric_freq = getattr(self.config, 'redo_metric_freq', 1)
        #if self.critic_redo_enabled:
        #    self.critic_gradredo = VerlGradientReDo(self.critic_module, tau=self.critic_redo_tau, frequency=self.critic_redo_reset_freq, optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)

        # Store initial critic state for layer reset functionality
        self._store_initial_critic_state()

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        # Periodically print plasticity metrics and do redo
        self.critic_update_step += 1
        metrics = {}
        # Print metrics & perform neuron reset only if enabled
        # Print metrics & perform neuron reset only if enabled
        #if getattr(self, 'critic_redo_enabled', True):
        #    if self.critic_update_step % self.critic_redo_reset_freq == 0:
        #        print(f"[Critic] Step {self.critic_update_step}: Performing Gradient-based neuron reset (tau={self.critic_redo_tau})")
        #        if self.critic_redo_enabled:
        #            self.critic_gradredo = VerlGradientReDo(self.critic_module, tau=self.critic_redo_tau, frequency=self.critic_redo_reset_freq, optimizer=self.critic_optimizer)
        #            self.critic_gradredo.step()
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics.update(self.critic.update_critic(data=data))
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size


            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            # C_K-based layer reset logic for critic
            if self.rank == 0 and self.ck_reset_manager is not None:
                try:
                    # For critic, we use a simplified approach since we don't have Fisher analysis
                    # Check if reset should be performed based on global steps
                    should_reset_ck, layer_ck_weights = self.ck_reset_manager.should_reset_with_ck_analysis(
                        self.critic_update_step, None  # No Fisher stats for critic
                    )
                    
                    if should_reset_ck:
                        print(f"[CK_RESET_TRIGGER] Critic Step {self.critic_update_step}: C_K-based reset triggered!")
                        
                        # Log reset statistics before performing reset
                        self.ck_reset_manager.log_reset_statistics(self.critic_update_step)
                        
                        # Get selected layers for reset
                        transformer_layers = self.ck_reset_manager.get_transformer_layers(self.critic_module)
                        total_layers = len(transformer_layers)
                        
                        # Use critic-specific strategy configuration
                        critic_strategy = os.environ.get('CK_RESET_CRITIC_STRATEGY', 'random')
                        print(f"[CK_RESET_INFO] Critic using strategy: {critic_strategy}")
                        
                        if critic_strategy == 'ck_guided':
                            # For ck_guided strategy, try to use the same layers as actor
                            print(f"[CK_RESET_INFO] Critic using ck_guided strategy - attempting to sync with actor")
                            
                            # Try to load actor's selected layers
                            actor_selected_layers = self._load_actor_reset_layers(self.critic_update_step)
                            
                            if actor_selected_layers is not None:
                                # Use the exact same layers as actor
                                selected_layers = actor_selected_layers
                                print(f"[CK_RESET_INFO] Critic using same layers as actor: {selected_layers}")
                            else:
                                # Fallback: use synchronized random selection
                                print(f"[CK_RESET_INFO] Actor layers not available, using synchronized random fallback")
                                
                                # Temporarily change to ck_guided strategy
                                original_strategy = self.ck_reset_manager.reset_strategy
                                self.ck_reset_manager.reset_strategy = 'ck_guided'
                                
                                # Use force_random_for_sync=True to ensure synchronized random selection
                                selected_layers = self.ck_reset_manager.select_layers_to_reset(
                                    total_layers, {}, self.critic_update_step, force_random_for_sync=True
                                )
                                
                                # Restore original strategy
                                self.ck_reset_manager.reset_strategy = original_strategy
                        else:
                            # For other strategies, use critic-specific strategy
                            # Temporarily change to critic-specific strategy
                            original_strategy = self.ck_reset_manager.reset_strategy
                            self.ck_reset_manager.reset_strategy = critic_strategy
                            
                            selected_layers = self.ck_reset_manager.select_layers_to_reset(
                                total_layers, {}, self.critic_update_step
                            )
                            
                            # Restore original strategy
                            self.ck_reset_manager.reset_strategy = original_strategy
                        
                        print(f"[CK_RESET_INFO] Critic Step {self.critic_update_step}: Resetting {len(selected_layers)} layers: {selected_layers}")
                        print(f"[CK_RESET_INFO] Critic Strategy: {self.ck_reset_manager.reset_strategy}")
                        
                        # Perform actual reset using the C_K-based reset manager
                        try:
                            # Enhanced reference worker access - use Ray object store (same as actor)
                            ref_worker = None
                            
                            # Method 1: Check if we have access to reference worker via Ray object store
                            try:
                                # Try to get reference worker from Ray object store (set by trainer)
                                ref_worker_info = ray.get("ck_reset_ref_worker_info")
                                if ref_worker_info:
                                    ref_worker_ref = ref_worker_info.get('ref_worker_ref')
                                    ref_worker_type = ref_worker_info.get('type')
                                    print(f"[CK_RESET_DEBUG] Critic: Found ref_worker info in object store ({ref_worker_type})")
                                    
                                    try:
                                        ref_worker = ray.get(ref_worker_ref)
                                        print(f"[CK_RESET_DEBUG] Critic: Successfully retrieved ref_worker from object store")
                                    except Exception as e:
                                        print(f"[CK_RESET_DEBUG] Critic: Failed to get ref_worker from object store: {e}")
                                        ref_worker = None
                                else:
                                    print(f"[CK_RESET_DEBUG] Critic: No ref_worker info found in object store")
                            except Exception as e:
                                print(f"[CK_RESET_DEBUG] Critic: Failed to access object store for ref_worker: {e}")
                            
                            # Method 2: Fallback to trainer access (original logic)
                            if ref_worker is None:
                                print(f"[CK_RESET_DEBUG] Critic: Falling back to trainer access")
                                if hasattr(self.trainer, 'ref_policy_wg') and self.trainer.use_reference_policy:
                                    ref_worker = self.trainer.ref_policy_wg
                                    print(f"[CK_RESET_DEBUG] Critic: Using ref_policy_wg from trainer")
                                elif hasattr(self.trainer, 'actor_rollout_wg'):
                                    ref_worker = self.trainer.actor_rollout_wg
                                    print(f"[CK_RESET_DEBUG] Critic: Using actor_rollout_wg from trainer (fallback)")
                            
                            # Method 3: Check if current worker has reference model
                            if ref_worker is None and hasattr(self, '_is_ref') and self._is_ref:
                                print(f"[CK_RESET_DEBUG] Critic: Using self as reference worker (_is_ref=True)")
                                ref_worker = self
                            
                            if ref_worker is None:
                                print(f"[CK_RESET_ERROR] Critic: No reference worker available for reset")
                                metrics['critic/ck_reset_triggered'] = -1.0
                                metrics['critic/ck_reset_layers_count'] = 0.0
                            else:
                                # Perform actual layer reset with reference worker
                                reset_param_names = self.ck_reset_manager.reset_model_with_ck_analysis(
                                    current_model=self.critic_module,
                                    ref_worker=ref_worker,
                                    layer_ck_weights=layer_ck_weights,
                                    global_step=self.critic_update_step,
                                    model_name="critic"
                                )
                                
                                # Store detailed reset info in metrics
                                metrics['critic/ck_reset_triggered'] = 1.0
                                metrics['critic/ck_reset_layers_count'] = len(selected_layers)
                                metrics['critic/ck_reset_strategy'] = hash(self.ck_reset_manager.reset_strategy) % 1000  # Encode strategy as number
                                metrics['critic/ck_reset_params_count'] = len(reset_param_names)
                                
                                print(f"[CK_RESET_SUCCESS] Critic Step {self.critic_update_step}: Reset {len(reset_param_names)} parameters in {len(selected_layers)} layers")
                            
                        except Exception as reset_error:
                            print(f"[CK_RESET_ERROR] Critic Step {self.critic_update_step}: Reset failed: {reset_error}")
                            import traceback
                            traceback.print_exc()
                            metrics['critic/ck_reset_triggered'] = -1.0  # Indicate failure
                            metrics['critic/ck_reset_layers_count'] = 0.0
                            
                    else:
                        metrics['critic/ck_reset_triggered'] = 0.0
                        metrics['critic/ck_reset_layers_count'] = 0.0
                        
                except Exception as e:
                    import traceback
                    print(f"[CK_RESET_ERROR] Critic: Failed to perform CK reset: {e}")
                    print(f"[CK_RESET_ERROR] Critic: Traceback: {traceback.format_exc()}")
                    metrics['critic/ck_reset_triggered'] = -1.0  # Indicate failure
                    metrics['critic/ck_reset_layers_count'] = 0.0

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_learning_rate(self):
        """Reset learning rates to initial values while keeping optimizer state"""
        if self.critic_optimizer is not None and self.critic_lr_scheduler is not None:
            # Reset scheduler's internal state
            self.critic_lr_scheduler.last_epoch = -1
            # Update learning rate
            self.critic_lr_scheduler.step()

    def _store_initial_critic_state(self):
        """Store the initial critic state for layer reset functionality"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
        import copy
        
        logger.info(f"Storing initial critic state for layer reset on rank {self.rank}")
        
        # Store initial state dict for layer reset
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            self.initial_critic_state_dict = copy.deepcopy(self.critic_module.state_dict())
        
        logger.info(f"Stored initial critic state with {len(self.initial_critic_state_dict)} parameters")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_layers(self, layer_reset_manager, ref_worker=None) -> None:
        """
        Reset specific transformer layers of the critic model to reference model state.
        
        Args:
            layer_reset_manager: LayerResetManager instance with reset configuration
            ref_worker: Reference worker to get layer weights from (for memory efficiency)
        """
        if not layer_reset_manager.reset_critic:
            print(f"[LAYER_RESET_DEBUG] Critic reset disabled, skipping")
            return
        
        print(f"[LAYER_RESET_DEBUG] CriticWorker resetting layers: k_first={layer_reset_manager.reset_k_first}, k_last={layer_reset_manager.reset_k_last}")
        
        # Get transformer layers for debugging
        transformer_layers = layer_reset_manager.get_transformer_layers(self.critic_module)
        total_layers = len(transformer_layers)
        print(f"[LAYER_RESET_DEBUG] Critic model has {total_layers} transformer layers")
        
        # Get layer indices to reset
        layer_indices = layer_reset_manager.get_layer_indices_to_reset(total_layers)
        
        if not layer_indices:
            print(f"[LAYER_RESET_DEBUG] No layers to reset for critic")
            return
        
        print(f"[LAYER_RESET_DEBUG] Critic resetting layers: {layer_indices}")
        
        # Capture weights before reset for comparison
        weights_before = {}
        for layer_idx in layer_indices[:1]:  # Only check first layer to avoid spam
            layer = transformer_layers[layer_idx]
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    weights_before[f"layer_{layer_idx}.{name}"] = param.data.clone().detach().cpu()
                    print(f"[LAYER_RESET_DEBUG] Before reset - Critic Layer {layer_idx} {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                    break  # Only check one weight per layer
        
        # Memory-efficient approach: get reference layers from ref worker
        if ref_worker is not None:
            # Get reference layer state dict from ref worker
            ref_layer_state_dict = layer_reset_manager.get_reference_layer_state_dict(ref_worker, layer_indices)
            
            # Reset critic layers using reference state dict
            reset_param_names = layer_reset_manager.reset_model_layers_from_ref(
                model=self.critic_module,
                ref_layer_state_dict=ref_layer_state_dict,
                reset_k_first=layer_reset_manager.reset_k_first,
                reset_k_last=layer_reset_manager.reset_k_last
            )
            
            # Reset optimizer states for affected parameters
            if self.critic_optimizer is not None and layer_reset_manager.reset_optimizer_states:
                layer_reset_manager._reset_optimizer_states(self.critic_optimizer, reset_param_names, "critic")
            
            # Multi-GPU: Ensure FSDP parameters are properly synchronized after reset
            if torch.distributed.is_initialized():
                print(f"[LAYER_RESET_DEBUG] Multi-GPU: Synchronizing critic FSDP parameters across ranks")
                torch.distributed.barrier()
                
                # If using FSDP offload, ensure parameters are properly loaded after reset
                if self._is_offload_param:
                    print(f"[LAYER_RESET_DEBUG] Multi-GPU: Reloading critic FSDP offloaded parameters after reset")
                    load_fsdp_param_and_grad(module=self.critic_module,
                                           device_id=torch.cuda.current_device(),
                                           load_grad=self._is_offload_grad)
            
            # Compare weights after reset
            print(f"[LAYER_RESET_DEBUG] Comparing critic weights after reset...")
            for layer_idx in layer_indices[:1]:  # Only check first layer to avoid spam
                layer = transformer_layers[layer_idx]
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        before_key = f"layer_{layer_idx}.{name}"
                        if before_key in weights_before:
                            before_weight = weights_before[before_key]
                            after_weight = param.data.detach().cpu()
                            
                            # Check if weights actually changed
                            weight_diff = torch.norm(after_weight - before_weight).item()
                            weights_equal = torch.allclose(after_weight, before_weight, atol=1e-6)
                            
                            print(f"[LAYER_RESET_DEBUG] After reset - Critic Layer {layer_idx} {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                            print(f"[LAYER_RESET_DEBUG] Weight change - Critic Layer {layer_idx} {name}: diff_norm={weight_diff:.6f}, weights_equal={weights_equal}")
                            
                            if weights_equal:
                                print(f"[LAYER_RESET_DEBUG] WARNING: Critic weights did not change for layer {layer_idx} {name}!")
                            else:
                                print(f"[LAYER_RESET_DEBUG] SUCCESS: Critic weights changed for layer {layer_idx} {name}")
                        break  # Only check one weight per layer
        
        elif hasattr(self, 'initial_critic_state_dict'):
            # Fallback: use initial critic state (less correct but available)
            print(f"[LAYER_RESET_DEBUG] Warning: Using initial critic state instead of reference model")
            # TODO: Implement reset from initial state dict if needed
            print(f"[LAYER_RESET_DEBUG] Error: Initial critic state reset not implemented yet")
            return
        else:
            print(f"[LAYER_RESET_DEBUG] Error: No reference model available for critic reset")
            return
        
        print(f"[LAYER_RESET_DEBUG] Critic layer reset completed")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_layers_with_ref_dict(self, layer_reset_manager, ref_layer_state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Reset specific transformer layers of the critic model using pre-extracted reference state dict.
        This avoids Ray deadlock by not calling other Ray workers from within this worker.
        
        Args:
            layer_reset_manager: LayerResetManager instance with reset configuration
            ref_layer_state_dict: Pre-extracted reference layer state dict (on CPU)
        """
        if not layer_reset_manager.reset_critic:
            return  # Only reset if critic reset is enabled
        
        print(f"[LAYER_RESET_DEBUG] CriticWorker resetting layers with pre-extracted reference: k_first={layer_reset_manager.reset_k_first}, k_last={layer_reset_manager.reset_k_last}")
        
        # Get transformer layers for debugging
        transformer_layers = layer_reset_manager.get_transformer_layers(self.critic_module)
        total_layers = len(transformer_layers)
        print(f"[LAYER_RESET_DEBUG] Critic model has {total_layers} transformer layers")
        
        # Get layer indices to reset
        layer_indices = layer_reset_manager.get_layer_indices_to_reset(total_layers)
        
        if not layer_indices:
            print(f"[LAYER_RESET_DEBUG] No layers to reset for critic")
            return
        
        print(f"[LAYER_RESET_DEBUG] Critic resetting layers: {layer_indices}")
        
        # Capture weights before reset for comparison
        weights_before = {}
        for layer_idx in layer_indices[:1]:  # Only check first layer to avoid spam
            layer = transformer_layers[layer_idx]
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    weights_before[f"layer_{layer_idx}.{name}"] = param.data.clone().detach().cpu()
                    print(f"[LAYER_RESET_DEBUG] Before reset - Critic Layer {layer_idx} {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                    break  # Only check one weight per layer
        
        # Reset critic layers using pre-extracted reference state dict
        print(f"[LAYER_RESET_DEBUG] Using pre-extracted reference layers for critic (avoiding Ray deadlock)")
        reset_param_names = layer_reset_manager.reset_model_layers_from_ref(
            model=self.critic_module,
            ref_layer_state_dict=ref_layer_state_dict,
            reset_k_first=layer_reset_manager.reset_k_first,
            reset_k_last=layer_reset_manager.reset_k_last
        )
        
        # Reset optimizer states for affected parameters
        if self.critic_optimizer is not None and layer_reset_manager.reset_optimizer_states:
            layer_reset_manager._reset_optimizer_states(self.critic_optimizer, reset_param_names, "critic")
        
        # Multi-GPU: Ensure FSDP parameters are properly synchronized after reset
        if torch.distributed.is_initialized():
            print(f"[LAYER_RESET_DEBUG] Multi-GPU: Synchronizing critic FSDP parameters across ranks")
            
            # Force FSDP to reshard parameters after reset
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                if isinstance(self.critic_module, FSDP) or hasattr(self.critic_module, '_fsdp_wrapped_module'):
                    print(f"[LAYER_RESET_DEBUG] Multi-GPU: Forcing FSDP reshard after parameter reset")
                    # Summon full parameters to ensure all ranks have updated values
                    with FSDP.summon_full_params(self.critic_module, writeback=True):
                        pass  # This forces synchronization of parameters across all ranks
            except Exception as e:
                print(f"[LAYER_RESET_DEBUG] Multi-GPU: FSDP reshard failed: {e}")
            
            torch.distributed.barrier()
            
            # If using FSDP offload, ensure parameters are properly loaded after reset
            if self._is_offload_param:
                print(f"[LAYER_RESET_DEBUG] Multi-GPU: Reloading critic FSDP offloaded parameters after reset")
                load_fsdp_param_and_grad(module=self.critic_module,
                                       device_id=torch.cuda.current_device(),
                                       load_grad=self._is_offload_grad)
        
        # Compare weights after reset
        print(f"[LAYER_RESET_DEBUG] Comparing critic weights after reset...")
        for layer_idx in layer_indices[:1]:  # Only check first layer to avoid spam
            layer = transformer_layers[layer_idx]
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    before_key = f"layer_{layer_idx}.{name}"
                    if before_key in weights_before:
                        before_weight = weights_before[before_key]
                        after_weight = param.data.detach().cpu()
                        
                        # Check if weights actually changed
                        weight_diff = torch.norm(after_weight - before_weight).item()
                        weights_equal = torch.allclose(after_weight, before_weight, atol=1e-6)
                        
                        print(f"[LAYER_RESET_DEBUG] After reset - Critic Layer {layer_idx} {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                        print(f"[LAYER_RESET_DEBUG] Weight change - Critic Layer {layer_idx} {name}: diff_norm={weight_diff:.6f}, weights_equal={weights_equal}")
                        
                        if weights_equal:
                            print(f"[LAYER_RESET_DEBUG] WARNING: Critic weights did not change for layer {layer_idx} {name}!")
                        else:
                            print(f"[LAYER_RESET_DEBUG] SUCCESS: Critic weights changed for layer {layer_idx} {name}")
                    break  # Only check one weight per layer
        
        print(f"[LAYER_RESET_DEBUG] Critic layer reset with pre-extracted reference completed")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        self.actor_update_step = 0
        self.critic_update_step = 0
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        # Debug: print all original parameter names and shapes before FSDP wrapping
        print("[DEBUG] Original model parameters before FSDP wrapping:")
        for name, param in reward_module.named_parameters():
            print(f"[DEBUG]   {name}: shape={tuple(param.shape)}")
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            forward_prefetch=False)

        # Debug: print FSDP sharding strategy after wrapping
        print(f"[DEBUG] FSDP sharding_strategy: {getattr(reward_module, 'sharding_strategy', 'N/A')}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
