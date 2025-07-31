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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import torch
from torch import nn
import torch.distributed as dist
import ray
import logging
import time

from verl import DataProto
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean, entropy_from_logits
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.single_controller.base.decorator import register, Dispatch
import torch.nn as nn
import torch.nn.functional as F
from verl.utils.redo_utils.fsdp_flat_utils import analyze_all_fsdp_zero_grad_space
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        original_param_shapes: dict = None, 
        grad_analyzer: "ray.actor.ActorHandle" = None,
        fisher_info_analyzer: "ray.actor.ActorHandle" = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.global_steps = 0  
        self.fsdp_grad_metric_enabled = True  
        print("[DEBUG][Actor] Config keys at init:", list(config.keys()) if hasattr(config, 'keys') else type(config))
        print("[DEBUG][Actor] fsdp_grad_metric_enabled in config:", getattr(config, "fsdp_grad_metric_enabled", None))
        self.logger = logging.getLogger(__name__)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.original_param_shapes = original_param_shapes
        
        # Initialize unified analyzer metrics storage
        try:
            from verl.utils.analyzer_metrics_storage import AnalyzerMetricsStorage
            import os
            
            # Use RUN_NAME from environment if available, otherwise generate timestamp-based name
            run_name = os.environ.get('RUN_NAME', f"actor_analysis_{int(time.time())}")
            
            self.analyzer_storage = AnalyzerMetricsStorage(
                base_dir="./analyzer_metrics",  # This will be auto-detected and changed
                experiment_name=run_name,
                enable_wandb=True,
                auto_detect_script_type=True  # Enable automatic path detection
            )
            print(f"[DataParallelPPOActor] Initialized analyzer metrics storage: {run_name}")
        except Exception as e:
            print(f"[DataParallelPPOActor] Warning: Failed to initialize analyzer storage: {e}")
            self.analyzer_storage = None 
        self.grad_analyzer = grad_analyzer
        self.fisher_info_analyzer = fisher_info_analyzer
        
        # Debug: Print analyzer status at initialization
        print(f"[DEBUG][DataParallelPPOActor] Initialized with grad_analyzer: {self.grad_analyzer is not None} (type: {type(self.grad_analyzer)})")
        print(f"[DEBUG][DataParallelPPOActor] Initialized with fisher_info_analyzer: {self.fisher_info_analyzer is not None} (type: {type(self.fisher_info_analyzer)})")
        if self.grad_analyzer is not None:
            print(f"[DEBUG][DataParallelPPOActor] GradientAnalyzer handle: {self.grad_analyzer}")
        if self.fisher_info_analyzer is not None:
            print(f"[DEBUG][DataParallelPPOActor] FisherInfoAnalyzer handle: {self.fisher_info_analyzer}")
        self.fisher_analysis_freq = self.config.get("fisher_analysis_freq", 1)
        self.fisher_components_to_analyze = self.config.get("fisher_components_to_analyze", None)
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        
        self._redo_step = 0
        self.redo_enabled = getattr(self.config, 'redo_enabled', False)
        self.redo_metric_freq = getattr(self.config, 'redo_metric_freq', 1)
        self.redo_reset_freq = getattr(self.config, 'redo_reset_freq', 1000)
        self.redo_mode = getattr(self.config, 'redo_mode', 'threshold')
        self.redo_tau = getattr(self.config, 'redo_tau', 0.3)
        print(f'[DEBUG][Actor] ReDo config: enabled={self.redo_enabled}, metric_freq={self.redo_metric_freq}, '
              f'reset_freq={self.redo_reset_freq}, mode={self.redo_mode}, tau={self.redo_tau}')

        self.optim_config = None
        if hasattr(self.config, 'optim'):
            self.optim_config = self.config.optim

        self.lr_scheduler = None
        if self.actor_optimizer is not None and self.optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            total_steps = self.optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = self.optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)
            self.lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.actor_optimizer,
                num_warmup_steps=num_warmup_steps
            )

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.debug_fqn_printed = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_optimizer_learning_rate(self):
        """Reset learning rates to initial values while keeping optimizer state"""
        if self.actor_optimizer is not None and self.lr_scheduler is not None:
            print("Before reset - Learning rates:", [group['lr'] for group in self.actor_optimizer.param_groups])
            
            self.lr_scheduler.last_epoch = -1
            self.lr_scheduler.step()
            
            print("After reset - Learning rates:", [group['lr'] for group in self.actor_optimizer.param_groups])

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  

                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  

                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  
                logits_rmpad = output.logits.squeeze(0)  

                logits_rmpad.div_(temperature)

                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  

                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  

            else:  
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm



    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        """
        Update the policy with the given data. Accepts global_steps from trainer for correct frequency control.
        If global_steps is None, increments internal counter. Otherwise, uses the provided value.
        """
        if 'global_steps' in data.meta_info:
            self.global_steps = data.meta_info['global_steps']
        else:
            self.global_steps += 1
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        run_fisher_analysis = self.fisher_info_analyzer and self.global_steps % self.fisher_analysis_freq == 0
        if run_fisher_analysis:
            rank = dist.get_rank()
            if rank == 0:
                # The history is intentionally not reset here to allow C_K and L_K to accumulate across steps.
                pass
            if isinstance(self.actor_module, FSDP):
                dist.barrier()

            components_to_analyze = {
                "embed_tokens": self.actor_module.model.embed_tokens,
                "final_norm": self.actor_module.model.norm,
                "lm_head": self.actor_module.lm_head,
            }
            for i, layer in enumerate(self.actor_module.model.layers):
                components_to_analyze[f"layer_{i}"] = layer
            
            component_param_ids = {name: {id(p) for p in module.parameters()} for name, module in components_to_analyze.items()}
            collected_grads_for_fisher = {name: [] for name in components_to_analyze}

        for batch_idx, data in enumerate(dataloader):
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  
                responses, response_mask = data['responses'], data['attention_mask'][:, -data['responses'].size(1):]
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=data['old_log_probs'],
                    log_prob=log_prob,
                    advantages=data['advantages'],
                    eos_mask=response_mask,
                    cliprange=self.config.clip_ratio
                )
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
                policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                kl_loss = None
                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    kl_loss = verl_F.masked_mean(log_prob - ref_log_prob, response_mask)
                    policy_loss += kl_loss * self.config.kl_coeff

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

            if run_fisher_analysis:
                for component_name, component_module in components_to_analyze.items():
                    with FSDP.summon_full_params(component_module, writeback=False, rank0_only=True, with_grads=True):
                        if rank == 0:
                            grad_dict = {}
                            for fqn, p in self.actor_module.named_parameters():
                                if id(p) in component_param_ids[component_name]:
                                    if p.grad is not None:
                                        clean_fqn = fqn.replace('_fsdp_wrapped_module.', '').replace('._fsdp_wrapped_module', '')
                                        grad_dict[clean_fqn] = p.grad.clone().cpu()
                            if grad_dict:
                                collected_grads_for_fisher[component_name].append(grad_dict)

            # Capture the learning rate that will be used for this optimizer step.
            # This is critical because some schedulers might update the LR after the step.
            current_lr = self.actor_optimizer.param_groups[0]['lr']

            if isinstance(self.actor_module, FSDP):
                self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            self.actor_optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                new_lr = self.actor_optimizer.param_groups[0]['lr']

            with torch.no_grad():
                metrics['actor/pg_loss'] = pg_loss.item()
                metrics['actor/entropy_loss'] = entropy_loss.item()
                metrics['actor/pg_clipfrac'] = pg_clipfrac.item()
                metrics['actor/ppo_kl'] = ppo_kl.item()
                if kl_loss is not None:
                    metrics['actor/kl_loss'] = kl_loss.item()

        if run_fisher_analysis and rank == 0:
            # --- Trigger asynchronous analysis for each component ---
            analysis_tasks = []
            for component_name, per_mini_batch_grads in collected_grads_for_fisher.items():
                if per_mini_batch_grads:
                    # print(f"[Fisher Debug] Finished collecting grads for {component_name}. Sending {len(per_mini_batch_grads)} mini-batch grads to analyzer.")
                    # print(f"[Fisher Debug] Using current_lr={current_lr} for l_k calculation.")
                    task = self.fisher_info_analyzer.analyze_component_grads.remote(
                        identifier='actor',
                        component_name=component_name,
                        per_micro_batch_grads=per_mini_batch_grads,
                        original_param_shapes=self.original_param_shapes,
                        micro_batch_size=self.config.ppo_mini_batch_size,
                        current_lr=current_lr,
                        global_step=self.global_steps
                    )
                    analysis_tasks.append(task)
            
            # --- Wait for all analyses to complete, then aggregate and log stats ---
            if analysis_tasks:
                ray.get(analysis_tasks) # Ensure all component analyses are done
                fisher_stats_ref = self.fisher_info_analyzer.get_aggregated_stats.remote(identifier='actor')
                fisher_stats = ray.get(fisher_stats_ref)
                print(f"[DP Actor Debug] Received Fisher Stats: {fisher_stats}")
                if fisher_stats:
                    # Add detailed Fisher Info logging
                    self.logger.info(f"--- 📈 Fisher Information Analysis Results (Step {self.global_steps}) ---")
                    
                    # Log global Fisher Info metrics
                    global_fisher = fisher_stats.get('global', {})
                    if global_fisher:
                        self.logger.info(f"Global Fisher Metrics:")
                        for metric_name, value in global_fisher.items():
                            if isinstance(value, (int, float)):
                                self.logger.info(f"  - {metric_name}: {value:.6g}")
                    
                    # Log per-component Fisher Info metrics
                    components_fisher = fisher_stats.get('components', {})
                    if components_fisher:
                        self.logger.info(f"--- Per-Component Fisher Info Breakdown ---")
                        for component_name, comp_stats in sorted(components_fisher.items()):
                            self.logger.info(f"  - Component: {component_name}")
                            params_stats = comp_stats.get('params', {})
                            if params_stats:
                                for param_name, param_metrics in sorted(params_stats.items()):
                                    short_name = '.'.join(param_name.split('.')[-3:])
                                    c_k = param_metrics.get('c_k', 0.0)
                                    l_k = param_metrics.get('l_k', 0.0)
                                    sigma_max = param_metrics.get('sigma_max', 0.0)
                                    sigma_min = param_metrics.get('sigma_min', 0.0)
                                    self.logger.info(f"    - {short_name:<40} | c_k: {c_k:.4f} | l_k: {l_k:.6g} | σ_max: {sigma_max:.6g} | σ_min: {sigma_min:.6g}")
                    
                    self.logger.info("-" * 60)
                    
                    # Update metrics for logging/tracking
                    metrics.update(fisher_stats)
                    
                    # Add summary Fisher metrics to main metrics dict
                    if global_fisher:
                        for key, value in global_fisher.items():
                            if isinstance(value, (int, float)):
                                metrics[f'actor/fisher_{key}'] = value

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_fsdp = isinstance(self.actor_module, FSDP)
        final_stats = None
        zero_grad_stats = None

        if self.grad_analyzer is not None and self.global_steps % self.config.get("redo_analysis_freq", 1) == 0:
            if rank == 0:
                print(f"[INFO][Actor][Step {self.global_steps}] Resetting remote gradient analyzer state.")
                ray.get(self.grad_analyzer.reset.remote(identifier='actor'))

            components_to_analyze = {
                "embed_tokens": self.actor_module.model.embed_tokens,
                "final_norm": self.actor_module.model.norm,
                "lm_head": self.actor_module.lm_head,
            }
            for i, layer in enumerate(self.actor_module.model.layers):
                components_to_analyze[f"layer_{i}"] = layer

            component_param_ids = {name: {id(p) for p in module.parameters()} for name, module in components_to_analyze.items()}
            for component_name, component_module in components_to_analyze.items():
                if rank == 0:
                    self.logger.info(f"--- Analyzing component: {component_name} ---")

                with FSDP.summon_full_params(component_module, writeback=False, rank0_only=True, with_grads=True):
                    if rank == 0:
                        grad_state_dict = {
                            f"model.{fqn.replace('._fsdp_wrapped_module', '')}": param.grad.cpu()
                            for fqn, param in self.actor_module.model.named_parameters()
                            if id(param) in component_param_ids[component_name] and param.grad is not None
                        }

                        if grad_state_dict:
                            self.grad_analyzer.analyze_component_gradients.remote(
                                identifier='actor',
                                component_name=component_name,
                                gradients=grad_state_dict,
                                original_param_shapes=self.original_param_shapes,
                                tau=self.redo_tau,
                                verbose=True
                            )
                        else:
                            print(f"[INFO][Actor][Step {self.global_steps}] No gradients found for component {component_name}.")
                    
        if rank == 0:
            print(f"[INFO][Actor][Step {self.global_steps}] Clearing CUDA cache on Rank 0 to free memory for gradient gathering.")
            torch.cuda.empty_cache()

        if is_fsdp:
            # All ranks must wait for rank 0 to finish before proceeding.
            dist.barrier()

        with torch.no_grad():
            # Debug: Check Gradient Analyzer conditions
            print(f"[DEBUG][Actor][Step {self.global_steps}] Gradient Analyzer check: grad_analyzer={self.grad_analyzer is not None}, freq_check={self.global_steps % self.config.get('redo_analysis_freq', 1) == 0}, redo_analysis_freq={self.config.get('redo_analysis_freq', 1)}")
            
            # Check if any analysis should be performed
            should_analyze_gradients = (self.grad_analyzer is not None and 
                                      self.global_steps % self.config.get("redo_analysis_freq", 1) == 0)
            should_analyze_fisher = (self.fisher_info_analyzer is not None and 
                                   self.global_steps % self.fisher_analysis_freq == 0)
            
            if should_analyze_gradients or should_analyze_fisher:
                print(f"[INFO][Actor][Step {self.global_steps}] 🚀 STARTING Parallel Analysis workflow")
                print(f"[INFO][Actor][Step {self.global_steps}] Gradient Analysis: {should_analyze_gradients}, Fisher Analysis: {should_analyze_fisher}")
                
                # Initialize futures for parallel execution
                grad_analysis_futures = []
                fisher_analysis_future = None
                
                # Step 1: Reset analyzers in parallel on rank 0
                if rank == 0:
                    reset_futures = []
                    if should_analyze_gradients:
                        print(f"[INFO][Actor][Step {self.global_steps}] Resetting gradient analyzer state.")
                        reset_futures.append(self.grad_analyzer.reset.remote(identifier='actor'))
                    if should_analyze_fisher:
                        print(f"[INFO][Actor][Step {self.global_steps}] Resetting Fisher analyzer state.")
                        reset_futures.append(self.fisher_info_analyzer.reset.remote(identifier='actor'))
                    
                    # Wait for all resets to complete
                    if reset_futures:
                        ray.get(reset_futures)
                        print(f"[INFO][Actor][Step {self.global_steps}] All analyzer resets completed.")

                # Synchronize all ranks to ensure reset is complete before analysis begins.
                if is_fsdp:
                    dist.barrier()

                if should_analyze_gradients:
                    # Define components to analyze. This must match the model architecture.
                    # Assumes a standard HuggingFace transformer structure like Llama/Qwen.
                    components_to_analyze = {
                        "embed_tokens": self.actor_module.model.embed_tokens,
                        "final_norm": self.actor_module.model.norm,
                        "lm_head": self.actor_module.lm_head,
                    }
                    # Add all transformer layers.
                    for i, layer in enumerate(self.actor_module.model.layers):
                        components_to_analyze[f"layer_{i}"] = layer

                    print(f"[INFO][Actor][Step {self.global_steps}] Starting parallel component analysis for {len(components_to_analyze)} components")
                    
                    # Step 2: Analyze each component and collect futures for parallel execution
                    for component_name, component_module in components_to_analyze.items():
                        if rank == 0:
                            self.logger.info(f"--- Analyzing component: {component_name} ---")

                        # Get the set of parameter IDs for the current component for efficient lookup.
                        component_param_ids = {id(p) for p in component_module.parameters()}

                        # Summon gradients for only this component. This is memory-safe.
                        with FSDP.summon_full_params(component_module, writeback=False, rank0_only=True, with_grads=True):
                            if rank == 0:
                                # We must use FQNs that match the keys in `original_param_shapes`.
                                # We iterate over the full model's parameters to get the FQN,
                                # but only include the ones that are part of the current component.
                                # FSDP inserts `_fsdp_wrapped_module` into parameter names. We must remove
                                # it. We also prepend `model.` to match the keys in `original_param_shapes`.
                                grad_state_dict = {
                                    f"model.{fqn.replace('._fsdp_wrapped_module', '')}": param.grad.cpu()
                                    for fqn, param in self.actor_module.model.named_parameters()
                                    if id(param) in component_param_ids and param.grad is not None
                                }

                                if grad_state_dict:
                                    # Fire-and-forget the analysis for this component.
                                    self.grad_analyzer.analyze_component_gradients.remote(
                                        identifier='actor',
                                        component_name=component_name,
                                        gradients=grad_state_dict,
                                        original_param_shapes=self.original_param_shapes,
                                        tau=self.redo_tau,
                                        verbose=True
                                    )
                                else:
                                    print(f"[INFO][Actor][Step {self.global_steps}] No gradients found for component {component_name}.")
                        
                        # Barrier to ensure all ranks are done with a component before the next.
                        if is_fsdp:
                            dist.barrier()

                # Step 3: Start Fisher analysis in parallel (if enabled)
                if should_analyze_fisher and rank == 0:
                    print(f"[INFO][Parallel Analysis][Step {self.global_steps}] 🔄 Starting Fisher analysis")
                    fisher_analysis_future = self.fisher_info_analyzer.get_aggregated_stats.remote(identifier='actor')
                
                # Step 4: Get aggregated results from gradient analyzer (if enabled)
                # Step 5: Start parallel analysis aggregation
                grad_stats_future = None
                fisher_stats_future = None
                
                if rank == 0:
                    print(f"[INFO][Parallel Analysis][Step {self.global_steps}] 🚀 Starting parallel execution of Gradient and Fisher analyzers")
                    
                    # Start gradient analysis aggregation (non-blocking) if enabled
                    if should_analyze_gradients:
                        print(f"[INFO][Actor][Step {self.global_steps}] Getting aggregated gradient analysis results.")
                        grad_stats_future = self.grad_analyzer.get_aggregated_stats.remote(identifier='actor', verbose=True)
                    
                    # Start Fisher analysis in parallel if enabled
                    if run_fisher_analysis and analysis_tasks:
                        print(f"[INFO][Parallel Analysis][Step {self.global_steps}] 🔄 Starting Fisher analysis in parallel")
                        # Wait for Fisher component analyses to complete
                        ray.get(analysis_tasks)
                        # Start Fisher aggregation in parallel with gradient analysis
                        fisher_stats_future = self.fisher_info_analyzer.get_aggregated_stats.remote(identifier='actor')
                    
                # Step 5: Process results from both analyzers in parallel
                zero_gradspace_ratio_avg = 0.0
                
                if rank == 0:
                    # Collect all futures for parallel waiting
                    all_futures = []
                    future_types = []
                    
                    if grad_stats_future is not None:
                        all_futures.append(grad_stats_future)
                        future_types.append('gradient')
                    
                    if fisher_analysis_future is not None:
                        all_futures.append(fisher_analysis_future)
                        future_types.append('fisher')
                    
                    if all_futures:
                        print(f"[INFO][Parallel Analysis][Step {self.global_steps}] ⏳ Waiting for {len(all_futures)} analysis tasks to complete")
                        
                        # Wait for all analysis tasks to complete in parallel
                        results = ray.get(all_futures)
                        
                        # Process gradient analysis results
                        for i, (result, future_type) in enumerate(zip(results, future_types)):
                            if future_type == 'gradient':
                                final_stats = result
                                if not final_stats:
                                    self.logger.warning(f"[Actor][Step {self.global_steps}] Failed to get zero-grad analysis results.")
                                    zero_gradspace_ratio_avg = 0.0
                                else:
                                    global_stats = final_stats.get('__global__', {})
                                    global_ratio = global_stats.get('ratio', 0.0)
                                    zero_gradspace_ratio_avg = global_ratio
                                    
                                    self.logger.info(f"--- 📊 Gradient Analysis Results (Step {self.global_steps}, Tau: {self.redo_tau}) ---")
                                    self.logger.info(f"Global Dormant Neuron Ratio: {global_ratio:.4%}")
                                    
                                    # Store detailed metrics to unified storage
                                    if self.analyzer_storage:
                                        try:
                                            self.analyzer_storage.store_gradient_metrics(
                                                step=self.global_steps,
                                                gradient_stats=final_stats,
                                                tau=self.redo_tau,
                                                additional_info={
                                                    "rank": rank,
                                                    "device": str(self.device) if hasattr(self, 'device') else "unknown"
                                                }
                                            )
                                            print(f"[INFO][Storage] Gradient metrics saved to JSON for step {self.global_steps}")
                                        except Exception as e:
                                            print(f"[ERROR][Storage] Failed to save gradient metrics: {e}")
                                    
                                    component_stats = final_stats.get('components', {})
                                    if component_stats:
                                        self.logger.info(f"--- Per-Component & Per-Matrix Breakdown ---")
                                        for component_name, comp_stats in sorted(component_stats.items()):
                                            self.logger.info(f"  - Component: {component_name} ({comp_stats.get('ratio', 0.0):.4%})")
                                            matrix_stats = comp_stats.get('matrices', {})
                                            if not matrix_stats:
                                                self.logger.info("    (No eligible matrices found in this component)")
                                            else:
                                                for matrix_name, mat_stats in sorted(matrix_stats.items()):
                                                    short_name = '.'.join(matrix_name.split('.')[-4:])
                                                    min_norm = mat_stats.get('min_row_norm', 0.0)
                                                    avg_norm = mat_stats.get('avg_row_norm', 0.0)
                                                    max_norm = mat_stats.get('max_row_norm', 0.0)
                                                    self.logger.info(f"    - {short_name:<40} | Ratio: {mat_stats.get('ratio', 0.0):.4%} | Norms (min/avg/max): {min_norm:.4e} / {avg_norm:.4e} / {max_norm:.4e}")
                                    self.logger.info("-" * 60)
                            
                            elif future_type == 'fisher':
                                fisher_stats = result
                                if fisher_stats:
                                    self.logger.info(f"--- 📈 Fisher Information Analysis Results (Step {self.global_steps}) ---")
                                    
                                    # Store detailed Fisher metrics to unified storage
                                    if self.analyzer_storage:
                                        try:
                                            self.analyzer_storage.store_fisher_metrics(
                                                step=self.global_steps,
                                                fisher_stats=fisher_stats,
                                                additional_info={
                                                    "rank": rank,
                                                    "device": str(self.device) if hasattr(self, 'device') else "unknown"
                                                }
                                            )
                                            print(f"[INFO][Storage] Fisher metrics saved to JSON for step {self.global_steps}")
                                        except Exception as e:
                                            print(f"[ERROR][Storage] Failed to save Fisher metrics: {e}")
                                    
                                    # Log detailed Fisher statistics
                                    for key, value in fisher_stats.items():
                                        if isinstance(value, (int, float)):
                                            self.logger.info(f"  {key}: {value:.6f}")
                                    
                                    print(f"[INFO][Parallel Analysis][Step {self.global_steps}] ✅ Fisher analysis completed successfully")
                                else:
                                    self.logger.warning(f"[Actor][Step {self.global_steps}] Failed to get Fisher analysis results.")
                        
                        print(f"[INFO][Parallel Analysis][Step {self.global_steps}] 🎉 All parallel analysis tasks completed")
                    
                    # Set the metrics with the correct value
                    metrics['actor/zero_gradspace_ratio'] = zero_gradspace_ratio_avg
                    print(f"[ZeroGradV2-Metrics][After Optim Step][Step {self.global_steps}] Aggregated Zero Grad Space Ratio: {zero_gradspace_ratio_avg:.4f}")
                    print(f"[INFO][Parallel Analysis][Step {self.global_steps}] 🎉 Parallel analysis execution completed")
                else:
                    # Non-rank 0 processes set zero
                    metrics['actor/zero_gradspace_ratio'] = 0.0
            else:
                # When no analysis is performed, set zero
                print(f"[INFO][Actor][Step {self.global_steps}] ❌ SKIPPING Analysis - conditions not met (grad: {should_analyze_gradients}, fisher: {should_analyze_fisher})")
                if rank == 0:
                    metrics['actor/zero_gradspace_ratio'] = 0.0
                    print(f"[ZeroGradV2-Metrics][After Optim Step][Step {self.global_steps}] Aggregated Zero Grad Space Ratio: 0.0000")
        # --- END FSDP analysis/reset ---

        self.actor_optimizer.zero_grad()
        return metrics
