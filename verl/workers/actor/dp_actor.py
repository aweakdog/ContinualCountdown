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
from typing import Iterable, Tuple, List, Dict, Any
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
from verl.utils.redo_utils.ck_based_reset_manager import create_ck_based_reset_manager
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
        
        self.redo_tau = getattr(self.config, 'redo_tau', 0.3)

        # Initialize C_K-based reset manager
        self.ck_reset_manager = None
        
        # Check for ck_reset in different possible locations
        ck_reset_config = None
        
        # Try actor.ck_reset path (most likely location)
        if hasattr(self.config, 'actor') and hasattr(self.config.actor, 'ck_reset'):
            ck_reset_config = self.config.actor.ck_reset
        # Try direct ck_reset path
        elif hasattr(self.config, 'ck_reset'):
            ck_reset_config = self.config.ck_reset
        # Try _content paths
        elif hasattr(self.config, '_content') and hasattr(self.config._content, 'get'):
            ck_reset_config = self.config._content.get('ck_reset')
            if not ck_reset_config and 'actor' in self.config._content:
                actor_config = self.config._content.get('actor', {})
                if hasattr(actor_config, 'get'):
                    ck_reset_config = actor_config.get('ck_reset')
        # Try get method
        elif hasattr(self.config, 'get'):
            ck_reset_config = self.config.get('ck_reset')
        
        if ck_reset_config and ck_reset_config.get('enable_reset', False):
            from verl.utils.redo_utils.ck_based_reset_manager import create_ck_based_reset_manager
            self.ck_reset_manager = create_ck_based_reset_manager(ck_reset_config)
            print(f"[DataParallelPPOActor] Initialized C_K-based reset manager with strategy: {ck_reset_config.get('reset_strategy', 'ck_guided')}")
        else:
            print(f"[DataParallelPPOActor] C_K-based reset manager disabled - config not found or disabled")
        
        # Initialize Ray shared state manager for reset synchronization
        try:
            from verl.utils.redo_utils.shared_reset_state import SharedResetStateManager
            self._shared_reset_manager = SharedResetStateManager()
            print(f"[DataParallelPPOActor] Using Ray shared state for reset synchronization")
        except Exception as e:
            print(f"[DataParallelPPOActor] Failed to initialize Ray shared state: {e}")
            raise RuntimeError(f"Ray shared state synchronization is required but failed to initialize: {e}")
        
        # Initialize optimizer configuration
        self._init_optimizer_config()
        
        # Initialize entropy computation function
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.debug_fqn_printed = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_ck_reset_status(self, global_step: int) -> Dict[str, Any]:
        """
        Check if CK reset should be performed and return status with C_K weights.
        
        Args:
            global_step: Current global step
            
        Returns:
            Dictionary with reset status and C_K weights
        """
        if not hasattr(self, 'ck_reset_manager') or self.ck_reset_manager is None:
            return {'should_reset': False, 'layer_ck_weights': {}}
        
        try:
            # Check if we have Fisher stats from the last analysis
            fisher_stats_for_reset = None
            if hasattr(self, 'fisher_detailed_stats'):
                fisher_stats_for_reset = self.fisher_detailed_stats
            
            # Check if reset should be performed and calculate C_K weights if needed
            should_reset_ck, _ = self.ck_reset_manager.should_reset_with_ck_analysis(
                global_step, fisher_stats_for_reset
            )
            
            # Calculate C_K weights if reset is needed and we have Fisher stats
            layer_ck_weights = {}
            if should_reset_ck and fisher_stats_for_reset and self.ck_reset_manager.reset_strategy == 'ck_guided':
                layer_ck_weights = self.ck_reset_manager.calculate_layer_ck_weights(
                    fisher_stats_for_reset, self.original_param_shapes or {}
                )
            
            return {
                'should_reset': should_reset_ck,
                'layer_ck_weights': layer_ck_weights or {}
            }
            
        except Exception as e:
            print(f"[CK_RESET_ERROR] Actor: Error in get_ck_reset_status: {e}")
            return {'should_reset': False, 'layer_ck_weights': {}}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_model_with_ck_analysis(self, layer_ck_weights: Dict[int, float], 
                                   global_step: int, ref_worker) -> Dict[str, Any]:
        """
        Perform CK-based layer reset using safe traditional calling pattern.
        
        Args:
            layer_ck_weights: C_K weighted values for each layer
            global_step: Current global step
            ref_worker: Reference worker to get layer weights from
            
        Returns:
            Dictionary with reset results and metrics
        """
        print(f"[CK_RESET_DEBUG] Actor: reset_model_with_ck_analysis called with ref_worker type: {type(ref_worker).__name__ if ref_worker else 'None'}")
        
        if not hasattr(self, 'ck_reset_manager') or self.ck_reset_manager is None:
            print(f"[CK_RESET_ERROR] Actor: No CK reset manager available")
            return {'reset_params_count': 0, 'reset_layers': []}
        
        try:
            # Get layers to reset based on C_K weights
            if not layer_ck_weights:
                return {'reset_params_count': 0, 'reset_layers': []}
            
            # Get transformer layers count
            transformer_layers = self.ck_reset_manager.get_transformer_layers(self.actor_module)
            total_layers = len(transformer_layers)
            
            # Select layers to reset using C_K weights
            layers_to_reset = self.ck_reset_manager.select_layers_to_reset(
                total_layers, layer_ck_weights, global_step
            )
            
            if not layers_to_reset:
                return {'reset_params_count': 0, 'reset_layers': []}
            
            print(f"[CK_RESET_DEBUG] Actor: Selected layers to reset: {layers_to_reset}")
            
            # Use safe traditional calling pattern: extract then apply
            print(f"[CK_RESET_DEBUG] Actor: Extracting reference layers from ref_worker")
            
            # Extract reference layers (safe trainer->worker call)
            ref_layer_state_dict_result = ref_worker.extract_layers_for_reset(layers_to_reset)
            
            # Handle RayWorkerGroup result (list) vs direct worker result (dict)
            if isinstance(ref_layer_state_dict_result, list):
                ref_layer_state_dict = ref_layer_state_dict_result[0] if ref_layer_state_dict_result else {}
            else:
                ref_layer_state_dict = ref_layer_state_dict_result or {}
            
            print(f"[CK_RESET_DEBUG] Actor: Received {len(ref_layer_state_dict)} reference parameters")
            
            if not ref_layer_state_dict:
                return {'reset_params_count': 0, 'reset_layers': []}
            
            # Apply parameters using traditional reset method (safe)
            reset_param_names = self.ck_reset_manager.reset_model_layers_from_ref(
                model=self.actor_module,
                ref_layer_state_dict=ref_layer_state_dict,
                reset_k_first=0,  # Not used in CK guided mode
                reset_k_last=0    # Not used in CK guided mode
            )
            
            print(f"[CK_RESET_DEBUG] Actor: Successfully reset {len(reset_param_names)} parameters")
            
            return {
                'reset_params_count': len(reset_param_names),
                'reset_layers': layers_to_reset
            }
            
        except Exception as e:
            import traceback
            print(f"[CK_RESET_ERROR] Actor: Failed to perform CK reset: {e}")
            print(f"[CK_RESET_ERROR] Actor: Traceback: {traceback.format_exc()}")
            return {'reset_params_count': 0, 'reset_layers': []}
    
    def _save_actor_reset_layers(self, selected_layers: List[int], global_step: int):
        """
        Save selected reset layers for critic synchronization via Ray shared state.
        """
        try:
            reset_info = {
                'layers': selected_layers,
                'step': global_step,
                'timestamp': time.time()
            }
            ray.put(reset_info)
            print(f"[ACTOR_RESET_SYNC] Saved reset layers {selected_layers} for step {global_step} via Ray")
        except Exception as e:
            print(f"[ACTOR_RESET_SYNC] Failed to save reset layers: {e}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_ck_reset_status(self, global_step: int) -> Dict[str, Any]:
        """
        Check if CK reset should be performed and return status with C_K weights.
        
        Args:
            global_step: Current global step
            
        Returns:
            Dictionary with reset status and C_K weights
        """
        if not hasattr(self, 'ck_reset_manager') or self.ck_reset_manager is None:
            return {'should_reset': False, 'layer_ck_weights': {}}
        
        try:
            # Check if we have Fisher stats from the last analysis
            fisher_stats_for_reset = None
            if hasattr(self, 'fisher_detailed_stats'):
                fisher_stats_for_reset = self.fisher_detailed_stats
            
            # Check if reset should be performed and calculate C_K weights if needed
            should_reset_ck, _ = self.ck_reset_manager.should_reset_with_ck_analysis(
                global_step, fisher_stats_for_reset
            )
            
            # Calculate C_K weights if reset is needed and we have Fisher stats
            layer_ck_weights = {}
            if should_reset_ck and fisher_stats_for_reset and self.ck_reset_manager.reset_strategy == 'ck_guided':
                layer_ck_weights = self.ck_reset_manager.calculate_layer_ck_weights(
                    fisher_stats_for_reset, self.original_param_shapes or {}
                )
            
            return {
                'should_reset': should_reset_ck,
                'layer_ck_weights': layer_ck_weights or {}
            }
            
        except Exception as e:
            print(f"[CK_RESET_ERROR] Actor: Error in get_ck_reset_status: {e}")
            return {'should_reset': False, 'layer_ck_weights': {}}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_layer_parameters(self, layer_params: Dict[str, torch.Tensor]) -> int:
        """
        Apply layer parameters to actor model.
        FSDP-aware: Only rank 0 applies parameters to avoid conflicts and reduce memory usage.
        
        Args:
            layer_params: Dictionary of parameters to apply
            
        Returns:
            Number of parameters applied
        """
        if not layer_params:
            return 0
        
        # Only apply on rank 0 to avoid FSDP conflicts and reduce memory usage
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank != 0:
            return 0
        
        print(f"[CK_RESET_DEBUG] Actor rank {rank}: Applying {len(layer_params)} layer parameters")
        
        # Get actor base model
        actor_base = self.actor_module._fsdp_wrapped_module if hasattr(self.actor_module, '_fsdp_wrapped_module') else self.actor_module
        
        # Find transformer layers
        actor_layers = None
        for pattern in ['model.layers', 'transformer.h', 'transformer.layers', 'layers']:
            try:
                actor_layers = actor_base
                for attr in pattern.split('.'):
                    actor_layers = getattr(actor_layers, attr)
                if isinstance(actor_layers, (list, torch.nn.ModuleList)):
                    break
            except AttributeError:
                continue
        
        if actor_layers is None:
            return 0
        
        # Apply parameters
        applied_count = 0
        for key, ref_param_data in layer_params.items():
            # Parse layer index and parameter name from key
            parts = key.split('.', 1)
            if len(parts) != 2 or not parts[0].startswith('layer_'):
                continue
                
            layer_idx = int(parts[0].replace('layer_', ''))
            param_name = parts[1]
            
            if layer_idx < len(actor_layers):
                actor_layer = actor_layers[layer_idx]
                for name, param in actor_layer.named_parameters():
                    if name == param_name:
                        with torch.no_grad():
                            param.data.copy_(ref_param_data.to(param.device))
                        applied_count += 1
                        break
        
        return applied_count

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_model_with_ck_analysis(self, layer_ck_weights: Dict[int, float], 
                                   global_step: int, ref_worker) -> Dict[str, Any]:
        """
        Perform CK-based layer reset with reference worker passed from trainer.
        
        Args:
            layer_ck_weights: C_K weighted values for each layer
            global_step: Current global step
            ref_worker: Reference worker to get layer weights from
            
        Returns:
            Dictionary with reset results and metrics
        """
        print(f"[CK_RESET_DEBUG] Actor: reset_model_with_ck_analysis called with ref_worker type: {type(ref_worker).__name__ if ref_worker else 'None'}")
        
        if not hasattr(self, 'ck_reset_manager') or self.ck_reset_manager is None:
            print(f"[CK_RESET_ERROR] Actor: No CK reset manager available")
            return {'reset_params_count': 0, 'reset_layers': []}
        
        try:
            # Get layers to reset
            if not layer_ck_weights:
                return {'reset_params_count': 0, 'reset_layers': []}
            
            sorted_layers = sorted(layer_ck_weights.items(), key=lambda x: x[1], reverse=True)
            reset_count = min(self.ck_reset_manager.reset_k_layers, len(sorted_layers))
            layers_to_reset = [layer_idx for layer_idx, _ in sorted_layers[:reset_count]]
            
            # Get parameters from reference worker using proper RayWorkerGroup method
            print(f"[CK_RESET_DEBUG] Actor: Calling get_layer_parameters on ref_worker for layers: {layers_to_reset}")
            
            # Debug: List all available methods on the ref_worker
            try:
                print(f"[CK_RESET_DEBUG] Actor: ref_worker type: {type(ref_worker).__name__}")
                print(f"[CK_RESET_DEBUG] Actor: ref_worker has _workers: {hasattr(ref_worker, '_workers')}")
                
                if hasattr(ref_worker, '_workers') and ref_worker._workers:
                    print(f"[CK_RESET_DEBUG] Actor: Number of workers in ref_worker: {len(ref_worker._workers)}")
                    
                    # Check all workers in the group
                    for i, worker in enumerate(ref_worker._workers):
                        worker_type = type(worker).__name__
                        print(f"[CK_RESET_DEBUG] Actor: Worker {i}: type={worker_type}")
                        
                        # Check if this worker has reference methods
                        ref_methods = [attr for attr in dir(worker) if 'ref_' in attr.lower() and not attr.startswith('_')]
                        if ref_methods:
                            print(f"[CK_RESET_DEBUG] Actor: Worker {i} ref methods: {ref_methods[:10]}")
                        
                        # Check for get_layer_parameters
                        has_get_layer = hasattr(worker, 'get_layer_parameters')
                        print(f"[CK_RESET_DEBUG] Actor: Worker {i} has get_layer_parameters: {has_get_layer}")
                        
                        # Check worker role if available
                        if hasattr(worker, 'role'):
                            print(f"[CK_RESET_DEBUG] Actor: Worker {i} role: {getattr(worker, 'role', 'unknown')}")
                        if hasattr(worker, '_is_ref'):
                            print(f"[CK_RESET_DEBUG] Actor: Worker {i} _is_ref: {getattr(worker, '_is_ref', 'unknown')}")
                else:
                    print(f"[CK_RESET_DEBUG] Actor: ref_worker has no _workers or empty _workers list")
            except Exception as debug_e:
                print(f"[CK_RESET_DEBUG] Actor: Error debugging ref_worker: {debug_e}")
            
            # Use execute_all_async to ensure we get parameters from the actual reference worker
            layer_params_futures = ref_worker.execute_all_async('ref_get_layer_parameters', layers_to_reset)
            layer_params_results = ray.get(layer_params_futures)
            
            # Find the first non-empty result (from the actual reference worker)
            layer_params = {}
            for result in layer_params_results:
                if result:  # Non-empty dictionary
                    layer_params = result
                    break
            print(f"[CK_RESET_DEBUG] Actor: Received {len(layer_params)} layer parameters from ref_worker")
            
            if not layer_params:
                return {'reset_params_count': 0, 'reset_layers': []}
            
            # Apply parameters to actor
            applied_count = self.apply_layer_parameters(layer_params)
            
            return {
                'reset_params_count': applied_count,
                'reset_layers': layers_to_reset
            }
            
        except Exception as e:
            import traceback
            print(f"[CK_RESET_ERROR] Actor: Failed to perform CK reset: {e}")
            print(f"[CK_RESET_ERROR] Actor: Traceback: {traceback.format_exc()}")
            return {'reset_params_count': 0, 'reset_layers': []}

    def _init_optimizer_config(self):
        """Initialize optimizer configuration."""
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

    def _collect_fisher_gradients_per_sample(self, mini_batch, components_to_analyze, component_param_ids, collected_grads_for_fisher):
        """
        在训练前收集每个样本独立的梯度用于Fisher分析
        注意：在分布式训练中，每个rank只看到部分样本，需要所有rank都参与收集
        """
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        if rank == 0:
            print(f"[FisherCollection][Step {self.global_steps}] Starting per-sample gradient collection for Fisher analysis")
            print(f"[FisherCollection][Step {self.global_steps}] Distributed setup: rank {rank}/{world_size}, world_size={world_size}")
        
        # 将mini-batch拆分成单个样本
        batch_size = mini_batch['responses'].size(0)
        total_samples = batch_size * world_size  # 全局样本总数
        
        if rank == 0:
            print(f"[FisherCollection][Step {self.global_steps}] Local batch size: {batch_size} samples, Total samples across all ranks: {total_samples}")
        
        for sample_idx in range(batch_size):
            # 提取单个样本
            single_sample = {
                key: value[sample_idx:sample_idx+1] for key, value in mini_batch.items()
            }
            single_sample = {k: v.cuda() for k, v in single_sample.items()}
            
            # 清零梯度
            self.actor_optimizer.zero_grad()
            
            # 单样本前向传播
            responses, response_mask = single_sample['responses'], single_sample['attention_mask'][:, -single_sample['responses'].size(1):]
            entropy, log_prob = self._forward_micro_batch(micro_batch=single_sample, temperature=1.0)
            
            # 计算loss（不除以gradient_accumulation，因为这是单样本）
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                old_log_prob=single_sample['old_log_probs'],
                log_prob=log_prob,
                advantages=single_sample['advantages'],
                eos_mask=response_mask,
                cliprange=self.config.clip_ratio
            )
            entropy_loss = verl_F.masked_mean(entropy, response_mask)
            policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff
            
            if self.config.use_kl_loss:
                ref_log_prob = single_sample['ref_log_prob']
                kl_loss = verl_F.masked_mean(log_prob - ref_log_prob, response_mask)
                policy_loss += kl_loss * self.config.kl_coeff
            
            # 反向传播（获得单样本梯度）
            policy_loss.backward()
            
            # 收集单样本梯度
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
                            current_count = len(collected_grads_for_fisher[component_name])
                            if sample_idx % 8 == 0:  # 每8个样本打印一次
                                print(f"[FisherCollection][Step {self.global_steps}][Rank {rank}] Sample {sample_idx}: collected gradients for component '{component_name}', local count: {current_count}")
        
        # 同步所有rank，确保收集完成
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # 汇总所有rank的梯度到rank 0
        if world_size > 1:
            # 创建临时存储用于汇总
            if rank == 0:
                all_rank_grads = {component_name: [] for component_name in collected_grads_for_fisher}
            
            # 每个rank将自己的梯度发送给rank 0
            for component_name in collected_grads_for_fisher:
                local_grads = collected_grads_for_fisher[component_name]
                
                if rank == 0:
                    # rank 0收集所有rank的梯度
                    all_rank_grads[component_name].extend(local_grads)  # 先添加自己的
                    
                    # 接收其他rank的梯度
                    for src_rank in range(1, world_size):
                        # 这里需要使用Ray的分布式通信或其他方式
                        # 暂时先保持当前逻辑，但添加警告
                        pass
                    
            
            # 更新collected_grads_for_fisher为汇总结果
            if rank == 0:
                collected_grads_for_fisher.update(all_rank_grads)
        
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
        
        # Check if any analysis should be performed (defined early for use in mini-batch loop)
        gradient_analysis_enabled = self.config.get("enable_gradient_analysis", True)
        gradient_analysis_freq = self.config.get("gradient_analysis_freq", 1)
        fisher_analysis_enabled = self.config.get("enable_fisher_analysis", True)
        fisher_analysis_freq = self.config.get("fisher_analysis_freq", 1)
        
        should_analyze_gradients = (gradient_analysis_enabled and 
                                  self.grad_analyzer is not None and 
                                  self.global_steps % gradient_analysis_freq == 0)
        should_analyze_fisher = (fisher_analysis_enabled and 
                               self.fisher_info_analyzer is not None and 
                               self.global_steps % fisher_analysis_freq == 0)
        
        #should_analyze_gradients = False
        #should_analyze_fisher = False
        # Legacy variable for backward compatibility
        run_fisher_analysis = should_analyze_fisher
        
        # Initialize gradient collection for gradient analysis (only if enabled)
        collected_grads_for_gradient = {}
        if should_analyze_gradients:
            grad_components_to_analyze = {
                "embed_tokens": self.actor_module.model.embed_tokens,
                "final_norm": self.actor_module.model.norm,
                "lm_head": self.actor_module.lm_head,
            }
            # Add all transformer layers
            for i, layer in enumerate(self.actor_module.model.layers):
                grad_components_to_analyze[f"layer_{i}"] = layer
            
            grad_component_param_ids = {name: {id(p) for p in module.parameters()} for name, module in grad_components_to_analyze.items()}
            collected_grads_for_gradient = {name: [] for name in grad_components_to_analyze}
        
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

        # 将dataloader转换为list以便重复使用第一个mini-batch
        dataloader_list = list(dataloader)
        
        # Fisher分析：在训练前对第一个mini-batch进行逐样本梯度收集
        if run_fisher_analysis and len(dataloader_list) > 0:
            first_mini_batch = dataloader_list[0]  # 获取第一个mini-batch但不消耗它
            self._collect_fisher_gradients_per_sample(first_mini_batch, components_to_analyze, component_param_ids, collected_grads_for_fisher)
        
        for batch_idx, data in enumerate(dataloader_list):
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for micro_batch_idx, data in enumerate(micro_batches):
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
                
                # Fisher分析已移至训练前的逐样本收集阶段
            
            # Gradient分析：在整个mini-batch处理完后收集一次累积梯度 (only if enabled)
            if batch_idx == 0 and should_analyze_gradients:
                # 获取当前进程的rank用于多GPU协调
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                
                if rank == 0:
                    print(f"[GradientCollection] Collecting accumulated gradients for mini-batch {batch_idx} (after all micro-batches)")
                
                for component_name, component_module in grad_components_to_analyze.items():
                    with FSDP.summon_full_params(component_module, writeback=False, rank0_only=True, with_grads=True):
                        if rank == 0:
                            grad_dict = {}
                            total_params = 0
                            collected_params = 0
                            
                            for fqn, p in self.actor_module.named_parameters():
                                if id(p) in grad_component_param_ids[component_name]:
                                    total_params += 1
                                    if p.grad is not None:
                                        clean_fqn = fqn.replace('_fsdp_wrapped_module.', '').replace('._fsdp_wrapped_module', '')
                                        # Gradient分析：收集累积梯度
                                        grad_dict[clean_fqn] = p.grad.clone().cpu()
                                        collected_params += 1
                            
                            if grad_dict:
                                collected_grads_for_gradient[component_name].append(grad_dict)
                                print(f"[GradientCollection] Component '{component_name}': collected {collected_params}/{total_params} parameters (accumulated gradients)")
                            else:
                                print(f"[GradientCollection] WARNING: No accumulated gradients found for component '{component_name}'")

            # Apply gradient clipping after all micro-batches
            if isinstance(self.actor_module, FSDP):
                self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)

            self.actor_optimizer.step()
            if self.lr_scheduler is not None:
                old_lr = self.actor_optimizer.param_groups[0]['lr']
                self.lr_scheduler.step()
                new_lr = self.actor_optimizer.param_groups[0]['lr']
                print(f"[LEARNING_RATE][Actor][Step {self.global_steps}] LR before scheduler: {old_lr:.8f}, LR after scheduler: {new_lr:.8f}")
            else:
                current_lr = self.actor_optimizer.param_groups[0]['lr']
                print(f"[LEARNING_RATE][Actor][Step {self.global_steps}] No scheduler - Current LR: {current_lr:.8f}")

            with torch.no_grad():
                metrics['actor/pg_loss'] = pg_loss.item()
                metrics['actor/entropy_loss'] = entropy_loss.item()
                metrics['actor/pg_clipfrac'] = pg_clipfrac.item()
                metrics['actor/ppo_kl'] = ppo_kl.item()
                if kl_loss is not None:
                    metrics['actor/kl_loss'] = kl_loss.item()

        # AFTER ALL MINI-BATCHES: Analysis happens only once per global step
        # Capture the learning rate that will be used for this optimizer step.
        current_lr = self.actor_optimizer.param_groups[0]['lr']
        
        # Variables already defined at the beginning of the function

        # Note: Gradient collection now happens during mini-batch loop (after clipping)
        # This section will submit analysis tasks using collected gradient data
        print(f"[DEBUG][GradCollection][Step {self.global_steps}] Gradient collection completed during mini-batch loop")
        
        # Single barrier after all gradient collection is complete
        if isinstance(self.actor_module, FSDP):
            dist.barrier()

        # Fisher analysis will be handled in the unified analysis section below
        # This removes the duplicate Fisher analysis logic

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_fsdp = isinstance(self.actor_module, FSDP)
        final_stats = None
        zero_grad_stats = None

        # Gradient analysis already performed above after gradient clipping - no duplicate analysis needed
                    
        if rank == 0:
            print(f"[INFO][Actor][Step {self.global_steps}] Clearing CUDA cache on Rank 0 to free memory for gradient gathering.")
            torch.cuda.empty_cache()

        if is_fsdp:
            # All ranks must wait for rank 0 to finish before proceeding.
            dist.barrier()

        with torch.no_grad():
            # Debug: Check Gradient Analyzer conditions
            print(f"[DEBUG][Actor][Step {self.global_steps}] Gradient Analyzer check: grad_analyzer={self.grad_analyzer is not None}, freq_check={self.global_steps % self.config.get('redo_analysis_freq', 1) == 0}, redo_analysis_freq={self.config.get('redo_analysis_freq', 1)}")
            
            # Variables already defined earlier after gradient clipping
            
            if should_analyze_gradients or should_analyze_fisher:
                
                # Initialize futures for parallel execution
                grad_analysis_futures = []
                fisher_analysis_future = None
                
                # Step 1: Both analyzers now preserve history for accumulation
                if rank == 0:
                    if should_analyze_gradients:
                        print(f"[INFO][Actor][Step {self.global_steps}] Gradient analyzer will accumulate history (no reset).")
                    if should_analyze_fisher:
                        print(f"[INFO][Actor][Step {self.global_steps}] Fisher analyzer will accumulate history (no reset).")
                        # NOTE: Both analyzers should NOT be reset as they need to accumulate history for running averages

                # Synchronize all ranks to ensure reset is complete before analysis begins.
                if is_fsdp:
                    dist.barrier()

                # NOTE: Gradient analysis has been moved to happen AFTER gradient clipping
                # for consistency with Fisher analysis and to use clipped gradients

                # Step 3: Submit Fisher analysis tasks if needed
                fisher_analysis_tasks = []
                if rank == 0 and should_analyze_fisher:
                    print(f"[INFO][Actor][Step {self.global_steps}] Starting PARALLEL Fisher analysis for {len(collected_grads_for_fisher)} components")
                    
                    valid_components = []
                    # Step 1: Validate and prepare all components for batch submission
                    for component_name, per_sample_grads in collected_grads_for_fisher.items():
                        if per_sample_grads:
                            valid_components.append((component_name, per_sample_grads))
                            print(f"[Fisher Debug][Step {self.global_steps}] Component {component_name}: {len(per_sample_grads)} sample grads ready for analysis (should be 32 for per-sample collection)")
                        else:
                            print(f"[Fisher Debug][Step {self.global_steps}] WARNING: Component {component_name} has no gradients collected!")
                    
                    # Step 2: Batch submit ALL Fisher analysis tasks simultaneously
                    for component_name, per_sample_grads in valid_components:
                        self.logger.info(f"--- Submitting parallel Fisher analysis for component: {component_name} ---")
                        task = self.fisher_info_analyzer.analyze_component_grads.remote(
                            identifier='actor',
                            component_name=component_name,
                            per_micro_batch_grads=per_sample_grads,
                            original_param_shapes=self.original_param_shapes,
                            micro_batch_size=self.config.ppo_mini_batch_size,
                            current_lr=current_lr,
                            global_step=self.global_steps
                        )
                        fisher_analysis_tasks.append((component_name, task))
                    
                    print(f"[INFO][Actor][Step {self.global_steps}] Submitted {len(fisher_analysis_tasks)} Fisher analysis tasks in parallel")
                
                # Step 4: Initialize futures for parallel analysis aggregation
                grad_stats_future = None
                fisher_analysis_future = None
                
                if rank == 0:
                    # Submit gradient analysis tasks using collected mini-batch gradients
                    if should_analyze_gradients and collected_grads_for_gradient:
                        print(f"[INFO][Actor][Step {self.global_steps}] Starting PARALLEL gradient analysis for {len(collected_grads_for_gradient)} components")
                        
                        gradient_analysis_tasks = []
                        for component_name, per_mini_batch_grads in collected_grads_for_gradient.items():
                            if per_mini_batch_grads:
                                print(f"[Gradient Debug] Component {component_name}: {len(per_mini_batch_grads)} mini-batch grads ready for analysis")
                                task = self.grad_analyzer.analyze_component_gradients_batched.remote(
                                    identifier='actor',
                                    component_name=component_name,
                                    per_mini_batch_grads=per_mini_batch_grads,
                                    original_param_shapes=self.original_param_shapes,
                                    tau=self.redo_tau,
                                    verbose=True
                                )
                                gradient_analysis_tasks.append((component_name, task))
                        
                        if gradient_analysis_tasks:
                            # Wait for all gradient analysis tasks to complete
                            task_refs = [task for component_name, task in gradient_analysis_tasks]
                            print(f"[INFO][Actor][Step {self.global_steps}] Waiting for {len(task_refs)} gradient analysis tasks to complete...")
                            ray.get(task_refs)
                            print(f"[INFO][Actor][Step {self.global_steps}] All gradient analysis tasks completed!")
                        
                        # Start gradient analysis aggregation (non-blocking) if enabled
                        grad_stats_future = self.grad_analyzer.get_aggregated_stats.remote(identifier='actor', verbose=True)
                    elif should_analyze_gradients:
                        print(f"[WARNING][Gradient][Step {self.global_steps}] No gradient data collected for analysis!")
                        grad_stats_future = None
                    
                    # Start Fisher analysis aggregation if enabled
                    if should_analyze_fisher and fisher_analysis_tasks:
                        # First wait for Fisher component analyses to complete
                        task_refs = [task for component_name, task in fisher_analysis_tasks]
                        print(f"[INFO][Actor][Step {self.global_steps}] Waiting for {len(task_refs)} Fisher analysis tasks to complete...")
                        ray.get(task_refs)  # Ensure all component analyses are done
                        print(f"[INFO][Actor][Step {self.global_steps}] All Fisher analysis tasks completed!")
                        
                        # Get both detailed component stats (for C_K reset) and aggregated stats (for logging)
                        fisher_detailed_future = self.fisher_info_analyzer.get_detailed_component_stats.remote(identifier='actor')
                        fisher_aggregated_future = self.fisher_info_analyzer.get_aggregated_stats.remote(identifier='actor')
                    
                # Step 5: Process results from both analyzers in parallel
                zero_gradspace_ratio_avg = 0.0
                
                if rank == 0:
                    # Collect all futures for parallel waiting
                    all_futures = []
                    future_types = []
                    
                    if grad_stats_future is not None:
                        all_futures.append(grad_stats_future)
                        future_types.append('gradient')
                    
                    if fisher_detailed_future is not None:
                        all_futures.append(fisher_detailed_future)
                        future_types.append('fisher_detailed')
                    
                    if fisher_aggregated_future is not None:
                        all_futures.append(fisher_aggregated_future)
                        future_types.append('fisher_aggregated')
                    
                    if all_futures:
                        
                        # Wait for all analysis tasks to complete in parallel
                        results = ray.get(all_futures)
                        
                        # Process gradient analysis results
                        for i, (result, future_type) in enumerate(zip(results, future_types)):
                            if future_type == 'gradient':
                                final_stats = result
                                
                                # Initialize default values to prevent undefined variable usage
                                global_stats = {}
                                zero_gradspace_ratio_avg = 0.0
                                
                                if not final_stats:
                                    self.logger.warning(f"[Actor][Step {self.global_steps}] Failed to get zero-grad analysis results.")
                                elif isinstance(final_stats, dict):
                                    global_stats = final_stats.get('__global__', {})
                                    global_ratio = global_stats.get('ratio', 0.0)
                                    zero_gradspace_ratio_avg = global_ratio
                                    
                                    if '__global__' not in final_stats:
                                        print(f"[DEBUG][Gradient][Step {self.global_steps}] WARNING: '__global__' key not found in final_stats")
                                    
                                    self.logger.info(f"--- 📊 Gradient Analysis Results (Step {self.global_steps}, Tau: {self.redo_tau}) ---")
                                    self.logger.info(f"Global Dormant Neuron Ratio: {global_ratio:.4%}")
                                    
                                    # Store detailed metrics to unified storage
                                    if self.analyzer_storage:
                                        try:
                                            layer_name = "layer1"
                                            sample_params = {}
                                            for name, param in self.actor_module.named_parameters():
                                                try:
                                                    if layer_name in name and "weight" in name:
                                                        # Store sample values before reset with FSDP safety check
                                                        try:
                                                            # Check FSDP state and parameter availability
                                                            if hasattr(param, '_fsdp_flattened') and param._fsdp_flattened:
                                                                sample_params[name] = f"[FSDP_FLATTENED] Parameter is flattened by FSDP"
                                                            elif param.data.numel() > 0:  # Check if parameter has data on this rank
                                                                sample_values = param.data.flatten()[:5].clone().cpu().tolist()
                                                                sample_params[name] = sample_values
                                                            else:
                                                                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                                                                sample_params[name] = f"[FSDP_SHARD] No data on rank {rank}, numel={param.data.numel()}"
                                                        except Exception as e:
                                                            sample_params[name] = f"[ERROR] {str(e)}"
                                                        break
                                                except Exception as e:
                                                    sample_params[name] = f"[ERROR] {str(e)}"
                                            self.analyzer_storage.store_gradient_metrics(
                                                step=self.global_steps,
                                                gradient_stats=final_stats,
                                                tau=self.redo_tau,
                                                additional_info={
                                                    "rank": rank,
                                                    "device": str(self.device) if hasattr(self, 'device') else "unknown",
                                                    "sample_params": sample_params
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
                            
                            elif future_type == 'fisher_detailed':
                                fisher_detailed_stats = result
                                # Store detailed stats for C_K reset analysis
                                self.fisher_detailed_stats = fisher_detailed_stats
                                print(f"[INFO][Fisher] Detailed component stats received for C_K reset analysis")
                                
                            elif future_type == 'fisher_aggregated':
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
                                    
                                else:
                                    self.logger.warning(f"[Actor][Step {self.global_steps}] Failed to get Fisher analysis results.")
                        
                    # CK reset is now handled centrally by trainer via get_ck_reset_status and reset_model_with_ck_analysis
                    
                    # Set the metrics with the correct value
                    metrics['actor/zero_gradspace_ratio'] = zero_gradspace_ratio_avg
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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_ck_reset_status(self, global_step: int) -> Dict[str, Any]:
        """
        Check if CK reset should be performed and return C_K weights.
        Uses existing Fisher analysis results to avoid duplicate computation.
        
        Args:
            global_step: Current global step
            
        Returns:
            Dictionary with reset status and C_K weights
        """
        print(f"[CK_RESET_DEBUG] Actor: get_ck_reset_status called for step {global_step}")
        
        # Check if we have Fisher stats from recent analysis
        fisher_stats_for_reset = None
        has_fisher_stats = hasattr(self, 'fisher_detailed_stats')
        print(f"[CK_RESET_DEBUG] Actor: has_fisher_detailed_stats = {has_fisher_stats}")
        
        if has_fisher_stats:
            fisher_stats_for_reset = self.fisher_detailed_stats
            if fisher_stats_for_reset:
                print(f"[CK_RESET_DEBUG] Actor: fisher_detailed_stats has {len(fisher_stats_for_reset)} components")
                # Show first few component names for debugging
                component_names = list(fisher_stats_for_reset.keys())[:5]
                print(f"[CK_RESET_DEBUG] Actor: Sample component names: {component_names}")
                
                # DETAILED STRUCTURE INSPECTION
                for comp_name, comp_data in fisher_stats_for_reset.items():
                    print(f"[CK_RESET_DEBUG] Actor: === COMPONENT '{comp_name}' STRUCTURE ===")
                    print(f"[CK_RESET_DEBUG] Actor: Component type: {type(comp_data)}")
                    
                    if isinstance(comp_data, dict):
                        print(f"[CK_RESET_DEBUG] Actor: Component has {len(comp_data)} keys")
                        param_names = list(comp_data.keys())[:10]  # Show first 10 parameter names
                        print(f"[CK_RESET_DEBUG] Actor: Sample parameter names: {param_names}")
                        
                        # Inspect first parameter's structure
                        if param_names:
                            first_param = param_names[0]
                            first_param_data = comp_data[first_param]
                            print(f"[CK_RESET_DEBUG] Actor: First param '{first_param}' type: {type(first_param_data)}")
                            
                            if isinstance(first_param_data, dict):
                                param_keys = list(first_param_data.keys())
                                print(f"[CK_RESET_DEBUG] Actor: First param keys: {param_keys}")
                                
                                # Show c_k value if exists
                                if 'c_k' in first_param_data:
                                    c_k_val = first_param_data['c_k']
                                    print(f"[CK_RESET_DEBUG] Actor: First param c_k value: {c_k_val} (type: {type(c_k_val)})")
                                else:
                                    print(f"[CK_RESET_DEBUG] Actor: First param does NOT have 'c_k' key")
                            else:
                                print(f"[CK_RESET_DEBUG] Actor: First param data is not dict: {first_param_data}")
                    else:
                        print(f"[CK_RESET_DEBUG] Actor: Component data is not dict: {comp_data}")
                    print(f"[CK_RESET_DEBUG] Actor: === END COMPONENT '{comp_name}' ===")
            else:
                print(f"[CK_RESET_DEBUG] Actor: fisher_detailed_stats is None or empty")
        
        # Get reset steps from configuration (passed from training script)
        reset_steps = getattr(self.config, 'reset_steps', [1, 40, 80, 120])  # Default fallback
        if hasattr(self.config, 'ck_reset') and hasattr(self.config.ck_reset, 'reset_steps'):
            reset_steps = self.config.ck_reset.reset_steps
        should_reset = global_step in reset_steps
        print(f"[CK_RESET_DEBUG] Actor: should_reset = {should_reset} (step {global_step} in {reset_steps})")
        
        # Calculate C_K weights if we have Fisher stats and should reset
        layer_ck_weights = {}
        if should_reset and fisher_stats_for_reset:
            print(f"[CK_RESET_DEBUG] Actor: Attempting to extract C_K weights from pre-calculated data...")
            try:
                # Check if Fisher analyzer pre-calculated layer weights are available
                if 'layer_ck_weights' in fisher_stats_for_reset:
                    layer_ck_weights = fisher_stats_for_reset['layer_ck_weights']
                    print(f"[CK_RESET_DEBUG] Actor: Using pre-calculated layer C_K weights: {len(layer_ck_weights)} layers")
                    for layer_idx, ck_weight in layer_ck_weights.items():
                        print(f"[CK_RESET_DEBUG] Actor: Layer {layer_idx} C_K_historical = {ck_weight:.6f}")
                else:
                    raise RuntimeError(f"[CK_RESET_ERROR] Actor: No pre-calculated layer weights found in fisher_stats_for_reset. "
                                     f"Expected 'layer_ck_weights' key but got keys: {list(fisher_stats_for_reset.keys())}. "
                                     f"Fisher analyzer must provide pre-calculated layer C_K weights.")
                
                print(f"[CK_RESET_DEBUG] Actor: Final C_K weights: {len(layer_ck_weights)} layers")
            except Exception as e:
                print(f"[CK_RESET_DEBUG] Actor: Error calculating C_K weights: {e}")
                import traceback
                print(f"[CK_RESET_DEBUG] Actor: Traceback: {traceback.format_exc()}")
                layer_ck_weights = {}
        elif should_reset and not fisher_stats_for_reset:
            print(f"[CK_RESET_DEBUG] Actor: Should reset but no Fisher stats available")
        elif not should_reset:
            print(f"[CK_RESET_DEBUG] Actor: Not a reset step")
        
        result = {
            'should_reset': should_reset,
            'layer_ck_weights': layer_ck_weights
        }
        print(f"[CK_RESET_DEBUG] Actor: Returning {result}")
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_sample_layer_weights(self, layer_indices: List[int]) -> Dict[str, Any]:
        """
        Get sample weights from specified layers for verification.
        
        Args:
            layer_indices: List of layer indices to sample
            
        Returns:
            Dictionary with sample weights from specified layers
        """
        try:
            sample_weights = {}
            
            # Get actor base model
            actor_base = self.actor_module._fsdp_wrapped_module if hasattr(self.actor_module, '_fsdp_wrapped_module') else self.actor_module
            
            # Find transformer layers
            actor_layers = None
            for pattern in ['model.layers', 'transformer.h', 'transformer.layers', 'layers']:
                try:
                    actor_layers = actor_base
                    for attr in pattern.split('.'):
                        actor_layers = getattr(actor_layers, attr)
                    if isinstance(actor_layers, (list, torch.nn.ModuleList)):
                        break
                except AttributeError:
                    continue
            
            if actor_layers is None:
                return {}
            
            # Sample weights from specified layers
            for layer_idx in layer_indices[:3]:  # Limit to first 3 layers to avoid spam
                if layer_idx < len(actor_layers):
                    layer = actor_layers[layer_idx]
                    layer_weights = {}
                    
                    # Sample first few parameters from this layer
                    param_count = 0
                    for name, param in layer.named_parameters():
                        if param_count >= 3:  # Limit to 3 parameters per layer
                            break
                        try:
                            # Get a small sample of the parameter values
                            if param.data.numel() > 0:
                                sample_values = param.data.flatten()[:5].detach().cpu().tolist()
                                layer_weights[f"layer_{layer_idx}.{name}"] = sample_values
                            param_count += 1
                        except Exception as e:
                            layer_weights[f"layer_{layer_idx}.{name}"] = f"Error: {e}"
                    
                    sample_weights.update(layer_weights)
            
            return sample_weights
            
        except Exception as e:
            print(f"[SAMPLE_WEIGHTS_ERROR] Failed to get sample weights: {e}")
            return {}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_transformer_layer_count(self) -> int:
        """
        Get the number of transformer layers in the actor model.
        This is used by the trainer to determine layer indices for reset.
        
        Returns:
            int: Number of transformer layers in the actor model
        """
        try:
            if hasattr(self, 'ck_reset_manager') and self.ck_reset_manager is not None:
                transformer_layers = self.ck_reset_manager.get_transformer_layers(self.actor_module)
                return len(transformer_layers)
            else:
                # Fallback: try to get layers directly
                if hasattr(self.actor_module, 'model') and hasattr(self.actor_module.model, 'layers'):
                    return len(self.actor_module.model.layers)
                else:
                    print(f"[ACTOR_DEBUG] Cannot determine transformer layer count")
                    return 0
        except Exception as e:
            print(f"[ACTOR_DEBUG] Error getting transformer layer count: {e}")
            return 0
