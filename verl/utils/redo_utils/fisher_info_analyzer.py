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

import ray
import torch
import numpy as np
import collections
from typing import List, Dict

@ray.remote(num_gpus=2, num_cpus=8)  # Multi-GPU support for parallel Fisher analysis
class FisherInfoAnalyzer:
    """
    A stateful Ray actor that computes Empirical Fisher Information Matrix (EFIM) metrics
    on a per-parameter basis, aggregated by component.
    """
    def __init__(self, config):
        self.config = config
        import os
        self.actor_pid = os.getpid()
        
        # Multi-GPU device management
        if torch.cuda.is_available():
            self.available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
            self.primary_device = self.available_devices[0]
            torch.cuda.set_device(self.primary_device)
            print(f"[FisherInfoAnalyzer] Actor initialized with PID {self.actor_pid} on {len(self.available_devices)} GPUs: {self.available_devices}")
        else:
            self.available_devices = [torch.device('cpu')]
            self.primary_device = self.available_devices[0]
            print(f"[FisherInfoAnalyzer] Actor initialized with PID {self.actor_pid} on CPU (no CUDA available)")
        # self.stats stores metrics for the CURRENT analysis step
        self.stats = collections.defaultdict(lambda: {'params': {}})
        # self.global_history stores aggregated metrics from ALL past analysis steps to compute running global stats
        self.global_history = collections.defaultdict(lambda: {'c_k_normalized_history': [], 'l_k_sums': []})
        # self.param_history stores per-parameter metrics from ALL past analysis steps
        self.param_history = collections.defaultdict(lambda: {'params': collections.defaultdict(lambda: {'c_k_history': [], 'l_k_history': []})})
        # self.param_shapes stores the original shapes of parameters for normalization
        self.param_shapes = collections.defaultdict(dict)

    def reset(self, identifier: str):
        """Resets all statistics and history for a given analysis identifier (e.g., 'actor')."""
        # Debug: Check what we're about to reset
        old_history_len = len(self.global_history.get(identifier, {}).get('c_k_normalized_history', []))
        
        self.stats.pop(identifier, None)
        self.global_history.pop(identifier, None)
        self.param_history.pop(identifier, None)
        self.param_shapes.pop(identifier, None)
        print(f"[FisherInfoAnalyzer] PID {self.actor_pid} - Reset all statistics and history for identifier '{identifier}'.")

    def analyze_component_grads(self, identifier: str, component_name: str, per_micro_batch_grads: List[Dict[str, torch.Tensor]], original_param_shapes: Dict[str, torch.Size], micro_batch_size: int, current_lr: float, global_step: int):
        """
        Analyzes gradients for a specific model component to compute EFIM metrics for each parameter.
        """
        if not per_micro_batch_grads or not per_micro_batch_grads[0]:
            print(f"[FisherInfoAnalyzer] No gradients for component '{component_name}'. Skipping.")
            return

        # Reorganize from a list of dicts to a dict of lists.
        grads_by_param = collections.defaultdict(list)
        for i, grad_dict in enumerate(per_micro_batch_grads):
            for name, grad_tensor in grad_dict.items():
                grads_by_param[name].append(grad_tensor)

        if not grads_by_param:
            print(f"[DEBUG][Fisher] 'grads_by_param' is empty for component '{component_name}'. Exiting analysis for this component.")
            return

        component_stats = {}
        
        # OPTIMIZED PARALLEL PROCESSING: Use CUDA streams for true GPU parallelism
        num_devices = len(self.available_devices)
        device_batches = {device: [] for device in self.available_devices}
        device_streams = {device: torch.cuda.Stream(device) for device in self.available_devices if device.type == 'cuda'}
        
        # Step 1: Distribute parameters across devices
        for i, (name, grads) in enumerate(grads_by_param.items()):
            device_idx = i % num_devices
            device_to_use = self.available_devices[device_idx]
            device_batches[device_to_use].append((name, grads))
            
            # Store parameter shape for normalization calculations
            original_shape = original_param_shapes.get(name)
            if original_shape:
                self.param_shapes[identifier][name] = original_shape
        
        # Step 2: Process all device batches in parallel using CUDA streams
        device_results = {}
        
        for device, param_batch in device_batches.items():
            if not param_batch:  # Skip empty batches
                continue
                
            if device.type == 'cuda':
                stream = device_streams[device]
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    device_results[device] = self._process_fisher_device_batch(
                        device, param_batch, original_param_shapes, current_lr, micro_batch_size
                    )
            else:
                # CPU fallback
                device_results[device] = self._process_fisher_device_batch(
                    device, param_batch, original_param_shapes, current_lr, micro_batch_size
                )
        
        # Step 3: Synchronize all streams and collect results
        for device in device_streams:
            torch.cuda.synchronize(device)
        
        # Step 4: Aggregate results from all devices
        for device, device_stats in device_results.items():
            component_stats.update(device_stats)
        
        # Step 5: Update history and statistics after parallel processing
        for name, param_stats in component_stats.items():
            param_hist = self.param_history[identifier]['params'][name]
            param_hist['c_k_history'].append(param_stats['c_k'])
            param_hist['l_k_history'].append(param_stats['l_k'])
            
            C_K_param = np.mean(param_hist['c_k_history'])
            L_K_param = np.sum(param_hist['l_k_history'])
            
        
        self.stats[identifier]['params'][component_name] = component_stats
    
    def _process_fisher_device_batch(self, device, param_batch, original_param_shapes, current_lr, micro_batch_size):
        """Optimized Fisher information computation for a single device batch"""
        device_stats = {}
        
        for name, grads in param_batch:
            try:
                original_shape = original_param_shapes.get(name)
                if not original_shape:
                    continue

                # Multi-GPU parallel gradient processing
                reshaped_then_flattened_grads = []
                
                for g in grads:
                    # Ensure gradient is on the correct device
                    if g.device != device:
                        g = g.to(device, non_blocking=True)
                    
                    if g.numel() == original_shape.numel():
                        g_reshaped = g.reshape(original_shape)
                        reshaped_then_flattened_grads.append(g_reshaped.flatten())
                    else:
                        # If shape mismatch, just flatten what we have.
                        reshaped_then_flattened_grads.append(g.flatten())

                if not reshaped_then_flattened_grads:
                    continue

                jacobian = torch.stack(reshaped_then_flattened_grads)
                
                # Compute reduced Fisher matrix: F_tilde = J @ J.T
                fisher_tilde = jacobian @ jacobian.T

                eigenvalues = torch.linalg.eigvalsh(fisher_tilde)
                
                eig_threshold = self.config.actor.get('fsdp_component_analysis', {}).get('fisher_eig_threshold', 1e-8)
                non_zero_eigenvalues = eigenvalues[eigenvalues > eig_threshold]

                if len(non_zero_eigenvalues) == 0:
                    continue
                elif len(non_zero_eigenvalues) == 1:
                    sigma_max = torch.sqrt(non_zero_eigenvalues.max())
                    sigma_min = sigma_max
                    c_k = torch.tensor(1.0, device=device)
                else:
                    sigma_max = torch.sqrt(non_zero_eigenvalues.max())
                    sigma_min = torch.sqrt(non_zero_eigenvalues.min())
                    c_k = sigma_max / sigma_min
                    
                    # Diagnostic logging for extreme condition numbers
                    if c_k > 1000:  # Threshold for "extremely high"
                        print(f"[WARNING][Fisher] Param '{name}' on {device}: EXTREME c_k={c_k:.2f}!")
                        print(f"  - sigma_max={sigma_max:.6g}, sigma_min={sigma_min:.6g}")
                        print(f"  - eigenvalue_max={non_zero_eigenvalues.max():.6g}, eigenvalue_min={non_zero_eigenvalues.min():.6g}")
                        print(f"  - num_eigenvalues={len(non_zero_eigenvalues)}, jacobian_shape={jacobian.shape}")
                        print(f"  - gradient_norms: max={torch.stack([g.norm() for g in reshaped_then_flattened_grads]).max():.6g}, min={torch.stack([g.norm() for g in reshaped_then_flattened_grads]).min():.6g}")
                        print(f"  - current_lr={current_lr}, micro_batch_size={micro_batch_size}")

                trace_F = torch.sum(non_zero_eigenvalues)
                # L_k now only depends on Fisher information trace, not learning rate
                l_k = torch.sqrt(trace_F) / micro_batch_size

                param_stats = {
                    'c_k': c_k.item(),
                    'l_k': l_k.item(),
                    'trace_F': trace_F.item(),
                    'sigma_max': sigma_max.item(),
                    'sigma_min': sigma_min.item(),
                }

                device_stats[name] = param_stats
                print(f"[FisherInfo][{device}] Param '{name}': c_k={param_stats['c_k']:.4f}, l_k={param_stats['l_k']:.6g}, sigma_max={param_stats['sigma_max']:.6g}, sigma_min={param_stats['sigma_min']:.6g}")

            except torch.linalg.LinAlgError as e:
                print(f"[FisherInfoAnalyzer][{device}] LinAlgError for param '{name}': {e}")
                continue
        
        return device_stats

    def get_aggregated_stats(self, identifier: str):
        """
        Computes and returns time-aggregated statistics for c_k and l_k.
        - c_k is the running average of the mean c_k from each step.
        - l_k is the cumulative sum of the l_k values from each step.
        - C_K_normalized is the sum of C_K values normalized by total parameter count.
        """
        # 1. Calculate stats for the CURRENT step from self.stats
        current_stats = self.stats.get(identifier)
        if not current_stats or 'params' not in current_stats:
            return {}

        current_all_c_k = []
        current_all_l_k = []
        total_c_k_weighted_sum = 0.0
        total_l_k_weighted_sum = 0.0
        total_current_c_k_weighted_sum = 0.0
        total_current_l_k_weighted_sum = 0.0
        total_current_sigma_max_weighted_sum = 0.0
        total_current_sigma_min_weighted_sum = 0.0
        total_param_count = 0
        
        for component_name, component_data in current_stats['params'].items():
            for param_name, param_stats in component_data.items():
                current_all_c_k.append(param_stats['c_k'])
                current_all_l_k.append(param_stats['l_k'])
                
                # Get the number of parameters in this matrix
                param_shape = self.param_shapes[identifier].get(param_name)
                if param_shape:
                    num_params = param_shape.numel()
                    
                    # Weight current step values by parameter count
                    total_current_c_k_weighted_sum += param_stats['c_k'] * num_params
                    total_current_l_k_weighted_sum += param_stats['l_k'] * num_params
                    total_current_sigma_max_weighted_sum += param_stats['sigma_max'] * num_params
                    total_current_sigma_min_weighted_sum += param_stats['sigma_min'] * num_params
                    total_param_count += num_params
                    
                    # Calculate C_K and L_K for this parameter from its history
                    param_hist = self.param_history[identifier]['params'][param_name]
                    if param_hist['c_k_history']:
                        C_K_param = np.mean(param_hist['c_k_history'])
                        total_c_k_weighted_sum += C_K_param * num_params
                    if param_hist['l_k_history']:
                        L_K_param = np.sum(param_hist['l_k_history'])
                        total_l_k_weighted_sum += L_K_param * num_params

        if not current_all_c_k:
            return {}

        current_c_k_mean = np.mean(current_all_c_k)
        current_l_k_sum_for_this_step = np.sum(current_all_l_k)
        
        # Calculate normalized metrics: weighted by parameter count
        if total_param_count > 0:
            C_K_normalized = total_c_k_weighted_sum / total_param_count
            L_K_normalized = total_l_k_weighted_sum / total_param_count
            c_k_normalized = total_current_c_k_weighted_sum / total_param_count
            l_k_normalized = total_current_l_k_weighted_sum / total_param_count
            sigma_max_normalized = total_current_sigma_max_weighted_sum / total_param_count
            sigma_min_normalized = total_current_sigma_min_weighted_sum / total_param_count
        else:
            C_K_normalized = 0.0
            L_K_normalized = 0.0
            c_k_normalized = 0.0
            l_k_normalized = 0.0
            sigma_max_normalized = 0.0
            sigma_min_normalized = 0.0

        # 2. Update global history (with step-level deduplication)
        self.global_history.setdefault(identifier, {'c_k_normalized_history': [], 'l_k_sums': [], 'last_step_added': -1})
        
        # Add step tracking to prevent duplicate entries in the same step
        current_step = len(self.global_history[identifier]['c_k_normalized_history'])
        last_step_added = self.global_history[identifier].get('last_step_added', -1)
        
        # Only add to history if this is a new step (prevent duplicate calls in same step)
        if current_step != last_step_added:
            self.global_history[identifier]['c_k_normalized_history'].append(c_k_normalized)
            self.global_history[identifier]['l_k_sums'].append(current_l_k_sum_for_this_step)
            self.global_history[identifier]['last_step_added'] = current_step
        else:
            print(f"[DEBUG][Fisher] PID {self.actor_pid} - Skipping duplicate history update for step {current_step}")

        # Debug: Print history state
        c_k_history_list = self.global_history[identifier]['c_k_normalized_history']
        l_k_history_list = self.global_history[identifier]['l_k_sums']
        print(f"[DEBUG][Fisher] PID {self.actor_pid} - History length for '{identifier}': {len(c_k_history_list)} steps")
        print(f"[DEBUG][Fisher] PID {self.actor_pid} - c_k_normalized_history: {c_k_history_list[-3:] if len(c_k_history_list) > 3 else c_k_history_list}")
        print(f"[DEBUG][Fisher] PID {self.actor_pid} - Current c_k_normalized: {c_k_normalized:.6f}")
        print(f"[DEBUG][Fisher] PID {self.actor_pid} - Global history keys: {list(self.global_history.keys())}")
        
        # Check if this is the first call for this identifier
        if len(c_k_history_list) == 1:
            print(f"[DEBUG][Fisher] PID {self.actor_pid} - WARNING: This appears to be the first call for identifier '{identifier}' - actor may be getting recreated!")
        
        K = len(c_k_history_list) # K is the number of steps we have history for

        # C_K = sigma(past c_k_normalized)/K -> This is the mean of the historical normalized c_k values
        final_c_k_running_avg = np.mean(c_k_history_list)
        
        # l_k is the sum of the history l_k -> This is the sum of the historical sums
        final_l_k_cumulative_sum = np.sum(l_k_history_list)

        summary_stats = {
            'fisher/c_k_running_avg': final_c_k_running_avg,
            'fisher/l_k_cumulative_sum': final_l_k_cumulative_sum,
            'fisher/l_k_cumulative_sum_e6': final_l_k_cumulative_sum * 1e6, # Scaled for visibility in logs
            'fisher/c_k_mean_current': current_c_k_mean,
            'fisher/l_k_sum_current': current_l_k_sum_for_this_step,
            'fisher/C_K_normalized': C_K_normalized,
            'fisher/L_K_normalized': L_K_normalized,
            'fisher/c_k_normalized': c_k_normalized,
            'fisher/l_k_normalized': l_k_normalized,
            'fisher/sigma_max_normalized': sigma_max_normalized,
            'fisher/sigma_min_normalized': sigma_min_normalized,
        }

        print(f"[FisherInfoAnalyzer] Aggregated Stats for '{identifier}': "
              f"c_k_running_avg={summary_stats['fisher/c_k_running_avg']:.4f}, "
              f"l_k_cumulative_sum={summary_stats['fisher/l_k_cumulative_sum']:.6g}, "
              f"C_K_normalized={summary_stats['fisher/C_K_normalized']:.4f}, "
              f"c_k_normalized={summary_stats['fisher/c_k_normalized']:.4f}, "
              f"l_k_normalized={summary_stats['fisher/l_k_normalized']:.6g}")
        return summary_stats
