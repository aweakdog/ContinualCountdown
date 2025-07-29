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

@ray.remote(num_gpus=1, num_cpus=1)
class FisherInfoAnalyzer:
    """
    A stateful Ray actor that computes Empirical Fisher Information Matrix (EFIM) metrics
    on a per-parameter basis, aggregated by component.
    """
    def __init__(self, config):
        self.config = config
        print("[FisherInfoAnalyzer] Actor initialized.")
        # self.stats stores metrics for the CURRENT analysis step
        self.stats = collections.defaultdict(lambda: {'params': {}})
        # self.global_history stores aggregated metrics from ALL past analysis steps to compute running global stats
        self.global_history = collections.defaultdict(lambda: {'c_k_means': [], 'l_k_sums': []})
        # self.param_history stores per-parameter metrics from ALL past analysis steps
        self.param_history = collections.defaultdict(lambda: {'params': collections.defaultdict(lambda: {'c_k_history': [], 'l_k_history': []})})
        # self.param_shapes stores the original shapes of parameters for normalization
        self.param_shapes = collections.defaultdict(dict)

    def reset(self, identifier: str):
        """Resets all statistics and history for a given analysis identifier (e.g., 'actor')."""
        self.stats.pop(identifier, None)
        self.global_history.pop(identifier, None)
        self.param_history.pop(identifier, None)
        self.param_shapes.pop(identifier, None)
        print(f"[FisherInfoAnalyzer] Reset all statistics and history for identifier '{identifier}'.")

    def analyze_component_grads(self, identifier: str, component_name: str, per_micro_batch_grads: List[Dict[str, torch.Tensor]], original_param_shapes: Dict[str, torch.Size], micro_batch_size: int, current_lr: float, global_step: int):
        """
        Analyzes gradients for a specific model component to compute EFIM metrics for each parameter.
        """
        # print(f"[DEBUG][Fisher] Received request for component '{component_name}' with {len(per_micro_batch_grads)} micro-batch grads.")
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
        for name, grads in grads_by_param.items():
            try:
                original_shape = original_param_shapes.get(name)
                if not original_shape:
                    continue
                
                # Store parameter shape for normalization calculations
                self.param_shapes[identifier][name] = original_shape

                # Per user instruction, reshape the flattened gradients before stacking them into the Jacobian.
                reshaped_then_flattened_grads = []
                for g in grads:
                    if g.numel() == original_shape.numel():
                        g_reshaped = g.reshape(original_shape)
                        reshaped_then_flattened_grads.append(g_reshaped.flatten().cuda())
                    else:
                        # If shape mismatch, just flatten what we have.
                        reshaped_then_flattened_grads.append(g.flatten().cuda())

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
                    c_k = torch.tensor(1.0)
                else:
                    sigma_max = torch.sqrt(non_zero_eigenvalues.max())
                    sigma_min = torch.sqrt(non_zero_eigenvalues.min())
                    c_k = sigma_max / sigma_min

                trace_F = torch.sum(non_zero_eigenvalues)
                l_k = (current_lr / micro_batch_size) * torch.sqrt(trace_F)

                param_stats = {
                    'c_k': c_k.item(),
                    'l_k': l_k.item(),
                    'trace_F': trace_F.item(),
                    'sigma_max': sigma_max.item(),
                    'sigma_min': sigma_min.item(),
                }

                param_hist = self.param_history[identifier]['params'][name]
                param_hist['c_k_history'].append(param_stats['c_k'])
                param_hist['l_k_history'].append(param_stats['l_k'])
                
                C_K_param = np.mean(param_hist['c_k_history'])
                L_K_param = np.sum(param_hist['l_k_history'])

                print(f"[FisherInfo] Param '{name}': c_k={param_stats['c_k']:.4f}, l_k={param_stats['l_k']:.6g}, C_K={C_K_param:.4f}, L_K={L_K_param:.6g}, sigma_max={param_stats['sigma_max']:.6g}, sigma_min={param_stats['sigma_min']:.6g}")
                #print(f'[FisherInfo] fisher_tilde shape: {fisher_tilde.shape}, eigenvalues: {len(non_zero_eigenvalues)} non-zero')
                component_stats[name] = param_stats

            except torch.linalg.LinAlgError as e:
                print(f"[FisherInfoAnalyzer] LinAlgError for param '{name}' in component '{component_name}': {e}")
                continue
        
        self.stats[identifier]['params'][component_name] = component_stats
        print(f"[FisherInfoAnalyzer] Step {global_step} | Component '{component_name}': Analyzed {len(component_stats)} params.")

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

        # 2. Update global history
        self.global_history.setdefault(identifier, {'c_k_means': [], 'l_k_sums': []})
        self.global_history[identifier]['c_k_means'].append(current_c_k_mean)
        self.global_history[identifier]['l_k_sums'].append(current_l_k_sum_for_this_step)

        # 3. Calculate and return final time-aggregated metrics
        c_k_history_list = self.global_history[identifier]['c_k_means']
        l_k_history_list = self.global_history[identifier]['l_k_sums']
        
        K = len(c_k_history_list) # K is the number of steps we have history for

        # C_K = sigma(past c_k)/K -> This is the mean of the historical means
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
