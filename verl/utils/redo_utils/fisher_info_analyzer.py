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
    def __init__(self):
        print("[FisherInfoAnalyzer] Actor initialized.")
        self.stats = collections.defaultdict(lambda: {'params': {}, 'summary': {}})
        self.current_step_idx = collections.defaultdict(int)
        self.running_C = collections.defaultdict(float)
        self.cumulative_L = collections.defaultdict(float)

    def reset(self, identifier: str):
        """Resets the statistics for a given analysis identifier (e.g., 'actor')."""
        self.stats[identifier] = {'params': {}, 'summary': {}}
        self.current_step_idx[identifier] = 0
        self.running_C[identifier] = 0.0
        self.cumulative_L[identifier] = 0.0
        print(f"[FisherInfoAnalyzer] Statistics reset for identifier: '{identifier}'.")

    def analyze_component_grads(self, identifier: str, component_name: str, per_micro_batch_grads: List[Dict[str, torch.Tensor]], original_param_shapes: Dict[str, torch.Size], micro_batch_size: int, current_lr: float, global_step: int):
        """
        Analyzes gradients for a specific model component to compute EFIM metrics for each parameter.
        """
        print(f"[DEBUG][Fisher] Received request for component '{component_name}' with {len(per_micro_batch_grads)} micro-batch grads.")
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
        print(f"[DEBUG][Fisher] Analyzing {len(grads_by_param)} params for component '{component_name}': {list(grads_by_param.keys())}")
        if original_param_shapes:
            print(f"[DEBUG][Fisher] Received original_param_shapes with {len(original_param_shapes)} entries. Keys: {list(original_param_shapes.keys())[:5]}...")

        for name, grads in grads_by_param.items():
            try:
                print(f"[DEBUG][Fisher] Param '{name}': processing {len(grads)} gradients.")
                
                original_shape = original_param_shapes.get(name)
                if not original_shape:
                    print(f"[DEBUG][Fisher] Param '{name}': No original shape found. Skipping.")
                    continue

                # Per user instruction, reshape the flattened gradients before stacking them into the Jacobian.
                # This mimics the GradientAnalyzer's behavior.
                reshaped_then_flattened_grads = []
                for g in grads:
                    # Ensure the number of elements matches before reshaping
                    if g.numel() == original_shape.numel():
                        g_reshaped = g.reshape(original_shape)
                        reshaped_then_flattened_grads.append(g_reshaped.flatten().cuda())
                    else:
                        # If shape mismatch, just flatten what we have.
                        if self.rank == 0:
                            print(f"[Fisher WARN] Mismatch for {name}: grad numel {g.numel()} vs original shape numel {original_shape.numel()}. Using as-is.")
                        reshaped_then_flattened_grads.append(g.flatten().cuda())

                if not reshaped_then_flattened_grads:
                    print(f"[DEBUG][Fisher] Param '{name}': No valid gradients to stack after reshape/flatten. Skipping.")
                    continue

                jacobian = torch.stack(reshaped_then_flattened_grads)
                print(f"[DEBUG][Fisher] Param '{name}': Jacobian shape: {jacobian.shape}")
                
                # Compute reduced Fisher matrix: F_tilde = J @ J.T
                fisher_tilde = jacobian @ jacobian.T
                
                eigenvalues = torch.linalg.eigvalsh(fisher_tilde)
                non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-8]

                if len(non_zero_eigenvalues) == 0:
                    print(f"[DEBUG][Fisher] Param '{name}': Skipping due to 0 non-zero eigenvalues.")
                    continue
                elif len(non_zero_eigenvalues) == 1:
                    # Handle the case of a single gradient vector where c_k is not meaningful.
                    print(f"[DEBUG][Fisher] Param '{name}': Only 1 non-zero eigenvalue found. Reporting default c_k=1.0.")
                    sigma_max = torch.sqrt(non_zero_eigenvalues.max())
                    sigma_min = sigma_max  # With one value, min is the same as max
                    c_k = torch.tensor(1.0)
                else:
                    # Original logic for 2 or more eigenvalues
                    sigma_max = torch.sqrt(non_zero_eigenvalues.max())
                    sigma_min = torch.sqrt(non_zero_eigenvalues.min())
                    c_k = sigma_max / sigma_min
                trace_F = torch.trace(fisher_tilde)
                l_k = (current_lr / micro_batch_size) * torch.sqrt(trace_F)

                param_stats = {
                    'c_k': c_k.item(),
                    'l_k': l_k.item(),
                    'trace_F': trace_F.item(),
                    'sigma_max': sigma_max.item(),
                    'sigma_min': sigma_min.item(),
                }
                print(f"[FisherInfo] Param '{name}': c_k={param_stats['c_k']:.4f}, l_k={param_stats['l_k']:.4f}")
                component_stats[name] = param_stats

            except torch.linalg.LinAlgError as e:
                print(f"[FisherInfoAnalyzer] LinAlgError for param '{name}' in component '{component_name}': {e}")
                continue
        
        self.stats[identifier]['params'][component_name] = component_stats
        print(f"[FisherInfoAnalyzer] Step {global_step} | Component '{component_name}': Analyzed {len(component_stats)} params.")

    def get_aggregated_stats(self, identifier: str):
        """
        Computes and returns aggregated statistics (mean, max, min) for c_k and l_k
        across all analyzed components and parameters for a given identifier.
        """
        stats = self.stats.get(identifier)
        if not stats or 'params' not in stats:
            return {}

        all_c_k = []
        all_l_k = []

        for component_name, component_data in stats['params'].items():
            for param_name, param_stats in component_data.items():
                if 'c_k' in param_stats:
                    all_c_k.append(param_stats['c_k'])
                if 'l_k' in param_stats:
                    all_l_k.append(param_stats['l_k'])
        
        if not all_c_k: # If no params were analyzed, return empty
            return {}

        summary_stats = {
            'fisher/c_k_mean': np.mean(all_c_k),
            'fisher/c_k_max': np.max(all_c_k),
            'fisher/c_k_min': np.min(all_c_k),
            'fisher/c_k_std': np.std(all_c_k),
            'fisher/l_k_mean': np.mean(all_l_k),
            'fisher/l_k_max': np.max(all_l_k),
            'fisher/l_k_min': np.min(all_l_k),
            'fisher/l_k_std': np.std(all_l_k),
        }
        
        # Store summary and return
        self.stats[identifier]['summary'] = summary_stats
        print(f"[FisherInfoAnalyzer] Aggregated Stats for '{identifier}': c_k_mean={summary_stats['fisher/c_k_mean']:.4f}, l_k_mean={summary_stats['fisher/l_k_mean']:.4f}")
        return summary_stats
