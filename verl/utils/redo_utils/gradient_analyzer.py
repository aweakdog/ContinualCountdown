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
import collections
from typing import Dict

@ray.remote(num_gpus=0.25, num_cpus=1)
class GradientAnalyzer:
    """
    A stateful Ray actor that analyzes gradients component-wise to avoid OOM errors.
    It accumulates statistics over multiple calls and provides an aggregated result, including
    per-component breakdowns.
    """
    def __init__(self):
        self.stats = collections.defaultdict(self._get_default_stats)
        print("[GradientAnalyzer] Actor initialized.")

    def _get_default_stats(self):
        """Helper to initialize the nested dictionary structure."""
        return {
            '__global__': {'zero': 0, 'total': 0},
            'components': {}
        }

    def reset(self, identifier: str):
        """Resets the statistics for a given analysis identifier."""
        self.stats[identifier] = self._get_default_stats()
        print(f"[GradientAnalyzer] Statistics reset for identifier: '{identifier}'.")

    def analyze_component_gradients(self, gradients: Dict[str, torch.Tensor], original_param_shapes: dict, tau: float, verbose: bool, identifier: str, component_name: str):
        """
        Calculates the zero-gradient space ratio for a single component, stores the per-component
        result, and accumulates the global total.
        """
        if verbose: print(f"[Analyzer] Analyzing component '{component_name}' for identifier '{identifier}'.")

        local_total_rows, local_zero_rows = self._calculate_stats_for_grads(gradients, original_param_shapes, tau, verbose)

        # Store component-specific stats
        component_ratio = local_zero_rows / local_total_rows if local_total_rows > 0 else 0.0
        self.stats[identifier]['components'][component_name] = {
            'zero': local_zero_rows,
            'total': local_total_rows,
            'ratio': component_ratio
        }

        # Accumulate global stats
        self.stats[identifier]['__global__']['zero'] += local_zero_rows
        self.stats[identifier]['__global__']['total'] += local_total_rows

        if verbose:
            print(f"  [Analyzer] Stats for '{component_name}': {self.stats[identifier]['components'][component_name]}.")

    def get_aggregated_stats(self, identifier: str, verbose: bool):
        """Computes and returns the final aggregated statistics, including the per-component breakdown."""
        full_stats = self.stats.get(identifier)
        if not full_stats or full_stats['__global__']['total'] == 0:
            if verbose: print(f"[Analyzer] No stats or total_rows is zero for identifier '{identifier}'.")
            return {}

        global_stats = full_stats['__global__']
        total_rows = global_stats['total']
        zero_rows = global_stats['zero']
        aggregated_ratio = zero_rows / total_rows if total_rows > 0 else 0.0
        
        # Add the final calculated ratio to the global stats dict before returning
        full_stats['__global__']['ratio'] = aggregated_ratio
        full_stats['__global__']['aggregated_ratio'] = aggregated_ratio  # for backward compatibility
        
        if verbose: print(f"[Analyzer] Returning final aggregated stats for '{identifier}': {full_stats}")

        return full_stats

    def _calculate_stats_for_grads(self, gradients, original_param_shapes, tau, verbose):
        total_rows = 0
        zero_rows = 0
        for name, grad in gradients.items():
            if grad is None: continue

            original_shape = original_param_shapes.get(name)
            if not original_shape: continue

            is_bias = name.endswith(".bias")
            effective_dim = len(original_shape)
            is_eligible = (effective_dim == 2) or (effective_dim == 1 and not is_bias)
            if not is_eligible: continue

            grad_to_process = grad.float()
            original_shape_size = torch.Size(original_shape)

            if grad_to_process.dim() == 1:
                if grad_to_process.numel() == original_shape_size.numel():
                    grad_to_process = grad_to_process.view(original_shape_size)
                elif len(original_shape_size) == 2:
                    H, W = original_shape_size
                    shard_numel = grad_to_process.numel()
                    if W > 0 and shard_numel % W == 0:
                        grad_to_process = grad_to_process.view(shard_numel // W, W)
                    elif W > 0:
                        num_full_rows = shard_numel // W
                        if num_full_rows > 0:
                            grad_to_process = grad_to_process[:num_full_rows * W].view(num_full_rows, W)
                        else:
                            grad_to_process = torch.empty((0, W), device=grad_to_process.device, dtype=grad_to_process.dtype)

            if grad_to_process.dim() == 1 and len(original_shape_size) == 1:
                grad_to_process = grad_to_process.unsqueeze(1)

            if grad_to_process.dim() != 2 or grad_to_process.shape[0] == 0: continue

            row_norms = torch.norm(grad_to_process, p=1, dim=1)
            H = grad_to_process.shape[0]
            avg_row_norm = row_norms.mean()
            s_i = row_norms / (avg_row_norm + 1e-9)
            num_dormant_neurons = (s_i < tau).sum().item()

            total_rows += H
            zero_rows += num_dormant_neurons
        return total_rows, zero_rows