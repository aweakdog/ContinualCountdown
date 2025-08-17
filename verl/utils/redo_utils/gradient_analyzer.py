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

@ray.remote(num_gpus=2, num_cpus=8)  # Multi-GPU support for parallel gradient analysis
class GradientAnalyzer:
    """
    A stateful Ray actor that analyzes gradients component-wise to avoid OOM errors.
    It accumulates statistics over multiple calls and provides an aggregated result, including
    per-component breakdowns.
    """
    def __init__(self):
        self.stats = collections.defaultdict(self._get_default_stats)
        
        # Multi-GPU device management
        if torch.cuda.is_available():
            self.available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
            self.primary_device = self.available_devices[0]
            torch.cuda.set_device(self.primary_device)
            print(f"[GradientAnalyzer] Actor initialized on {len(self.available_devices)} GPUs: {self.available_devices}")
        else:
            self.available_devices = [torch.device('cpu')]
            self.primary_device = self.available_devices[0]
            print("[GradientAnalyzer] Actor initialized on CPU (no CUDA available)")

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
        
        # Ensure stats are properly initialized for this identifier
        if identifier not in self.stats:
            self.stats[identifier] = self._get_default_stats()

        local_total_rows, local_zero_rows, per_matrix_stats = self._calculate_stats_for_grads(gradients, original_param_shapes, tau, verbose)

        # Store component-specific stats, now including per-matrix details
        component_ratio = local_zero_rows / local_total_rows if local_total_rows > 0 else 0.0
        self.stats[identifier]['components'][component_name] = {
            'zero': local_zero_rows,
            'total': local_total_rows,
            'ratio': component_ratio,
            'matrices': per_matrix_stats
        }

        # Accumulate global stats
        old_zero = self.stats[identifier]['__global__']['zero']
        old_total = self.stats[identifier]['__global__']['total']
        self.stats[identifier]['__global__']['zero'] += local_zero_rows
        self.stats[identifier]['__global__']['total'] += local_total_rows
        
        if verbose:
            print(f"  [Analyzer] Component '{component_name}': local_zero={local_zero_rows}, local_total={local_total_rows}")
            print(f"  [Analyzer] Global stats updated: zero {old_zero} -> {self.stats[identifier]['__global__']['zero']}, total {old_total} -> {self.stats[identifier]['__global__']['total']}")
            print(f"  [Analyzer] Component '{component_name}' stored. Total components now: {len(self.stats[identifier]['components'])}")
            print(f"  [Analyzer] Current component names: {list(self.stats[identifier]['components'].keys())}")
            print(f"  [Analyzer] Finished component '{component_name}'.")

    def analyze_component_gradients_batched(self, per_mini_batch_grads: list, original_param_shapes: dict, tau: float, verbose: bool, identifier: str, component_name: str):
        """
        Analyzes gradients from multiple mini-batches for a single component.
        Aggregates gradients across mini-batches before analysis to get a representative gradient.
        
        Args:
            per_mini_batch_grads: List of gradient dictionaries, one per mini-batch
            original_param_shapes: Dictionary of original parameter shapes
            tau: Threshold for zero gradient detection
            verbose: Whether to print debug information
            identifier: Analysis identifier (e.g., 'actor')
            component_name: Name of the component being analyzed
        """
        if verbose: 
            print(f"[Analyzer] Analyzing component '{component_name}' with {len(per_mini_batch_grads)} mini-batches for identifier '{identifier}'.")
        
        # Ensure stats are properly initialized for this identifier
        if identifier not in self.stats:
            self.stats[identifier] = self._get_default_stats()
        
        if not per_mini_batch_grads:
            print(f"[WARNING][Analyzer] No mini-batch gradients provided for component '{component_name}'")
            return
        
        # Aggregate gradients across mini-batches
        # Strategy: Sum all mini-batch gradients to get the total accumulated gradient
        aggregated_gradients = {}
        
        # Initialize with first mini-batch
        first_batch = per_mini_batch_grads[0]
        for param_name, grad_tensor in first_batch.items():
            aggregated_gradients[param_name] = grad_tensor.clone()
        
        # Add remaining mini-batches
        for batch_idx, batch_grads in enumerate(per_mini_batch_grads[1:], 1):
            for param_name, grad_tensor in batch_grads.items():
                if param_name in aggregated_gradients:
                    aggregated_gradients[param_name] += grad_tensor
                else:
                    print(f"[WARNING][Analyzer] Parameter '{param_name}' not found in previous batches, adding separately")
                    aggregated_gradients[param_name] = grad_tensor.clone()
        
        if verbose:
            print(f"[Analyzer] Aggregated {len(per_mini_batch_grads)} mini-batches for component '{component_name}'")
            print(f"[Analyzer] Total parameters in aggregated gradients: {len(aggregated_gradients)}")
        
        # Analyze the aggregated gradients using existing logic
        local_total_rows, local_zero_rows, per_matrix_stats = self._calculate_stats_for_grads(aggregated_gradients, original_param_shapes, tau, verbose)
        
        # Store component-specific stats, now including per-matrix details
        component_ratio = local_zero_rows / local_total_rows if local_total_rows > 0 else 0.0
        self.stats[identifier]['components'][component_name] = {
            'zero': local_zero_rows,
            'total': local_total_rows,
            'ratio': component_ratio,
            'matrices': per_matrix_stats,
            'mini_batches_analyzed': len(per_mini_batch_grads)  # Track how many mini-batches were aggregated
        }
        
        # Accumulate global stats
        old_zero = self.stats[identifier]['__global__']['zero']
        old_total = self.stats[identifier]['__global__']['total']
        self.stats[identifier]['__global__']['zero'] += local_zero_rows
        self.stats[identifier]['__global__']['total'] += local_total_rows
        
        if verbose:
            print(f"  [Analyzer] Component '{component_name}': aggregated from {len(per_mini_batch_grads)} mini-batches")
            print(f"  [Analyzer] Component '{component_name}': local_zero={local_zero_rows}, local_total={local_total_rows}")
            print(f"  [Analyzer] Global stats updated: zero {old_zero} -> {self.stats[identifier]['__global__']['zero']}, total {old_total} -> {self.stats[identifier]['__global__']['total']}")
            print(f"  [Analyzer] Component '{component_name}' stored. Total components now: {len(self.stats[identifier]['components'])}")
            print(f"  [Analyzer] Finished batched analysis for component '{component_name}'.")

    def get_aggregated_stats(self, identifier: str, verbose: bool):
        """Computes and returns the final aggregated statistics, including the per-component breakdown."""
        full_stats = self.stats.get(identifier)
        
        # Handle multi-instance Ray scenario: if global stats are empty but components exist,
        # manually aggregate from component statistics
        if not full_stats or full_stats['__global__']['total'] == 0:
            if verbose: print(f"[Analyzer] Global stats empty for '{identifier}', checking for component-level aggregation...")
            
            # Check if we have component-level statistics to aggregate
            if full_stats and full_stats.get('components'):
                if verbose: print(f"[Analyzer] Found {len(full_stats['components'])} components, manually aggregating...")
                
                # Manually aggregate from all components
                total_zero = 0
                total_rows = 0
                
                for comp_name, comp_stats in full_stats['components'].items():
                    comp_zero = comp_stats.get('zero', 0)
                    comp_total = comp_stats.get('total', 0)
                    total_zero += comp_zero
                    total_rows += comp_total
                    if verbose: print(f"[Analyzer] Component '{comp_name}': zero={comp_zero}, total={comp_total}")
                
                # Update global stats with aggregated values
                full_stats['__global__']['zero'] = total_zero
                full_stats['__global__']['total'] = total_rows
                
                aggregated_ratio = total_zero / total_rows if total_rows > 0 else 0.0
                full_stats['__global__']['ratio'] = aggregated_ratio
                full_stats['__global__']['aggregated_ratio'] = aggregated_ratio
                
                if verbose: print(f"[Analyzer] Manual aggregation complete: zero={total_zero}, total={total_rows}, ratio={aggregated_ratio:.6f}")
                return full_stats
            else:
                if verbose: print(f"[Analyzer] No stats or components found for identifier '{identifier}'.")
                return {}

        global_stats = full_stats['__global__']
        total_rows = global_stats['total']
        zero_rows = global_stats['zero']
        aggregated_ratio = zero_rows / total_rows if total_rows > 0 else 0.0
        
        # Add the final calculated ratio to the global stats dict before returning
        full_stats['__global__']['ratio'] = aggregated_ratio
        full_stats['__global__']['aggregated_ratio'] = aggregated_ratio  # for backward compatibility
        
        return full_stats

    def _calculate_stats_for_grads(self, gradients, original_param_shapes, tau, verbose):
        total_rows = 0
        zero_rows = 0
        per_matrix_stats = {}
        if verbose: print(f"    [Analyzer Internals] Processing {len(gradients)} gradients with OPTIMIZED PARALLEL processing.")

        # OPTIMIZED PARALLEL PROCESSING: Use CUDA streams for true GPU parallelism
        num_devices = len(self.available_devices)
        device_batches = {device: [] for device in self.available_devices}
        device_streams = {device: torch.cuda.Stream(device) for device in self.available_devices if device.type == 'cuda'}
        
        # Step 1: Distribute parameters across devices
        for i, (name, grad) in enumerate(gradients.items()):
            if grad is None: continue
            
            device_idx = i % num_devices
            device_to_use = self.available_devices[device_idx]
            device_batches[device_to_use].append((name, grad))
            
            if verbose: 
                print(f"      [Param: {name}] - Assigned to device: {device_to_use} (batch size: {len(device_batches[device_to_use])})")
        
        # Step 2: Process all device batches in parallel using CUDA streams
        device_results = {}
        
        for device, param_batch in device_batches.items():
            if not param_batch:  # Skip empty batches
                continue
                
            if device.type == 'cuda':
                stream = device_streams[device]
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    device_results[device] = self._process_device_batch_optimized(
                        device, param_batch, original_param_shapes, tau, verbose
                    )
            else:
                # CPU fallback
                device_results[device] = self._process_device_batch_optimized(
                    device, param_batch, original_param_shapes, tau, verbose
                )
        
        # Step 3: Synchronize all streams and collect results
        for device in device_streams:
            torch.cuda.synchronize(device)
        
        # Step 4: Aggregate results from all devices
        for device, (device_stats, device_total_rows, device_zero_rows) in device_results.items():
            per_matrix_stats.update(device_stats)
            total_rows += device_total_rows
            zero_rows += device_zero_rows
            
            if verbose:
                print(f"    [Device {device}] Completed: {device_total_rows} total rows, {device_zero_rows} zero rows")
        
        if verbose: print(f"    [Analyzer] PARALLEL processing complete: {total_rows} total rows, {zero_rows} zero rows")
        return total_rows, zero_rows, per_matrix_stats
    
    def _process_device_batch_optimized(self, device, param_batch, original_param_shapes, tau, verbose):
        """Optimized batch processing for a single device"""
        device_stats = {}
        device_total_rows = 0
        device_zero_rows = 0
        
        if verbose: print(f"    [Device {device}] Processing {len(param_batch)} parameters")
        
        for name, grad in param_batch:
            original_shape = original_param_shapes.get(name)
            if not original_shape:
                if verbose: print(f"        -> Skipping: Name not found in original_param_shapes map.")
                continue

            is_bias = name.endswith(".bias")
            effective_dim = len(original_shape)
            is_eligible = (effective_dim == 2) or (effective_dim == 1 and not is_bias)
            if not is_eligible:
                if verbose: print(f"        -> Skipping: Not eligible (dim={effective_dim}, is_bias={is_bias}).")
                continue

            # Move gradient to assigned device for multi-GPU processing
            if grad.device != device:
                grad_to_process = grad.float().to(device, non_blocking=True)
            else:
                grad_to_process = grad.float()
            original_shape_size = torch.Size(original_shape)
            
            if verbose: 
                print(f"        -> [Device {device}] Processing {name}: {original_shape_size}")

            # --- Reshaping Logic for FSDP --- 
            if grad_to_process.dim() == 1:
                if verbose: print(f"        -> Is 1D tensor, attempting reshape.")
                if grad_to_process.numel() == original_shape_size.numel():
                    grad_to_process = grad_to_process.view(original_shape_size)
                elif len(original_shape_size) == 2: # Shard of a 2D tensor
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
                grad_to_process = grad_to_process.unsqueeze(1) # Reshape 1D LayerNorm weights to (N, 1)

            if verbose: print(f"        -> Shape after reshape: {grad_to_process.shape}")

            if grad_to_process.dim() != 2 or grad_to_process.shape[0] == 0:
                if verbose: print(f"        -> Skipping: Final shape is not a non-empty 2D tensor.")
                continue

            # --- Neuron Analysis ---
            row_norms = torch.norm(grad_to_process, p=1, dim=1)
            H = grad_to_process.shape[0]
            
            if verbose:
                print(f"        -> Computing on device: {row_norms.device} (grad_to_process: {grad_to_process.device})")

            if H > 0:
                min_row_norm = row_norms.min()
                max_row_norm = row_norms.max()
                avg_row_norm = row_norms.mean()
                s_i = row_norms / (avg_row_norm + 1e-9)
                num_dormant_neurons = (s_i < tau).sum().item()
                matrix_ratio = num_dormant_neurons / H
                
                # Additional debugging info
                s_i_min = s_i.min().item()
                s_i_max = s_i.max().item()
                s_i_mean = s_i.mean().item()
                below_tau_count = (s_i < tau).sum().item()
                
                if verbose: 
                    print(f"        -> Analysis: {num_dormant_neurons}/{H} dormant ({matrix_ratio:.2%}). Norms (min/avg/max): {min_row_norm:.4e} / {avg_row_norm:.4e} / {max_row_norm:.4e}")
                    print(f"        -> s_i stats (min/mean/max): {s_i_min:.4f} / {s_i_mean:.4f} / {s_i_max:.4f}, tau={tau}, below_tau={below_tau_count}")
            else:
                min_row_norm, max_row_norm, avg_row_norm = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                num_dormant_neurons = 0
                matrix_ratio = 0.0
                
                if verbose: print(f"        -> Analysis: {num_dormant_neurons}/{H} dormant (empty matrix)")

            # Store per-matrix stats
            device_stats[name] = {
                'zero': num_dormant_neurons,
                'total': H,
                'ratio': matrix_ratio,
                'avg_row_norm': avg_row_norm.item(),
                'min_row_norm': min_row_norm.item(),
                'max_row_norm': max_row_norm.item(),
            }

            device_total_rows += H
            device_zero_rows += num_dormant_neurons

        if verbose: print(f"    [Device {device}] Batch complete: {device_total_rows} total rows, {device_zero_rows} zero rows")
        return device_stats, device_total_rows, device_zero_rows