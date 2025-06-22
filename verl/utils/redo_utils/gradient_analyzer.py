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

def calculate_zero_grad_ratio_from_full_grad(gradients: Dict[str, torch.Tensor], original_param_shapes: dict, tau: float, verbose: bool):
    """
    Calculates the zero-gradient space ratio from a dictionary of full, unsharded gradients.
    
    Args:
        gradients (dict): A dictionary mapping parameter names to their full gradient tensors.
        original_param_shapes (dict): A map from FQN to original torch.Size.
        tau (float): The threshold for considering a gradient norm to be zero.
        verbose (bool): Whether to print detailed analysis for each tensor.

    Returns:
        dict: A dictionary containing statistics like total rows, zero rows, and the ratio.
    """
    if verbose:
        print(f"[Analyzer] Received {len(gradients)} gradients for analysis with tau={tau}.")

    total_rows = 0
    zero_rows = 0

    for name, grad in gradients.items():
        if grad is None:
            if verbose:
                print(f"  [Analyzer] Skipping {name}: Gradient is None.")
            continue

        original_shape = original_param_shapes.get(name)
        if not original_shape:
            if verbose:
                print(f"  [Analyzer] Skipping {name}: No original shape found.")
            continue

        # Determine eligibility based on original shape
        is_bias = name.endswith(".bias")
        effective_dim = len(original_shape)
        
        # We process 2D matrices and 1D non-bias vectors (like LayerNorm weights)
        is_eligible = (effective_dim == 2) or (effective_dim == 1 and not is_bias)

        if not is_eligible:
            if verbose:
                print(f"  [Analyzer] Skipping '{name}' (original shape: {original_shape}). Not eligible (dim={effective_dim}, is_bias={is_bias}).")
            continue

        grad_to_process = grad.float()

        # Reshape flattened tensors back to their original shape if necessary
        if grad_to_process.dim() == 1 and grad_to_process.numel() == torch.Size(original_shape).numel():
             grad_to_process = grad_to_process.view(original_shape)
        
        # Reshape 1D non-bias tensors to be processed like 2D tensors
        if grad_to_process.dim() == 1:
            grad_to_process = grad_to_process.unsqueeze(1) # Shape (N) -> (N, 1)

        if grad_to_process.dim() != 2:
            if verbose:
                print(f"  [Analyzer] Skipping '{name}' after reshape attempt. Final dim is not 2 (shape: {grad_to_process.shape}).")
            continue

        if grad_to_process.shape[0] == 0:
            continue

        # Calculate the L1 norm for each row
        row_norms = torch.norm(grad_to_process, p=1, dim=1)
        
        # Count rows where the norm is below the absolute threshold tau
        num_zero_rows = (row_norms < tau).sum().item()
        H = grad_to_process.shape[0]
        ratio = num_zero_rows / H if H > 0 else 0

        if verbose:
            print(f"  [Analyzer] Analyzed '{name}' (shape: {grad_to_process.shape}): {num_zero_rows} / {H} zero-grad rows ({ratio:.2%}).")

        total_rows += H
        zero_rows += num_zero_rows

    aggregated_ratio = zero_rows / total_rows if total_rows > 0 else 0
    if verbose:
        print(f"[Analyzer] Global Stats: Zero Rows: {zero_rows}, Total Rows: {total_rows}, Ratio: {aggregated_ratio:.4f}")
        
    if total_rows == 0:
        if verbose:
            print("[Analyzer] No eligible parameters found for analysis. Returning empty stats.")
        return {}

    return {
        '__global__': {
            'zero': zero_rows,
            'total': total_rows,
            'ratio': aggregated_ratio,
            'aggregated_ratio': aggregated_ratio
        }
    }


@ray.remote
class GradientAnalyzer:
    """
    A Ray actor that performs gradient analysis on a dedicated device (preferably a GPU).
    """
    def __init__(self):
        self.device = None # Lazy initialization

    def analyze_gradients(self, gradients: dict, original_param_shapes: dict, tau: float, verbose: bool, identifier: str):
        try:
            if self.device is None:
                gpu_ids = ray.get_gpu_ids()
                if gpu_ids and torch.cuda.is_available():
                    # Inside a Ray actor, the assigned GPU is always ordinal 0 from torch's perspective
                    self.device = torch.device("cuda:0")
                else:
                    self.device = torch.device("cpu")
                
                if verbose: 
                    print(f"[GradientAnalyzer] Actor '{identifier}' initialized on device: {self.device}")

            device_gradients = {name: grad.to(self.device) for name, grad in gradients.items()}
            
            results = calculate_zero_grad_ratio_from_full_grad(
                device_gradients,
                original_param_shapes,
                tau=tau,
                verbose=verbose
            )
            return results
        except Exception as e:
            if verbose:
                print(f"[ERROR][GradientAnalyzer] Analysis failed for '{identifier}': {e}")
                import traceback
                traceback.print_exc()
            return None
