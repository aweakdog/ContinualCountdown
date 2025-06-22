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

def calculate_zero_grad_ratio_from_full_grad(gradients: dict, original_param_shapes: dict, tau=0.1, verbose=True):
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
    total_rows = 0
    zero_rows = 0
    
    if verbose:
        print(f"[GradientAnalyzer] Starting analysis with tau={tau}. Analyzing {len(gradients)} tensors.")

    for name, grad in gradients.items():
        if grad is None:
            continue

        is_bias = name.endswith(".bias")
        original_shape = original_param_shapes.get(name)

        # Determine effective dimension for analysis using original shapes if available
        effective_dim = len(original_shape) if original_shape else grad.dim()
        
        is_eligible = (effective_dim == 2) or (effective_dim == 1 and not is_bias)

        if not is_eligible:
            if verbose:
                print(f"[GradientAnalyzer] Skipping '{name}' (shape: {grad.shape}, original_dim: {effective_dim}, is_bias: {is_bias}). Not eligible.")
            continue

        grad_to_process = grad.float()

        # Reshape flattened 2D tensors back to their original shape
        if original_shape and len(original_shape) > 1 and grad_to_process.dim() == 1:
            if grad_to_process.numel() == original_shape.numel():
                grad_to_process = grad_to_process.reshape(original_shape)
            else:
                if verbose:
                    print(f"[GradientAnalyzer] Skipping reshape for '{name}' due to numel mismatch: grad ({grad_to_process.numel()}) vs original ({original_shape.numel()})")
                continue

        # Reshape 1D non-bias tensors to be processed like 2D tensors
        if grad_to_process.dim() == 1:
            grad_to_process = grad_to_process.unsqueeze(1) # Shape (N) -> (N, 1)

        if grad_to_process.shape[0] == 0:
            continue

        # Calculate the L1 norm for each row (neuron's output gradient)
        A_local_row_tensor = torch.norm(grad_to_process, p=1, dim=1)
        
        H = grad_to_process.shape[0]
        B = A_local_row_tensor.sum()
        
        avg_norm = B / H if H > 0 else 0
        threshold = tau * avg_norm
        
        num_zero_rows = (A_local_row_tensor < threshold).sum().item()
        ratio = num_zero_rows / H if H > 0 else 0

        if verbose:
            print(f"[GradientAnalyzer] Analyzed '{name}' (shape: {grad.shape}, reshaped_to: {grad_to_process.shape}): {num_zero_rows} / {H} zero-grad rows ({ratio:.2%}). Avg norm: {avg_norm:.4e}, Threshold: {threshold:.4e}")

        total_rows += H
        zero_rows += num_zero_rows

    aggregated_ratio = zero_rows / total_rows if total_rows > 0 else 0
    if verbose:
        print(f"[GradientAnalyzer] Global Stats: Zero Rows: {zero_rows}, Total Rows: {total_rows}, Ratio: {aggregated_ratio:.4f}")
        
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
        if self.device is None:
            # Automatically select the device assigned by Ray
            self.device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if verbose: print(f"[GradientAnalyzer] Actor '{identifier}' initialized on device: {self.device}")

        # Move gradients to the actor's device
        device_gradients = {name: grad.to(self.device) for name, grad in gradients.items()}
        
        results = calculate_zero_grad_ratio_from_full_grad(
            device_gradients,
            original_param_shapes,
            tau=tau,
            verbose=verbose
        )
        return results
