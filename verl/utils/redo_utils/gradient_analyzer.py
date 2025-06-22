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

def calculate_zero_grad_ratio_from_full_grad(gradients, tau=0.1, verbose=True):
    """
    Calculates the zero-gradient space ratio from full, unsharded gradient tensors.

    Args:
        gradients (dict): A dictionary mapping parameter names (str) to their full gradient tensors (torch.Tensor).
        tau (float): The threshold below which a gradient norm is considered zero.
        verbose (bool): Whether to print detailed logs.

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

        # We are interested in weights (2D) and 1D non-bias parameters (e.g., LayerNorm)
        is_bias = name.endswith(".bias")
        
        # Determine effective dimension for analysis
        effective_dim = grad.dim()
        
        is_eligible = (effective_dim == 2) or (effective_dim == 1 and not is_bias)

        if not is_eligible:
            if verbose:
                print(f"[GradientAnalyzer] Skipping '{name}' (shape: {grad.shape}, dim: {grad.dim()}, is_bias: {is_bias}). Not a 2D or 1D non-bias parameter.")
            continue

        grad_to_process = grad.float()

        # Reshape 1D non-bias tensors to be processed like 2D tensors
        if grad_to_process.dim() == 1:
            grad_to_process = grad_to_process.unsqueeze(1) # Shape (N) -> (N, 1)

        if grad_to_process.shape[0] == 0:
            continue

        # Calculate the L1 norm for each row (neuron's output gradient)
        # For a (N, 1) tensor, this is just the absolute value.
        A_local_row_tensor = torch.norm(grad_to_process, p=1, dim=1)
        
        # H is the number of rows (neurons)
        H = grad_to_process.shape[0]
        
        # B is the sum of all row norms
        B = A_local_row_tensor.sum()
        
        # The metric `si` is defined as A / (B/H), which simplifies to A * H / B
        # We check if si < tau, which is equivalent to A < tau * B / H
        threshold = tau * B / H
        
        local_zero_rows = (A_local_row_tensor < threshold).sum().item()
        
        if verbose:
            print(f"[GradientAnalyzer] Analyzed '{name}' (shape: {grad.shape}): "
                  f"{local_zero_rows} / {H} zero-grad rows. "
                  f"Avg norm: {B/H:.4e}, Threshold: {threshold:.4e}")

        total_rows += H
        zero_rows += local_zero_rows

    ratio = zero_rows / (total_rows + 1e-8)
    
    if verbose:
        print(f"[GradientAnalyzer] Global Stats: Zero Rows: {zero_rows}, Total Rows: {total_rows}, Ratio: {ratio:.4f}")

    return {
        'zero': zero_rows,
        'total': total_rows,
        'ratio': ratio,
        'aggregated_ratio': ratio # For compatibility with dp_actor
    }


@ray.remote(num_cpus=1, num_gpus=1)
class GradientAnalyzer:
    """
    A Ray actor dedicated to analyzing gradients on a single GPU.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GradientAnalyzer] Actor initialized on device: {self.device}")

    def analyze_gradients(self, component_name: str, grad_state_dict: dict, tau: float, original_shapes_map: dict):
        """
        Receives gradients from a worker, moves them to its own device, and analyzes them.

        Args:
            component_name (str): The name of the model component being analyzed (e.g., 'actor').
            grad_state_dict (dict): A state dict of the gradients, where keys are param FQNs.
            tau (float): The threshold for zero-grad analysis.
            original_shapes_map (dict): A map from FQN to original torch.Size.

        Returns:
            dict: The analysis results.
        """
        print(f"[GradientAnalyzer] Received gradients for component '{component_name}' for analysis.")
        
        # Move gradients to the analyzer's device
        local_grads = {name: grad.to(self.device) for name, grad in grad_state_dict.items()}

        # Here, we can filter or select specific gradients if needed.
        # For now, we analyze all provided gradients.
        
        # We don't need the complex logic from the old function because we have full gradients.
        # We can write a simpler analysis function.
        stats = calculate_zero_grad_ratio_from_full_grad(local_grads, tau=tau, verbose=True)

        # Structure the results similarly to the old function for compatibility
        results = {
            component_name: stats,
            '__global__': stats
        }
        
        print(f"[GradientAnalyzer] Analysis complete for '{component_name}'. Ratio: {stats.get('ratio'):.4f}")
        return results
