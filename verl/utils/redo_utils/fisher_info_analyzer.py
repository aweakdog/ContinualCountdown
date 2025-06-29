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
from typing import List

@ray.remote(num_gpus=1, num_cpus=1)
class FisherInfoAnalyzer:
    """
    A stateful Ray actor to compute and track Empirical Fisher Information Matrix (EFIM)
    based metrics, specifically the condition number (C_K) and cumulative energy (L_K),
    as described in the user's pseudocode.
    """
    def __init__(self):
        """
        Initializes the analyzer.
        """
        print("[FisherInfoAnalyzer] Actor initialized.")
        self.running_C = 0.0
        self.cumulative_L = 0.0
        self.current_step_idx = 0

    def reset(self):
        """Resets the internal state of the analyzer."""
        self.running_C = 0.0
        self.cumulative_L = 0.0
        self.current_step_idx = 0
        print(f"[FisherInfoAnalyzer] State has been reset.")

    def analyze_fisher_info(self, per_micro_batch_grads: List[torch.Tensor], micro_batch_size: int, current_lr: float, global_step: int):
        """
        Computes C_K and L_K from a list of per-micro-batch gradients.

        Args:
            per_micro_batch_grads: A list of flattened gradient tensors, one for each micro-batch.
            micro_batch_size: The number of samples in each micro-batch.
            current_lr: The learning rate at the current step.
            global_step: The global training step, used for logging.
        """
        if not per_micro_batch_grads:
            print("[FisherInfoAnalyzer] Received an empty list of gradients. Skipping analysis.")
            return None

        # Move gradients to the GPU where the actor is running and stack into a Jacobian matrix.
        try:
            jacobian = torch.stack([g.cuda() for g in per_micro_batch_grads])  # Shape: [num_micro_batches, param_dim]
        except Exception as e:
            print(f"[FisherInfoAnalyzer] Error stacking gradients: {e}")
            return None

        # Compute the reduced Fisher matrix F_tilde = J @ J.T
        # This results in a small matrix of shape [num_micro_batches, num_micro_batches].
        fisher_tilde = jacobian @ jacobian.T

        try:
            # Compute eigenvalues. eigvalsh is for symmetric matrices and is more efficient.
            eigenvalues = torch.linalg.eigvalsh(fisher_tilde)
            # Filter out near-zero eigenvalues to avoid numerical instability.
            non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-8]

            if len(non_zero_eigenvalues) < 2:
                print(f"[FisherInfoAnalyzer] Not enough non-zero eigenvalues ({len(non_zero_eigenvalues)}) to compute condition number. Skipping.")
                return None

            # Calculate c_k (condition number of the EFIM)
            sigma_max = torch.sqrt(non_zero_eigenvalues.max())
            sigma_min = torch.sqrt(non_zero_eigenvalues.min())
            c_k = sigma_max / sigma_min

            # Calculate the trace of the Fisher matrix
            trace_F = torch.trace(fisher_tilde)

            # Update the running average for C_K (average condition number)
            self.running_C += c_k.item()
            C_K = self.running_C / (self.current_step_idx + 1)

            # Update the cumulative L_K (total energy)
            l_k = (current_lr / micro_batch_size) * torch.sqrt(trace_F)
            self.cumulative_L += l_k.item()
            L_K = self.cumulative_L

            self.current_step_idx += 1

            stats = {
                'fisher/c_k_step': c_k.item(),
                'fisher/trace_F_step': trace_F.item(),
                'fisher/C_K_running_avg': C_K,
                'fisher/L_K_cumulative': L_K,
                'fisher/sigma_max': sigma_max.item(),
                'fisher/sigma_min': sigma_min.item(),
            }
            print(f"[FisherInfoAnalyzer] Step {global_step}: {stats}")
            return stats

        except torch.linalg.LinAlgError as e:
            print(f"[FisherInfoAnalyzer] A linear algebra error occurred during eigenvalue computation: {e}. Skipping analysis for this step.")
            return None
