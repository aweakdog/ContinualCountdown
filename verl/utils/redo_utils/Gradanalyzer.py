import bisect
import numpy as np
import torch
from collections import defaultdict

class GradientAnalyzer:
    def extract_full_state_gradients(self, fsdp_model):
        pass

    def compute_nullspace_ratio(self, grad_dict, batch_size=32, tol=1e-5, max_params_per_layer=100):
        """
        Compute the nullspace ratio using SVD on a sampled gradient matrix.
        Args:
            grad_dict: dict mapping parameter names to gradients (from extract_full_state_gradients)
            batch_size: number of samples (rows) in the gradient matrix (default: 32)
            tol: relative tolerance for singular value threshold
            max_params_per_layer: maximum number of parameters to sample per layer
        Returns:
            nullspace_ratio (float), zero_vector_ratio (float)
        Example:
            grad_dict = analyzer.extract_full_state_gradients(fsdp_model)
            nullspace, zero_ratio = analyzer.compute_nullspace_ratio(grad_dict)
        """
        import torch
        import numpy as np
        # Flatten and sample gradients per parameter (layer)
        grad_arrays = []
        for name, grad in grad_dict.items():
            if grad is not None:
                arr = grad.detach().cpu().view(-1)
                n = min(max_params_per_layer, arr.numel())
                if n > 0:
                    idx = torch.randperm(arr.numel())[:n]
                    grad_arrays.append(arr[idx])
        if not grad_arrays:
            print("[NullspaceMetric] No gradients found.")
            return None, None
        grad_matrix = torch.stack(grad_arrays, dim=1)  # shape: [num_samples, num_layers]
        grad_matrix = grad_matrix[:batch_size, :] if grad_matrix.size(0) > batch_size else grad_matrix
        # Normalize
        grad_matrix = grad_matrix / (grad_matrix.norm(dim=1, keepdim=True) + 1e-8)
        # Zero vector ratio
        zero_vectors = (grad_matrix.abs().sum(dim=1) < 1e-8).float().mean().item()
        # SVD
        U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
        relative_tol = tol * S[0]
        nullspace_dim = (S < relative_tol).sum().item()
        nullspace_ratio = nullspace_dim / grad_matrix.size(0)
        print(f"[NullspaceMetric] Nullspace ratio: {nullspace_ratio:.6f}, Zero vector ratio: {zero_vectors:.6f}")
        return nullspace_ratio, zero_vectors

    def print_dormant_metrics(self, grad_dict, top_n=5):
        """
        Compute and print dormant-neuron-related metrics from a grad_dict.
        Metrics:
          - Proportion of zero gradients (per parameter and global)
          - Top-N parameters with highest proportion of zero gradients
        Args:
            grad_dict: dict mapping parameter names to gradients (as returned by extract_full_state_gradients)
            top_n: number of most dormant parameters to print
        """
        import numpy as np
        param_zero_ratios = {}
        total_numel = 0
        total_zeros = 0
        for name, grad in grad_dict.items():
            if grad is not None:
                arr = grad.detach().cpu().numpy()
                zero_count = np.sum(arr == 0)
                numel = arr.size
                zero_ratio = zero_count / numel
                param_zero_ratios[name] = zero_ratio
                total_numel += numel
                total_zeros += zero_count
            else:
                param_zero_ratios[name] = None
        # Global dormant metric
        if total_numel > 0:
            global_zero_ratio = total_zeros / total_numel
            print(f"[DormantMetric] Global zero-gradient ratio: {global_zero_ratio:.6f}")
        else:
            print("[DormantMetric] No gradients found.")
        # Top-N most dormant parameters
        filtered = [(k, v) for k, v in param_zero_ratios.items() if v is not None]
        filtered.sort(key=lambda x: x[1], reverse=True)
        print(f"[DormantMetric] Top {top_n} most dormant parameters:")
        for i, (name, ratio) in enumerate(filtered[:top_n]):
            print(f"  {i+1}. {name}: zero-ratio={ratio:.6f}")

        """
        Extract per-parameter gradients using FSDP's full state dict.
        This is slow (all-gather across ranks) and should be called infrequently (e.g., every N steps).
        Only call this on rank 0 after backward.
        Returns: dict mapping parameter names to gradients (CPU tensors)
        """
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullyShardedDataParallel as FSDP
        import torch
        grad_dict = {}
        # Use FSDP context to get full state dict (unflattened params)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = fsdp_model.state_dict()
        for k, v in full_state_dict.items():
            # v is parameter tensor; try to get .grad if available
            if hasattr(v, 'grad') and v.grad is not None:
                grad_dict[k] = v.grad.cpu()
            else:
                # Try to get grad from model.named_parameters()
                param = dict(fsdp_model.named_parameters()).get(k, None)
                if param is not None and param.grad is not None:
                    grad_dict[k] = param.grad.detach().cpu()
                else:
                    grad_dict[k] = None
        return grad_dict

    def __init__(self, model, seed=None):
        """
        Initialize the gradient analyzer
        Args:
            model: The PyTorch model to analyze
            seed: Optional random seed
        """
        if seed is not None:
            print(f"Setting seed to {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.model = model
        self.params_info = []
        self.start_indices = []
        self._build_params_index()
        

    def _build_params_index(self):
        """Build parameter metadata index"""
        self.params_info = []
        current_idx = 0
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{name}.{param_name}" if name else param_name
                    param_size = param.numel()
                    self.params_info.append((
                        full_name,
                        current_idx,
                        current_idx + param_size
                    ))
                    current_idx += param_size
        self.start_indices = [start for (_, start, _) in self.params_info]