import torch
import numpy as np

class VerlGradientAnalyzer:
    """
    verl-compatible Gradient Analyzer for distributed model analysis.
    Computes nullspace_ratio and zero_grad_ratio, and can be extended for distributed use.
    """
    def __init__(self, model, seed=None):
        self.model = model
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._build_params_index()
        self.grad_buffer = None
        self._register_grad_hooks()

    def _build_params_index(self):
        self.params_info = []
        current_idx = 0
        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{name}.{param_name}" if name else param_name
                    param_size = param.numel()
                    self.params_info.append((full_name, current_idx, current_idx + param_size))
                    current_idx += param_size
        self.start_indices = [start for (_, start, _) in self.params_info]

    def _register_grad_hooks(self):
        self.grad_handles = []
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.grad_buffer = torch.zeros(total_params, device=next(self.model.parameters()).device)
        current_idx = 0
        for p in self.model.parameters():
            if p.requires_grad:
                param_numel = p.numel()
                start_idx = current_idx
                end_idx = start_idx + param_numel
                def hook(grad, start=start_idx, end=end_idx):
                    self.grad_buffer[start:end] = grad.contiguous().view(-1)
                handle = p.register_hook(hook)
                self.grad_handles.append(handle)
                current_idx += param_numel

    def analyze_gradients(self, return_counts=False):
        """
        Returns: (nullspace_ratio, zero_grad_ratio)
        If return_counts=True, also returns (zero_grad_count, total)
        """
        grad_flat = self.grad_buffer
        total = grad_flat.numel()
        zero_grad_count = (grad_flat == 0).sum().item()
        zero_grad_ratio = zero_grad_count / total
        nullspace_ratio = zero_grad_ratio  # For distributed, would need allreduce
        if return_counts:
            return nullspace_ratio, zero_grad_ratio, zero_grad_count, total
        return nullspace_ratio, zero_grad_ratio
