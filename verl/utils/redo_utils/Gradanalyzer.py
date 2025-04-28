import bisect
import numpy as np
import torch
from collections import defaultdict

class GradientAnalyzer:
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
        
        # Pre-register parameter hooks
        self.grad_buffer = None
        self._register_grad_hooks()

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

    def _register_grad_hooks(self):
        """Register gradient collection hooks"""
        self.grad_handles = []
        self.grad_shapes = []
        current_idx = 0
        
        # Pre-calculate total parameter count
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.grad_buffer = torch.zeros(total_params, device=next(self.model.parameters()).device)

        for p in self.model.parameters():
            if p.requires_grad:
                param_numel = p.numel()
                start_idx = current_idx
                end_idx = start_idx + param_numel

                # Use closure to capture index range
                def hook(grad, start=start_idx, end=end_idx):
                    self.grad_buffer[start:end] = grad.contiguous().view(-1)

                handle = p.register_hook(hook)
                self.grad_handles.append(handle)
                current_idx = end_idx

    def _compute_batch_gradients(self, inputs, param_indices):
        """Vectorized computation of batch gradients"""
        if isinstance(inputs, tuple):
            batch_size = inputs[0].size(0)
            outputs = self.model(*inputs)
        else:
            batch_size = inputs.size(0)
            outputs = self.model(inputs)


        
        # Create a pseudo loss function: sum of each sample's output
        if isinstance(outputs,tuple):
            outputs = torch.cat(outputs,dim=-1)
        pseudo_loss = outputs.sum(dim=tuple(range(1, outputs.ndim)))

        # Batch gradient computation
        gradients = []
        for i in range(batch_size):
            # Clear the gradient buffer
            self.grad_buffer.zero_()
            
            # Compute the gradient of a single sample
            grad_outputs = torch.zeros_like(outputs)
            grad_outputs[i] = 1.0
            torch.autograd.backward(
                outputs, 
                grad_tensors=grad_outputs,
                retain_graph=True
            )
            
            # Collect sampled gradients
            gradients.append(self.grad_buffer[param_indices].clone())

        return torch.stack(gradients)

    def analyze_gradients(self, inputs, num_params_to_sample=1000, tol=1e-5):
        # Here we calculate our plasticity metrics -- nullspace ratio
        # Set fixed random seed, we set num_params_to_sample to 1000 for fast computation, this can be changed via practice in other algorithms
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)

        
        # Device consistency verification
        model_device = next(self.model.parameters()).device
        if isinstance(inputs, tuple):
            assert inputs[0].device == model_device 
            batch_size = inputs[0].shape[0]
        else:
            assert inputs.device == model_device
            batch_size = inputs.shape[0]

        # Hierarchical proportional sampling
        layer_samples = []
        for layer_info in self.params_info:
            name, start, end = layer_info
            layer_size = end - start
            samples = min(100, layer_size)  # At least sample 100
            indices = torch.randint(start, end, (samples,))
            layer_samples.append(indices)
        param_indices = torch.cat(layer_samples)

        # increase sample count check
        if num_params_to_sample < param_indices.size(0) * 0.1:
            print(f"Warning: Sampling only {num_params_to_sample}/{param_indices.size(0)} parameters, consider increasing sample size")

        #  gradient computation use current batch
        grad_matrix = self._compute_batch_gradients(inputs, param_indices)

        # Add normalization before SVD
        grad_matrix = grad_matrix / (grad_matrix.norm(dim=1, keepdim=True) + 1e-8)

        # Calculate the proportion of zero vectors in the column vectors （ this is  a strcit style measearment for our gradian-based metrics）
        zero_vectors = (grad_matrix.abs().sum(dim=0) < 1e-8).float().mean().item()


        # SVD, we use singular values to evaluate the degradation of the gradient matrix (loss of plasticity), 
        # we call it nullspace ratio(Number of singular values below the threshold).
        # U: Left singular vectors (batch_size x batch_size matrix)
        # S: Singular values in descending order (vector of length min(batch_size, num_params))
        # Vh: Right singular vectors transposed (num_params x batch_size matrix)
        # Together these decompose grad_matrix as: grad_matrix = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(grad_matrix)
        relative_tol = tol * S[0]
        small_singular_indices = torch.where(S < relative_tol)[0]
        nullspace_dim = len(small_singular_indices)
        nullspace_ratio = nullspace_dim / batch_size


        importance = Vh.abs().sum(dim=0)  # Sum along the singular vector dimension

        # Map parameter index to layer name (using binary search for acceleration)
        selected_layers = []
        cpu_indices = param_indices.cpu().numpy()
        for idx in cpu_indices:
            pos = bisect.bisect_right(self.start_indices, idx) - 1
            selected_layers.append(
                self.params_info[pos][0] 
                if pos >=0 and self.params_info[pos][1] <= idx < self.params_info[pos][2]
                else "unknown"
            )

        # Aggregate layer contributions
        layer_contribution = {}
        for param_idx, layer_name in enumerate(selected_layers):
            contrib = importance[param_idx].item()
            layer_contribution[layer_name] = layer_contribution.get(layer_name, 0.0) + contrib

        # Normalize processing
        total_contribution = sum(layer_contribution.values())
        if total_contribution > 0:
            layer_contribution = {k: v/total_contribution for k, v in layer_contribution.items()}


        return nullspace_ratio, layer_contribution, zero_vectors

    def __del__(self):
        """Clean up hooks"""
        for handle in self.grad_handles:
            handle.remove()