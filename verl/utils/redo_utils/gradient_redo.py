"""
Utilities for dormant neuron detection and reset, adapted from grad-based-plasticity-metrics/utils/ReDo.py.
Includes: ModifiedGradientReDo with mask and reset logic for use in actor/critic modules.
"""
import math
from typing import Union
import torch
import torch.nn as nn
from torch import optim

class ModifiedGradientReDo:
    """
    Implements dormant neuron mask and reset logic based on gradients for Linear, Conv2d, LayerNorm layers.
    """
    @staticmethod
    def _get_layer_mask(layer: nn.Module, tau: float, mode: str = 'threshold', percentage: float = 0.01, max_percentage: float = 0.01):
        if layer.weight.grad is None:
            return torch.zeros(layer.out_features if hasattr(layer, 'out_features') else layer.weight.shape[0], dtype=torch.bool, device=layer.weight.device)
        if isinstance(layer, nn.Conv2d):
            grad_magnitude = layer.weight.grad.abs().mean(dim=(1, 2, 3))
        elif isinstance(layer, nn.Linear):
            grad_magnitude = layer.weight.grad.abs().mean(dim=1)
        elif isinstance(layer, nn.LayerNorm):
            grad_magnitude = layer.weight.grad.abs()
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
        if mode == 'threshold':
            mask = grad_magnitude < tau
        elif mode == 'percentage':
            k = max(1, int(len(grad_magnitude) * percentage))
            _, indices = torch.topk(grad_magnitude, k, largest=False)
            mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
            mask[indices] = True
        elif mode == 'hybrid':
            threshold_mask = grad_magnitude < tau
            k = max(1, int(len(grad_magnitude) * percentage))
            max_k = max(1, int(len(grad_magnitude) * max_percentage))
            if threshold_mask.sum() > max_k:
                combined_grad = grad_magnitude.clone()
                combined_grad[~threshold_mask] = float('inf')
                _, indices = torch.topk(combined_grad, max_k, largest=False)
                mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                mask[indices] = True
            else:
                mask = threshold_mask
        else:
            raise ValueError(f"Unknown redo_mode: {mode}")
        return mask

    @staticmethod
    def _reset_single_layer(layer: nn.Module, mask: torch.Tensor, use_lecun_init: bool = False):
        if torch.all(~mask):
            return
        with torch.no_grad():
            if isinstance(layer, nn.LayerNorm):
                layer.weight.data[mask] = torch.ones_like(layer.weight.data[mask])
                if layer.bias is not None:
                    layer.bias.data[mask] = 0.0
                return
            fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
            if use_lecun_init:
                stddev = 1. / math.sqrt(fan_in) if fan_in > 0 else 1.0
                torch.nn.init.trunc_normal_(layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0)
                layer.weight[mask] *= stddev
                if layer.bias is not None:
                    layer.bias.data[mask] = 0.0
            else:
                gain = nn.init.calculate_gain('relu', param=math.sqrt(5))
                std = gain / math.sqrt(fan_in) if fan_in > 0 else 1.0
                bound = math.sqrt(3.0) * std
                layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)
                if layer.bias is not None:
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)
            if layer.weight.grad is not None:
                layer.weight.grad[mask] = 0.0
            if layer.bias is not None and layer.bias.grad is not None:
                layer.bias.grad[mask] = 0.0
