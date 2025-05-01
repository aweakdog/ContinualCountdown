"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
"""
import torch
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDP
import torch.nn as nn


def get_fsdp_flat_param_index_map(fsdp_module):
    """
    Returns a list of dicts with start, end, shape, and param reference for each original parameter in the flat param.
    """
    flat_param = next((p for p in fsdp_module.parameters() if isinstance(p, FlatParameter)), None)
    if flat_param is None:
        raise RuntimeError("No FlatParameter found in FSDP module")
    param_shapes = flat_param._param_shapes  # List of shapes for each original param
    param_numels = [torch.Size(shape).numel() for shape in param_shapes]
    offsets = [0] + list(torch.cumsum(torch.tensor(param_numels[:-1]), dim=0).tolist())
    # Optionally, get the original param refs (if available)
    param_refs = getattr(flat_param, '_params', [None]*len(param_shapes))
    index_map = []
    for i, (shape, start) in enumerate(zip(param_shapes, offsets)):
        end = start + param_numels[i]
        index_map.append({'index': i, 'start': start, 'end': end, 'shape': shape, 'param': param_refs[i]})
    return index_map, flat_param


def compute_fsdp_dormant_mask_only(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01):
    """
    Computes and returns the dormant neuron mask for the FSDP flat parameter, but does NOT reset weights.
    """
    index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
    flat_grad = flat_param.grad
    global_mask = torch.zeros_like(flat_param.data, dtype=torch.bool)
    for entry in index_map:
        shape = entry['shape']
        grad_slice = flat_grad[entry['start']:entry['end']].view(shape)
        if len(shape) == 2:  # Linear
            grad_magnitude = grad_slice.abs().mean(dim=1)
            if mode == 'threshold':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                mask = normalized_grad <= tau
            elif mode == 'percentage':
                k = max(1, int(len(grad_magnitude) * percentage))
                threshold = torch.kthvalue(grad_magnitude, k).values if k < len(grad_magnitude) else torch.min(grad_magnitude)
                mask = grad_magnitude <= threshold
            elif mode == 'hybrid':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                threshold_mask = normalized_grad <= tau
                k_max = max(1, int(len(grad_magnitude) * max_percentage))
                if threshold_mask.sum() > k_max:
                    combined_grad = grad_magnitude.clone()
                    combined_grad[~threshold_mask] = float('inf')
                    _, indices = torch.topk(combined_grad, k_max, largest=False)
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                    mask[indices] = True
                else:
                    mask = threshold_mask
            else:
                raise ValueError(f"Unknown mode: {mode}")
            mask_full = mask[:, None].expand_as(grad_slice)
            global_mask[entry['start']:entry['end']] = mask_full.reshape(-1)
        elif len(shape) == 4:  # Conv2d
            grad_magnitude = grad_slice.abs().mean(dim=(1,2,3))
            if mode == 'threshold':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                mask = normalized_grad <= tau
            elif mode == 'percentage':
                k = max(1, int(len(grad_magnitude) * percentage))
                threshold = torch.kthvalue(grad_magnitude, k).values if k < len(grad_magnitude) else torch.min(grad_magnitude)
                mask = grad_magnitude <= threshold
            elif mode == 'hybrid':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                threshold_mask = normalized_grad <= tau
                k_max = max(1, int(len(grad_magnitude) * max_percentage))
                if threshold_mask.sum() > k_max:
                    combined_grad = grad_magnitude.clone()
                    combined_grad[~threshold_mask] = float('inf')
                    _, indices = torch.topk(combined_grad, k_max, largest=False)
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                    mask[indices] = True
                else:
                    mask = threshold_mask
            else:
                raise ValueError(f"Unknown mode: {mode}")
            mask_full = mask[:, None, None, None].expand_as(grad_slice)
            global_mask[entry['start']:entry['end']] = mask_full.reshape(-1)
        elif len(shape) == 1:  # LayerNorm or bias
            grad_magnitude = grad_slice.abs()
            mask = grad_magnitude <= tau if mode == 'threshold' else torch.zeros_like(grad_magnitude, dtype=torch.bool)
            global_mask[entry['start']:entry['end']] = mask
    return global_mask

def lecun_reset(param_slice, mask, shape):
    """LeCun normal initialization for masked neurons."""
    if len(shape) == 2:  # Linear
        fan_in = shape[1]
        stddev = 1. / (fan_in ** 0.5) if fan_in > 0 else 1.0
        param_slice[mask, :] = torch.empty_like(param_slice[mask, :]).normal_(0, stddev)
    elif len(shape) == 4:  # Conv2d
        fan_in = shape[1] * shape[2] * shape[3]
        stddev = 1. / (fan_in ** 0.5) if fan_in > 0 else 1.0
        param_slice[mask, :, :, :] = torch.empty_like(param_slice[mask, :, :, :]).normal_(0, stddev)


def kaiming_reset(param_slice, mask, shape):
    """Kaiming uniform initialization for masked neurons."""
    if len(shape) == 2:  # Linear
        fan_in = shape[1]
        bound = (6.0 / fan_in) ** 0.5 if fan_in > 0 else 1.0
        param_slice[mask, :] = torch.empty_like(param_slice[mask, :]).uniform_(-bound, bound)
    elif len(shape) == 4:  # Conv2d
        fan_in = shape[1] * shape[2] * shape[3]
        bound = (6.0 / fan_in) ** 0.5 if fan_in > 0 else 1.0
        param_slice[mask, :, :, :] = torch.empty_like(param_slice[mask, :, :, :]).uniform_(-bound, bound)


def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True):
    """
    Computes the dormant neuron mask for the FSDP flat parameter and resets the weights of dormant neurons.
    Args:
        fsdp_module: The FSDP-wrapped module.
        mode: Dormant neuron selection mode ('threshold', 'percentage', 'hybrid').
        tau: Threshold for dormant selection.
        percentage: Fraction of neurons to consider dormant.
        max_percentage: Max fraction for hybrid mode.
        use_lecun_init: If True, use LeCun initialization; else Kaiming uniform.
    Returns:
        global_mask: The boolean mask of dormant neurons (same shape as flat param).
    """
    index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
    flat_grad = flat_param.grad
    global_mask = torch.zeros_like(flat_param.data, dtype=torch.bool)
    device = flat_param.device
    for entry in index_map:
        shape = entry['shape']
        grad_slice = flat_grad[entry['start']:entry['end']].view(shape)
        param_slice = flat_param.data[entry['start']:entry['end']].view(shape)
        if len(shape) == 2:  # Linear
            grad_magnitude = grad_slice.abs().mean(dim=1)
            if mode == 'threshold':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                mask = normalized_grad <= tau
            elif mode == 'percentage':
                k = max(1, int(len(grad_magnitude) * percentage))
                threshold = torch.kthvalue(grad_magnitude, k).values if k < len(grad_magnitude) else torch.min(grad_magnitude)
                mask = grad_magnitude <= threshold
            elif mode == 'hybrid':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                threshold_mask = normalized_grad <= tau
                k_max = max(1, int(len(grad_magnitude) * max_percentage))
                if threshold_mask.sum() > k_max:
                    combined_grad = grad_magnitude.clone()
                    combined_grad[~threshold_mask] = float('inf')
                    _, indices = torch.topk(combined_grad, k_max, largest=False)
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                    mask[indices] = True
                else:
                    mask = threshold_mask
            else:
                raise ValueError(f"Unknown mode: {mode}")
            mask_full = mask[:, None].expand_as(grad_slice)
            global_mask[entry['start']:entry['end']] = mask_full.reshape(-1)
            # Reset weights for masked neurons
            if mask.any():
                with torch.no_grad():
                    if use_lecun_init:
                        lecun_reset(param_slice, mask, shape)
                    else:
                        kaiming_reset(param_slice, mask, shape)
        elif len(shape) == 4:  # Conv2d
            grad_magnitude = grad_slice.abs().mean(dim=(1,2,3))
            if mode == 'threshold':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                mask = normalized_grad <= tau
            elif mode == 'percentage':
                k = max(1, int(len(grad_magnitude) * percentage))
                threshold = torch.kthvalue(grad_magnitude, k).values if k < len(grad_magnitude) else torch.min(grad_magnitude)
                mask = grad_magnitude <= threshold
            elif mode == 'hybrid':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                threshold_mask = normalized_grad <= tau
                k_max = max(1, int(len(grad_magnitude) * max_percentage))
                if threshold_mask.sum() > k_max:
                    combined_grad = grad_magnitude.clone()
                    combined_grad[~threshold_mask] = float('inf')
                    _, indices = torch.topk(combined_grad, k_max, largest=False)
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                    mask[indices] = True
                else:
                    mask = threshold_mask
            else:
                raise ValueError(f"Unknown mode: {mode}")
            mask_full = mask[:, None, None, None].expand_as(grad_slice)
            global_mask[entry['start']:entry['end']] = mask_full.reshape(-1)
            # Reset weights for masked neurons
            if mask.any():
                with torch.no_grad():
                    if use_lecun_init:
                        lecun_reset(param_slice, mask, shape)
                    else:
                        kaiming_reset(param_slice, mask, shape)
        elif len(shape) == 1:  # LayerNorm or bias
            grad_magnitude = grad_slice.abs()
            mask = grad_magnitude <= tau if mode == 'threshold' else torch.zeros_like(grad_magnitude, dtype=torch.bool)
            global_mask[entry['start']:entry['end']] = mask
            # Reset weights for masked neurons
            if mask.any():
                with torch.no_grad():
                    param_slice[mask] = 0.0  # LayerNorm weights/biases are typically reset to zero
    return global_mask
