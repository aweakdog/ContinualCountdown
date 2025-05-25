"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
This version includes optimizer state reset for dormant neurons.
"""
import torch
import math
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDP

# ... (other utility functions unchanged, copy from original file)

def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=False, optimizer=None):
    """
    Computes the dormant neuron mask for the FSDP flat parameter and resets the weights of dormant neurons.
    Also resets optimizer state for those neurons if optimizer is provided.
    Works with both FSDP FlatParameters and regular Parameters.
    Args:
        fsdp_module: The FSDP-wrapped module.
        mode: Dormant neuron selection mode ('threshold', 'percentage', 'hybrid').
        tau: Threshold for dormant selection.
        percentage: Fraction of neurons to consider dormant.
        max_percentage: Max fraction for hybrid mode.
        use_lecun_init: Whether to use LeCun initialization for resetting dormant neurons.
    """
    try:
        from .fsdp_flat_utils import get_fsdp_flat_param_index_map, get_shard_overlap_slices, lecun_reset, kaiming_reset
        index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
        all_masks = []
        all_ratios = []
        my_global_start = getattr(flat_param, '_shard_start_idx', None)
        my_global_end = None
        if my_global_start is None:
            try:
                import torch.distributed as dist
                dist_ok = dist.is_available() and dist.is_initialized()
                if dist_ok:
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    if len(index_map) > 0:
                        global_flat_param_numel = max(entry['end'] for entry in index_map)
                        shard_size = (global_flat_param_numel + world_size - 1) // world_size
                        my_global_start = rank * shard_size
                        my_global_end = min((rank + 1) * shard_size, global_flat_param_numel)
                        my_global_end = min(my_global_end, my_global_start + flat_param.numel())
                    else:
                        my_global_start = 0
                        my_global_end = 0
                else:
                    my_global_start = 0
                    my_global_end = flat_param.numel()
            except Exception:
                my_global_start = 0
                my_global_end = flat_param.numel()
        else:
            my_global_end = my_global_start + flat_param.numel()
        for idx, entry in enumerate(index_map):
            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
            if overlap_info is None:
                continue
            entry_name = overlap_info['entry_name'] or idx
            local_slice_start = overlap_info['local_slice_start']
            valid_numel = overlap_info['valid_numel']
            num_rows = overlap_info['num_rows']
            num_cols = overlap_info['num_cols']
            sub_shape = overlap_info['sub_shape']
            if not isinstance(sub_shape, (tuple, list)) or len(sub_shape) != 2 or sub_shape[0] <= 0 or sub_shape[1] <= 0:
                if verbose:
                    print(f"[ERROR] Invalid sub_shape {sub_shape} for entry {entry_name}, skipping.")
                continue
            try:
                param_mask = torch.zeros(sub_shape, dtype=torch.bool, device=flat_param.device)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Could not create param_mask of shape {sub_shape} for entry {entry_name}: {e}")
                continue
            if flat_param.grad is None:
                if verbose:
                    print(f"[ERROR] flat_param.grad is None for entry {entry_name}, skipping.")
                continue
            grad_raw = flat_param.grad[local_slice_start: local_slice_start + valid_numel]
            if grad_raw.numel() == 0:
                continue
            if grad_raw.numel() != valid_numel or valid_numel != num_rows * num_cols:
                if verbose:
                    print(f"[WARN] grad_raw.numel()={grad_raw.numel()} does not match valid_numel={valid_numel} or sub_shape={sub_shape} (num_rows*num_cols={num_rows*num_cols}), skipping entry {entry_name}.")
                continue
            try:
                grad_slice = grad_raw.view(sub_shape)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Could not reshape grad_raw to {sub_shape} for entry {entry_name}: {e}")
                continue
            grad_magnitude = grad_slice.abs().mean(dim=1)
            if mode == 'threshold':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                mask = normalized_grad <= tau
            elif mode == 'percentage':
                k = max(1, int(percentage * grad_magnitude.numel()))
                threshold = torch.kthvalue(grad_magnitude, k).values
                mask = grad_magnitude <= threshold
            elif mode == 'hybrid':
                normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                threshold_mask = normalized_grad <= tau
                k = max(1, min(int(max_percentage * grad_magnitude.numel()), threshold_mask.sum().item()))
                if k < threshold_mask.sum().item():
                    values, indices = torch.topk(grad_magnitude[threshold_mask], k, largest=False)
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                    mask[indices] = True
                else:
                    mask = threshold_mask
            else:
                if verbose:
                    print(f"[WARN] Unknown mode: {mode}, using empty mask for entry {entry_name}")
                mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
            if mask.shape == param_mask.shape[:1]:
                param_mask[mask] = True
                if mask.any():
                    with torch.no_grad():
                        param_slice = flat_param.data[local_slice_start: local_slice_start + valid_numel]
                        param_slice = param_slice.view(sub_shape)
                        if use_lecun_init:
                            lecun_reset(param_slice, mask, sub_shape)
                        else:
                            kaiming_reset(param_slice, mask, sub_shape)
                        # Reset optimizer state for dormant neurons if optimizer is not None
                        if optimizer is not None:
                            try:
                                for param_group in optimizer.param_groups:
                                    for p in param_group['params']:
                                        if p is flat_param:
                                            if 'exp_avg' in optimizer.state[p]:
                                                exp_avg = optimizer.state[p]['exp_avg']
                                                exp_avg_slice = exp_avg[local_slice_start: local_slice_start + valid_numel].view(sub_shape)
                                                exp_avg_slice[mask] = 0.0
                                            if 'exp_avg_sq' in optimizer.state[p]:
                                                exp_avg_sq = optimizer.state[p]['exp_avg_sq']
                                                exp_avg_sq_slice = exp_avg_sq[local_slice_start: local_slice_start + valid_numel].view(sub_shape)
                                                exp_avg_sq_slice[mask] = 0.0
                                            if verbose and mask.sum().item() > 0:
                                                print(f"[INFO] Reset optimizer state for {mask.sum().item()} dormant neurons in {entry_name}")
                                            break
                            except Exception as e:
                                if verbose:
                                    print(f"[WARN] Failed to reset optimizer state: {e}")
            else:
                if verbose:
                    print(f"[ERROR] Mask shape {mask.shape} does not match param_mask shape {param_mask.shape} for entry {entry_name}, skipping assignment.")
            ratio = mask.float().mean().item()
            all_masks.append(param_mask.flatten())
            all_ratios.append(ratio)
        if all_masks:
            global_mask = torch.cat(all_masks)
            avg_ratio = sum(all_ratios) / len(all_ratios)
            return global_mask
        else:
            return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)
    except Exception as e:
        import traceback
        print(f"Error in fsdp_dormant_neuron_mask_and_reset: {e}")
        traceback.print_exc()
        return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)

# Copy all other functions from the original fsdp_flat_utils.py as needed.
