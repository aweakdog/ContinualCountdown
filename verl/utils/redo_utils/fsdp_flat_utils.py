"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
"""
import torch
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDP


def get_flat_param_to_layer_map(fsdp_module):
    """
    Build index map from flat parameter to original parameters using params and _unflattened_param_names.
    Returns a list of dicts: [{'name': ..., 'start': ..., 'end': ..., 'shape': ...}, ...]
    """
    index_map = []
    if not hasattr(fsdp_module, "_flat_param") or fsdp_module._flat_param is None:
        raise ValueError("No flat parameter found. Check if `use_orig_params=False`.")
    flat_param = fsdp_module._flat_param

    if hasattr(fsdp_module, "params"):
        offset = 0
        for param in fsdp_module.params:
            num_elements = param.numel()
            if hasattr(param, "_unflattened_param_names"):
                name = param._unflattened_param_names[0]
            else:
                name = None
            index_map.append({
                'name': name,
                'start': offset,
                'end': offset + num_elements,
                'shape': tuple(param.shape),
            })
            print(f"[DEBUG] flat_param slice: name={name}, start={offset}, end={offset + num_elements}, shape={tuple(param.shape)}")
            offset += num_elements
        # print(f"[DEBUG] Built flat_param to layer map using params/_unflattened_param_names with {len(index_map)} entries")
        return index_map, flat_param
    else:
        raise AttributeError("`params` not found on FSDP module. Check PyTorch/FSDP version.")


def try_print_flat_param_to_layer_map(fsdp_module):
    """
    Try to print the contents of _flat_param_to_layer_map if it exists on the FSDP module.
    """
    if hasattr(fsdp_module, '_flat_param_to_layer_map'):
        flat_param_to_layer_map = getattr(fsdp_module, '_flat_param_to_layer_map')
        print(f"[DEBUG] _flat_param_to_layer_map found with {len(flat_param_to_layer_map)} entries:")
        for idx, (param_name, submodule) in enumerate(flat_param_to_layer_map.items()):
            print(f"Flat param index {idx} -> Layer: {getattr(submodule, '__class__', type(submodule)).__name__}, Param: {param_name}")
        return flat_param_to_layer_map
    else:
        print("[DEBUG] _flat_param_to_layer_map attribute not found on this FSDP module.")
        return None

def iter_leaf_fsdp_modules(module):
    """
    Yield (name, submodule) pairs for all leaf FSDP modules in the model.
    A leaf FSDP module is an FSDP module whose direct children are not FSDP.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    for name, submodule in module.named_modules():
        if isinstance(submodule, FSDP):
            # If none of the direct children are FSDP, this is a leaf
            if not any(isinstance(child, FSDP) for child in submodule.children()):
                yield name, submodule


def analyze_all_fsdp_dormant_neurons(module, mode='threshold', tau=0.04, verbose=True):
    """
    Analyze all leaf FSDP-wrapped submodules for dormant neurons.
    Returns a dict mapping module names to their dormant neuron mask and statistics.
    """
    from verl.utils.redo_utils.fsdp_flat_utils import compute_fsdp_dormant_mask_only
    results = {}
    total_dormant = 0
    total_count = 0
    for name, submodule in iter_leaf_fsdp_modules(module):
        try:
            mask = compute_fsdp_dormant_mask_only(submodule, mode=mode, tau=tau)
            if mask is not None:
                dormant = mask.sum().item()
                count = mask.numel()
                total_dormant += dormant
                total_count += count
                results[name] = {
                    'mask': mask,
                    'dormant': dormant,
                    'total': count,
                    'ratio': dormant / (count + 1e-8)
                }
                if verbose:
                    print(f"[DormantNeuron][{name}] dormant={dormant}, total={count}, ratio={dormant/(count+1e-8):.6f}")
            else:
                results[name] = None
        except Exception as e:
            print(f"[WARN] Could not analyze dormant neurons for {name}: {e}")
    if verbose and total_count > 0:
        print(f"[DormantNeuron][ALL] total_dormant={total_dormant}, total_params={total_count}, ratio={total_dormant/(total_count+1e-8):.6f}")
    return results

def redo_reset_all_fsdp_layers(module, mode='threshold', tau=0.04, verbose=True, use_lecun_init=True):
    """
    Reset dormant neurons for all leaf FSDP-wrapped submodules.
    Returns a dict mapping module names to the number of resets performed.
    Args:
        module: The root module to search for FSDP submodules.
        mode: Dormant neuron selection mode ('threshold', 'percentage', 'hybrid').
        tau: Threshold for dormant selection.
        verbose: If True, print reset stats for each layer.
        use_lecun_init: If True, use LeCun normal; if False, use Kaiming uniform for reset.
    """
    from verl.utils.redo_utils.fsdp_flat_utils import fsdp_dormant_neuron_mask_and_reset
    results = {}
    total_reset = 0
    for name, submodule in iter_leaf_fsdp_modules(module):
        try:
            mask = fsdp_dormant_neuron_mask_and_reset(submodule, mode=mode, tau=tau, use_lecun_init=use_lecun_init)
            if mask is not None:
                reset_count = mask.sum().item()
                total_reset += reset_count
                results[name] = reset_count
                if verbose:
                    print(f"[ReDo-Reset][{name}] reset {reset_count} dormant neurons.")
        except Exception as e:
            print(f"[WARN] Could not reset dormant neurons for {name}: {e}")
    if verbose:
        print(f"[ReDo-Reset][ALL] total_reset={total_reset}")
    return results

def get_fsdp_flat_param_index_map(fsdp_module):
    """
    Returns a list of dicts with start, end, shape, param, and fqn (fully qualified name, if available) for each original parameter in the flat param.
    Handles both use_orig_params=True (modern FSDP), param_infos/_param_infos (legacy FSDP), and non-FSDP modules.
    """
    index_map = []
    flat_param = None
    ## --- Preferred: use use_orig_params=True and _fsdp_prev (PyTorch 2.0+) ---
    #if not hasattr(fsdp_module, "named_parameters"):
    #    print(f"[WARN][get_fsdp_flat_param_index_map] Module {type(fsdp_module).__name__} has no named_parameters method; skipping preferred mapping block.")
    #else:
    #    for name, param in fsdp_module.named_parameters():
    #        #print(f"[DEBUG][param attributes] name={name}, attributes={dir(param)}")
    #        has_prev = hasattr(param, "_fsdp_prev")
    #        has_offset = has_prev and hasattr(param._fsdp_prev, "offset_in_flat_param")
    #        print(f"[DEBUG][param attr] name={name}, type={type(param)}, shape={tuple(param.shape)}, has__fsdp_prev={has_prev}, has_offset_in_flat_param={has_offset}")
    #        if has_offset:
    #            offset = param._fsdp_prev.offset_in_flat_param
    #            numel = param.numel()
    #            if flat_param is None and hasattr(param._fsdp_prev, "flat_param"):
    #                flat_param = param._fsdp_prev.flat_param
    #            fqn = getattr(param, 'fqn', name)  # Prefer fqn if available
    #            print(f"[DEBUG][get_fsdp_flat_param_index_map] name={name}, fqn={fqn}, offset={offset}, numel={numel}, shape={tuple(param.shape)}, flat_param_set={flat_param is not None}")
    #            index_map.append({
    #                "name": name,
    #                "fqn": fqn,
    #                "start": offset,
    #                "end": offset + numel,
    #                "shape": tuple(param.shape),
    #                "param": param,
    #            })
    #if flat_param is not None and len(index_map) > 0:
    #    index_map = sorted(index_map, key=lambda x: x["start"])
    #    print(f"[DEBUG] Using use_orig_params=True mapping for {len(index_map)} params")
    #    return index_map, flat_param

    # --- Fallback: legacy logic (for older FSDP or use_orig_params=False) ---
    from torch.distributed.fsdp import FlatParameter
    flat_param = next((p for p in fsdp_module.parameters() if isinstance(p, FlatParameter)), None)
    if flat_param is not None:
        #print(f"[DEBUG] flat_param attributes: {dir(flat_param)}")
        shapes = getattr(flat_param, '_shapes', None)
        if shapes is not None:
            print(f"[DEBUG] flat_param._shapes type: {type(shapes)}, length: {len(shapes)}")
            if len(shapes) > 0:
                print(f"[DEBUG] flat_param._shapes[0] type: {type(shapes[0])}, value: {shapes[0]}")
                if hasattr(shapes[0], '__dict__'):
                    print(f"[DEBUG] flat_param._shapes[0] attributes: {dir(shapes[0])}")
        names = getattr(flat_param, '_unflattened_param_names', None)
        # If shapes provides (global_start, global_end, shape) tuples, use them directly
        #if shapes is not None and len(shapes) > 0 and isinstance(shapes[0], tuple) and len(shapes[0]) == 3:
        #    for i, (global_start, global_end, shape) in enumerate(shapes):
        #        name = names[i] if names is not None and i < len(names) else f"param_{i}"
        #        index_map.append({
        #            'index': i,
        #            'start': global_start,
        #            'end': global_end,
        #            'shape': shape,
        #            'param': None,
        #            'fqn': name,
        #        })
        #        print(f"[DEBUG] flat_param global slice: name={name}, start={global_start}, end={global_end}, shape={shape}")
        #    print(f"[DEBUG] Built index map from flat_param._shapes (global indices) with {len(index_map)} entries")
        #    return index_map, flat_param
        # Otherwise, fall back to offset-based logic
        offset = 0
        if shapes is not None:
            for i, shape in enumerate(shapes):
                numel = 1
                for s in shape:
                    numel *= s
                name = names[i] if names is not None and i < len(names) else f"param_{i}"
                start = offset
                end = offset + numel
                print(f"[DEBUG] Computed flat_param indices for layer {name}: start={start}, end={end}, shape={shape}")
                index_map.append({
                    'index': i,
                    'start': start,
                    'end': end,
                    'shape': shape,
                    'param': None,  # original param ref not available here
                    'fqn': name,
                })
                offset = end
                # print(f"[DEBUG] flat_param slice: name={name}, start={offset}, end={offset + numel}, shape={shape}")
                offset += numel
            # print(f"[DEBUG] Built index map from flat_param._shapes/_unflattened_param_names with {len(index_map)} entries")
            return index_map, flat_param

    # --- Final fallback: regular parameter (non-FSDP) ---
    param = next((p for p in fsdp_module.parameters() if p.requires_grad), None)
    if param is None:
        param = next(fsdp_module.parameters(), None)
        if param is None:
            raise RuntimeError("No parameters found in module")
    shape = param.shape
    index_map = [{'index': 0, 'start': 0, 'end': param.numel(), 'shape': shape, 'param': param, 'fqn': getattr(param, 'fqn', None)}]
    # print(f"[DEBUG] Successfully created simplified index map for regular Parameter")
    return index_map, param


def compute_fsdp_zero_grad_space_ratio(fsdp_module, eps=1e-8, verbose=False):
    """
    Computes the fraction of output neurons (rows) in each 2D param whose gradient is (almost) exactly zero.
    Handles global-to-local index mapping for FSDP shards.
    Returns the global ratio: (total zero-grad rows) / (total rows across all 2D params)
    """
    index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
    grad = flat_param.grad
    if grad is None:
        if verbose:
            # print("[WARN] flat_param.grad is None, skipping zero grad space calculation.")
            pass
        return {'zero': 0, 'total': 0, 'ratio': 0.0}

    my_global_start = getattr(flat_param, '_shard_start_idx', None)
    my_global_end = None
    if my_global_start is None:
        # Manual computation fallback (same as compute_fsdp_dormant_mask_only)
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
                    if verbose:
                        # print(f"[DEBUG] rank={rank}, my_global_start={my_global_start}, my_global_end={my_global_end}, global_flat_param_numel={global_flat_param_numel}, shard_size={shard_size}")
                        pass
                else:
                    my_global_start = 0
                    my_global_end = 0
                    if verbose:
                        # print(f"[DEBUG] rank={rank}, my_global_start={my_global_start}, my_global_end={my_global_end}, global_flat_param_numel=0")
                        pass
            else:
                my_global_start = 0
                my_global_end = flat_param.numel()
        except Exception as e:
            if verbose:
                # print(f"[WARN] Could not infer global shard start idx: {e}")
                pass
            my_global_start = 0
            my_global_end = flat_param.numel()
    else:
        my_global_end = my_global_start + flat_param.numel()
    if verbose:
        # print(f"[DEBUG] my_global_start={my_global_start}, my_global_end={my_global_end}, local flat_param.numel={flat_param.numel()}")
        pass

    total_zero = 0
    total_rows = 0
    for entry in index_map:
        shape = entry['shape']
        if len(shape) != 2:
            continue
        param_start = entry['start']
        param_end = entry['end']

        overlap_start = max(param_start, my_global_start)
        overlap_end = min(param_end, my_global_end)
        if overlap_start >= overlap_end:
            if verbose:
                # print(f"[WARN] Entry {entry.get('fqn', entry['index'])} not present in this shard.")
                pass
            continue
        if (overlap_start == param_start) and (overlap_end == param_end):
            local_slice_start = overlap_start - my_global_start
            local_slice_end = overlap_end - my_global_start
            grad_raw = grad[local_slice_start:local_slice_end]
            if grad_raw.numel() != torch.Size(shape).numel():
                if verbose:
                    # print(f"[WARN] Mismatch in grad slice size for {entry.get('fqn', entry['index'])}: got {grad_raw.numel()}, expected {torch.Size(shape).numel()}")
                    pass
                continue
            grad_matrix = grad_raw.view(shape)
            zero_vectors = (grad_matrix.abs().sum(dim=1) < eps)
            total_zero += zero_vectors.sum().item()
            total_rows += grad_matrix.size(0)
            if verbose:
                # print(f"[DEBUG] Zero grad vectors for {entry.get('fqn', entry['index'])}: {zero_vectors.sum().item()}/{grad_matrix.size(0)}")
                pass
        else:
            if verbose:
                # print(f"[WARN] Entry {entry.get('fqn', entry['index'])} only partially present in this shard. Skipping.")
                pass
            continue

    ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    if verbose:
        pass
        # print(f"[DEBUG] Zero grad space: {total_zero}/{total_rows} ({ratio:.6f})")
    return {'zero': total_zero, 'total': total_rows, 'ratio': ratio}

def compute_fsdp_dormant_mask_only(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01):
    """
    Computes and returns the dormant neuron mask for the FSDP flat parameter, but does NOT reset weights.
    Works with both FSDP FlatParameters and regular Parameters.
    """
    try:
        index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
        
        # all_masks = []  # Unused debug variable
        # all_ratios = []  # Unused debug variable
        # print(f"[DEBUG] compute_fsdp_dormant_mask_only: Processing module {type(fsdp_module).__name__}")
        if verbose:
            pass
        # Process each parameter individually
        # Get the global start offset for this local flat_param shard
        # Try to get the global start offset for this local flat_param shard
        my_global_start = getattr(flat_param, '_shard_start_idx', None)
        my_global_end = None
        if my_global_start is None:
            # Manual computation fallback
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
                        # print(f"[DEBUG] rank={rank}, my_global_start={my_global_start}, my_global_end={my_global_end}, global_flat_param_numel={global_flat_param_numel}, shard_size={shard_size}")
                        pass
                    else:
                        my_global_start = 0
                        my_global_end = 0
                        # print(f"[DEBUG] rank={rank}, my_global_start={my_global_start}, my_global_end={my_global_end}, global_flat_param_numel=0")
                        pass
                else:
                    my_global_start = 0
                    my_global_end = flat_param.numel()
            except Exception as e:
                # print(f"[WARN] Could not infer global shard start idx: {e}")
                pass
                my_global_start = 0
                my_global_end = flat_param.numel()
        else:
            my_global_end = my_global_start + flat_param.numel()
        # print(f"[DEBUG] my_global_start={my_global_start}, my_global_end={my_global_end}, local flat_param.numel={flat_param.numel()}")
        pass
        for idx, entry in enumerate(index_map):
            entry_name = entry.get('fqn', None) or entry.get('name', None) or idx
            param_start = entry['start']
            param_end = entry['end']
            # Check for overlap between param and local shard
            overlap_start = max(param_start, my_global_start)
            overlap_end = min(param_end, my_global_end)
            if overlap_start >= overlap_end:
                print(f"[WARN] Entry {entry_name} is not present in this shard. Global param [{param_start}:{param_end}), local shard [{my_global_start}:{my_global_end})")
                print(f"[WARN] Full entry info: {entry}")
                continue
            # Compute local indices
            local_slice_start = overlap_start - my_global_start
            local_slice_end = overlap_end - my_global_start
            param_slice_start = overlap_start - param_start
            param_slice_end = overlap_end - param_start
            # Only process if the full param is present (most common FSDP case)
            if (overlap_start == param_start) and (overlap_end == param_end):
                numel = param_end - param_start
                expected_numel = torch.Size(entry['shape']).numel()
                # Guard: skip if local slice is zero-sized
                if local_slice_end <= local_slice_start or (local_slice_end - local_slice_start) == 0:
                    print(f"[WARN] Skipping zero-size local slice for {entry_name}: local [{local_slice_start}:{local_slice_end}]")
                    continue
                raw_slice = flat_param.data[local_slice_start:local_slice_end]
                print(f"[DEBUG] About to reshape param_slice for {entry_name}: raw numel={raw_slice.numel()}, target shape={entry['shape']}")
                if raw_slice.numel() != expected_numel:
                    print(f"[ERROR] Slice size mismatch for {entry_name}: got {raw_slice.numel()}, expected {expected_numel}. Skipping.")
                    print(f"[ERROR] Full entry info: {entry}")
                    continue
                param_slice = raw_slice.view(entry['shape'])
            else:
                # Partial param present (rare): print info and skip
                print(f"[WARN] Entry {entry_name} is only partially present in this shard. Skipping. ")
                print(f"[WARN] Overlap: global [{overlap_start}:{overlap_end}), local flat_param[{local_slice_start}:{local_slice_end}], param[{param_slice_start}:{param_slice_end}] of shape {entry['shape']}")
                continue
            param_mask = torch.zeros_like(param_slice, dtype=torch.bool)
            if len(entry['shape']) == 2:  # Linear layer
                if flat_param.grad is None:
                    print(f"[ERROR] flat_param.grad is None for entry {entry_name}, skipping.")
                    continue
                grad_raw = flat_param.grad[local_slice_start:local_slice_end]
                print(f"[DEBUG] About to reshape grad_slice for {entry_name}: raw numel={grad_raw.numel()}, target shape={entry['shape']}")
                if grad_raw.numel() == 0:
                    print(f"[WARN] Skipping zero-size grad slice for {entry_name}: local [{local_slice_start}:{local_slice_end}]")
                    continue
                grad_slice = grad_raw.view(entry['shape'])
                grad_magnitude = grad_slice.abs().mean(dim=1)  # Average across inputs
                
                if mode == 'threshold':
                    # Normalize by mean gradient magnitude
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    mask = normalized_grad <= tau
                elif mode == 'percentage':
                    # Select bottom percentage of neurons by gradient magnitude
                    k = max(1, int(percentage * grad_magnitude.numel()))
                    threshold = torch.kthvalue(grad_magnitude, k).values
                    mask = grad_magnitude <= threshold
                elif mode == 'hybrid':
                    # Threshold-based mask
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    threshold_mask = normalized_grad <= tau
                    
                    # Percentage-based mask (up to max_percentage)
                    k = max(1, min(int(max_percentage * grad_magnitude.numel()), threshold_mask.sum().item()))
                    if k < threshold_mask.sum().item():
                        values, indices = torch.topk(grad_magnitude[threshold_mask], k, largest=False)
                        mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                        mask[indices] = True
                    else:
                        mask = threshold_mask
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                param_mask[mask] = True
                ratio = mask.float().mean().item()
                all_masks.append(param_mask.flatten())
                all_ratios.append(ratio)
                print(f"[DEBUG] Created mask for entry {entry_name}: {mask.sum().item()}/{mask.numel()} neurons marked dormant ({ratio:.6f})")
            else:
                # If not a 2D weight, skip but append a default ratio
                all_masks.append(param_mask)
                all_ratios.append(0.0)
                print(f"[DEBUG] Skipped entry {entry_name}: shape={entry['shape']} (not 2D)")
        
        # Combine all masks into one global mask
        if all_masks:
            global_mask = torch.cat(all_masks)
            avg_ratio = sum(all_ratios) / len(all_ratios)
            print(f"[DEBUG] Combined mask has {global_mask.sum().item()}/{global_mask.numel()} elements marked dormant ({avg_ratio*100:.2f}%)")
            return global_mask
        else:
            print(f"[DEBUG] No valid parameters found, returning empty mask")
            return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)
    
    except Exception as e:
        import traceback
        print(f"Error in compute_fsdp_dormant_mask_only: {e}")
        traceback.print_exc()
        # Return an empty mask as fallback
        return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)


def lecun_reset(param_slice, mask, shape):
    """LeCun normal initialization for masked neurons."""
    if len(shape) == 2:  # Linear layer
        fan_in = shape[1]
        std = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            # Only reset the masked neurons
            param_slice[mask] = torch.randn_like(param_slice[mask]) * std


def kaiming_reset(param_slice, mask, shape):
    """Kaiming uniform initialization for masked neurons."""
    if len(shape) == 2:  # Linear layer
        fan_in = shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        with torch.no_grad():
            # Only reset the masked neurons
            param_slice[mask] = torch.rand_like(param_slice[mask]).uniform_(-bound, bound)


def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True):
    """
    Computes the dormant neuron mask for the FSDP flat parameter and resets the weights of dormant neurons.
    Works with both FSDP FlatParameters and regular Parameters.
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
    try:
        # Simpler approach: just iterate through all parameters and process each one individually
        # This avoids the complexity of trying to handle FSDP FlatParameters specially
        all_masks = []
        reset_count = 0
        total_count = 0
        
        print(f"[DEBUG] fsdp_dormant_neuron_mask_and_reset: Processing module {type(fsdp_module).__name__}")
        
        # Process each parameter individually
        for name, param in fsdp_module.named_parameters():
            if not param.requires_grad or param.grad is None:
                print(f"[DEBUG] Skipping parameter {name} (no gradient)")
                continue
                
            print(f"[DEBUG] Processing parameter {name} with shape {param.shape}")
            
            # Create mask for this parameter
            param_mask = torch.zeros_like(param.data, dtype=torch.bool)
            
            # Only process parameters that could have neurons (weight matrices)
            if len(param.shape) == 2:  # Linear layer weights
                # For linear layers, neurons are represented by rows
                grad_magnitude = param.grad.abs().mean(dim=1)  # Average across inputs
                
                if mode == 'threshold':
                    # Normalize by mean gradient magnitude
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    mask = normalized_grad <= tau
                elif mode == 'percentage':
                    # Select bottom percentage of neurons by gradient magnitude
                    k = max(1, int(percentage * grad_magnitude.numel()))
                    threshold = torch.kthvalue(grad_magnitude, k).values
                    mask = grad_magnitude <= threshold
                elif mode == 'hybrid':
                    # Threshold-based mask
                    normalized_grad = grad_magnitude / (grad_magnitude.mean() + 1e-9)
                    threshold_mask = normalized_grad <= tau
                    
                    # Percentage-based mask (up to max_percentage)
                    k = max(1, min(int(max_percentage * grad_magnitude.numel()), threshold_mask.sum().item()))
                    if k < threshold_mask.sum().item():
                        values, indices = torch.topk(grad_magnitude[threshold_mask], k, largest=False)
                        mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                        mask[indices] = True
                    else:
                        mask = threshold_mask
                else:
                    print(f"[DEBUG] Unknown mode: {mode}, using empty mask")
                    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
                
                # Reset weights for masked neurons
                if mask.any():
                    with torch.no_grad():
                        if use_lecun_init:
                            fan_in = param.shape[1]
                            std = 1.0 / math.sqrt(fan_in)
                            param.data[mask] = torch.randn_like(param.data[mask]) * std
                        else:
                            fan_in = param.shape[1]
                            bound = 1.0 / math.sqrt(fan_in)
                            param.data[mask] = torch.rand_like(param.data[mask]).uniform_(-bound, bound)
                # Expand mask to match parameter shape
                param_mask = mask.unsqueeze(1).expand_as(param)
                reset_count += mask.sum().item()
                total_count += mask.numel()
        # No fallback: only perform LeCun or Kaiming reset for masked neurons
        return None

    except Exception as e:
        print(f"Error in fsdp_dormant_neuron_mask_and_reset: {e}")
        # Return an empty mask as fallback
        param = next(fsdp_module.parameters())
        return torch.zeros_like(param.data, dtype=torch.bool)
