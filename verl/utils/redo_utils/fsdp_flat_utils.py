"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
"""
import torch
import math
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
            #print(f"[DEBUG] flat_param slice: name={name}, start={offset}, end={offset + num_elements}, shape={tuple(param.shape)}")
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
        #print(f"[DEBUG] _flat_param_to_layer_map found with {len(flat_param_to_layer_map)} entries:")
        for idx, (param_name, submodule) in enumerate(flat_param_to_layer_map.items()):
            print(f"Flat param index {idx} -> Layer: {getattr(submodule, '__class__', type(submodule)).__name__}, Param: {param_name}")
        return flat_param_to_layer_map
    else:
        #print("[DEBUG] _flat_param_to_layer_map attribute not found on this FSDP module.")
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


def analyze_all_fsdp_dormant_neurons(module, mode='threshold', tau=0.1, verbose=True):
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
                #if verbose:
                #    print(f"[DormantNeuron][{name}] dormant={dormant}, total={count}, ratio={dormant/(count+1e-8):.6f}")
            else:
                results[name] = None
        except Exception as e:
            print(f"[WARN] Could not analyze dormant neurons for {name}: {e}")
    #if verbose and total_count > 0:
    #    print(f"[DormantNeuron][ALL] total_dormant={total_dormant}, total_params={total_count}, ratio={total_dormant/(total_count+1e-8):.6f}")
    return results

def analyze_all_fsdp_zero_grad_space(module, tau=0.1, verbose=True):
    """
    Analyze all leaf FSDP-wrapped submodules for zero grad space ratio.
    Returns a dict mapping module names to their zero grad stats and the global aggregate.
    """
    from verl.utils.redo_utils.fsdp_flat_utils import compute_fsdp_zero_grad_space_ratio
    results = {}
    total_zero = 0
    total_rows = 0
    for name, submodule in iter_leaf_fsdp_modules(module):
        try:
            stats = compute_fsdp_zero_grad_space_ratio(submodule, tau=tau, verbose=verbose)
            if stats is not None:
                zero = stats.get('zero', 0)
                rows = stats.get('total', 0)
                total_zero += zero
                total_rows += rows
                results[name] = stats
            else:
                results[name] = None
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not analyze zero grad space for {name}: {e}")
    global_ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    results['__global__'] = {'zero': total_zero, 'total': total_rows, 'ratio': global_ratio}
    return results

def redo_reset_all_fsdp_layers(module, mode='threshold', tau=0.1, verbose=True, use_lecun_init=True):
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
            #print(f"[DEBUG] flat_param._shapes type: {type(shapes)}, length: {len(shapes)}")
            if len(shapes) > 0:
                #print(f"[DEBUG] flat_param._shapes[0] type: {type(shapes[0])}, value: {shapes[0]}")
                if hasattr(shapes[0], '__dict__'):
                    pass
                    #print(f"[DEBUG] flat_param._shapes[0] attributes: {dir(shapes[0])}")
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
                #print(f"[DEBUG] Computed flat_param indices for layer {name}: start={start}, end={end}, shape={shape}")
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


def get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param_numel, verbose=False):
    """
    Given an index_map entry and the global start/end for this shard, compute the overlap region and local/global slices.
    Returns None if there is no overlap or if the overlap does not contain any full rows (for 2D params).
    """
    param_start = entry['start']
    param_end = entry['end']
    overlap_start = max(param_start, my_global_start)
    overlap_end = min(param_end, my_global_end)
    if overlap_start >= overlap_end:
        return None
    local_slice_start = max(overlap_start - my_global_start, 0)
    local_slice_end = min(local_slice_start + (overlap_end - overlap_start), flat_param_numel)
    actual_numel = local_slice_end - local_slice_start
    if actual_numel < 0: # hacky here we skip the rank0 non-full-row param
        import torch.distributed as dist
        dist_ok = dist.is_available() and dist.is_initialized()
        if dist_ok:
            rank = dist.get_rank()
            if verbose:
                print('[RANK]current_rank:',rank)
        #if verbose:
        #    print("[WARN][slices]",'param_start',param_start,'param_end',param_end,'my_global_start',my_global_start,'my_global_end',my_global_end,'local_slice_start',local_slice_start,'local_slice_end',local_slice_end)
        return None
    param_slice_start = overlap_start - param_start
    param_slice_end = overlap_end - param_start
    if len(entry['shape']) != 2:
        return None
    num_cols = entry['shape'][1]
    valid_numel = (actual_numel // num_cols) * num_cols
    if valid_numel == 0:
        if verbose:
            print(f"[WARN] No full rows in local grad slice (actual_numel={actual_numel}, num_cols={num_cols}), skipping entry {entry.get('fqn', None) or entry.get('name', None)}.")
        return None
    num_rows = valid_numel // num_cols
    sub_shape = (num_rows, num_cols)
    return {
        'local_slice_start': local_slice_start,
        'valid_numel': valid_numel,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'sub_shape': sub_shape,
        'param_slice_start': param_slice_start,
        'param_slice_end': param_slice_end,
        'entry_name': entry.get('fqn', None) or entry.get('name', None)
    }

def compute_fsdp_zero_grad_space_ratio(fsdp_module, tau=0.1, verbose=False):
    """
    Computes the fraction of output neurons (rows) in each 2D param whose normalized gradient metric si = A/(B/H) is below tau.
    - H: number of neurons (rows) in the layer
    - B: sum of absolute gradients in this layer (all elements)
    - A: sum of absolute gradients for this neuron (row)
    - si = A / (B / H)
    Handles global-to-local index mapping for FSDP shards.
    Returns the global ratio: (total zero-grad rows) / (total rows across all 2D params)
    """
    #tau = 0.04
    index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
    grad = flat_param.grad
    if grad is None:
        print("[WARN] flat_param.grad is None, skipping zero grad space calculation.")
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
                    my_global_end = min((rank + 1) * shard_size, global_flat_param_numel) # for last rank
                    my_global_end = min(my_global_end, my_global_start + flat_param.numel())
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
            print(f"[WARN] Could not infer global shard start idx: {e}")
            my_global_start = 0
            my_global_end = flat_param.numel()
    else:
        print('[WARN][my_global_start] is Not None! Got it from param! ')
        my_global_end = my_global_start + flat_param.numel()
    #print(f"[DEBUG][ZeroGrad] my_global_start={my_global_start}, my_global_end={my_global_end}, local flat_param.numel={flat_param.numel()}")

    total_zero = 0
    total_rows = 0
    for idx, entry in enumerate(index_map):
        #print(f"[DEBUG][ZeroGrad][{idx}] entry:{entry}")
        overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
        if overlap_info is None:
            continue
        local_slice_start = overlap_info['local_slice_start']
        valid_numel = overlap_info['valid_numel']
        sub_shape = overlap_info['sub_shape']
        num_rows = overlap_info['num_rows']
        num_cols = overlap_info['num_cols']
        # Robust shape check
        if not isinstance(sub_shape, (tuple, list)) or len(sub_shape) != 2 or sub_shape[0] <= 0 or sub_shape[1] <= 0:
            if verbose:
                print(f"[ERROR][ZeroGrad] Invalid sub_shape {sub_shape} for entry {idx}, skipping.")
            continue
        grad_raw = grad[local_slice_start:local_slice_start + valid_numel]
        if grad_raw.numel() != valid_numel:
            if verbose:
                print(f"[WARN][ZeroGrad] grad_raw size mismatch ({grad_raw.numel()} != {valid_numel}), skipping entry {idx}.")
            continue
        try:
            grad_matrix = grad_raw.view(sub_shape)
        except Exception as e:
            if verbose:
                print(f"[ERROR][ZeroGrad] Could not reshape grad_raw to {sub_shape} for entry {idx}: {e}")
            continue
        grad_matrix_abs = grad_matrix.abs()
        H = grad_matrix_abs.size(0)  # number
        B = grad_matrix_abs.sum().item()  # sum of all gradients
        if H == 0 or B == 0:
            zero_vectors = (grad_matrix_abs.sum(dim=1) < float('inf'))  # all False if H==0 or B==0
        else:
            row_sums = grad_matrix_abs.sum(dim=1)  # A for each neuron
            avg_row_sum = B / H
            si = row_sums / avg_row_sum
            zero_vectors = (si < tau)
        total_zero += zero_vectors.sum().item()
        total_rows += grad_matrix.size(0)
        #print(f"[DEBUG][ZeroGrad]total_zero:{total_zero}")
        #print(f"[DEBUG][ZeroGrad]total_rows:{total_rows}")

    ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    # if verbose:
    #     print(f"[DEBUG] Zero grad space: {total_zero}/{total_rows} ({ratio:.6f})")
    return {'zero': total_zero, 'total': total_rows, 'ratio': ratio}

def compute_fsdp_dormant_mask_only(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=True):
    """
    Computes and returns the dormant neuron mask for the FSDP flat parameter, but does NOT reset weights.
    Works with both FSDP FlatParameters and regular Parameters.
    """
    try:
        index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
        
        all_masks = []
        all_ratios = []
        # print(f"[DEBUG] compute_fsdp_dormant_mask_only: Processing module {type(fsdp_module).__name__}")
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
                        my_global_end = min(my_global_end, my_global_start + flat_param.numel())
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
                my_global_start = 0
                my_global_end = flat_param.numel()
        else:
            print('[WARN][my_global_start] is Not None! Got it from param! ')
            my_global_end = my_global_start + flat_param.numel()
        #print(f"[DEBUG][nullspace] my_global_start={my_global_start}, my_global_end={my_global_end}, local flat_param.numel={flat_param.numel()}")
        #pass
        for idx, entry in enumerate(index_map):
            #print(f"[DEBUG][nullspace]entry:{entry}")
            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
            if overlap_info is None:
                continue
            entry_name = overlap_info['entry_name'] or idx
            local_slice_start = overlap_info['local_slice_start']
            valid_numel = overlap_info['valid_numel']
            num_rows = overlap_info['num_rows']
            num_cols = overlap_info['num_cols']
            sub_shape = overlap_info['sub_shape']
            # Guard against invalid/negative/zero shapes
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
                if verbose:
                    print(f"[WARN] Unknown mode: {mode}, using empty mask for entry {entry_name}")
                mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
            # Only assign to param_mask if mask shape matches
            if mask.shape == param_mask.shape[:1]:
                param_mask[mask] = True
                # Perform the reset for masked neurons
                if mask.any():
                    with torch.no_grad():
                        param_slice = flat_param.data[local_slice_start: local_slice_start + valid_numel]
                        param_slice = param_slice.view(sub_shape)
                        if use_lecun_init:
                            lecun_reset(param_slice, mask, sub_shape)
                        else:
                            kaiming_reset(param_slice, mask, sub_shape)
            else:
                if verbose:
                    print(f"[ERROR] Mask shape {mask.shape} does not match param_mask shape {param_mask.shape} for entry {entry_name}, skipping assignment.")
            ratio = mask.float().mean().item()
            all_masks.append(param_mask.flatten())
            all_ratios.append(ratio)

        #print(f"[DEBUG] Created mask for entry {entry_name}: {mask.sum().item()}/{mask.numel()} neurons marked dormant ({ratio:.6f})")

        # Combine all masks into one global mask
        if all_masks:
            global_mask = torch.cat(all_masks)
            avg_ratio = sum(all_ratios) / len(all_ratios)
            #print(f"[DEBUG] Combined mask has {global_mask.sum().item()}/{global_mask.numel()} elements marked dormant ({avg_ratio*100:.2f}%)")
            return global_mask
        else:
            #print(f"[DEBUG] No valid parameters found, returning empty mask")
            return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)

    except Exception as e:
        import traceback
        print(f"Error in compute_fsdp_dormant_mask_only: {e}")
        traceback.print_exc()
        # Return an empty mask as fallback
        return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)


def lecun_reset(param_slice, mask, shape, is_layernorm=False, bias_slice=None):
    """
    LeCun normal/truncated normal reset for masked neurons. Supports linear, conv2d, and layernorm slices.
    If is_layernorm is True, sets weights to 1 and bias to 0.
    If bias_slice is provided, resets bias for masked neurons as well.
    """
    with torch.no_grad():
        if is_layernorm:
            param_slice[mask] = 1.0
            if bias_slice is not None:
                bias_slice[mask] = 0.0
            return
        if len(shape) == 2:  # Linear
            fan_in = shape[1]
        elif len(shape) == 4:  # Conv2d
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            raise NotImplementedError(f"Unsupported shape for LeCun reset: {shape}")
        variance = 1.0 / fan_in
        stddev = math.sqrt(variance) / 0.87962566103423978
        # Truncated normal [-2, 2]
        param_slice[mask] = torch.empty_like(param_slice[mask]).normal_()
        param_slice[mask] = torch.clamp(param_slice[mask], -2.0, 2.0)
        param_slice[mask] *= stddev
        if bias_slice is not None:
            bias_slice[mask] = 0.0


def kaiming_reset(param_slice, mask, shape, is_layernorm=False, bias_slice=None):
    """
    Kaiming uniform reset for masked neurons. Supports linear, conv2d, and layernorm slices.
    If is_layernorm is True, sets weights to 1 and bias to 0.
    If bias_slice is provided, resets bias for masked neurons as well.
    """
    with torch.no_grad():
        if is_layernorm:
            param_slice[mask] = 1.0
            if bias_slice is not None:
                bias_slice[mask] = 0.0
            return
        if len(shape) == 2:  # Linear
            fan_in = shape[1]
        elif len(shape) == 4:  # Conv2d
            fan_in = shape[1] * shape[2] * shape[3]
        else:
            raise NotImplementedError(f"Unsupported shape for Kaiming reset: {shape}")
        gain = math.sqrt(2.0)  # relu default
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        param_slice[mask] = torch.empty_like(param_slice[mask]).uniform_(-bound, bound)
        if bias_slice is not None:
            # PyTorch's Kaiming bias: uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
            if fan_in > 0:
                bias_bound = 1.0 / math.sqrt(fan_in)
            else:
                bias_bound = 0.0
            bias_slice[mask] = torch.empty_like(bias_slice[mask]).uniform_(-bias_bound, bias_bound)


def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=False):
    """
    Computes the dormant neuron mask for the FSDP flat parameter and resets the weights of dormant neurons.
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
        index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
        
        all_masks = []
        all_ratios = []
        # print(f"[DEBUG] compute_fsdp_dormant_mask_only: Processing module {type(fsdp_module).__name__}")
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
                        my_global_end = min(my_global_end, my_global_start + flat_param.numel())
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
                my_global_start = 0
                my_global_end = flat_param.numel()
        else:
            print('[WARN][my_global_start] is Not None! Got it from param! ')
            my_global_end = my_global_start + flat_param.numel()
        #print(f"[DEBUG][nullspace] my_global_start={my_global_start}, my_global_end={my_global_end}, local flat_param.numel={flat_param.numel()}")
        #pass
        for idx, entry in enumerate(index_map):
            #print(f"[DEBUG][nullspace]entry:{entry}")
            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
            if overlap_info is None:
                continue
            entry_name = overlap_info['entry_name'] or idx
            local_slice_start = overlap_info['local_slice_start']
            valid_numel = overlap_info['valid_numel']
            num_rows = overlap_info['num_rows']
            num_cols = overlap_info['num_cols']
            sub_shape = overlap_info['sub_shape']
            # Guard against invalid/negative/zero shapes
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
                if verbose:
                    print(f"[WARN] Unknown mode: {mode}, using empty mask for entry {entry_name}")
                mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
            # Only assign to param_mask if mask shape matches
            if mask.shape == param_mask.shape[:1]:
                param_mask[mask] = True
                # Perform the reset for masked neurons
                if mask.any():
                    with torch.no_grad():
                        param_slice = flat_param.data[local_slice_start: local_slice_start + valid_numel]
                        param_slice = param_slice.view(sub_shape)
                        if use_lecun_init:
                            lecun_reset(param_slice, mask, sub_shape)
                        else:
                            kaiming_reset(param_slice, mask, sub_shape)
            else:
                if verbose:
                    print(f"[ERROR] Mask shape {mask.shape} does not match param_mask shape {param_mask.shape} for entry {entry_name}, skipping assignment.")
            ratio = mask.float().mean().item()
            all_masks.append(param_mask.flatten())
            all_ratios.append(ratio)

        #print(f"[DEBUG] Created mask for entry {entry_name}: {mask.sum().item()}/{mask.numel()} neurons marked dormant ({ratio:.6f})")

        # Combine all masks into one global mask
        if all_masks:
            global_mask = torch.cat(all_masks)
            avg_ratio = sum(all_ratios) / len(all_ratios)
            #print(f"[DEBUG] Combined mask has {global_mask.sum().item()}/{global_mask.numel()} elements marked dormant ({avg_ratio*100:.2f}%)")
            return global_mask
        else:
            #print(f"[DEBUG] No valid parameters found, returning empty mask")
            return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)

    except Exception as e:
        import traceback
        print(f"Error in fsdp_dormant_neuron_mask_and_reset: {e}")
        traceback.print_exc()
        # Return an empty mask as fallback
        return torch.zeros(1, dtype=torch.bool, device=next(fsdp_module.parameters()).device)
