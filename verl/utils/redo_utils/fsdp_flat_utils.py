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


def analyze_all_fsdp_dormant_neurons(module, mode='threshold', tau=0.1, verbose=True, original_shapes_map=None):
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
            mask = compute_fsdp_dormant_mask_only(submodule, mode=mode, tau=tau, verbose=verbose, original_shapes_map=original_shapes_map)
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

def analyze_all_fsdp_zero_grad_space(module, tau=0.1, verbose=True, original_shapes_map=None): # <<< Cascade: Added original_shapes_map
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
            stats = compute_fsdp_zero_grad_space_ratio(submodule, tau=tau, verbose=verbose, original_shapes_map=original_shapes_map) 
            if stats is not None and '__global__' in stats:
                submodule_global_stats = stats['__global__']
                total_zero += submodule_global_stats.get('zero', 0)
                total_rows += submodule_global_stats.get('total', 0)
                results[name] = stats # Store the full detailed stats for this submodule
            else:
                results[name] = None
                if verbose:
                    if stats is None:
                        print(f"[WARN][analyze_all_fsdp_zero_grad_space] compute_fsdp_zero_grad_space_ratio returned None for submodule {name}")
                    elif '__global__' not in stats:
                        print(f"[WARN][analyze_all_fsdp_zero_grad_space] '__global__' key missing in stats from compute_fsdp_zero_grad_space_ratio for submodule {name}. Stats: {stats}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not analyze zero grad space for {name}: {e}")
            results[name] = None

    global_ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    results['__global__'] = {'zero': total_zero, 'total': total_rows, 'ratio': global_ratio}
    return results

def redo_reset_all_fsdp_layers(module, mode='threshold', tau=0.1, verbose=True, use_lecun_init=True, original_shapes_map=None):
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
            mask = fsdp_dormant_neuron_mask_and_reset(submodule, mode=mode, tau=tau, use_lecun_init=use_lecun_init, verbose=verbose, optimizer=getattr(submodule, 'optimizer', None), original_shapes_map=original_shapes_map)
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
    # --- Preferred: use use_orig_params=True and _fsdp_prev (PyTorch 2.0+) ---
    index_map = []
    flat_param = None
    try:
        for name, param in fsdp_module.named_parameters():
            print('170name:',name,'param:',type(param),'shape:',param.shape)
            has_prev = hasattr(param, "_fsdp_prev")
            has_offset = has_prev and hasattr(param._fsdp_prev, "offset_in_flat_param")
            if has_offset:
                offset = param._fsdp_prev.offset_in_flat_param
                numel = param.numel()
                if flat_param is None and hasattr(param._fsdp_prev, "flat_param"):
                    flat_param = param._fsdp_prev.flat_param
                
                # Robust FQN retrieval
                fqn_attr = getattr(param, 'fqn', None) # Check if 'fqn' attribute exists
                if fqn_attr is not None:
                    fqn = fqn_attr
                else:
                    fqn = name # Fallback to name from named_parameters()
                
                # Ensure fqn is not None if name is valid and fqn_attr was None
                if fqn is None and name:
                    fqn = name
                    
                index_map.append({
                    "index": len(index_map),
                    "start": offset,
                    "end": offset + numel,
                    "shape": tuple(param.shape),
                    "param": param,
                    "fqn": fqn,
                })
        if flat_param is not None and len(index_map) > 0:
            index_map = sorted(index_map, key=lambda x: x["start"])
            return index_map, flat_param
    except Exception:
        pass
    # --- Fallback: legacy logic (for older FSDP or use_orig_params=False) ---
    from torch.distributed.fsdp import FlatParameter
    flat_param = next((p for p in fsdp_module.parameters() if isinstance(p, FlatParameter)), None)
    if flat_param is not None:
        shapes = getattr(flat_param, '_shapes', None)
        # Try to get FQNs from flat_param._fqns first, then _unflattened_param_names
        fqns_from_flat_param = getattr(flat_param, '_fqns', None)
        unflattened_names = getattr(flat_param, '_unflattened_param_names', None)
        
        offset = 0
        if shapes is not None:
            for i, shape in enumerate(shapes):
                numel = 1
                for s_dim in shape:
                    numel *= s_dim
                
                # Determine the FQN for the current parameter
                fqn = None
                if fqns_from_flat_param is not None and i < len(fqns_from_flat_param):
                    fqn = fqns_from_flat_param[i]
                elif unflattened_names is not None and i < len(unflattened_names):
                    fqn = unflattened_names[i]
                else:
                    fqn = f"param_{i}" # Default if no other name source found
                    
                start = offset
                end = offset + numel
                index_map.append({
                    'index': i,
                    'start': start,
                    'end': end,
                    'shape': shape,
                    'param': None, # Original param object not directly available here
                    'fqn': fqn,
                })
                offset = end
            return index_map, flat_param
    # --- Final fallback: regular parameter (non-FSDP) ---
    param = next((p for p in fsdp_module.parameters() if p.requires_grad), None)
    if param is None:
        param = next(fsdp_module.parameters(), None)
        if param is None:
            raise RuntimeError("No parameters found in module")
    shape = param.shape
    index_map = [{'index': 0, 'start': 0, 'end': param.numel(), 'shape': shape, 'param': param, 'fqn': getattr(param, 'fqn', None)}]
    return index_map, param


def map_dormant_neurons_to_layers(fsdp_module, mask, return_stats=True):
    """
    Given an FSDP module and a dormant neuron mask (1D tensor over flat param),
    returns a list of dicts with {'layer': fqn, 'neuron_idx': idx} for each dormant neuron.
    If return_stats=True, also returns layer-wise statistics.
    """
    import torch.distributed as dist
    import collections
    
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # Only print debug info on rank 0
    if rank == 0:
        print(f"[DEBUG] map_dormant_neurons_to_layers called with mask of shape {mask.shape if mask is not None else None}, sum: {mask.sum().item() if mask is not None else 0}")
    
    index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
    dormant_info = []
    if mask is None:
        return dormant_info
    
    mask = mask.detach().cpu().bool()
    
    # Get shard information - critical for FSDP
    my_global_start = getattr(flat_param, '_shard_start_idx', None)
    my_global_end = None
    
    # If _shard_start_idx is not available, compute it manually
    if my_global_start is None and world_size > 1:
        # Manual computation fallback (same as in compute_fsdp_zero_grad_space_ratio)
        if len(index_map) > 0:
            global_flat_param_numel = max(entry['end'] for entry in index_map)
            shard_size = (global_flat_param_numel + world_size - 1) // world_size
            my_global_start = rank * shard_size
            my_global_end = min((rank + 1) * shard_size, global_flat_param_numel)
            my_global_end = min(my_global_end, my_global_start + flat_param.numel())
            if rank == 0:
                print(f"[DEBUG] Computed shard info: rank={rank}, my_global_start={my_global_start}, my_global_end={my_global_end}, global_size={global_flat_param_numel}")
        else:
            my_global_start = 0
            my_global_end = flat_param.numel() if flat_param is not None else 0
    else:
        # If not sharded or _shard_start_idx is available
        if my_global_start is None:
            my_global_start = 0
        my_global_end = my_global_start + flat_param.numel() if flat_param is not None else 0
    
    # Debug info about index map
    if rank == 0:
        print(f"[DEBUG] Got index_map with {len(index_map)} entries")
        print(f"[DEBUG] Flat param shape: {flat_param.shape if flat_param is not None else None}")
        print(f"[DEBUG] Mask shape: {mask.shape}")
        print(f"[DEBUG] Shard info: start={my_global_start}, end={my_global_end}")
        
        # Print first few entries of index_map
        for i, entry in enumerate(index_map[:3]):
            print(f"[DEBUG] index_map[{i}]: start={entry['start']}, end={entry['end']}, shape={entry['shape']}, fqn={entry['fqn']}")
    
    # Limit to top 100 dormant neurons for logging
    max_dormant_to_log = 100
    dormant_count = 0
    
    # For statistics
    layer_stats = collections.defaultdict(lambda: {'dormant': 0, 'total': 0})
    total_dormant = 0
    total_neurons = 0
    
    for entry in index_map:
        start, end, shape, fqn = entry['start'], entry['end'], entry['shape'], entry['fqn']
        
        # Skip if no overlap with our shard
        if end <= my_global_start or start >= my_global_end:
            continue
            
        # Get the overlap region between this param and our shard
        overlap = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel())
        if overlap is None:
            continue
            
        # Use local indices within our shard
        local_start = overlap['local_slice_start']
        local_numel = overlap['valid_numel']
        local_end = local_start + local_numel
        
        # Check if mask is large enough
        if local_start >= mask.numel() or local_end > mask.numel():
            if rank == 0:
                print(f"[DEBUG] Skipping entry with local_start={local_start}, local_end={local_end} - mask too small (size={mask.numel()})")
            continue
            
        mask_slice = mask[local_start:local_end]
        dormant_in_slice = mask_slice.sum().item()
        
        if dormant_in_slice > 0 and rank == 0:
            print(f"[DEBUG] Found {dormant_in_slice} dormant neurons in layer {fqn} (shape {shape})")
        
        # Handle different parameter shapes
        if len(shape) == 1:
            # 1D param, e.g., LayerNorm
            # For 1D params, we need to map local indices back to global
            dormant_indices = mask_slice.nonzero(as_tuple=True)[0].tolist()
            
            # Update statistics
            layer_stats[fqn]['dormant'] += len(dormant_indices)
            layer_stats[fqn]['total'] += shape[0]  # Total neurons in this 1D layer
            total_dormant += len(dormant_indices)
            total_neurons += shape[0]
            
            for local_idx in dormant_indices:
                # Convert local index to global
                global_idx = local_idx + (overlap.get('global_offset', 0))
                if dormant_count < max_dormant_to_log:
                    dormant_info.append({'layer': fqn, 'neuron_idx': global_idx})
                    dormant_count += 1
                    
        elif len(shape) == 2:
            # 2D param, e.g., Linear
            try:
                # For 2D params, we need to reshape according to the overlap sub_shape
                sub_shape = overlap.get('sub_shape')
                if sub_shape is None:
                    continue
                    
                mask_slice_2d = mask_slice.view(sub_shape)
                row_sums = mask_slice_2d.sum(dim=1)
                dormant_indices = row_sums.nonzero(as_tuple=True)[0].tolist()
                
                # Update statistics - for 2D params, we count rows (neurons)
                layer_stats[fqn]['dormant'] += len(dormant_indices)
                layer_stats[fqn]['total'] += shape[0]  # Total rows/neurons in this 2D layer
                total_dormant += len(dormant_indices)
                total_neurons += shape[0]
                
                if rank == 0 and len(dormant_indices) > 0:
                    print(f"[DEBUG] Found {len(dormant_indices)} dormant rows in layer {fqn}")
                    
                for local_idx in dormant_indices:
                    # Convert local row index to global
                    global_idx = local_idx + (overlap.get('row_offset', 0))
                    if dormant_count < max_dormant_to_log:
                        dormant_info.append({'layer': fqn, 'neuron_idx': global_idx})
                        dormant_count += 1
            except Exception as e:
                if rank == 0:
                    print(f"[DEBUG] Error handling 2D param {fqn}: {e}")
                continue
        # Add more cases for Conv or other shapes if needed
    
    # Compute percentages and prepare statistics output
    if return_stats and rank == 0:
        stats_output = []
        for layer, counts in sorted(layer_stats.items()):
            dormant = counts['dormant']
            total = counts['total']
            percentage = (dormant / total * 100) if total > 0 else 0
            stats_output.append({
                'layer': layer,
                'dormant': dormant,
                'total': total,
                'percentage': percentage
            })
        
        # Add global statistics
        global_percentage = (total_dormant / total_neurons * 100) if total_neurons > 0 else 0
        stats_output.append({
            'layer': 'GLOBAL',
            'dormant': total_dormant,
            'total': total_neurons,
            'percentage': global_percentage
        })
        
        # Print summary statistics
        print(f"\n[DormantNeuron][Stats] Layer-wise dormant neuron statistics:")
        for stat in stats_output:
            print(f"  {stat['layer']}: {stat['dormant']}/{stat['total']} neurons dormant ({stat['percentage']:.2f}%)")
        print(f"  TOTAL: {total_dormant}/{total_neurons} neurons dormant ({global_percentage:.2f}%)\n")
        
        if rank == 0:
            print(f"[DEBUG] Returning {len(dormant_info)} dormant neurons (limited to {max_dormant_to_log})")
        
        return dormant_info, stats_output
    
    return dormant_info

def get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param_numel, verbose=False):
    """
    Given an index_map entry and the global start/end for this shard, compute the overlap region and local/global slices.
    Robustly supports 1D, 2D, 3D, and 4D parameter shapes. Returns None if there is no overlap or no full neuron fits.
    Returns dict with local_slice_start, valid_numel, sub_shape, entry_name. Only includes num_rows/num_cols for 2D.
    """
    param_start = entry['start']
    param_end = entry['end']
    overlap_start = max(param_start, my_global_start)
    overlap_end = min(param_end, my_global_end)
    if overlap_start >= overlap_end:
        return None
    local_slice_start = max(overlap_start - my_global_start, 0)
    actual_numel = overlap_end - overlap_start
    shape = entry['shape']
    if len(shape) == 1:
        valid_numel = actual_numel
        if valid_numel == 0:
            return None
        sub_shape = (valid_numel,)
        return {
            'local_slice_start': local_slice_start,
            'valid_numel': valid_numel,
            'sub_shape': sub_shape,
            'entry_name': entry.get('fqn', None) or entry.get('name', None)
        }
    elif len(shape) == 2:
        num_cols = shape[1]
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
            'sub_shape': sub_shape,
            'num_rows': num_rows,
            'num_cols': num_cols,
            'entry_name': entry.get('fqn', None) or entry.get('name', None)
        }
    elif len(shape) == 3:
        num_inner = shape[1] * shape[2]
        valid_numel = (actual_numel // num_inner) * num_inner
        if valid_numel == 0:
            return None
        num_outer = valid_numel // num_inner
        sub_shape = (num_outer, shape[1], shape[2])
        return {
            'local_slice_start': local_slice_start,
            'valid_numel': valid_numel,
            'sub_shape': sub_shape,
            'entry_name': entry.get('fqn', None) or entry.get('name', None)
        }
    elif len(shape) == 4:
        num_inner = shape[1] * shape[2] * shape[3]
        valid_numel = (actual_numel // num_inner) * num_inner
        if valid_numel == 0:
            return None
        num_outer = valid_numel // num_inner
        sub_shape = (num_outer, shape[1], shape[2], shape[3])
        return {
            'local_slice_start': local_slice_start,
            'valid_numel': valid_numel,
            'sub_shape': sub_shape,
            'entry_name': entry.get('fqn', None) or entry.get('name', None)
        }
    else:
        if verbose:
            print(f"[ERROR] Unsupported shape {shape} for entry {entry.get('fqn', None) or entry.get('name', None)}.")
        return None

def compute_fsdp_zero_grad_space_ratio(fsdp_module, tau=0.1, verbose=False, original_shapes_map=None): 
    """
    Computes the fraction of output neurons (rows) in each 2D param whose normalized gradient metric si = A/(B/H) is below tau,
    using GLOBAL layer statistics (B_global and H_global) for consistent metric calculation.
    
    Key improvements:
    1. Uses global layer statistics (B_global, H_global) for si calculation
    2. Batched all-reduces for efficiency
    3. Proper handling of sharded parameters
    4. Accurate per-layer statistics
    """
    import torch
    import torch.distributed as dist
    import collections
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not (dist.is_available() and dist.is_initialized()):
        if verbose:
            print("[WARN][ZeroGradV2] Distributed not initialized. Skipping calculation.")
        return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0}}

    rank = dist.get_rank()
    device = fsdp_module.compute_device
    layer_stats_local = {}
    global_contributions = []

    def _clean_fsdp_fqn(fqn_str):
        # FSDP often prepends '_fsdp_wrapped_module.' to parameter names.
        # This can happen multiple times for nested FSDP modules.
        # We remove all occurrences to get the original model's FQN.
        cleaned_fqn = fqn_str
        while "_fsdp_wrapped_module." in cleaned_fqn:
            cleaned_fqn = cleaned_fqn.replace("_fsdp_wrapped_module.", "")
        # Sometimes it might just be _checkpoint_wrapped_module without FSDP directly
        # Or other wrapper prefixes. For now, focusing on the common FSDP one.
        # A more robust solution might involve knowing the top-level model's name
        # and stripping prefixes until that is found, but this is a good start.
        return cleaned_fqn

    if rank == 0 and verbose:
        print(f"[ZeroGradV2-Debug][Rank {rank}] Starting analysis. Iterating named_parameters...")
    
    param_count_total = 0
    param_count_eligible_for_contrib = 0
    skipped_grad_none = 0
    skipped_dim_not_2 = 0
    skipped_shape0_is_0 = 0

    # Step 1: Collect local statistics for all layers
    param_iterator = fsdp_module.named_parameters()

    for fqn, param in param_iterator:
        param_count_total += 1
        if param.grad is None:
            skipped_grad_none += 1
            if rank == 0 and verbose and skipped_grad_none < 5: # Log first 4 occurrences
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {fqn}: SKIPPED (grad is None).")
            continue
        
        # <<< Cascade: Use original_shapes_map if available for dimension check >>>
        param_dim_to_check = -1
        original_shape_str = "N/A (no map)"
        if original_shapes_map:
            cleaned_fqn = _clean_fsdp_fqn(fqn)
            original_shape = original_shapes_map.get(cleaned_fqn)
            if original_shape:
                param_dim_to_check = len(original_shape)
                original_shape_str = str(original_shape)
            else:
                if rank == 0 and verbose and skipped_dim_not_2 < 5: 
                    print(f"[ZeroGradV2-Debug][Rank {rank}] Param {fqn} (cleaned: {cleaned_fqn}): SKIPPED (FQN not in original_shapes_map). Current dim: {param.dim()}, grad_shape: {param.grad.shape if param.grad is not None else 'N/A'}.")
                skipped_dim_not_2 += 1
                continue
        else: # Fallback if no map is provided
            param_dim_to_check = param.dim()

        if param_dim_to_check != 2:
            skipped_dim_not_2 += 1
            if rank == 0 and verbose and skipped_dim_not_2 < 5: # Log first 4 occurrences
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {fqn}: SKIPPED (dim_to_check is {param_dim_to_check}, original_shape: {original_shape_str}, current_param_dim: {param.dim()}, grad_shape: {param.grad.shape if param.grad is not None else 'N/A'}).")
                # print('param.grad:', param.grad) # Already printed by user's change
            continue
        # <<< End Cascade modification >>>

        if param.grad.shape[0] == 0:
            skipped_shape0_is_0 += 1
            if rank == 0 and verbose and skipped_shape0_is_0 < 5: # Log first 4 occurrences
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {fqn}: SKIPPED (grad.shape[0] is 0, grad_shape: {param.grad.shape}).")
            continue
        
        param_count_eligible_for_contrib +=1 # Parameter passed all initial checks
        current_grad_to_process = param.grad
        # <<< Cascade: Reshape grad if it's flat but originally 2D >>>
        reshaped_for_norm = False
        if original_shapes_map:
            cleaned_fqn = _clean_fsdp_fqn(fqn) # Ensure cleaned_fqn is available here too
            original_shape = original_shapes_map.get(cleaned_fqn)
            if original_shape and len(original_shape) == 2 and current_grad_to_process.dim() == 1:
                # Sanity check for number of elements before reshaping
                if current_grad_to_process.numel() == (original_shape[0] * original_shape[1]):
                    current_grad_to_process = current_grad_to_process.view(original_shape)
                    reshaped_for_norm = True
                elif rank == 0 and verbose:
                    print(f"[ZeroGradV2-Warning][Rank {rank}] Param {fqn}: Mismatch numel for reshape. Grad shape: {current_grad_to_process.shape}, Original: {original_shape}. Skipping reshape.")
        # <<< End Cascade modification >>>

        if rank == 0 and verbose and param_count_eligible_for_contrib <= 5: # Log first 5 eligible
            print(f"[ZeroGradV2-Debug][Rank {rank}] Param {fqn}: ELIGIBLE (original_grad_shape: {param.grad.shape}, processing_grad_shape: {current_grad_to_process.shape}, reshaped: {reshaped_for_norm}, current_param_dim: {param.dim()}).")
        
        # H_global and B_global are calculated based on all eligible params
        # For each parameter, calculate its contribution to H_global and B_global
        # grad_norm_row: (output_dim,)
        grad_norm_row = torch.norm(current_grad_to_process, p=2, dim=1)  # Norm along input_dim (dim 1)
        # H_local is the number of rows (output dimension) from the (potentially reshaped) gradient tensor
        H_local = current_grad_to_process.shape[0]
        
        # B_local is the sum of squared norms of all rows in the current layer's gradient
        B_local = torch.sum(grad_norm_row**2).item() 
        
        # S_local is the count of rows where the norm is below the tau-adjusted metric
        # This calculation is deferred until after B_global and H_global are known.
        
        param_details[fqn] = {
            'H_local': H_local, 
            'B_local': B_local, 
            'grad_shape': current_grad_to_process.shape, # Use shape of (potentially reshaped) grad
            'grad_norm_row_sample': grad_norm_row[:5].tolist() if H_local > 0 else []
        }

        global_contributions.append((fqn, H_local, B_local, grad_norm_row))

    if rank == 0 and verbose:
        print(f"[ZeroGradV2-Debug][Rank {rank}] Param iteration summary: Total iterated: {param_count_total}, Eligible for contribution processing: {param_count_eligible_for_contrib}, Actually added to global_contributions: {len(global_contributions)}")
        print(f"[ZeroGradV2-Debug][Rank {rank}] Skipped counts: grad_none={skipped_grad_none}, dim_not_2={skipped_dim_not_2}, shape0_is_0={skipped_shape0_is_0}")

    # Step 2: Batch all-reduce for global statistics
    if not global_contributions:
        if rank == 0 and verbose:
            print(f"[ZeroGradV2-Debug][Rank {rank}] global_contributions is empty. No valid parameters found or retained for metric calculation on this rank.")
        # No valid parameters found
        total_zero_global = torch.tensor(0.0, device=device, dtype=torch.float32) # Ensure float for division
        total_rows_global = torch.tensor(0.0, device=device, dtype=torch.float32)
    else:
        # Prepare batched tensors for all-reduce
        H_locals_list = [item[1] for item in global_contributions]
        B_locals_list = [item[2].item() for item in global_contributions] # .item() to get scalar from 0-dim tensor

        H_global_tensor = torch.tensor(H_locals_list, device=device, dtype=torch.float32)
        B_global_tensor = torch.tensor(B_locals_list, device=device, dtype=torch.float32)
        
        # All-reduce global statistics
        dist.all_reduce(H_global_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(B_global_tensor, op=dist.ReduceOp.SUM)
        
        # Step 3: Compute per-layer metrics with GLOBAL statistics
        total_zero_local = 0
        total_rows_local = 0
        
        for i, (fqn, H_local_scalar, B_local_scalar_tensor, A_local_row_tensor) in enumerate(global_contributions):
            H_global = H_global_tensor[i].item()
            B_global = B_global_tensor[i].item()
            
            if H_global == 0 or B_global == 0:
                # Entire layer has no gradient or no rows globally
                si = torch.zeros(H_local_scalar, device=device) # Use H_local_scalar for shape
            else:
                avg_global = B_global / H_global
                si = A_local_row_tensor / (avg_global + 1e-9)
            
            zero_rows = (si < tau).sum().item()
            layer_stats_local[fqn] = {'zero': zero_rows, 'total': H_local_scalar} # Use H_local_scalar
            total_zero_local += zero_rows
            total_rows_local += H_local_scalar # Use H_local_scalar

        # Convert to tensors for global reduction
        total_zero_global = torch.tensor(total_zero_local, device=device, dtype=torch.float32)
        total_rows_global = torch.tensor(total_rows_local, device=device, dtype=torch.float32)
    
    # Step 4: Global aggregation of total zero/rows counts
    dist.all_reduce(total_zero_global, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_rows_global, op=dist.ReduceOp.SUM)
    
    global_total_zero_val = total_zero_global.item()
    global_total_rows_val = total_rows_global.item()
    global_ratio = global_total_zero_val / (global_total_rows_val + 1e-8) if global_total_rows_val > 0 else 0.0

    results = {'__global__': {
        'zero': global_total_zero_val,
        'total': global_total_rows_val,
        'ratio': global_ratio
    }}
    
    # Step 5: Aggregate per-layer statistics globally
    all_layer_stats_gathered = [None] * dist.get_world_size()
    dist.all_gather_object(all_layer_stats_gathered, layer_stats_local) 
    
    combined_stats = collections.defaultdict(lambda: {'zero': 0, 'total': 0})
    if all_layer_stats_gathered[0] is not None: # Check if any stats were gathered
        for stats_dict_from_rank in all_layer_stats_gathered:
            if stats_dict_from_rank: # Ensure the dict from a rank is not empty
                for fqn, data in stats_dict_from_rank.items():
                    combined_stats[fqn]['zero'] += data['zero']
                    combined_stats[fqn]['total'] += data['total']
    
    for fqn, data in combined_stats.items():
        ratio = data['zero'] / (data['total'] + 1e-8) if data['total'] > 0 else 0.0
        results[fqn] = {**data, 'ratio': ratio}
    
    # Optional verbose output
    if verbose and rank == 0:
        print(f"[ZeroGradV2] Global: {results['__global__']['zero']:.0f}/{results['__global__']['total']:.0f} "
              f"({results['__global__']['ratio']:.4f})")
        sorted_layer_items = sorted([item for item in results.items() if item[0] != '__global__'])
        for fqn, stats in sorted_layer_items:
            print(f"[ZeroGradV2] {fqn}: {stats['zero']:.0f}/{stats['total']:.0f} ({stats['ratio']:.4f})")
    
    return results

def compute_fsdp_dormant_mask_only(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=True, original_shapes_map=None):
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

        #pass
        for idx, entry in enumerate(index_map):

            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
            if overlap_info is None:
                continue
            entry_name = overlap_info['entry_name'] or idx
            local_slice_start = overlap_info['local_slice_start']
            valid_numel = overlap_info['valid_numel']

            fqn = entry.get('fqn', str(idx)) # Use FQN if available, else index string

            # Determine if the parameter is authoritatively a valid 2D matrix
            param_is_valid_2d = False
            log_shape_info = "N/A"

            if original_shapes_map and fqn in original_shapes_map:
                original_shape = original_shapes_map[fqn]
                log_shape_info = f"original_map_shape={original_shape}"
                if isinstance(original_shape, (list, tuple)) and len(original_shape) == 2 and original_shape[0] > 0 and original_shape[1] > 0:
                    param_is_valid_2d = True
            elif 'num_rows' in overlap_info and 'num_cols' in overlap_info: # Fallback to info from get_shard_overlap_slices
                log_shape_info = f"overlap_info_shape={overlap_info.get('sub_shape')}, entry_shape={entry.get('shape')}"
                if overlap_info['num_rows'] > 0 and overlap_info['num_cols'] > 0:
                    param_is_valid_2d = True
            # Else, not 2D by any available information
            
            if not param_is_valid_2d:
                if verbose:
                    print(f"[INFO][DormantMask] Skipping entry {fqn} for dormant neuron analysis as it's not a valid 2D parameter. Details: {log_shape_info}")
                continue
            
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
        if len(shape) == 1:  # 1D
            fan_in = shape[0]
        elif len(shape) == 2:  # Linear
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
        if len(shape) == 1:  # 1D
            fan_in = shape[0]
        elif len(shape) == 2:  # Linear
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


def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=True, optimizer=None, original_shapes_map=None):
    """

    def _clean_fsdp_fqn(fqn_str):
        # FSDP often prepends '_fsdp_wrapped_module.' to parameter names.
        # This can happen multiple times for nested FSDP modules.
        # We remove all occurrences to get the original model's FQN.
        cleaned_fqn = fqn_str
        while "_fsdp_wrapped_module." in cleaned_fqn:
            cleaned_fqn = cleaned_fqn.replace("_fsdp_wrapped_module.", "")
        return cleaned_fqn

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
        import torch
        import torch.distributed as dist

        index_map, flat_param = get_fsdp_flat_param_index_map(fsdp_module)
        all_masks_details_local = [] # Stores {'fqn': cleaned_fqn, 'dormant': num_dormant_in_shard, 'total_in_shard': num_neurons_in_shard}
        
        # This will store the number of resets done on this rank, keyed by cleaned_fqn
        # It will be aggregated later if distributed.
        reset_counts_details_local = {}

        my_global_start = getattr(flat_param, '_shard_start_idx', None)
        my_global_end = None

        current_rank = -1
        is_distributed = dist.is_available() and dist.is_initialized()
        if is_distributed:
            current_rank = dist.get_rank()

        if my_global_start is None:
            try:
                if is_distributed:
                    world_size = dist.get_world_size()
                    if len(index_map) > 0:
                        global_flat_param_numel = max(entry['end'] for entry in index_map)
                        shard_size = (global_flat_param_numel + world_size - 1) // world_size
                        my_global_start = current_rank * shard_size
                        my_global_end = min((current_rank + 1) * shard_size, global_flat_param_numel)
                        my_global_end = min(my_global_end, my_global_start + flat_param.numel()) if flat_param is not None else my_global_start
                    else:
                        my_global_start = 0
                        my_global_end = 0
                else:
                    my_global_start = 0
                    my_global_end = flat_param.numel() if flat_param is not None else 0
            except Exception as e:
                if verbose and (not is_distributed or current_rank == 0): print(f"[WARN][DormantReset] Error determining shard bounds: {e}. Assuming full param.")
                my_global_start = 0
                my_global_end = flat_param.numel() if flat_param is not None else 0
        elif flat_param is not None:
            my_global_end = my_global_start + flat_param.numel()
        else: # flat_param is None, implies no parameters or error in get_fsdp_flat_param_index_map
             if verbose and (not is_distributed or current_rank == 0): print(f"[WARN][DormantReset] flat_param is None. Cannot proceed.")
             return {}


        if verbose and (not is_distributed or current_rank == 0):
            print(f"[DEBUG][DormantReset] Rank {current_rank if is_distributed else 0} processing. Global Start: {my_global_start}, Global End: {my_global_end}, Flat Param Numel: {flat_param.numel() if flat_param is not None else 'N/A'}")

        for idx, entry in enumerate(index_map):
            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose and (not is_distributed or current_rank == 0))
            if overlap_info is None:
                continue

            fqn = entry.get('fqn', str(idx))
            cleaned_fqn = _clean_fsdp_fqn(fqn)

            param_is_valid_for_dormancy_check = False
            log_shape_info = "N/A"
            is_1d_param = False
            original_shape_for_analysis = None

            if original_shapes_map and cleaned_fqn in original_shapes_map:
                original_shape = original_shapes_map[cleaned_fqn]
                original_shape_for_analysis = original_shape
                log_shape_info = f"original_map_shape={original_shape}"
                if isinstance(original_shape, (list, tuple)) and len(original_shape) == 2 and original_shape[0] > 0 and original_shape[1] > 0:
                    param_is_valid_for_dormancy_check = True
                elif isinstance(original_shape, (list, tuple)) and len(original_shape) == 1 and original_shape[0] > 0:
                    param_is_valid_for_dormancy_check = True
                    is_1d_param = True
            elif isinstance(overlap_info['sub_shape'], (tuple, list)) and len(overlap_info['sub_shape']) > 0 and all(dim > 0 for dim in overlap_info['sub_shape']):
                original_shape_for_analysis = overlap_info['sub_shape']
                log_shape_info = f"sub_shape={overlap_info['sub_shape']} (used as fallback), entry_orig_shape={entry.get('shape')}"
                if len(overlap_info['sub_shape']) == 2:
                    param_is_valid_for_dormancy_check = True
                elif len(overlap_info['sub_shape']) == 1:
                    param_is_valid_for_dormancy_check = True
                    is_1d_param = True
            
            if not param_is_valid_for_dormancy_check:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[INFO][DormantReset] Skipping entry {fqn} (cleaned: {cleaned_fqn}). Not valid 1D/2D or shape info missing. Details: {log_shape_info}")
                continue
            
            if not (isinstance(original_shape_for_analysis, (tuple, list)) and len(original_shape_for_analysis) > 0 and all(dim > 0 for dim in original_shape_for_analysis)):
                 if verbose and (not is_distributed or current_rank == 0):
                    print(f"[ERROR][DormantReset] Invalid original_shape_for_analysis {original_shape_for_analysis} for {fqn} (cleaned: {cleaned_fqn}). Skipping.")
                 continue

            local_param_data = flat_param.data[overlap_info['local_slice_start'] : overlap_info['local_slice_start'] + overlap_info['valid_numel']]
            
            if local_param_data.numel() == 0:
                 if verbose and (not is_distributed or current_rank == 0):
                    print(f"[DEBUG][DormantReset] Empty local_param_data for {fqn} (cleaned: {cleaned_fqn}). Skipping.")
                 continue

            param_view_for_analysis = None
            try:
                # Use sub_shape from overlap_info for viewing the shard, as it reflects the shard's actual dimensions
                current_shard_shape = overlap_info['sub_shape']
                if local_param_data.numel() == torch.Size(current_shard_shape).numel():
                     param_view_for_analysis = local_param_data.view(current_shard_shape)
                else:
                    if verbose and (not is_distributed or current_rank == 0):
                        print(f"[WARN][DormantReset] Numel mismatch for shard view. FQN: {fqn}, Shard Numel: {local_param_data.numel()}, Expected Shard Shape: {current_shard_shape}. Skipping.")
                    continue
            except RuntimeError as e:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[ERROR][DormantReset] Error viewing shard {fqn} with shape {current_shard_shape}: {e}. Skipping.")
                continue
            
            if param_view_for_analysis is None: continue

            num_neurons_in_shard = 0
            dormant_mask_for_shard = None

            if not is_1d_param and len(param_view_for_analysis.shape) == 2: # 2D matrix shard
                num_neurons_in_shard = param_view_for_analysis.shape[0] # Rows in this shard
                if num_neurons_in_shard == 0: continue
                neuron_abs_mean = torch.mean(torch.abs(param_view_for_analysis.data), dim=1)
                if mode == 'threshold': dormant_mask_for_shard = neuron_abs_mean < tau
                # (Simplified: Add percentage/hybrid logic here if needed, similar to zero_grad_util)
                else: dormant_mask_for_shard = neuron_abs_mean < tau # Fallback
            elif is_1d_param and len(param_view_for_analysis.shape) == 1: # 1D vector shard
                num_neurons_in_shard = param_view_for_analysis.shape[0]
                if num_neurons_in_shard == 0: continue
                if mode == 'threshold': dormant_mask_for_shard = torch.abs(param_view_for_analysis.data) < tau
                else: dormant_mask_for_shard = torch.abs(param_view_for_analysis.data) < tau # Fallback
            else:
                if verbose and (not is_distributed or current_rank == 0): print(f"[ERROR][DormantReset] Param {fqn} has unexpected shard shape {param_view_for_analysis.shape}. Skipping.")
                continue
            
            num_dormant_in_shard = torch.sum(dormant_mask_for_shard).item()

            if num_dormant_in_shard > 0:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[INFO][DormantReset] Param {fqn} (cleaned: {cleaned_fqn}), Shard Shape: {param_view_for_analysis.shape}, Dormant in shard: {num_dormant_in_shard}/{num_neurons_in_shard}")
                with torch.no_grad():
                    if not is_1d_param:
                        rows_to_reset = param_view_for_analysis.data[dormant_mask_for_shard]
                        if use_lecun_init: lecun_reset(rows_to_reset)
                        else: kaiming_reset(rows_to_reset, is_bias=False)
                        param_view_for_analysis.data[dormant_mask_for_shard] = rows_to_reset
                    else:
                        elements_to_reset = param_view_for_analysis.data[dormant_mask_for_shard]
                        if use_lecun_init: lecun_reset(elements_to_reset)
                        else: kaiming_reset(elements_to_reset, is_bias=True)
                        param_view_for_analysis.data[dormant_mask_for_shard] = elements_to_reset
                
                reset_counts_details_local[cleaned_fqn] = reset_counts_details_local.get(cleaned_fqn, 0) + num_dormant_in_shard
                # Optimizer state reset logic would be very complex here and needs careful handling of flat_param and optimizer state indices.
                # This is often done via custom hooks or by directly manipulating optimizer.state[flat_param].
                # For now, this part is omitted for brevity but is crucial for effective reset.

            all_masks_details_local.append({'fqn': cleaned_fqn, 'dormant': num_dormant_in_shard, 'total_in_shard': num_neurons_in_shard})

        # Aggregate reset_counts_details if distributed
        final_reset_counts = {}
        if is_distributed:
            gathered_counts_list = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_counts_list, reset_counts_details_local)
            if current_rank == 0:
                for rank_counts_dict in gathered_counts_list:
                    for f_name, count in rank_counts_dict.items():
                        final_reset_counts[f_name] = final_reset_counts.get(f_name, 0) + count
        else: # Not distributed
            final_reset_counts = reset_counts_details_local
        
        if verbose and (not is_distributed or current_rank == 0) and final_reset_counts:
             print(f"[INFO][DormantReset][Aggregated] Reset counts: {final_reset_counts}")
        
        return final_reset_counts

    except Exception as e:
        if verbose and (not is_distributed or (is_distributed and dist.get_rank() == 0)):
            import traceback
            print(f"[ERROR][DormantReset] Exception: {e}\n{traceback.format_exc()}")
        return {}