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

def compute_fsdp_zero_grad_space_ratio(fsdp_module, tau=0.1, verbose=False):
    """
    Computes the fraction of output neurons (rows) in each 2D param whose normalized gradient metric si = A/(B/H) is below tau,
    iterating over fsdp_module.named_parameters() (expects use_orig_params=True).
    - H_local: number of neurons (rows) in the local shard of the layer's gradient.
    - B_local: sum of absolute gradients in the local shard of the layer's gradient.
    - A_local_row: sum of absolute gradients for a row within the local shard.
    - si_local_row = A_local_row / (B_local / H_local)
    Counts of zero-gradient rows are aggregated globally.
    Returns a dictionary with per-layer and global statistics.
    """
    import torch
    import torch.distributed as dist
    import collections

    if not (dist.is_available() and dist.is_initialized()):
        if verbose:
            print("[WARN][ZeroGradV2] Distributed not initialized. Skipping calculation.")
        # Return a structure consistent with the populated case, but with zero values.
        return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0}}

    rank = dist.get_rank()
    total_zero_local = 0  # Renamed to clarify it's rank-local before aggregation
    total_rows_local = 0    # Renamed to clarify it's rank-local before aggregation
    layer_stats_local = collections.defaultdict(lambda: {'zero': 0, 'total': 0})
    # world_size = dist.get_world_size() # Not explicitly needed with all_gather_object

    for fqn, param in fsdp_module.named_parameters():
        if param.grad is not None and param.dim() == 2 and param.grad.shape[0] > 0:
            grad_matrix = param.grad
            H = grad_matrix.shape[0]
            
            grad_matrix_abs = grad_matrix.abs()
            B_local_param = grad_matrix_abs.sum()
            
            if H == 0:
                si_local_row_param = torch.empty(0, device=param.device)
            elif B_local_param == 0:
                # If total sum of grads is 0, all rows are effectively 'zero-activity' w.r.t this metric
                # so si would be 0/0 effectively. Treat as all rows < tau.
                si_local_row_param = torch.zeros(H, device=param.device)
            else:
                avg_row_grad_sum_metric = B_local_param / H
                A_local_row_param = grad_matrix_abs.sum(dim=1)
                si_local_row_param = A_local_row_param / (avg_row_grad_sum_metric + 1e-9) # Add epsilon for stability
            
            zero_vectors = (si_local_row_param < tau)
            
            current_param_zero = zero_vectors.sum().item()
            current_param_rows = H

            layer_stats_local[fqn] = {'zero': current_param_zero, 'total': current_param_rows}
            
            total_zero_local += current_param_zero
            total_rows_local += current_param_rows

    # Global aggregation for total_zero and total_rows
    device = fsdp_module.compute_device # FSDP module's compute device

    total_zero_tensor = torch.tensor([total_zero_local], dtype=torch.float32, device=device)
    total_rows_tensor = torch.tensor([total_rows_local], dtype=torch.float32, device=device)

    dist.all_reduce(total_zero_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_rows_tensor, op=dist.ReduceOp.SUM)

    global_total_zero = total_zero_tensor.item()
    global_total_rows = total_rows_tensor.item()
    global_ratio = global_total_zero / (global_total_rows + 1e-8) if global_total_rows > 0 else 0.0

    # Per-layer statistics aggregation
    all_rank_layer_stats = [None] * dist.get_world_size()
    dist.all_gather_object(all_rank_layer_stats, layer_stats_local)

    final_layer_stats = collections.defaultdict(lambda: {'zero': 0, 'total': 0, 'ratio': 0.0})
    # Process all_rank_layer_stats on all ranks, so all ranks have the full picture if needed
    # Alternatively, can restrict this processing and subsequent printing to rank == 0
    for rank_data in all_rank_layer_stats:
        for fqn, stats in rank_data.items():
            final_layer_stats[fqn]['zero'] += stats['zero']
            final_layer_stats[fqn]['total'] += stats['total']
    
    for fqn in final_layer_stats: # Calculate ratios after summing up zero/total from all ranks
        layer_zero = final_layer_stats[fqn]['zero']
        layer_total = final_layer_stats[fqn]['total']
        final_layer_stats[fqn]['ratio'] = layer_zero / (layer_total + 1e-8) if layer_total > 0 else 0.0

    results = {'__global__': {'zero': global_total_zero, 'total': global_total_rows, 'ratio': global_ratio}}
    results.update(dict(final_layer_stats)) # Convert defaultdict to dict for cleaner output/serialization

    if verbose and rank == 0:
        print(f"[INFO][ZeroGradV2][Rank {rank}] Global Zero Grad Space: {global_total_zero:.0f}/{global_total_rows:.0f} ({global_ratio:.6f})")
        for fqn, stats in sorted(results.items()): # Iterate over results to include __global__ if desired in sorted output
            if fqn == '__global__': continue # Already printed
            print(f"[INFO][ZeroGradV2][Rank {rank}] Layer {fqn}: Zero {stats['zero']:.0f}/{stats['total']:.0f} ({stats.get('ratio',0.0):.6f})")
    
    return results

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

        #pass
        for idx, entry in enumerate(index_map):

            overlap_info = get_shard_overlap_slices(entry, my_global_start, my_global_end, flat_param.numel(), verbose=verbose)
            if overlap_info is None:
                continue
            entry_name = overlap_info['entry_name'] or idx
            local_slice_start = overlap_info['local_slice_start']
            valid_numel = overlap_info['valid_numel']

            # Ensure this is a 2D parameter for which num_rows/num_cols would be present
            # The current dormant neuron logic (si calculation) is specific to 2D matrices.
            if 'num_rows' not in overlap_info or 'num_cols' not in overlap_info:
                if verbose:
                    entry_name_for_log = overlap_info.get('entry_name', idx)
                    param_shape_for_log = overlap_info.get('sub_shape', 'Unknown shape')
                    print(f"[INFO][DormantMask] Skipping entry {entry_name_for_log} for dormant neuron analysis as it's not 2D (or num_rows/num_cols missing). Shape: {param_shape_for_log}")
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

def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=True, optimizer=None):
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
            sub_shape = overlap_info['sub_shape']
            if not isinstance(sub_shape, (tuple, list)) or len(sub_shape) < 1 or any(dim <= 0 for dim in sub_shape):
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
            if grad_raw.numel() != valid_numel:
                if verbose:
                    print(f"[WARN] grad_raw.numel()={grad_raw.numel()} does not match valid_numel={valid_numel} or sub_shape={sub_shape}, skipping entry {entry_name}.")
                continue
            try:
                grad_slice = grad_raw.view(sub_shape)
            except Exception as e:
                if verbose:
                    print(f"[ERROR] Could not reshape grad_raw to {sub_shape} for entry {entry_name}: {e}")
                continue
            grad_magnitude = grad_slice.abs()
            if len(sub_shape) == 1:
                mask = grad_magnitude <= tau
            else:
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
            if mask.shape == param_mask.shape:
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
                                                if verbose and mask.sum().item() > 0:
                                                    before_mean = exp_avg_slice[mask].mean().item() if exp_avg_slice[mask].numel() > 0 else 0.0
                                                    print(f"[DEBUG] exp_avg before reset (mean over masked): {before_mean}")
                                                exp_avg_slice[mask] = 0.0
                                                if verbose and mask.sum().item() > 0:
                                                    after_mean = exp_avg_slice[mask].mean().item() if exp_avg_slice[mask].numel() > 0 else 0.0
                                                    print(f"[DEBUG] exp_avg after reset (mean over masked): {after_mean}")
                                            if 'exp_avg_sq' in optimizer.state[p]:
                                                exp_avg_sq = optimizer.state[p]['exp_avg_sq']
                                                exp_avg_sq_slice = exp_avg_sq[local_slice_start: local_slice_start + valid_numel].view(sub_shape)
                                                if verbose and mask.sum().item() > 0:
                                                    before_mean_sq = exp_avg_sq_slice[mask].mean().item() if exp_avg_sq_slice[mask].numel() > 0 else 0.0
                                                    print(f"[DEBUG] exp_avg_sq before reset (mean over masked): {before_mean_sq}")
                                                exp_avg_sq_slice[mask] = 0.0
                                                if verbose and mask.sum().item() > 0:
                                                    after_mean_sq = exp_avg_sq_slice[mask].mean().item() if exp_avg_sq_slice[mask].numel() > 0 else 0.0
                                                    print(f"[DEBUG] exp_avg_sq after reset (mean over masked): {after_mean_sq}")
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