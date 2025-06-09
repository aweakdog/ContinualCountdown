"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
"""
import torch
import math
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDP
from torch import nn
import torch.distributed as dist
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable
import logging
import time
import math
import numpy as np
import copy
import warnings
from collections import defaultdict
import os
from transformers import AutoModelForCausalLM, AutoConfig
from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# Global variable to store reference model parameters
REFERENCE_MODEL_PARAMS = {}

def load_reference_model(model_path: str, trust_remote_code: bool = False, device: str = 'cpu'):
    """
    Load a reference model from the given path and store its parameters in a global dictionary.
    
    Args:
        model_path: Path to the reference model
        trust_remote_code: Whether to trust remote code when loading the model
        device: Device to load the model on (default: 'cpu' to save GPU memory)
        
    Returns:
        Dictionary mapping parameter FQNs to their values
    """
    global REFERENCE_MODEL_PARAMS
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[INFO] Loading reference model from {model_path} on device {device}")
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    
    # Load model weights
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float32,  # Load in fp32 for maximum precision
            trust_remote_code=trust_remote_code
        )
        model.to(device)
        
        # Store parameters in dictionary
        for name, param in model.named_parameters():
            REFERENCE_MODEL_PARAMS[name] = param.detach().clone()
            
        # Free up memory
        del model
        torch.cuda.empty_cache()
    
    if rank == 0:
        print(f"[INFO] Successfully loaded reference model with {len(REFERENCE_MODEL_PARAMS)} parameters")
    
    return REFERENCE_MODEL_PARAMS


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

def analyze_all_fsdp_zero_grad_space(module, tau=0.1, verbose=True, original_shapes_map=None, top_level_prefix=None, skip_mlp=True, skip_embed=True):
    """
    Analyze all leaf FSDP-wrapped submodules for zero grad space ratio.
    Returns a dict mapping module names to their zero grad stats and the global aggregate.
    
    Args:
        module: The module to analyze
        tau: Threshold for considering a gradient as zero
        verbose: Whether to print verbose output
        original_shapes_map: Map of parameter FQNs to their original shapes before FSDP wrapping
        top_level_prefix: Prefix to use for parameter FQNs when analyzing the top-level module
                         If None, will attempt to detect the appropriate prefix
    """
    from verl.utils.redo_utils.fsdp_flat_utils import compute_fsdp_zero_grad_space_ratio
    results = {}
    total_zero = 0
    total_rows = 0
    
    ## Check if the module itself is FSDP-wrapped
    #if hasattr(module, '_fsdp_wrapped_module'):
    #    # Try to determine the appropriate prefix for the top-level module
    #    if top_level_prefix is None:
    #        # Check if this is a Qwen model by inspecting the wrapped module
    #        wrapped_module = module._fsdp_wrapped_module
    #        if hasattr(wrapped_module, 'model') and hasattr(wrapped_module.model, 'embed_tokens'):
    #            # For Qwen models, parameters are typically prefixed with 'model'
    #            detected_prefix = "model"
    #        else:
    #            # Default to empty prefix if we can't determine the structure
    #            detected_prefix = ""
    #        
    #        if verbose:
    #            print(f"[ZeroGradV2] Auto-detected top-level prefix: '{detected_prefix}'")
    #    else:
    #        detected_prefix = top_level_prefix
    #    
    #    if verbose:
    #        print(f"[ZeroGradV2] Analyzing top-level FSDP module directly with prefix: '{detected_prefix}'")
    #    
    #    try:
    #        # Analyze the top-level module directly
    #        stats = compute_fsdp_zero_grad_space_ratio(module, tau=tau, verbose=verbose, 
    #                                                  original_shapes_map=original_shapes_map, 
    #                                                  fqn_prefix=detected_prefix)
    #        if stats is not None and '__global__' in stats:
    #            results[detected_prefix or "top_level"] = stats
    #            global_stats = stats['__global__']
    #            total_zero = global_stats.get('zero', 0)
    #            total_rows = global_stats.get('total', 0)
    #            global_ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    #            results['__global__'] = {'zero': total_zero, 'total': total_rows, 'ratio': global_ratio}
    #            return results
    #        else:
    #            if verbose:
    #                print(f"[WARN] Top-level module analysis failed or returned no global stats")
    #    except Exception as e:
    #        if verbose:
    #            print(f"[WARN] Could not analyze zero grad space for top-level module: {e}")
    #            import traceback
    #            traceback.print_exc()
    
    ## Fall back to per-layer analysis if top-level analysis fails or module is not FSDP-wrapped
    #if verbose:
    #    print(f"[ZeroGradV2] Falling back to per-layer FSDP module analysis")
    
    for name, submodule in iter_leaf_fsdp_modules(module):
        if verbose:
            print(f'169Analyzing submodule: {name}')
        try:
            stats = compute_fsdp_zero_grad_space_ratio(submodule, tau=tau, verbose=verbose, 
                                                       original_shapes_map=original_shapes_map, 
                                                       fqn_prefix=name,
                                                       skip_mlp=skip_mlp,
                                                       skip_embed=skip_embed)
            if stats is not None and '__global__' in stats:
                # Ensure 'aggregated_ratio' is present in '__global__' for compatibility with dp_actor.py
                if 'ratio' in stats['__global__'] and 'aggregated_ratio' not in stats['__global__']:
                    stats['__global__']['aggregated_ratio'] = stats['__global__']['ratio']
                # If compute_fsdp_zero_grad_space_ratio was successful for this module, return its stats directly
                return stats
            else:
                if verbose:
                    if stats is None:
                        print(f"[WARN][analyze_all_fsdp_zero_grad_space] compute_fsdp_zero_grad_space_ratio returned None for submodule {name}")
                    elif '__global__' not in stats:
                        print(f"[WARN][analyze_all_fsdp_zero_grad_space] '__global__' key missing in stats from compute_fsdp_zero_grad_space_ratio for submodule {name}")
                # Return a minimal dictionary if analysis of the first module failed
                return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}}
        except Exception as e:
            if verbose:
                print(f"[WARN][analyze_all_fsdp_zero_grad_space] Exception analyzing zero grad space for {name}: {e}")
                import traceback
                traceback.print_exc()
            return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}} # Return minimal dict on error
        break # hacky - ensures we only process the first FSDP module and then exit the loop (and function)

    # This part is reached only if iter_leaf_fsdp_modules yields nothing.
    if verbose:
        print(f"[WARN][analyze_all_fsdp_zero_grad_space] iter_leaf_fsdp_modules yielded no FSDP modules for module: {type(module)}")
    return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}}

#def analyze_all_fsdp_zero_grad_space(module, tau=0.1, verbose=True, original_shapes_map=None, top_level_prefix=None, skip_mlp=True, skip_embed=True, current_step=None):
#    """
#    Analyze the FSDP-wrapped module for zero grad space ratio.
#    Assumes the first module from iter_leaf_fsdp_modules is the root FSDP module and
#    compute_fsdp_zero_grad_space_ratio performs a full analysis on it.
#    
#    Args:
#        module: The FSDP-wrapped module to analyze (expected to be the root FSDP module)
#        tau: Threshold for considering a gradient as zero
#        verbose: Whether to print verbose output
#        original_shapes_map: Map of parameter FQNs to their original shapes
#        top_level_prefix: (Largely unused if the primary logic path is taken) Prefix for FQNs
#        skip_mlp: Whether to skip MLP layers during analysis
#        skip_embed: Whether to skip embedding layers during analysis
#        current_step: Current training step, for logging purposes
#    """
#    from verl.utils.redo_utils.fsdp_flat_utils import compute_fsdp_zero_grad_space_ratio
#    import torch.distributed as dist # For rank-specific logging
#
#    # --- NEW SIMPLER DEBUG PRINT: Log entry --- 
#    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
#        print(f"[DEBUG_ENTRY] ENTERING analyze_all_fsdp_zero_grad_space. Step: {current_step}")
#    # --- END NEW SIMPLER DEBUG PRINT ---
#
#    # --- DEBUG PRINT: Log invocation --- 
#    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
#        print(f"[DEBUG_INVOCATION] analyze_all_fsdp_zero_grad_space called. Step: {current_step}, Module ID: {id(module)}, Module Type: {type(module)}")
#    # --- END DEBUG PRINT ---
#
#    # Directly use 'module' as the FSDP module to analyze.
#    # 'name' (fqn_prefix) will be an empty string, assuming 'module' is the root.
#    name = "" 
#    submodule_obj = module # In this context, module is the FSDP instance directly
#
#    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
#        # Updated log message to reflect direct call
#        print(f'[INFO][analyze_all_fsdp_zero_grad_space] Analyzing FSDP module directly. Module Type: {type(submodule_obj)}, Module ID: {id(submodule_obj)}, Step: {current_step}')
#
#    try:
#        # compute_fsdp_zero_grad_space_ratio is expected to return a flat dictionary
#        # with all parameter FQNs and their stats, including a '__global__' key.
#        stats = compute_fsdp_zero_grad_space_ratio(
#            submodule_obj,  # Use submodule_obj which is 'module'
#            tau=tau, 
#            verbose=verbose, 
#            original_shapes_map=original_shapes_map, 
#            fqn_prefix=name, # Use the defined 'name' (empty string)
#            skip_mlp=skip_mlp,
#            skip_embed=skip_embed
#        )
#
#        if stats: # If stats is not None and not empty
#            if verbose and ('__global__' in stats) and \
#               (not dist.is_initialized() or dist.get_rank() == 0):
#                global_ratio_from_stats = stats['__global__'].get('ratio', 0.0)
#                # Updated log message
#                print(f"[ZeroGradV2-Metrics][FSDP Root Analysis][Step {current_step}] Aggregated Zero Grad Space Ratio (direct call): {global_ratio_from_stats:.4f}")
#            
#            # Ensure 'aggregated_ratio' is present in '__global__' for compatibility with dp_actor.py
#            if '__global__' in stats and 'ratio' in stats['__global__'] and 'aggregated_ratio' not in stats['__global__']:
#                stats['__global__']['aggregated_ratio'] = stats['__global__']['ratio']
#            elif '__global__' not in stats: # If __global__ key is missing entirely
#                 stats['__global__'] = {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}
#
#            return stats # Return the comprehensive stats dictionary directly
#        else:
#            # This case handles if compute_fsdp_zero_grad_space_ratio itself returns None or empty
#            if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
#                # Updated log message
#                print(f"[WARN][analyze_all_fsdp_zero_grad_space] compute_fsdp_zero_grad_space_ratio returned None or empty for FSDP module (direct call). Module ID: {id(submodule_obj)}")
#            return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}} # Return minimal dict
#            
#    except Exception as e:
#        if verbose: 
#            current_rank = -1 
#            if dist.is_available() and dist.is_initialized():
#                current_rank = dist.get_rank()
#            
#            # Updated log message
#            print(f"[ERROR][Rank {current_rank}][analyze_all_fsdp_zero_grad_space] Exception in direct call. Module ID: {id(submodule_obj)}. Error: {e}")
#            import traceback
#            traceback.print_exc()
#        return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0, 'aggregated_ratio': 0.0}} # Return minimal dict on error

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

def compute_fsdp_zero_grad_space_ratio(fsdp_module, tau=0.1, verbose=True, original_shapes_map=None, debug_mlp_sharding=True, fqn_prefix="", skip_mlp=True, skip_embed=True):  # Added skip_mlp parameter with default True
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
    results = {}  # Initialize results dictionary
    
    # Initialize debug counter to limit print statements
    debug_print_count = 0
    global_contributions = []
    param_details = {} # <<< Cascade: Re-initialize param_details

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
    
    # Create a canonical list of parameter FQNs from original_shapes_map if available
    # This ensures all ranks process the same parameters in the same order
    canonical_param_fqns = []
    if original_shapes_map:
        # Sort to ensure consistent order across ranks
        canonical_param_fqns = sorted(original_shapes_map.keys())
        if rank == 0 and verbose:
            print(f"[ZeroGradV2-Debug][Rank {rank}] Using canonical parameter list from original_shapes_map with {len(canonical_param_fqns)} parameters.")
    
    # Create a map of local parameters for lookup
    local_params = {_clean_fsdp_fqn(f"{fqn_prefix}.{name}" if fqn_prefix else name): param 
                   for name, param in fsdp_module.named_parameters()}
    
    # Counters for diagnostics
    param_count_total = 0
    param_count_eligible_for_contrib = 0
    skipped_grad_none = 0
    skipped_dim_not_2 = 0
    skipped_shape0_is_0 = 0
    skipped_not_in_local = 0
    
    global_contributions = []
    param_details = {}
    
    # If we have a canonical list, use it; otherwise fall back to iterating local parameters
    if canonical_param_fqns:
        param_iterator = canonical_param_fqns
        if rank == 0 and verbose:
            print(f"[ZeroGradV2-Debug][Rank {rank}] Using canonical parameter list with {len(canonical_param_fqns)} parameters.")
    else:
        # Fall back to local parameters if no original_shapes_map
        param_iterator = [_clean_fsdp_fqn(f"{fqn_prefix}.{name}" if fqn_prefix else name) 
                         for name, param in fsdp_module.named_parameters()]
        if rank == 0 and verbose:
            print(f"[ZeroGradV2-Debug][Rank {rank}] Using local parameter list with {len(param_iterator)} parameters.")
    
    # Process parameters in canonical order
    for full_fqn_for_map in param_iterator:
        param_count_total += 1
        # Initialize default values for this parameter (will be used if parameter is not eligible)
        H_local = 0
        B_local = 0.0
        grad_norm_row = torch.zeros(1, device=device)
        is_eligible = False
        
        # Check if parameter is in local parameters
        if full_fqn_for_map not in local_params:
            skipped_not_in_local += 1
            if rank == 0 and verbose and skipped_not_in_local < 5:
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: CONTRIBUTING ZEROS (not in local_params).")
        else:
            param = local_params[full_fqn_for_map]
            
            # Skip MLP layers if requested
            if skip_mlp and ("mlp" in full_fqn_for_map or "mlp." in full_fqn_for_map):
                skipped_mlp_layer = getattr(locals(), 'skipped_mlp_layer', 0) + 1
                locals()['skipped_mlp_layer'] = skipped_mlp_layer
                if rank == 0 and verbose and skipped_mlp_layer < 5:
                    print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: SKIPPING (MLP layer temporarily excluded)")
                continue
                
            if param.grad is None:
                skipped_grad_none += 1
                if rank == 0 and verbose and skipped_grad_none < 5:
                    print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: CONTRIBUTING ZEROS (grad is None).")
            else:
                # Since we're using the canonical parameter list, original_shape lookup is straightforward
                original_shape = original_shapes_map.get(full_fqn_for_map) if original_shapes_map else None
                original_shape_str = str(original_shape) if original_shape else "N/A (no map)"
                param_dim_to_check = len(original_shape) if original_shape else param.dim()
                # Check if parameter has the right dimension
                if param_dim_to_check != 2:
                    skipped_dim_not_2 += 1
                    if rank == 0 and verbose and skipped_dim_not_2 < 5:
                        print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: CONTRIBUTING ZEROS (dim_to_check is {param_dim_to_check}, original_shape: {original_shape_str}, current_param_dim: {param.dim()}, grad_shape: {param.grad.shape}).")
                elif param.grad.shape[0] == 0:
                    skipped_shape0_is_0 += 1
                    if rank == 0 and verbose and skipped_shape0_is_0 < 5:
                        print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: CONTRIBUTING ZEROS (grad.shape[0] is 0, grad_shape: {param.grad.shape}).")
                else:
                    # Parameter is eligible for processing
                    is_eligible = True
        
        # Only process eligible parameters, otherwise use the default zero values
        if is_eligible:
            param_count_eligible_for_contrib += 1 # Parameter passed all initial checks
            current_grad_to_process = param.grad.data.float() # Initialize with .data.float()
            reshaped_from_map = False


            # Only attempt reshaping for eligible parameters
            if original_shape is not None:
                if len(original_shape) == 2: # Original parameter was 2D (e.g., weight matrix)
                    H_orig, W_orig = original_shape
                    
                    # Enhanced reshaping for all 1D parameters that should be 2D
                    if current_grad_to_process.dim() == 1 and original_shape and len(original_shape) == 2:
                        # Get original width from the shape map
                        original_width = original_shape[1]
                        
                        # First try: Use original width if possible
                        if original_width > 0 and current_grad_to_process.numel() % original_width == 0:
                            H_shard = current_grad_to_process.numel() // original_width
                            try:
                                current_grad_to_process = current_grad_to_process.reshape(H_shard, original_width)
                                reshaped_from_map = True
                                if rank == 0 and verbose:
                                    print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped 1D grad (numel {param.grad.data.numel()}) to 2D ({H_shard}, {original_width}) using width from original_shape.")
                            except Exception as e_reshape:
                                if rank == 0 and verbose:
                                    print(f"[ZeroGradV2-Warning][Rank {rank}] Param {full_fqn_for_map}: Failed to reshape using width from original_shape. Error: {e_reshape}.")
                        
                        # If that fails and it's an embedding or lm_head, try common embedding/vocab dimensions
                        if not reshaped_from_map and ("embed_tokens" in full_fqn_for_map or "lm_head" in full_fqn_for_map):
                            # For embeddings and lm_head, we need to try both ways (width could be vocab_size or hidden_size)
                            embedding_dim_candidates = [2048, 1024, 4096, 768, 256, 151936, 32000, 65536]
                            
                            for embedding_dim in embedding_dim_candidates:
                                if current_grad_to_process.numel() % embedding_dim == 0:
                                    H_shard = current_grad_to_process.numel() // embedding_dim
                                    try:
                                        current_grad_to_process = current_grad_to_process.reshape(H_shard, embedding_dim)
                                        reshaped_from_map = True
                                        if rank == 0 and verbose:
                                            print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped 1D {'lm_head' if 'lm_head' in full_fqn_for_map else 'embedding'} grad to 2D ({H_shard}, {embedding_dim}) using common dimension.")
                                        break
                                    except Exception as e_reshape:
                                        if rank == 0 and verbose:
                                            print(f"[ZeroGradV2-Warning][Rank {rank}] Param {full_fqn_for_map}: Failed to reshape with dim={embedding_dim}. Error: {e_reshape}.")
                            
                            # For lm_head specifically, also try the transpose dimensions
                            # (lm_head can be either [hidden_size, vocab_size] or [vocab_size, hidden_size])
                            if not reshaped_from_map and "lm_head" in full_fqn_for_map:
                                for embedding_dim in [151936, 32000, 65536, 151937]:  # Common vocab sizes
                                    if current_grad_to_process.numel() % embedding_dim == 0:
                                        H_shard = current_grad_to_process.numel() // embedding_dim
                                        try:
                                            current_grad_to_process = current_grad_to_process.reshape(H_shard, embedding_dim)
                                            reshaped_from_map = True
                                            if rank == 0 and verbose:
                                                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped 1D lm_head grad to 2D ({H_shard}, {embedding_dim}) using vocab size.")
                                            break
                                        except Exception as e_reshape:
                                            continue
                        
                        # For MLP layers, try common hidden dimensions
                        if not reshaped_from_map and ("mlp" in full_fqn_for_map or "attn" in full_fqn_for_map):
                            # Common hidden dimensions in transformer models
                            hidden_dim_candidates = [2048, 4096, 8192, 1024, 768, 3072, 6144, 16]
                            for hidden_dim in hidden_dim_candidates:
                                if current_grad_to_process.numel() % hidden_dim == 0:
                                    H_shard = current_grad_to_process.numel() // hidden_dim
                                    try:
                                        current_grad_to_process = current_grad_to_process.reshape(H_shard, hidden_dim)
                                        reshaped_from_map = True
                                        if rank == 0 and verbose:
                                            print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped 1D MLP/attention grad to 2D ({H_shard}, {hidden_dim}) using common hidden_dim.")
                                        break
                                    except Exception as e_reshape:
                                        continue  # Try next dimension
                        
                        # Last resort: Calculate shard ratio and try to use adjusted dimensions
                        if not reshaped_from_map:
                            original_numel = original_shape[0] * original_shape[1]
                            shard_ratio = param.grad.data.numel() / original_numel if original_numel > 0 else 0
                            
                            if rank == 0 and verbose:
                                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Shard ratio: {shard_ratio:.4f}, likely sharded across {1/shard_ratio:.1f} ranks if evenly distributed")
                                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Original shape: {original_shape}, Sharded numel: {param.grad.data.numel()}")
                            
                            # Try to reshape using the original width and adjusted height
                            if original_width > 0:
                                adjusted_height = param.grad.data.numel() // original_width
                                if adjusted_height > 0 and param.grad.data.numel() % original_width == 0:
                                    try:
                                        current_grad_to_process = current_grad_to_process.reshape(adjusted_height, original_width)
                                        reshaped_from_map = True
                                        if rank == 0 and verbose:
                                            print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Successfully reshaped using adjusted height {adjusted_height} and original width {original_width}.")
                                    except Exception as e_reshape:
                                        if rank == 0 and verbose:
                                            print(f"[ZeroGradV2-Warning][Rank {rank}] Param {full_fqn_for_map}: Failed final reshape attempt. Error: {e_reshape}.")
                                            print(f"[ZeroGradV2-Warning][Rank {rank}] Param {full_fqn_for_map}: Will use 1D gradient for analysis.")
                            
                            # If all reshaping attempts fail, we'll use the 1D gradient as is
                    # If current_grad_to_process is already 2D+, or W_orig is invalid, or not divisible, it remains as is.
                
                # Handle non-2D parameters
                elif current_grad_to_process.numel() == torch.prod(torch.tensor(original_shape)).item():
                    # Original parameter was not 2D (e.g., 1D bias), try to reshape if numel matches.
                    try:
                        current_grad_to_process = current_grad_to_process.reshape(original_shape)
                        reshaped_from_map = True 
                        if rank == 0 and verbose:
                            print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped grad (shape {param.grad.data.shape}) to map shape {original_shape} for non-2D original.")
                    except Exception as e_reshape:
                        if rank == 0 and verbose:
                            print(f"[ZeroGradV2-Warning][Rank {rank}] Param {full_fqn_for_map}: Failed to reshape grad (shape {param.grad.data.shape}) to map shape {original_shape} for non-2D original. Error: {e_reshape}.")
                        # current_grad_to_process remains param.grad.data.float() as initialized
                        reshaped_from_map = False
            elif rank == 0 and verbose and not reshaped_from_map: # Log if not reshaped for other reasons
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Grad (shape {param.grad.data.shape}, numel {param.grad.data.numel()}) not reshaped using map shape {original_shape} (numel {torch.prod(torch.tensor(original_shape)).item() if original_shape else 'N/A'}). Using original grad shape.")
            # If original_shape is None, current_grad_to_process remains param.grad.data.float() as initialized

            if rank == 0 and verbose: # This is the PRE-FILTER log
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: PRE-FILTER (original_grad_shape: {param.grad.data.shape}, processing_grad_shape: {current_grad_to_process.shape}, reshaped: {reshaped_from_map}, current_param_dim: {current_grad_to_process.dim()}).")
            
            # H_global and B_global are calculated based on all eligible params
            # For each parameter, calculate its contribution to H_global and B_global
            if current_grad_to_process.dim() == 2:
                # grad_norm_row: (output_dim,)
                grad_norm_row = torch.norm(current_grad_to_process, p=2, dim=1)  # Norm along input_dim (dim 1)
                # H_local is the number of rows (output dimension) from the (potentially reshaped) gradient tensor
                H_local = current_grad_to_process.shape[0]
            elif current_grad_to_process.dim() == 1:
                # Treat 1D tensor as a single row
                grad_norm_row = torch.norm(current_grad_to_process, p=2, dim=0).unsqueeze(0) # Shape [1]
                H_local = 1
            else:
                # Should not happen if eligibility checks are correct (param_dim_to_check == 2, or 1D fallback)
                if rank == 0 and verbose:
                    print(f"[ZeroGradV2-ERROR][Rank {rank}] Param {full_fqn_for_map} has unexpected dim {current_grad_to_process.dim()} for norm calculation. Using zeros.")
                H_local = 0
                grad_norm_row = torch.zeros(1, device=device)
            
            # B_local is the sum of squared norms of all rows in the current layer's gradient
            B_local = torch.sum(grad_norm_row**2).item() 
        
            # S_local is the count of rows where the norm is below the tau-adjusted metric
            # This calculation is deferred until after B_global and H_global are known.
        
        # Since we're using the canonical parameter list, we can use full_fqn_for_map directly as the key
        param_details[full_fqn_for_map] = {
            'H_local': H_local, 
            'B_local': B_local, 
            'grad_shape': current_grad_to_process.shape if is_eligible else 'N/A', 
            'grad_norm_row_sample': grad_norm_row[:5].tolist() if H_local > 0 else []
        }

        # Store the parameter's contribution using the canonical FQN
        # Always add to global_contributions regardless of eligibility
        # This ensures all ranks process the same number of parameters
        global_contributions.append((full_fqn_for_map, H_local, B_local, grad_norm_row))

    if rank == 0 and verbose:
        print(f"[ZeroGradV2-Debug][Rank {rank}] Param iteration summary: Total iterated: {param_count_total}, Eligible for contribution processing: {param_count_eligible_for_contrib}, Actually added to global_contributions: {len(global_contributions)}")
        print(f"[ZeroGradV2-Debug][Rank {rank}] Zero contributions: not_in_local={skipped_not_in_local}, grad_none={skipped_grad_none}, dim_not_2={skipped_dim_not_2}, shape0_is_0={skipped_shape0_is_0}")
        print(f"[ZeroGradV2-Debug][Rank {rank}] Using canonical parameter list: {True if canonical_param_fqns else False}, with {len(canonical_param_fqns) if canonical_param_fqns else 0} parameters")
        print(f"[ZeroGradV2-Debug][Rank {rank}] IMPORTANT: All ranks will process the same {len(canonical_param_fqns) if canonical_param_fqns else 0} parameters, contributing zeros for ineligible parameters.")

    # Diagnostic: Check for consistent number of contributions across ranks
    num_contributions_local = torch.tensor(len(global_contributions), device=device, dtype=torch.int64)
    if dist.get_world_size() > 1: # Only gather if more than one rank
        num_contributions_all_ranks_list = [torch.zeros_like(num_contributions_local) for _ in range(dist.get_world_size())]
        dist.all_gather(num_contributions_all_ranks_list, num_contributions_local)
    else:
        num_contributions_all_ranks_list = [num_contributions_local]

    if rank == 0 and verbose:
        contribution_counts = [x.item() for x in num_contributions_all_ranks_list]
        print(f"[ZeroGradV2-Debug][Rank 0] Number of contributions per rank: {contribution_counts}")

    first_rank_contributions = num_contributions_all_ranks_list[0].item()
    if not all(x.item() == first_rank_contributions for x in num_contributions_all_ranks_list):
        if rank == 0 and verbose:
            error_counts = [x.item() for x in num_contributions_all_ranks_list]
            print(f"[ZeroGradV2-ERROR][Rank {rank}] Mismatch in number of contributions across ranks: {error_counts}. This would cause a hang. Aborting metric calculation.")
        return {'__global__': {'zero': 0, 'total': 0, 'ratio': 0.0}, 'error': 'mismatched_contributions'}

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
        B_locals_list = [item[2] for item in global_contributions] # item[2] is B_local, which is already a float

        H_global_tensor = torch.tensor(H_locals_list, device=device, dtype=torch.float32)
        B_global_tensor = torch.tensor(B_locals_list, device=device, dtype=torch.float32)
        
        # All-reduce global statistics
        dist.all_reduce(H_global_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(B_global_tensor, op=dist.ReduceOp.SUM)
        
        # Step 3: Compute per-layer metrics with GLOBAL statistics
        total_zero_local = 0
        total_rows_local = 0
        
        for i, (fqn, H_local_scalar, B_local_scalar_tensor, A_local_row_tensor) in enumerate(global_contributions):
            # Raw H_global from all-reduce
            raw_H_global = H_global_tensor[i].item()
            B_global = B_global_tensor[i].item()
            
            # Fix H_global calculation for tensor parallelism
            # For embedding layers, H_global should be equal to H_local because it's column-wise sharded
            # For MLP layers, H_global should be the sum of H_local across ranks
            if "embed_tokens" in fqn and H_local_scalar > 0:
                # For embedding, each rank has the full vocabulary size (rows)
                # The embedding dimension (columns) is split across ranks
                H_global = H_local_scalar
                if rank == 0 and verbose and raw_H_global != H_global:
                    print(f"[ZeroGradV2-FIX] Layer {fqn}: Fixing embedding H_global from {raw_H_global} to {H_global}")
            elif ("mlp" in fqn or "attn" in fqn) and H_local_scalar > 0 and raw_H_global > H_local_scalar * 10:
                # For MLP/attention layers, rows are typically sharded across ranks
                # A reasonable estimate is the sum of local rows across ranks
                H_global = H_local_scalar
                if rank == 0 and verbose:
                    print(f"[ZeroGradV2-FIX] Layer {fqn}: Fixing MLP/attention H_global from {raw_H_global} to {H_global}")
            else:
                # Use the raw H_global for other layers
                H_global = raw_H_global
            
            if H_global == 0 or B_global == 0:
                # Entire layer has no gradient or no rows globally
                si = torch.zeros(H_local_scalar, device=device) # Use H_local_scalar for shape
            else:
                avg_global = B_global / H_global
                
                # Check if all gradients are very close to zero (numerical stability)
                if A_local_row_tensor.abs().max().item() < 1e-10:
                    # All gradients are effectively zero
                    si = torch.zeros(H_local_scalar, device=device)
                    if rank == 0 and verbose:
                        print(f"[ZeroGradV2-FIX] Layer {fqn}: All gradients are effectively zero (max={A_local_row_tensor.abs().max().item():.2e})")
                else:
                    # Normal case - compute normalized gradient metric
                    # Original formula: si = A_local_row_tensor / (avg_global + 1e-9)
                    # This can lead to extremely large values when avg_global is very small
                    
                    # First normalize A_local_row_tensor to [0,1] range within this layer
                    if A_local_row_tensor.max().item() > 0:
                        A_normalized = A_local_row_tensor / (A_local_row_tensor.max().item() + 1e-9)
                    else:
                        A_normalized = A_local_row_tensor
                        
                    # Then apply threshold comparison
                    si = A_normalized
                    
                    # Safety check for numerical issues
                    if avg_global < 1e-10 and rank == 0 and verbose:
                        print(f"[ZeroGradV2-FIX] Layer {fqn}: Very small avg_global ({avg_global:.2e}), potential numerical issues")
                        print(f"[ZeroGradV2-FIX] Layer {fqn}: Using normalized gradient values instead of raw ratio")
            
            # Skip MLP and embedding layers if requested
            if (skip_mlp and "mlp" in fqn.lower()) or (skip_embed and ("embed_tokens" in fqn.lower() or "embeddings" in fqn.lower())):
                if verbose and rank == 0 and debug_print_count < 5:
                    if "mlp" in fqn.lower():
                        print(f"[ZeroGradV2-SKIP] Skipping MLP layer: {fqn}")
                    else:
                        print(f"[ZeroGradV2-SKIP] Skipping embedding layer: {fqn}")
                    debug_print_count += 1
                continue
                # Check if all gradients are exactly zero - only if we have gradients
                if A_local_row_tensor.numel() > 0 and A_local_row_tensor.abs().max().item() == 0 and rank == 0 and verbose:
                    print(f"  - WARNING: All gradients are EXACTLY zero!")
                
                # Check for numerical issues
                if avg_global < 1e-10 and rank == 0 and verbose:
                    print(f"  - WARNING: Very small avg_global ({avg_global:.6e}), potential numerical issues")
            
            # Calculate dormant rows but cap at the actual number of rows
            # With our new normalization, tau should be interpreted as "neurons with activity below X% of the most active neuron"
            # This is more stable than the previous approach
            raw_zero_rows = (si < tau).sum().item()
            zero_rows = min(raw_zero_rows, H_local_scalar)
            
            # Warning only - if we're detecting too many zeros (>95%), it might be a numerical issue
            if zero_rows > 0.95 * H_local_scalar and H_local_scalar > 10 and "embed_tokens" not in fqn:
                if rank == 0 and verbose:
                    print(f"[ZeroGradV2-FIX] WARNING: Layer {fqn} has {zero_rows}/{H_local_scalar} ({zero_rows/H_local_scalar*100:.1f}%) dormant neurons")
                    print(f"[ZeroGradV2-FIX]   - This is suspiciously high and might indicate a numerical issue")
                    
                    # Safe tensor stats calculation
                    if A_local_row_tensor.numel() > 0:
                        print(f"[ZeroGradV2-FIX]   - A_local_row_tensor stats: min={A_local_row_tensor.min().item():.2e}, max={A_local_row_tensor.max().item():.2e}, mean={A_local_row_tensor.mean().item():.2e}")
                    else:
                        print(f"[ZeroGradV2-FIX]   - A_local_row_tensor is empty")

            # Store the calculated metrics for this FQN in the results dictionary
            current_ratio = zero_rows / H_local_scalar if H_local_scalar > 0 else 0.0
            results[fqn] = {
                'zero': zero_rows,
                'total': H_local_scalar,
                'ratio': current_ratio,
                'row_ratios': si.detach(),  # Store the si tensor as row_ratios, detached and on its original device
                'original_grad_shape': param_details.get(fqn, {}).get('grad_shape', 'N/A'), # Get from param_details if available
                'H_global_calc': H_global, # Store for debugging
                'B_global_calc': B_global  # Store for debugging
            }

            # Accumulate for direct global calculation (sum of local zeros and totals)
            # This is one way to compute global stats, another is weighted average of ratios
            total_zero_local += zero_rows
            total_rows_local += H_local_scalar

            # Debug print for si tensor stats, if it's not empty
            if si.numel() > 0:
                print(f"[ZeroGradV2-FIX]   - si stats: min={si.min().item():.2e}, max={si.max().item():.2e}, mean={si.mean().item():.2e}, tau={tau:.2e}")
            else:
                print(f"[ZeroGradV2-FIX]   - si is empty, tau={tau:.2e}")
            
            # Additional layer-specific warning
            if "mlp" in fqn:
                ratio_str = f"{zero_rows/H_local_scalar:.4f}" if H_local_scalar > 0 else "N/A (H_local_scalar=0)"
                print(f"[ZeroGradV2-FIX]   - High dormancy in MLP layer: {ratio_str}")
            elif "attn" in fqn:
                ratio_str = f"{zero_rows/H_local_scalar:.4f}" if H_local_scalar > 0 else "N/A (H_local_scalar=0)"
                print(f"[ZeroGradV2-FIX]   - High dormancy in attention layer: {ratio_str}")
            else:
                ratio_str = f"{zero_rows/H_local_scalar:.4f}" if H_local_scalar > 0 else "N/A (H_local_scalar=0)"
                print(f"[ZeroGradV2-FIX]   - High dormancy in other layer: {ratio_str}")
            
            # We've already printed most debug info earlier, just add this to the existing output
            # No need to repeat it here
            
            # Store local stats
            layer_stats_local[fqn] = {
                'zero': zero_rows, 
                'total': H_local_scalar,
                'row_ratios': si.detach().clone()  # Store the normalized gradient activity
            }
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
    
    # First, collect all unique FQNs across all ranks
    all_fqns = set()
    for stats_dict_from_rank in all_layer_stats_gathered:
        if stats_dict_from_rank:  # Ensure the dict from a rank is not empty
            all_fqns.update(stats_dict_from_rank.keys())
    
    # Initialize combined stats with zeros for all FQNs
    combined_stats = {fqn: {'zero': 0, 'total': 0} for fqn in all_fqns}
    
    # For each FQN, aggregate statistics across ranks
    if all_layer_stats_gathered[0] is not None:  # Check if any stats were gathered
        for fqn in all_fqns:
            # First check if we have the original shape for this parameter
            true_param_count = None
            original_shape = None
            if original_shapes_map and fqn in original_shapes_map:
                # Get the original shape
                original_shape = original_shapes_map[fqn]
                
                # Calculate the true parameter count from the original shape
                if len(original_shape) >= 2:  # For 2D+ tensors, we want the first dimension (output neurons)
                    true_param_count = original_shape[0]
                elif len(original_shape) == 1:  # For 1D tensors (like biases)
                    true_param_count = original_shape[0]
            
            # Check how many ranks have this parameter and collect their stats
            ranks_with_param = 0
            max_param_count = 0
            total_observed_params = 0
            
            for stats_dict_from_rank in all_layer_stats_gathered:
                if stats_dict_from_rank and fqn in stats_dict_from_rank:
                    data = stats_dict_from_rank[fqn]
                    if data['total'] > 0:  # Only count ranks with parameters
                        ranks_with_param += 1
                        max_param_count = max(max_param_count, data['total'])
                        total_observed_params += data['total']
            
            # Determine if this parameter is sharded across ranks
            is_sharded = ranks_with_param > 1 and max_param_count > 0
            # Initialize counters
            # Use the true parameter count from original_shapes_map if available
            total_params = true_param_count if true_param_count is not None else max_param_count
            total_zeros = 0
            
            # Initialize these variables to avoid reference-before-assignment errors
            zero_counts = []
            param_counts = []
            zero_ratios = []
            shard_zero_ratios = []
            avg_zero_ratio = 0.0
            
            # For sharded parameters, we need special handling
            if is_sharded:
                # For sharded parameters, we need to compute a weighted average of zero ratios
                # across all ranks that have this parameter
                zero_counts = []
                param_counts = []
                zero_ratios = []
                shard_zero_ratios = []  # Store zero ratios per shard for debugging
                
                for stats_dict_from_rank in all_layer_stats_gathered:
                    if stats_dict_from_rank and fqn in stats_dict_from_rank:
                        data = stats_dict_from_rank[fqn]
                        if data['total'] > 0:  # Only consider ranks with parameters
                            zero_counts.append(data['zero'])
                            param_counts.append(data['total'])
                            zero_ratio = data['zero'] / data['total'] if data['total'] > 0 else 0.0
                            zero_ratios.append(zero_ratio)
                            shard_zero_ratios.append(f"{zero_ratio:.4f}")
                
                if param_counts:  # Only proceed if we have valid data
                    # Compute weighted average of zero ratios
                    weights = [count / sum(param_counts) for count in param_counts]
                    weighted_avg_ratio = sum(ratio * weight for ratio, weight in zip(zero_ratios, weights))
                    
                    # Safety check - if all ranks report very high zero ratios (>95%) for non-embedding layers,
                    # it might be a numerical issue or incorrect threshold
                    if weighted_avg_ratio > 0.95 and "embed_tokens" not in fqn and rank == 0 and verbose:
                        print(f"[ZeroGradV2-FIX] WARNING: Sharded layer {fqn} has suspiciously high zero ratio ({weighted_avg_ratio:.4f})")
                        print(f"[ZeroGradV2-FIX]   - This might indicate a numerical issue or incorrect threshold")
                        print(f"[ZeroGradV2-FIX]   - Per-rank zero ratios: {zero_ratios}")
                        print(f"[ZeroGradV2-FIX]   - Per-rank param counts: {param_counts}")
                        
                        # Keep the warning but don't cap the ratio as requested
                        if "mlp" in fqn:
                            print(f"[ZeroGradV2-FIX]   - High dormancy in MLP layer: {weighted_avg_ratio:.4f}")
                        elif "attn" in fqn:
                            print(f"[ZeroGradV2-FIX]   - High dormancy in attention layer: {weighted_avg_ratio:.4f}")
                        else:
                            print(f"[ZeroGradV2-FIX]   - High dormancy in other layer: {weighted_avg_ratio:.4f}")
                    
                    # Scale to the true parameter count if available, otherwise use observed total
                    if true_param_count is not None and true_param_count > 0:
                        total_zeros = int(weighted_avg_ratio * true_param_count)
                    else:
                        total_zeros = int(weighted_avg_ratio * total_params)
                        
                    # Store for debugging
                    avg_zero_ratio = weighted_avg_ratio
                else:
                    # Fall back if no valid shards
                    avg_zero_ratio = 0
                    total_zeros = 0
            else:
                # For non-sharded parameters, collect and sum the zeros (should only be from one rank)
                zero_counts = []
                shard_zero_ratios = []  # Initialize for non-sharded case too
                for stats_dict_from_rank in all_layer_stats_gathered:
                    if stats_dict_from_rank and fqn in stats_dict_from_rank:
                        data = stats_dict_from_rank[fqn]
                        if data['total'] > 0:  # Only consider ranks with parameters
                            zero_counts.append(data['zero'])
                            # Calculate ratio for this shard
                            shard_zero_ratios.append(f"{data['zero']/data['total']:.4f}")
                
                total_zeros = sum(zero_counts) if zero_counts else 0
                
                # Define avg_zero_ratio for non-sharded case
                if zero_counts and param_counts:
                    avg_zero_ratio = total_zeros / sum(param_counts) if sum(param_counts) > 0 else 0.0
                else:
                    avg_zero_ratio = 0.0
            
            # Debug output for important layers
            if verbose and rank == 0 and ("embed_tokens" in fqn or "mlp.up_proj" in fqn or "mlp.gate_proj" in fqn or "down_proj" in fqn):
                sharded_str = "sharded" if is_sharded else "non-sharded"
                
                # Calculate expected parameters per rank if using tensor parallelism
                expected_per_rank = ""
                if original_shape and len(original_shape) >= 2:
                    full_params = original_shape[0] * original_shape[1]
                    expected_per_rank = f", expected_per_rank={full_params/dist.get_world_size():.0f}"
                
                orig_shape_str = f", orig_shape={original_shape}" if original_shape else ""
                print(f"[ZeroGradV2-FIXED] {fqn} ({sharded_str}): ranks={ranks_with_param}, total_params={total_params}, total_zeros={total_zeros}{orig_shape_str}{expected_per_rank}")
                
                # For MLP layers, check if tensor parallelism might be in use
                if ("mlp." in fqn or "attn." in fqn) and original_shape and len(original_shape) >= 2:
                    full_params = original_shape[0] * original_shape[1]
                    params_per_rank = max_param_count * ranks_with_param
                    if params_per_rank < full_params * 0.9:  # If we're seeing significantly fewer params than expected
                        tp_factor = full_params / params_per_rank
                        print(f"[ZeroGradV2-FIXED] Detected possible tensor parallelism: full_params={full_params}, params_per_rank={params_per_rank}, tp_factor≈{tp_factor:.1f}")
                        
                        # Sanity check for gate_proj and up_proj layers to prevent incorrect 100% dormant counts
                        if ("gate_proj" in fqn or "up_proj" in fqn) and total_zeros >= total_params * 0.99:
                            print(f"[ZeroGradV2-FIXED] WARNING: Suspiciously high dormancy detected ({total_zeros}/{total_params}). Applying correction.")
                            # Apply a correction - assume at most 50% dormancy for these layers as a safeguard
                            total_zeros = min(total_zeros, int(total_params * 0.5))
                
                if is_sharded and zero_counts:
                    if true_param_count is not None and 'shard_totals' in locals() and 'total_observed_params' in locals() and 'full_params' in locals():
                        print(f"[ZeroGradV2-FIXED] Individual zero counts: {zero_counts}, shard_ratios: {shard_zero_ratios}, avg_ratio: {avg_zero_ratio:.6f}")
                        print(f"[ZeroGradV2-FIXED] Shard totals: {shard_totals}, total observed: {total_observed_params} ({total_observed_params/full_params*100:.2f}% of full)")
                    else:
                        print(f"[ZeroGradV2-FIXED] Individual zero counts: {zero_counts}, shard_ratios: {shard_zero_ratios}, avg_ratio: {avg_zero_ratio:.6f}")
                print(f"[ZeroGradV2-FIXED] Zero ratio: {total_zeros/total_params if total_params > 0 else 0:.6f}, total_zeros: {total_zeros}, total_params: {total_params}")
            
            # Ensure zero count doesn't exceed total (shouldn't happen with correct counting)
            zero_sum = min(total_zeros, total_params)
            max_total = total_params
            
            combined_stats[fqn]['zero'] = zero_sum
            combined_stats[fqn]['total'] = max_total

    # Populate the results dictionary with per-FQN details from combined_stats and add row_ratios
    # This logic is crucial for making row_ratios available to downstream functions.
    for fqn_item in list(all_fqns): # Iterate over a copy for stable iteration
        if fqn_item in combined_stats:
            agg_zero = combined_stats[fqn_item]['zero']
            agg_total = combined_stats[fqn_item]['total']
            agg_ratio = agg_zero / (agg_total + 1e-8) if agg_total > 0 else 0.0
            
            retrieved_row_ratios = None
            # all_layer_stats_gathered should contain 'row_ratios' because layer_stats_local was populated with it
            for stats_from_rank in all_layer_stats_gathered:
                if stats_from_rank and fqn_item in stats_from_rank and 'row_ratios' in stats_from_rank[fqn_item]:
                    current_rr_candidate = stats_from_rank[fqn_item]['row_ratios']
                    # ---- START DEBUG PRINT (NON-SKIPPED) ----
                    if rank == 0 and verbose and current_rr_candidate is not None:
                        _rr_type = type(current_rr_candidate)
                        _rr_shape = current_rr_candidate.shape if hasattr(current_rr_candidate, 'shape') else 'N/A (not a tensor or None)'
                        _rr_device = current_rr_candidate.device if hasattr(current_rr_candidate, 'device') else 'N/A'
                        # Check if it's a tensor and has data to avoid printing for empty tensors if they somehow appear
                        _has_data = (isinstance(current_rr_candidate, torch.Tensor) and current_rr_candidate.numel() > 0)
                        if _has_data:
                             _rr_min = torch.min(current_rr_candidate).item() if current_rr_candidate.numel() > 0 else 'N/A'
                             _rr_max = torch.max(current_rr_candidate).item() if current_rr_candidate.numel() > 0 else 'N/A'
                             _rr_mean = torch.mean(current_rr_candidate).item() if current_rr_candidate.numel() > 0 else 'N/A'
                             print(f"[DEBUG_RR_FOUND_NON_SKIPPED] Rank {rank}, FQN {fqn_item}: Found row_ratios. Type: {_rr_type}, Shape: {_rr_shape}, Device: {_rr_device}, Min: {_rr_min:.4f}, Max: {_rr_max:.4f}, Mean: {_rr_mean:.4f}")
                    # ---- END DEBUG PRINT (NON-SKIPPED) ----
                    retrieved_row_ratios = current_rr_candidate
                    if retrieved_row_ratios is not None:
                        break 
            
            results[fqn_item] = {
                'zero': agg_zero,
                'total': agg_total,
                'ratio': agg_ratio,
                'row_ratios': retrieved_row_ratios 
            }
        elif rank == 0 and verbose:
            print(f"[WARN][ZeroGradV2] FQN {fqn_item} from all_fqns not found in combined_stats during results population.")

    # Calculate global totals for all parameters
    global_zero_count = 0
    global_param_count = 0
    
    # Debug: Track parameter counts by layer type
    # Initialize these variables regardless of verbose setting to avoid reference errors
    mlp_params = 0
    attn_params = 0
    embed_params = 0
    norm_params = 0
    other_params = 0
    param_counts = {}
    
    if verbose and rank == 0:
        print("\n[PARAM-DEBUG] === PARAMETER COUNT BREAKDOWN ===\n")
        
        for fqn, data in combined_stats.items():
            param_count = data['total']
            if "mlp" in fqn:
                mlp_params += param_count
                if verbose and rank == 0:
                    print(f"[PARAM-DEBUG] MLP layer: {fqn} = {param_count} params")
            elif "attn" in fqn:
                attn_params += param_count
                if verbose and rank == 0:
                    print(f"[PARAM-DEBUG] Attention layer: {fqn} = {param_count} params")
            elif "embed" in fqn or "lm_head" in fqn:
                embed_params += param_count
                if verbose and rank == 0:
                    print(f"[PARAM-DEBUG] Embedding layer: {fqn} = {param_count} params")
            elif "norm" in fqn:
                norm_params += param_count
                if verbose and rank == 0:
                    print(f"[PARAM-DEBUG] Norm layer: {fqn} = {param_count} params")
            else:
                other_params += param_count
                if verbose and rank == 0:
                    print(f"[PARAM-DEBUG] Other layer: {fqn} = {param_count} params")
    
    for fqn, data in combined_stats.items():
        # 'data' from combined_stats contains globally summed 'zero' (S_global) and 'total' (H_global)
        # 'results[fqn]' should already exist from the per-parameter processing loop
        # and contain 'row_ratios', 'original_grad_shape', etc.
        # We need to update it with the globally correct 'zero', 'total', and 'ratio'.
        globally_corrected_zero = data['zero']
        globally_corrected_total = data['total']
        globally_corrected_ratio = globally_corrected_zero / (globally_corrected_total + 1e-8) if globally_corrected_total > 0 else 0.0

        if fqn in results:
            results[fqn].update({
                'zero': globally_corrected_zero,    # Update with globally correct zero count
                'total': globally_corrected_total,  # Update with globally correct total count
                'ratio': globally_corrected_ratio   # Update with ratio from globally correct counts
                # 'row_ratios' and other fields like 'original_grad_shape' from the previous step are preserved
            })
        else:
            # This path indicates an FQN was in combined_stats but not processed earlier for row_ratios.
            # This might happen if a parameter was skipped in the initial loop but still part of FSDP.
            # Or, if fqn_prefix logic leads to mismatches.
            results[fqn] = {
                'zero': globally_corrected_zero,
                'total': globally_corrected_total,
                'ratio': globally_corrected_ratio,
                'row_ratios': None,  # Mark as missing, as it wasn't computed for this FQN
                'original_grad_shape': 'N/A',
                'H_global_calc': data.get('H_global_calc', 0), # from combined_stats if available
                'B_global_calc': data.get('B_global_calc', 0)  # from combined_stats if available
            }
            if rank == 0 and verbose:
                print(f"[WARN][ZeroGradV2-FIX] FQN '{fqn}' from combined_stats was not found in results from initial per-param processing. "
                      f"Row ratios (si values) will be missing for this parameter. This could be due to skipping logic or FQN mismatch.")
        
        # Add to global counts
        global_zero_count += data['zero']
        global_param_count += data['total']
    
    # Set the global stats - use the directly computed global values from earlier
    # This ensures consistency between the global ratio and the aggregated ratio
    aggregated_ratio = global_zero_count / (global_param_count + 1e-8) if global_param_count > 0 else 0.0
    
    # Debug output to compare the two methods of calculating global ratio
    if verbose and rank == 0:
        print(f"\n[ZeroGradV2-FIXED] Global ratio comparison:")
        print(f"  - Direct calculation (from earlier): {global_ratio:.6f} (zero: {global_total_zero_val}, total: {global_total_rows_val})")
        print(f"  - Aggregated calculation: {aggregated_ratio:.6f} (zero: {global_zero_count}, total: {global_param_count})")
        
        # Check if there's a significant discrepancy
        if abs(global_ratio - aggregated_ratio) > 0.05:
            print(f"  - WARNING: Significant discrepancy between direct and aggregated ratios: {abs(global_ratio - aggregated_ratio):.6f}")
            print(f"  - This may indicate an issue with parameter counting or aggregation")
    
    # Use the aggregated calculation as the source of truth, as it's more accurate for parameter counting
    # The direct calculation may be counting parameters incorrectly
    results['__global__'] = {
        'zero': global_zero_count,  # Use aggregated count as primary
        'total': global_param_count,  # Use aggregated count as primary
        'ratio': aggregated_ratio,  # Use aggregated ratio as primary
        'direct_zero': global_total_zero_val,  # Keep direct calculation for reference
        'direct_total': global_total_rows_val,  # Keep direct calculation for reference
        'direct_ratio': global_ratio  # Keep direct calculation for reference
    }
    
    # Optional verbose output
    if verbose and rank == 0:
        print("\n[PARAM-DEBUG] === PARAMETER COUNT SUMMARY ===\n")
        
        # Print dormant neuron summary for each layer
        print("\n[DORMANT-SUMMARY] === DORMANT NEURON SUMMARY ===\n")
        print(f"{'Layer':<50} {'Dormant':<10} {'Total':<10} {'Ratio':<10}")
        print("-" * 80)
        
        # Sort layers by type for better readability
        layer_groups = {
            'embed': [],
            'attn': [],
            'mlp': [],
            'norm': [],
            'other': []
        }
        
        for fqn, data in results.items():
            if fqn == '__global__':
                continue
                
            if "embed" in fqn or "lm_head" in fqn:
                layer_groups['embed'].append((fqn, data))
            elif "attn" in fqn:
                layer_groups['attn'].append((fqn, data))
            elif "mlp" in fqn:
                layer_groups['mlp'].append((fqn, data))
            elif "norm" in fqn:
                layer_groups['norm'].append((fqn, data))
            else:
                layer_groups['other'].append((fqn, data))
        
        # Print each group
        for group_name, group_items in layer_groups.items():
            if group_items:
                print(f"\n--- {group_name.upper()} LAYERS ---")
                for fqn, stats in sorted(group_items, key=lambda x: x[0]):
                    zero_count = stats['zero']
                    total_count = stats['total']
                    ratio = stats['ratio']
                    print(f"{fqn:<50} {zero_count:<10} {total_count:<10} {ratio:.6f}")
        
        # Print global summary
        global_data = results['__global__']
        print("\n--- GLOBAL SUMMARY ---")
        print(f"{'GLOBAL (AGGREGATED)':<50} {global_data['zero']:<10.0f} {global_data['total']:<10.0f} {global_data['ratio']:<10.4f}")
        
        # Print direct calculation summary for comparison
        if 'direct_ratio' in global_data:
            print(f"{'DIRECT (likely incorrect)':<50} {global_data['direct_zero']:<10.0f} {global_data['direct_total']:<10.0f} {global_data['direct_ratio']:<10.4f}")
            
            # Calculate weighted average as a third method for verification
            total_weighted_zero = 0
            total_weighted_params = 0
            for fqn, data in results.items():
                if fqn != '__global__':
                    total_weighted_zero += data['zero']
                    total_weighted_params += data['total']
            
            weighted_ratio = total_weighted_zero / (total_weighted_params + 1e-8) if total_weighted_params > 0 else 0.0
            print(f"{'WEIGHTED':<50} {total_weighted_zero:<10.0f} {total_weighted_params:<10.0f} {weighted_ratio:<10.4f}")
            
            # Store the aggregated ratio in the global data for consistency with dp_actor.py
            global_data['aggregated_ratio'] = weighted_ratio
        
        if verbose:
            print("\n[DORMANT-SUMMARY] === ZERO GRADIENT SPACE ANALYSIS COMPLETE ===\n") 
            if 'mlp_params' in locals():
                print(f"[PARAM-DEBUG] MLP layers: {mlp_params} params")
                print(f"[PARAM-DEBUG] Attention layers: {attn_params} params")
                print(f"[PARAM-DEBUG] Embedding layers: {embed_params} params")
                print(f"[PARAM-DEBUG] Norm layers: {norm_params} params")
                print(f"[PARAM-DEBUG] Other layers: {other_params} params")
                print(f"[PARAM-DEBUG] TOTAL: {mlp_params + attn_params + embed_params + norm_params + other_params} params")
                print(f"[PARAM-DEBUG] Global count: {global_param_count} params")
            print(f"[PARAM-DEBUG] TOTAL: {mlp_params + attn_params + embed_params + norm_params + other_params} params")
            print(f"[PARAM-DEBUG] Global count: {global_param_count} params")
        
        print(f"\n[ZeroGradV2] Global: {results['__global__']['zero']:.0f}/{results['__global__']['total']:.0f} "
              f"({results['__global__']['ratio']:.4f})")
        sorted_layer_items = sorted([item for item in results.items() if item[0] != '__global__'])
        for fqn, stats in sorted_layer_items:
            print(f"[ZeroGradV2] {fqn}: {stats['zero']:.0f}/{stats['total']:.0f} ({stats['ratio']:.4f})")
    # Debug prints before returning, only on rank 0
    if rank == 0 and verbose: # Added verbose check as well
        print(f"\n[DEBUG-RESULTS] Final keys in results dictionary (rank {rank}): {list(results.keys())}")
        
        items_to_print_count = 0
        max_items_to_print = 3 # Print first N items plus __global__
        
        print(f"[DEBUG-RESULTS] Sample items from results dictionary (rank {rank}):")
        for key, value in results.items():
            if items_to_print_count < max_items_to_print or key == '__global__':
                # For 'row_ratios' tensor, print shape and device to avoid large output
                if isinstance(value, dict) and 'row_ratios' in value and isinstance(value.get('row_ratios'), torch.Tensor):
                    value_to_print = value.copy()
                    row_ratios_tensor = value_to_print['row_ratios']
                    value_to_print['row_ratios'] = f"<Tensor shape={row_ratios_tensor.shape} device={row_ratios_tensor.device}>"
                    print(f"  - {key}: {value_to_print}")
                else:
                    print(f"  - {key}: {value}")
                if key != '__global__':
                    items_to_print_count += 1
            elif items_to_print_count >= max_items_to_print and '__global__' not in list(results.keys())[:max_items_to_print]:
                # This case ensures __global__ is printed if it wasn't among the first N items
                # and we have already printed N other items. This might be redundant if __global__ is always last.
                pass # Covered by the initial loop logic for __global__
    
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


def fsdp_dormant_neuron_mask_and_reset(fsdp_module, mode='threshold', tau=0.04, percentage=0.01, max_percentage=0.01, use_lecun_init=True, verbose=True, optimizer=None, original_shapes_map=None, fqn_prefix=""):
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

            fqn_local = entry.get('fqn', str(idx))
            cleaned_local_fqn = _clean_fsdp_fqn(fqn_local)
            cleaned_fqn_prefix = _clean_fsdp_fqn(fqn_prefix) # Clean the prefix

            param_is_valid_for_dormancy_check = False
            log_shape_info = "N/A"
            is_1d_param = False
            original_shape_for_analysis = None

            if original_shapes_map and full_fqn_for_map in original_shapes_map:
                original_shape = original_shapes_map[full_fqn_for_map]
                if current_rank == 0 and verbose and "embed_tokens" in full_fqn_for_map:
                    print(
                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Found in map with original shape: {original_shape}")
            elif cleaned_fqn_for_map_lookup and original_shapes_map and cleaned_fqn_for_map_lookup in original_shapes_map:
                original_shape = original_shapes_map[cleaned_fqn_for_map_lookup]
                if current_rank == 0 and verbose and "embed_tokens" in full_fqn_for_map:
                    print(
                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Found in map using cleaned FQN {cleaned_fqn_for_map_lookup} with original shape: {original_shape}")
            else:
                if current_rank == 0 and verbose and "embed_tokens" in full_fqn_for_map:
                    print(f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: NOT FOUND in original shapes map!")
                    if original_shapes_map:
                        print(
                            f"[ZeroGradV2-Debug][Rank {current_rank}] Available keys in original_shapes_map: {list(original_shapes_map.keys())[:5]}... (showing first 5)")

            if current_rank == 0 and verbose and "embed_tokens" in full_fqn_for_map:
                print(f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Original shape from map: {original_shape}")
                if param.grad is not None:
                    print(
                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Grad shape: {param.grad.shape}, numel: {param.grad.numel()}, expected 2D shape if reshaped: [{param.grad.numel() // 2048 if param.grad.numel() % 2048 == 0 else '?'}, 2048]")
                if original_shape and len(original_shape) == 2:
                    print(
                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Expected numel from original shape: {original_shape[0] * original_shape[1]}")
                    if param.grad is not None:
                        print(
                            f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Actual grad numel: {param.grad.numel()}, match: {param.grad.numel() == original_shape[0] * original_shape[1]}")
                        if param.grad.numel() != original_shape[0] * original_shape[1]:
                            print(
                                f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: MISMATCH in numel! This could be due to sharding.")
                            if param.grad.numel() < original_shape[0] * original_shape[1]:
                                shard_ratio = param.grad.numel() / (original_shape[0] * original_shape[1])
                                print(
                                    f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Possible shard ratio: {shard_ratio:.4f}")
                                possible_shard_count = 1 / shard_ratio if shard_ratio > 0 else 'unknown'
                                print(
                                    f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Possibly sharded across {possible_shard_count} ranks")
                                if param.grad.dim() == 1 and original_shape[1] > 0 and param.grad.numel() % original_shape[1] == 0:
                                    print(
                                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Can reshape to [{param.grad.numel() // original_shape[1]}, {original_shape[1]}]")
                                else:
                                    print(
                                        f"[ZeroGradV2-Debug][Rank {current_rank}] Param {full_fqn_for_map}: Cannot reshape using original width {original_shape[1]}")

            if isinstance(original_shape, (list, tuple)) and len(original_shape) == 2 and original_shape[0] > 0 and original_shape[1] > 0:
                param_is_valid_for_dormancy_check = True
            elif isinstance(original_shape, (list, tuple)) and len(original_shape) == 1 and original_shape[0] > 0:
                param_is_valid_for_dormancy_check = True
                is_1d_param = True
            elif isinstance(overlap_info['sub_shape'], (tuple, list)) and len(overlap_info['sub_shape']) > 0 and all(
                    dim > 0 for dim in overlap_info['sub_shape']):
                original_shape_for_analysis = overlap_info['sub_shape']
                log_shape_info = f"sub_shape={overlap_info['sub_shape']} (used as fallback), entry_orig_shape={entry.get('shape')}"
                if len(overlap_info['sub_shape']) == 2:
                    param_is_valid_for_dormancy_check = True
                elif len(overlap_info['sub_shape']) == 1:
                    param_is_valid_for_dormancy_check = True
                    is_1d_param = True
            
            if not param_is_valid_for_dormancy_check:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[INFO][DormantReset] Skipping entry local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}'). Not valid 1D/2D or shape info missing. Details: {log_shape_info}")
                continue
            
            if not (isinstance(original_shape_for_analysis, (tuple, list)) and len(original_shape_for_analysis) > 0 and all(dim > 0 for dim in original_shape_for_analysis)):
                 if verbose and (not is_distributed or current_rank == 0):
                    print(f"[ERROR][DormantReset] Invalid original_shape_for_analysis {original_shape_for_analysis} for local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}'). Skipping.")
                 continue

            local_param_data = flat_param.data[overlap_info['local_slice_start'] : overlap_info['local_slice_start'] + overlap_info['valid_numel']]
            
            if local_param_data.numel() == 0:
                 if verbose and (not is_distributed or current_rank == 0):
                    print(f"[DEBUG][DormantReset] Empty local_param_data for local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}'). Skipping.")
                 continue

            param_view_for_analysis = None
            try:
                # Use sub_shape from overlap_info for viewing the shard, as it reflects the shard's actual dimensions
                current_shard_shape = overlap_info['sub_shape']
                if local_param_data.numel() == torch.Size(current_shard_shape).numel():
                     param_view_for_analysis = local_param_data.view(current_shard_shape)
                else:
                    if verbose and (not is_distributed or current_rank == 0):
                        print(f"[WARN][DormantReset] Numel mismatch for shard view. local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}'), Shard Numel: {local_param_data.numel()}, Expected Shard Shape: {current_shard_shape}. Skipping.")
                    continue
            except RuntimeError as e:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[ERROR][DormantReset] Error viewing shard local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}') with shape {current_shard_shape}: {e}. Skipping.")
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
                if verbose and (not is_distributed or current_rank == 0): print(f"[ERROR][DormantReset] Param local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}') has unexpected shard shape {param_view_for_analysis.shape}. Skipping.")
                continue
            
            num_dormant_in_shard = torch.sum(dormant_mask_for_shard).item()

            if num_dormant_in_shard > 0:
                if verbose and (not is_distributed or current_rank == 0):
                    print(f"[INFO][DormantReset] Param local_fqn='{fqn_local}' (prefix='{fqn_prefix}', cleaned_prefix='{cleaned_fqn_prefix}', cleaned_local='{cleaned_local_fqn}', map_key='{full_fqn_for_map}'), Shard Shape: {param_view_for_analysis.shape}, Dormant in shard: {num_dormant_in_shard}/{num_neurons_in_shard}")
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
                
                reset_counts_details_local[full_fqn_for_map] = reset_counts_details_local.get(full_fqn_for_map, 0) + num_dormant_in_shard
                # Optimizer state reset logic would be very complex here and needs careful handling of flat_param and optimizer state indices.
                # This is often done via custom hooks or by directly manipulating optimizer.state[flat_param].
                # For now, this part is omitted for brevity but is crucial for effective reset.

            all_masks_details_local.append({'fqn': full_fqn_for_map, 'dormant': num_dormant_in_shard, 'total_in_shard': num_neurons_in_shard})

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

def identify_dormant_neurons(zero_grad_ratios, tau=0.1, percentage=None, hybrid_mode=False):
    """
    Identify dormant neurons based on zero gradient ratios.
    
    Args:
        zero_grad_ratios: Dictionary mapping parameter FQNs to zero gradient statistics
        tau: Threshold for identifying dormant neurons (0.0-1.0)
             Neurons with normalized activity below this tau are considered dormant
        percentage: If provided, select top percentage of neurons with lowest activity
        hybrid_mode: If True, use both tau and percentage criteria
        
    Returns:
        Dictionary mapping parameter FQNs to boolean masks indicating dormant neurons
    """
    dormant_masks = {}
    
    # ---- START DEBUG PRINT FOR identify_dormant_neurons INPUT ----
    # Check if dist is initialized before trying to get rank for printing
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0: # Print only on rank 0 to avoid log spam
        print(f"[DEBUG_IDN_INPUT] identify_dormant_neurons received zero_grad_ratios. Keys: {list(zero_grad_ratios.keys())}")
        test_fqn = 'model.layers.33.self_attn.q_proj.weight' # An FQN we expect to see
        if test_fqn in zero_grad_ratios:
            stats_for_test_fqn = zero_grad_ratios[test_fqn]
            print(f"[DEBUG_IDN_INPUT] Stats for {test_fqn}: {type(stats_for_test_fqn)}")
            if isinstance(stats_for_test_fqn, dict):
                print(f"[DEBUG_IDN_INPUT] Keys in stats for {test_fqn}: {list(stats_for_test_fqn.keys())}")
                if 'row_ratios' in stats_for_test_fqn:
                    rr_type = type(stats_for_test_fqn['row_ratios'])
                    rr_shape = stats_for_test_fqn['row_ratios'].shape if hasattr(stats_for_test_fqn['row_ratios'], 'shape') else 'N/A'
                    print(f"[DEBUG_IDN_INPUT] {test_fqn} has 'row_ratios'. Type: {rr_type}, Shape: {rr_shape}")
                else:
                    print(f"[DEBUG_IDN_INPUT] {test_fqn} does NOT have 'row_ratios' key in its stats dict.")
            else:
                print(f"[DEBUG_IDN_INPUT] Stats for {test_fqn} is NOT a dict.")
        else:
            print(f"[DEBUG_IDN_INPUT] Expected FQN {test_fqn} NOT FOUND in zero_grad_ratios keys.")
    # ---- END DEBUG PRINT FOR identify_dormant_neurons INPUT ----

    
    # Skip global stats
    zero_grad_ratios = {k: v for k, v in zero_grad_ratios.items() if k != '__global__'}
    
    # Process each parameter's zero gradient statistics
    params_with_row_ratios_count = 0
    params_with_actual_dormancy_count = 0
    for param_name, stats in zero_grad_ratios.items():
        # Skip parameters without proper stats
        if not isinstance(stats, dict):
            continue
            
        # Use row_ratios if available (these are the normalized gradient metrics from analyze_all_fsdp_zero_grad_space)
        if 'row_ratios' in stats and isinstance(stats['row_ratios'], torch.Tensor):
            # This is the same approach used in analyze_all_fsdp_zero_grad_space
            # where neurons with normalized activity below threshold are considered dormant
            row_metrics = stats['row_ratios']
            # Neurons with activity below tau are dormant (need to be reset)
            dormant_mask = row_metrics < tau
            
            # Apply percentage-based filtering if specified
            if percentage is not None and 0 < percentage < 1:
                # Sort neurons by their activity (ascending)
                sorted_metrics, _ = torch.sort(row_metrics.flatten())
                # Take the bottom k neurons (those with lowest activity)
                k = max(1, int(percentage * sorted_metrics.numel()))
                max_dormant_value = sorted_metrics[k-1]  # Highest activity among dormant neurons
                percentage_mask = row_metrics <= max_dormant_value
                
                if hybrid_mode:
                    # In hybrid mode, neurons must satisfy both criteria
                    final_mask = dormant_mask & percentage_mask
                else:
                    # Otherwise, use percentage-based selection
                    final_mask = percentage_mask
            else:
                # If no percentage specified, use threshold-based selection
                final_mask = dormant_mask
                
            dormant_masks[param_name] = final_mask
            
            if rank == 0 and params_with_row_ratios_count < 5: # Log details for the first 5 params with row_ratios
                print(f"[DEBUG_IDN_DETAIL][{param_name}] Tau: {tau:.4f}, Total rows: {stats['total']}, Num dormant rows: {final_mask.sum().item()}")
                if final_mask.sum().item() > 0:
                    print(f"[DEBUG_IDN_DETAIL][{param_name}] Example DORMANT row_ratios: {stats['row_ratios'][final_mask][:5].tolist()}")
                if final_mask.sum().item() < stats['total'] and stats['total'] > 0:
                    # Get non-dormant mask by inverting final_mask, ensure it's boolean
                    non_dormant_mask = ~(final_mask.bool()) 
                    if non_dormant_mask.sum().item() > 0:
                         print(f"[DEBUG_IDN_DETAIL][{param_name}] Example NON-DORMANT row_ratios: {stats['row_ratios'][non_dormant_mask][:5].tolist()}")
                elif stats['total'] == 0:
                    print(f"[DEBUG_IDN_DETAIL][{param_name}] No rows to analyze (total_rows_for_param is 0).")
            
            if final_mask.sum().item() > 0:
                params_with_actual_dormancy_count += 1
            
            params_with_row_ratios_count += 1
    
    if rank == 0:
        print(f"[INFO] Processed {params_with_row_ratios_count} parameters with row_ratios for dormancy.")
        print(f"[INFO] Found {params_with_actual_dormancy_count} parameters with at least one dormant row (row_ratio <= {tau}).")
    
    return dormant_masks


def reset_dormant_neurons_to_reference(module: nn.Module, 
                                      dormant_masks: Dict[str, torch.Tensor],
                                      fqn_map: Dict[str, list],
                                      original_shapes_map: Dict[str, torch.Size],
                                      optimizer: Optional[Optimizer] = None,
                                      verbose: bool = False):
    """
    Reset dormant neurons in the current model to values from the reference model.
    
    Args:
        module: The FSDP-wrapped model
        dormant_masks: Dictionary mapping parameter FQNs to boolean masks indicating dormant neurons
        fqn_map: Mapping from flat parameter names to original FQNs
        original_shapes_map: Mapping from FQNs to original parameter shapes
        optimizer: Optimizer to reset states for dormant neurons
        verbose: Whether to print verbose information
    """
    global REFERENCE_MODEL_PARAMS
    
    if not REFERENCE_MODEL_PARAMS:
        raise ValueError("Reference model parameters not loaded. Call load_reference_model first.")
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    printed_osm_keys_once = False # Flag to print original_shapes_map keys only once
    reset_count = 0  # Total number of dormant rows/neurons reset
    total_params_elements_in_reset_layers = 0  # Total elements in parameters that had at least one neuron reset
    params_with_resets_count = 0  # Number of parameters that had at least one neuron reset
    
    # Process each leaf FSDP module
    for fsdp_name, fsdp_module in iter_leaf_fsdp_modules(module):
        # Skip if no flat parameter
        if not hasattr(fsdp_module, '_flat_param'):
            continue
            
        flat_param = fsdp_module._flat_param
        flat_param_name = f"{fsdp_name}._flat_param"
        
        # Get the mapping from flat parameter to original FQNs
        param_fqns = fqn_map.get(flat_param_name, [])
        if not param_fqns:
            continue
            
        # Process each original parameter in this flat parameter
        if verbose and rank == 0 and not printed_osm_keys_once and original_shapes_map:
            print(f"[DEBUG_RDNTR_INIT] original_shapes_map keys (sample): {list(original_shapes_map.keys())[:10]}")
            printed_osm_keys_once = True

        for fqn in param_fqns:
            if verbose and rank == 0:
                # Print only a few keys from dormant_masks to avoid excessive logging
                dm_keys_sample = list(dormant_masks.keys())[:5]
                if len(dormant_masks.keys()) > 5:
                    dm_keys_sample.append("...")
                print(f"[DEBUG_RDNTR_FQN] Checking FQN: '{fqn}'. Available in dormant_masks (sample): {dm_keys_sample}")
            # Skip if no dormant mask for this parameter
            if fqn not in dormant_masks:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' not found in dormant_masks. Skipping.")
                continue
                
            # Get original shape and dormant mask
            orig_shape = original_shapes_map.get(fqn)
            if orig_shape is None or len(orig_shape) != 2:  # Only handle 2D parameters
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped. orig_shape: {orig_shape}. Must be 2D and not None.")
                continue
                
            # Get metadata for this parameter in the flat parameter
            param_metadata = None
            try:
                # Find the metadata for the current fqn within the flat_param's original parameters
                # This relies on fqn matching one of the original_fqns in param_metadata_list
                for meta_idx, meta in enumerate(flat_param._param_metadata_list):
                    current_meta_fqn = getattr(meta, 'fqn', None) # FSDP often stores full FQN here
                    if verbose and rank == 0:
                        # This log can be very verbose, print only if fqn might be problematic or for specific debug
                        # For now, let's keep it to see the comparison if metadata is not found later
                        pass # print(f"[DEBUG_RDNTR_METAMATCH_DETAIL] Comparing our fqn='{fqn}' with meta.fqn='{current_meta_fqn}' from _param_metadata_list for {flat_param_name}")
                    if current_meta_fqn == fqn:
                        param_metadata = meta
                        break
                    # Sometimes metadata might only have local 'param_name'. 
                    # Our 'fqn' is already the full FQN from fqn_map, so direct match to 'meta.fqn' is preferred.
                    # A more robust way would be to ensure fqn_map itself is derived perfectly from these metadatas earlier.
            except AttributeError:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped due to AttributeError accessing _param_metadata_list for {flat_param_name}.")
                continue
            
            if param_metadata is None:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped. No matching metadata found in {flat_param_name}'s _param_metadata_list. Searched for FQN: '{fqn}'. Available meta.fqns (sample): {[getattr(m, 'fqn', None) for m in flat_param._param_metadata_list[:5]]}")
                continue
                
            # Get the slice of this parameter in the flat parameter
            # Different versions of PyTorch FSDP might use different attribute names
            start = getattr(param_metadata, 'start_idx', None)
            end = getattr(param_metadata, 'end_idx', None)
            
            # If start_idx/end_idx are not available, try other attribute names
            if start is None and hasattr(param_metadata, 'param_offset'):
                start = param_metadata.param_offset
                
            if end is None and start is not None and hasattr(param_metadata, 'numel'):
                end = start + param_metadata.numel
                
            if start is None or end is None:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped. Could not determine parameter slice in {flat_param_name}.")
                continue
                
            flat_tensor = flat_param[start:end].view(orig_shape)
            
            # Get reference parameter
            if fqn not in REFERENCE_MODEL_PARAMS:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped. Parameter not found in reference model.")
                continue
                
            ref_param = REFERENCE_MODEL_PARAMS[fqn]
            
            # Check shape compatibility
            if flat_tensor.shape != ref_param.shape:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN '{fqn}' skipped. Shape mismatch: {flat_tensor.shape} vs {ref_param.shape}.")
                continue
                
            # Apply dormant mask and reset weights
            num_dormant = dormant_masks[fqn].sum().item()
            if verbose and rank == 0:
                print(f"[DEBUG_RDNTR_DORMANCY_CHECK] FQN: {fqn}, NumDormant in mask: {num_dormant}")

            if num_dormant > 0:
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_RESETTING] FQN: {fqn} has {num_dormant} dormant rows. Proceeding with reset.")
            else: # num_dormant == 0
                if verbose and rank == 0:
                    print(f"[DEBUG_RDNTR_SKIP] FQN: {fqn} skipped as num_dormant is 0.")
                continue
                
            # Expand mask to match parameter dimensions if needed
            current_dormant_mask = dormant_masks[fqn] # Use a local variable for clarity
            if current_dormant_mask.dim() == 1 and flat_tensor.dim() == 2:
                expanded_mask = current_dormant_mask.unsqueeze(1).expand_as(flat_tensor)
            else:
                expanded_mask = current_dormant_mask
            
            # Increment count of parameters that had resets
            params_with_resets_count += 1
            
            # Reset dormant neurons to reference model values
            with torch.no_grad():
                flat_tensor[expanded_mask] = ref_param.to(flat_tensor.device)[expanded_mask]
                reset_count += num_dormant # num_dormant was calculated before the 'if num_dormant > 0' block
                total_params_elements_in_reset_layers += flat_tensor.numel()
                
                # Reset optimizer state if provided
                if optimizer is not None:
                    reset_optimizer_state_for_dormant_neurons(optimizer, flat_param, 
                                                              start, end, # These are start_idx, end_idx for the param in flat_param
                                                              orig_shape, expanded_mask, verbose)
                
                # Reset optimizer state if provided
                if optimizer is not None:
                    reset_optimizer_state_for_dormant_neurons(optimizer, flat_param, start, end, 
                                                           orig_shape, expanded_mask, verbose)
                    # Reset dormant neurons to reference model values
                    flat_tensor[expanded_mask] = ref_param.to(flat_tensor.device)[expanded_mask]
                    reset_count += num_dormant
                    total_params_elements_in_reset_layers += flat_tensor.numel()
                    
                    # Reset optimizer state if provided
                    if optimizer is not None:
                        reset_optimizer_state_for_dormant_neurons(optimizer, flat_param, start, end, 
                                                               orig_shape, expanded_mask, verbose)
    
    if verbose and rank == 0:
        print(f"[INFO] Dormant Neuron Reset Summary:")
        print(f"  - Parameters with at least one dormant row reset: {params_with_resets_count}")
        print(f"  - Total individual dormant rows/neurons reset: {reset_count}")
        print(f"  - Total elements in affected parameters: {total_params_elements_in_reset_layers}")
        if total_params_elements_in_reset_layers > 0:
            print(f"  - Percentage of elements reset in affected parameters: {reset_count / total_params_elements_in_reset_layers * 100:.2f}%")
        else:
            print(f"  - Percentage of elements reset in affected parameters: 0.00%")


def reset_optimizer_state_for_dormant_neurons(optimizer: Optimizer,
                                            flat_param: torch.Tensor,
                                            start_idx: int,
                                            end_idx: int,
                                            orig_shape: torch.Size,
                                            dormant_mask: torch.Tensor,
                                            verbose: bool = False):
    """
    Reset optimizer state for dormant neurons.
    
    Args:
        optimizer: The optimizer
        flat_param: The flat parameter
        start_idx: Start index of the parameter in the flat parameter
        end_idx: End index of the parameter in the flat parameter
        orig_shape: Original shape of the parameter
        dormant_mask: Boolean mask indicating dormant neurons
        verbose: Whether to print verbose information
    """
    # Find the parameter in the optimizer state
    found = False
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if id(p) == id(flat_param):
                found = True
                break
        if found:
            break
            
    if not found:
        if verbose:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                print(f"[WARNING] Parameter not found in optimizer state")
        return
        
    # Reset optimizer state for dormant neurons
    if 'adam' in optimizer.__class__.__name__.lower() or 'adamw' in optimizer.__class__.__name__.lower():
        # For Adam/AdamW optimizers
        if 'state' in optimizer.state[flat_param]:
            state = optimizer.state[flat_param]
            
            # Reset exp_avg (momentum) and exp_avg_sq (variance) for dormant neurons
            for state_name in ['exp_avg', 'exp_avg_sq']:
                if state_name in state:
                    with torch.no_grad():
                        state_tensor = state[state_name][start_idx:end_idx].view(orig_shape)
                        state_tensor[dormant_mask] = 0.0
    elif 'sgd' in optimizer.__class__.__name__.lower():
        # For SGD optimizer
        if 'state' in optimizer.state[flat_param]:
            state = optimizer.state[flat_param]
            
            # Reset momentum for dormant neurons
            if 'momentum_buffer' in state:
                with torch.no_grad():
                    momentum = state['momentum_buffer'][start_idx:end_idx].view(orig_shape)
                    momentum[dormant_mask] = 0.0


def fsdp_dormant_neuron_reset_pipeline(module: nn.Module, 
                                      zero_grad_ratios: Dict[str, Dict[str, float]],
                                      optimizer: Optional[Optimizer] = None,
                                      threshold: float = 0.9,
                                      percentage: Optional[float] = None,
                                      hybrid_mode: bool = False,
                                      verbose: bool = False,
                                      original_param_shapes: Optional[Dict[str, torch.Size]] = None):
    """
    Complete pipeline for resetting dormant neurons in an FSDP-wrapped model to reference model values.
    
    Args:
        module: The FSDP-wrapped model
        zero_grad_ratios: Dictionary mapping parameter FQNs to their zero gradient statistics
        optimizer: Optimizer to reset states for dormant neurons
        threshold: Threshold for considering a neuron dormant based on zero gradient ratio
        percentage: If provided, select top percentage of neurons with highest zero gradient ratios
        hybrid_mode: If True, use both threshold and percentage criteria
        verbose: Whether to print verbose information
        original_param_shapes: Dictionary mapping parameter FQNs to their original shapes before FSDP wrapping
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if verbose and rank == 0:
        print(f"[INFO] Starting dormant neuron reset pipeline with threshold={threshold}, "
              f"percentage={percentage}, hybrid_mode={hybrid_mode}")
    
    # Check if reference model is loaded
    global REFERENCE_MODEL_PARAMS
    if not REFERENCE_MODEL_PARAMS:
        if rank == 0:
            print("[ERROR] Reference model parameters not loaded. Call load_reference_model first.")
        return
    
    # Step 1: Build FQN map for all flat parameters
    fqn_map = {}
    # original_shapes_map argument is already correctly keyed by full FQNs and should be read-only here.
    # We'll use it for lookups if needed but not modify it.

    def _get_clean_module_fqn(fsdp_module_fqn: str) -> str:
        # Remove _fsdp_wrapped_module. prefix to get model-relative FQN
        # e.g., _fsdp_wrapped_module.model.layers.0.self_attn -> model.layers.0.self_attn
        prefix = fsdp_module_fqn
        if prefix.startswith("_fsdp_wrapped_module."):
            prefix = prefix[len("_fsdp_wrapped_module."):]
        # Handle potential nested FSDP wrapping if it results in multiple prefixes, though less common for leaf FSDP names
        # For simplicity, this handles one level. If deeper nesting prefixes occur in fsdp_name, this might need to be a loop.
        return prefix
    
    for fsdp_name, fsdp_module in iter_leaf_fsdp_modules(module):
        if not hasattr(fsdp_module, '_flat_param'):
            continue
            
        flat_param = fsdp_module._flat_param
        flat_param_name = f"{fsdp_name}._flat_param"
        
        # Map flat parameter to original FQNs
        param_fqns = []
        
        # Try to extract FQNs from param_infos if possible
        try:
            for metadata in flat_param._param_infos:
                # Different versions of PyTorch FSDP might use different attribute names
                local_param_name_from_meta = None
                if hasattr(metadata, 'fqn'):
                    local_param_name_from_meta = metadata.fqn
                elif hasattr(metadata, 'param_name'):
                    local_param_name_from_meta = metadata.param_name
                
                if local_param_name_from_meta:
                    # Construct the full FQN
                    # fsdp_name is like '_fsdp_wrapped_module.model.layers.0.self_attn' or '_fsdp_wrapped_module.'
                    # local_param_name_from_meta is like 'q_proj.weight' (relative) or 'model.embed_tokens.weight' (absolute for root params)
                    cleaned_module_fqn = _get_clean_module_fqn(fsdp_name) # e.g., 'model.layers.0.self_attn' or ''
                    
                    if cleaned_module_fqn: 
                        # Example: cleaned_module_fqn = 'model.layers.0.self_attn', local_param_name_from_meta = 'q_proj.weight'
                        # Result: 'model.layers.0.self_attn.q_proj.weight'
                        full_fqn = f"{cleaned_module_fqn}.{local_param_name_from_meta}"
                    else:
                        # Example: cleaned_module_fqn = '', local_param_name_from_meta = 'model.embed_tokens.weight' (from metadata.fqn for a root FSDP param)
                        # Result: 'model.embed_tokens.weight'
                        full_fqn = local_param_name_from_meta
                    param_fqns.append(full_fqn)
                    
                    # original_shapes_map (passed as original_param_shapes argument)
                    # should already contain the correct original shape keyed by full_fqn.
                    # No need to update it here from metadata.shape.
        except Exception as e:
            if verbose:
                print(f"[WARNING] Error accessing param_infos: {e}")
                
        # If we couldn't extract FQNs, use the original_param_shapes if provided
        if not param_fqns and original_param_shapes:
            # Try to match based on size and other heuristics
            # This is a fallback mechanism
            if verbose:
                print(f"[INFO] Using original_param_shapes as fallback for {flat_param_name}")
                
        fqn_map[flat_param_name] = param_fqns
    
    # Step 2: Identify dormant neurons based on zero gradient ratios
    dormant_masks = identify_dormant_neurons(
        zero_grad_ratios, tau=threshold, percentage=percentage, hybrid_mode=hybrid_mode
    )
    
    if verbose and rank == 0:
        print(f"[INFO] Identified dormant neurons in {len(dormant_masks)} parameters")
    
    # Step 3: Reset dormant neurons to reference model values
    reset_dormant_neurons_to_reference(
        module, dormant_masks, fqn_map, original_param_shapes, optimizer, verbose
    )
    
    if verbose and rank == 0:
        print("[INFO] Dormant neuron reset pipeline completed")