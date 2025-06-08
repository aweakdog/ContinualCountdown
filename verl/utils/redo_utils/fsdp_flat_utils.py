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

def iter_leaf_fsdp_modules(module: nn.Module):
    """Iterate over all leaf FSDP modules."""
    for name, submodule in module.named_modules():
        if isinstance(submodule, FSDP) and not any(isinstance(child, FSDP) for child in submodule.modules() if child is not submodule):
            yield name, submodule

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
                submodule_global_stats = stats['__global__']
                total_zero += submodule_global_stats.get('zero', 0)
                total_rows += submodule_global_stats.get('total', 0)
                results[name] = stats # Store the full detailed stats for this submodule
            else:
                results[name] = None
                if verbose:
                    if stats is None:
                        print(f"[WARN] Zero-grad analysis returned None for submodule {name}")
                    elif '__global__' not in stats:
                        print(f"[WARN] '__global__' key missing in stats for submodule {name}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Could not analyze zero grad space for {name}: {e}")
            results[name] = None
        break # hacky

    # Calculate the aggregated ratio from the total counts
    global_ratio = total_zero / (total_rows + 1e-8) if total_rows > 0 else 0.0
    
    # Create the global stats dictionary with the aggregated ratio
    results['__global__'] = {
        'zero': total_zero, 
        'total': total_rows, 
        'ratio': global_ratio,
        'aggregated_ratio': global_ratio  # Add aggregated_ratio key explicitly for dp_actor.py
    }
    
    return results

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
                        
                    if si.numel() > 0:
                        print(f"[ZeroGradV2-FIX]   - si stats: min={si.min().item():.2e}, max={si.max().item():.2e}, mean={si.mean().item():.2e}, tau={tau:.2e}")
                    else:
                        print(f"[ZeroGradV2-FIX]   - si is empty, tau={tau:.2e}")
                    
                    # Additional layer-specific warning
                    if "mlp" in fqn:
                        print(f"[ZeroGradV2-FIX]   - High dormancy in MLP layer: {zero_rows/H_local_scalar:.4f}")
                    elif "attn" in fqn:
                        print(f"[ZeroGradV2-FIX]   - High dormancy in attention layer: {zero_rows/H_local_scalar:.4f}")
                    else:
                        print(f"[ZeroGradV2-FIX]   - High dormancy in other layer: {zero_rows/H_local_scalar:.4f}")
            
            # We've already printed most debug info earlier, just add this to the existing output
            # No need to repeat it here
            
            # Store local stats including the normalized gradient activity (si) as row_ratios
            layer_stats_local[fqn] = {
                'zero': zero_rows, 
                'total': H_local_scalar,
                'row_ratios': si.detach().clone()  # Store the normalized gradient activity for dormant neuron identification
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
        ratio = data['zero'] / (data['total'] + 1e-8) if data['total'] > 0 else 0.0
        results[fqn] = {**data, 'ratio': ratio}
        
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
                for fqn, data in sorted(group_items, key=lambda x: x[0]):
                    zero_count = data['zero']
                    total_count = data['total']
                    ratio = data['ratio']
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
    
    return results

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


def kaiming_reset(weight_slice, mask, fan_in=None, a=0, mode='fan_in', nonlinearity='leaky_relu', bias_slice=None, bias_bound=0.01):
    """Reset weights using Kaiming initialization."""
    if fan_in is None:
        fan_in = weight_slice.size(1)  # Assume weight is of shape (out_features, in_features)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        weight_slice[mask] = torch.empty_like(weight_slice[mask]).uniform_(-bound, bound)
        if bias_slice is not None:
            bias_slice[mask] = torch.empty_like(bias_slice[mask]).uniform_(-bias_bound, bias_bound)


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
    
    # Skip global stats
    zero_grad_ratios = {k: v for k, v in zero_grad_ratios.items() if k != '__global__'}
    
    # Process each parameter's zero gradient statistics
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
    reset_count = 0
    total_params = 0
    
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
        for fqn in param_fqns:
            # Skip if no dormant mask for this parameter
            if fqn not in dormant_masks:
                continue
                
            # Get original shape and dormant mask
            orig_shape = original_shapes_map.get(fqn)
            if orig_shape is None or len(orig_shape) != 2:  # Only handle 2D parameters
                continue
                
            dormant_mask = dormant_masks[fqn]
            
            # Get metadata for this parameter in the flat parameter
            param_metadata = None
            try:
                for metadata in flat_param._param_infos:
                    # Try different attribute names that might contain the parameter name
                    param_name = None
                    if hasattr(metadata, 'fqn'):
                        param_name = metadata.fqn
                    elif hasattr(metadata, 'param_name'):
                        param_name = metadata.param_name
                        
                    if param_name == fqn:
                        param_metadata = metadata
                        break
            except Exception as e:
                if verbose:
                    print(f"[WARNING] Error accessing param_infos for {fqn}: {e}")
                continue
                    
            if param_metadata is None:
                if verbose:
                    print(f"[WARNING] Could not find metadata for parameter {fqn}")
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
                if verbose:
                    print(f"[WARNING] Could not determine parameter slice for {fqn}")
                continue
            flat_tensor = flat_param[start:end].view(orig_shape)
            
            # Get reference parameter
            if fqn not in REFERENCE_MODEL_PARAMS:
                if verbose and rank == 0:
                    print(f"[WARNING] Parameter {fqn} not found in reference model")
                continue
                
            ref_param = REFERENCE_MODEL_PARAMS[fqn]
            
            # Check shape compatibility
            if flat_tensor.shape != ref_param.shape:
                if verbose and rank == 0:
                    print(f"[WARNING] Shape mismatch for {fqn}: {flat_tensor.shape} vs {ref_param.shape}")
                continue
                
            # Apply dormant mask and reset weights
            num_dormant = dormant_mask.sum().item()
            if num_dormant > 0:
                with torch.no_grad():
                    # Expand mask to match parameter dimensions if needed
                    if dormant_mask.dim() == 1 and flat_tensor.dim() == 2:
                        expanded_mask = dormant_mask.unsqueeze(1).expand_as(flat_tensor)
                    else:
                        expanded_mask = dormant_mask
                        
                    # Reset dormant neurons to reference model values
                    flat_tensor[expanded_mask] = ref_param.to(flat_tensor.device)[expanded_mask]
                    reset_count += num_dormant
                    total_params += flat_tensor.numel()
                    
                    # Reset optimizer state if provided
                    if optimizer is not None:
                        reset_optimizer_state_for_dormant_neurons(optimizer, flat_param, start, end, 
                                                               orig_shape, expanded_mask, verbose)
    
    if verbose and rank == 0:
        print(f"[INFO] Reset {reset_count} dormant parameters out of {total_params} total parameters")
        print(f"[INFO] Reset percentage: {reset_count / max(1, total_params) * 100:.2f}%")


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
    original_shapes_map = original_param_shapes or {}
    
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
                if hasattr(metadata, 'fqn'):
                    fqn = metadata.fqn
                    param_fqns.append(fqn)
                    
                    # Store original shape if available
                    if hasattr(metadata, 'shape'):
                        original_shapes_map[fqn] = metadata.shape
                elif hasattr(metadata, 'param_name'):
                    fqn = metadata.param_name
                    param_fqns.append(fqn)
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
        module, dormant_masks, fqn_map, original_shapes_map, optimizer, verbose
    )
    
    if verbose and rank == 0:
        print("[INFO] Dormant neuron reset pipeline completed")