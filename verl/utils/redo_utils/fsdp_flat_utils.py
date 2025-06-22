"""
Utilities for mapping and manipulating FSDP flat parameters for dormant neuron analysis and reset.
Supports extracting per-layer grad stats and applying neuron-wise resets directly on the flat parameter.
"""
import torch
import math
from torch.distributed.fsdp import FlatParameter, FullyShardedDataParallel as FSDP

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
                                                       )
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
                # Determine the effective dimension for checking eligibility
                # Use original_shape if available, otherwise the current param's dimension
                effective_dim = len(original_shape) if original_shape else param.dim()
                is_bias = full_fqn_for_map.endswith(".bias")

                # Parameter eligibility condition:
                # Must be 2D (handled by original_shape or param.dim()), OR
                # Must be 1D AND NOT a bias parameter (these will be reshaped to (N,1) later).
                # Bias parameters (1D or other) are typically not analyzed this way.
                is_eligible_based_on_dim_and_type = (effective_dim == 2) or \
                                                    (effective_dim == 1 and not is_bias)

                if not is_eligible_based_on_dim_and_type:
                    skipped_dim_not_2 += 1
                    if rank == 0 and verbose and skipped_dim_not_2 < 5: # Log only a few times
                        print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: SKIPPING (effective_dim: {effective_dim}, is_bias: {is_bias}. Original shape: {original_shape_str}, current_param_dim: {param.dim()}, grad_shape: {param.grad.shape}). Not 2D or 1D non-bias.")
                # For eligible dimensions, also check if grad.shape[0] is 0 (empty tensor along the main dim)
                # This check is more relevant for 2D+ tensors or 1D tensors that will be treated as (N,1)
                elif effective_dim > 0 and param.grad.shape[0] == 0 : 
                    skipped_shape0_is_0 += 1
                    if rank == 0 and verbose and skipped_shape0_is_0 < 5: # Log only a few times
                        print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: SKIPPING (grad.shape[0] is 0, grad_shape: {param.grad.shape}).")
                else:
                    # Parameter is eligible for processing based on dimension, type, and grad shape
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

            # After all other potential reshaping, if current_grad_to_process is 1D 
            # and it's a non-bias parameter (is_bias should be in scope from earlier check),
            # reshape it to (N, 1) to be treated as a 2D matrix.
            if current_grad_to_process.dim() == 1 and not is_bias: # `is_bias` refers to the current param
                current_grad_to_process = current_grad_to_process.unsqueeze(1) # Shape (N) -> (N, 1)
                if rank == 0 and verbose: # Add a log for this specific reshape
                    print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: Reshaped 1D non-bias grad to 2D ({current_grad_to_process.shape}) for analysis.")
                reshaped_from_map = True # Indicate that a reshape for analysis occurred

            if rank == 0 and verbose: # This is the PRE-FILTER log
                print(f"[ZeroGradV2-Debug][Rank {rank}] Param {full_fqn_for_map}: PRE-FILTER (original_grad_shape: {param.grad.data.shape}, processing_grad_shape: {current_grad_to_process.shape}, reshaped: {reshaped_from_map}, current_param_dim: {current_grad_to_process.dim()}).")
            
            # H_global and B_global are calculated based on all eligible params
            # For each parameter, calculate its contribution to H_global and B_global
            if current_grad_to_process.dim() == 2:
                # grad_norm_row: (output_dim,)
                grad_norm_row = torch.norm(current_grad_to_process, p=1, dim=1)  # Norm along input_dim (dim 1)
                # H_local is the number of rows (output dimension) from the (potentially reshaped) gradient tensor
                H_local = current_grad_to_process.shape[0]
            elif current_grad_to_process.dim() == 1:
                # Treat 1D tensor as a single row
                grad_norm_row = torch.norm(current_grad_to_process, p=1, dim=0).unsqueeze(0) # Shape [1]
                H_local = 1
            else:
                # Should not happen if eligibility checks are correct (param_dim_to_check == 2, or 1D fallback)
                if rank == 0 and verbose:
                    print(f"[ZeroGradV2-ERROR][Rank {rank}] Param {full_fqn_for_map} has unexpected dim {current_grad_to_process.dim()} for norm calculation. Using zeros.")
                H_local = 0
                grad_norm_row = torch.zeros(1, device=device)
            
            # B_local is the sum of squared norms of all rows in the current layer's gradient
            B_local = torch.sum(grad_norm_row).item() 
        
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
            
            # Store local stats
            current_avg_global_for_si = avg_global if 'avg_global' in locals() and H_global > 0 else 0.0
            layer_stats_local[fqn] = {
                'zero': zero_rows, 
                'total': H_local_scalar, 
                'avg_global_for_si_calc': current_avg_global_for_si,
                'B_global_val': B_global if 'B_global' in locals() else 0.0, 
                'H_global_val': H_global if 'H_global' in locals() else 0.0,
                'A_local_rows_part': grad_norm_row # grad_norm_row is A_local_row_tensor for this rank
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
    # Process gathered stats on rank 0 to fill combined_stats
    if rank == 0:
        for fqn in all_fqns:
            # Initialize sums for zero and total counts for this FQN
            current_fqn_zero_sum = 0
            current_fqn_total_sum = 0
            
            # Initialize B, H, B/H and A_parts for this fqn
            fqn_B_global_val = 0.0
            fqn_H_global_val = 0.0
            fqn_avg_global_for_si_calc = 0.0
            found_global_BH_stats_for_fqn = False
            all_A_parts_for_fqn = []

            for rank_idx, rank_specific_stats_dict in enumerate(all_layer_stats_gathered):
                if rank_specific_stats_dict and fqn in rank_specific_stats_dict:
                    data_for_fqn_from_rank = rank_specific_stats_dict[fqn]
                    
                    current_fqn_zero_sum += data_for_fqn_from_rank.get('zero', 0)
                    current_fqn_total_sum += data_for_fqn_from_rank.get('total', 0) # 'total' is H_local_scalar_for_si from that rank
                    
                    # Retrieve the globally consistent B, H, B/H. 
                    # These were stored in layer_stats_local from the global maps earlier.
                    # We only need to get them once from any rank that processed this FQN.
                    if not found_global_BH_stats_for_fqn and ('total' in data_for_fqn_from_rank and data_for_fqn_from_rank['total'] > 0): 
                        fqn_B_global_val = data_for_fqn_from_rank.get('B_global_val', 0.0)
                        fqn_H_global_val = data_for_fqn_from_rank.get('H_global_val', 0.0)
                        fqn_avg_global_for_si_calc = data_for_fqn_from_rank.get('avg_global_for_si_calc', 0.0)
                        found_global_BH_stats_for_fqn = True

                    # Collect A_local_rows_part if the parameter contributed rows on that rank
                    if data_for_fqn_from_rank.get('total', 0) > 0: # 'total' here is H_local_scalar_for_si for that rank
                        a_part = data_for_fqn_from_rank.get('A_local_rows_part')
                        if isinstance(a_part, torch.Tensor) and a_part.numel() > 0:
                            all_A_parts_for_fqn.append(a_part.to(device)) # Ensure on same device for cat
            
            combined_stats[fqn]['zero'] = current_fqn_zero_sum
            # The 'total' in combined_stats should be the true global H for this parameter, 
            # which is fqn_H_global_val if found, otherwise sum of local H contributions.
            combined_stats[fqn]['total'] = fqn_H_global_val if found_global_BH_stats_for_fqn and fqn_H_global_val > 0 else current_fqn_total_sum
            combined_stats[fqn]['B_global_val'] = fqn_B_global_val
            combined_stats[fqn]['H_global_val'] = fqn_H_global_val # This is the definitive H_global for this fqn
            combined_stats[fqn]['avg_global_for_si_calc'] = fqn_avg_global_for_si_calc

            if all_A_parts_for_fqn:
                # Concatenate all A_local_rows_part tensors for this FQN from all ranks
                combined_A_tensor_for_fqn = torch.cat(all_A_parts_for_fqn)
            else:
                # Ensure it's an empty tensor on the correct device if no parts were found
                combined_A_tensor_for_fqn = torch.empty(0, device=device) 
            combined_stats[fqn]['all_A_local_rows_gathered'] = combined_A_tensor_for_fqn

    # Step 6: Verbose per-parameter printing on rank 0 using combined_stats
    if rank == 0 and verbose:
        print(f"\n[ZeroGradV2-FINAL-STATS][Rank {rank}] Per-parameter zero gradient statistics (Tau: {tau}):")
        sorted_fqns = sorted(combined_stats.keys())
        for fqn in sorted_fqns:
            stats = combined_stats[fqn]
            zero_count = stats.get('zero', 0)
            total_count = stats.get('total', 0) # This should be H_global for the parameter
            ratio = zero_count / (total_count + 1e-8) if total_count > 0 else 0.0
            
            # Get B, H, and B/H from combined_stats (which should hold the true global values)
            B_global_print = stats.get('B_global_val', 0.0)
            H_global_print = stats.get('H_global_val', 0.0)
            avg_BH_print = stats.get('avg_global_for_si_calc', 0.0)

            # Get A stats (min, max, mean of per-neuron gradient magnitudes)
            A_tensor = stats.get('all_A_local_rows_gathered')
            A_stats_str = "A_tensor: N/A"
            if isinstance(A_tensor, torch.Tensor) and A_tensor.numel() > 0:
                A_min = A_tensor.min().item()
                A_max = A_tensor.max().item()
                A_mean = A_tensor.mean().item()
                A_stats_str = f"A_min: {A_min:.4e}, A_max: {A_max:.4e}, A_mean: {A_mean:.4e} (Size: {A_tensor.numel()})"
            elif isinstance(A_tensor, torch.Tensor) and A_tensor.numel() == 0:
                A_stats_str = "A_tensor: EMPTY"

            print(f"  Layer: {fqn:<80} | Zeros: {zero_count:<7} / Total: {total_count:<7} ({ratio:>7.2%}) | B_global: {B_global_print:.8e}, H_global: {H_global_print:.0f}, B/H_calc: {avg_BH_print:.4e} | {A_stats_str}")

    # Step 7: Populate results dictionary from combined_stats (on rank 0)
    # The '__global__' entry in results is already populated with overall zero/total/ratio from Step 4.
    # Here, we add per-parameter details.
    if rank == 0:
        for fqn, stats in combined_stats.items():
            # Ensure fqn entry exists
            if fqn not in results:
                results[fqn] = {}
            
            results[fqn]['zero'] = stats.get('zero',0)
            results[fqn]['total'] = stats.get('total',0) # This is H_global for the parameter
            results[fqn]['ratio'] = stats.get('zero',0) / (stats.get('total',0) + 1e-8) if stats.get('total',0) > 0 else 0.0
            results[fqn]['avg_global_for_si_calc'] = stats.get('avg_global_for_si_calc', 0.0)
            results[fqn]['B_global_val'] = stats.get('B_global_val', 0.0)
            results[fqn]['H_global_val'] = stats.get('H_global_val', 0.0)
            
            # Add A stats to results if you want them stored
            A_tensor_res = stats.get('all_A_local_rows_gathered')
            if isinstance(A_tensor_res, torch.Tensor) and A_tensor_res.numel() > 0:
                results[fqn]['A_min'] = A_tensor_res.min().item()
                results[fqn]['A_max'] = A_tensor_res.max().item()
                results[fqn]['A_mean'] = A_tensor_res.mean().item()
                results[fqn]['A_numel'] = A_tensor_res.numel()
            elif isinstance(A_tensor_res, torch.Tensor) and A_tensor_res.numel() == 0:
                results[fqn]['A_min'] = 0.0
                results[fqn]['A_max'] = 0.0
                results[fqn]['A_mean'] = 0.0
                results[fqn]['A_numel'] = 0

    # Ensure all ranks have the full results dictionary if needed elsewhere, or just return from rank 0
    # For now, we assume results are primarily used/returned by rank 0 after this function.
    # If other ranks need it, an all_gather_object or broadcast would be needed here for `results`.

            for stats_dict_from_rank_for_bh in all_layer_stats_gathered:
                if stats_dict_from_rank_for_bh and fqn in stats_dict_from_rank_for_bh:
                    # Initialize to ensure they are picked up if present
                    temp_B = stats_dict_from_rank_for_bh[fqn].get('B_global_val', None)
                    temp_H = stats_dict_from_rank_for_bh[fqn].get('H_global_val', None)
                    if temp_B is not None:
                        fqn_B_val = temp_B
                    if temp_H is not None:
                        fqn_H_val = temp_H
                    # If both found from one rank, assume consistency and break
                    if temp_B is not None and temp_H is not None:
                        break
            combined_stats[fqn]['B_global_val'] = fqn_B_val
            combined_stats[fqn]['H_global_val'] = fqn_H_val

            # Aggregate A_local_rows_part for this fqn from all ranks where it was processed
            all_A_parts_for_fqn = []
            for rank_stats_dict in all_layer_stats_gathered:
                if rank_stats_dict and fqn in rank_stats_dict:
                    # Only consider A_local_rows_part if the parameter contributed rows (H_local_scalar > 0) on that rank
                    if rank_stats_dict[fqn].get('total', 0) > 0:
                        a_part = rank_stats_dict[fqn].get('A_local_rows_part')
                        if isinstance(a_part, torch.Tensor) and a_part.numel() > 0:
                            all_A_parts_for_fqn.append(a_part.to(device)) # Ensure on same device for cat
            
            if all_A_parts_for_fqn:
                combined_A_tensor_for_fqn = torch.cat(all_A_parts_for_fqn)
            else:
                combined_A_tensor_for_fqn = torch.empty(0, device=device) 
            combined_stats[fqn]['all_A_local_rows_gathered'] = combined_A_tensor_for_fqn
    
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
            avg_global_val = stats.get('avg_global_for_si_calc')
            B_val = stats.get('B_global_val', 0.0)
            H_val = stats.get('H_global_val', 0.0)
            bh_stats_display_str = f", B/H: {avg_global_val:.4e} (B: {B_val:.4e}, H: {H_val:.1f})" if avg_global_val is not None else ""

            A_stats_display_str = ""
            # Assuming 'all_A_local_rows_gathered' is a torch.Tensor on rank 0 with all relevant values, based on memory 95a9f283.
            all_A_tensor = stats.get('all_A_local_rows_gathered') 
            if isinstance(all_A_tensor, torch.Tensor) and all_A_tensor.numel() > 0:
                min_A = all_A_tensor.min().item()
                max_A = all_A_tensor.max().item()
                mean_A = all_A_tensor.mean().item()
                A_stats_display_str = f", A(min/max/mean): {min_A:.2e}/{max_A:.2e}/{mean_A:.2e}"
            elif all_A_tensor is not None: # Key exists but content is not a usable tensor
                A_stats_display_str = ", A(stats): N/A (Err)"
            # If all_A_tensor is None, A_stats_display_str remains empty, so nothing extra is printed for A-stats.

            print(f"[ZeroGradV2] {fqn}: {stats['zero']:.0f}/{stats['total']:.0f} ({stats['ratio']:.4f}){bh_stats_display_str}{A_stats_display_str}")
    
    return results