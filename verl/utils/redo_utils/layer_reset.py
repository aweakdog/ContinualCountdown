# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Layer Reset Utilities for LLM Plasticity Experiments

This module provides utilities to reset specific transformer layers of LLMs 
to their reference model states during RLHF training to maintain plasticity.
"""

import logging
import torch
import ray
from typing import List, Optional, Dict, Any, Set
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

logger = logging.getLogger(__name__)


class LayerResetManager:
    """
    Manages layer reset operations for LLM plasticity experiments.
    
    This class handles resetting specific transformer layers (first k and last k)
    of both actor and critic models to their reference states at specified
    global training steps.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LayerResetManager.
        
        Args:
            config: Configuration dictionary with reset parameters
        """
        self.config = config or {}
        
        # Core reset settings
        self.enable_reset = self.config.get('enable_reset', False)
        self.reset_k_first = self.config.get('reset_k_first', 0)
        self.reset_k_last = self.config.get('reset_k_last', 0)
        
        # Parse reset_steps - handle both list and string formats
        reset_steps_raw = self.config.get('reset_steps', [])
        if isinstance(reset_steps_raw, str):
            try:
                # Handle string format like "[2,30]" from environment variables
                import ast
                self.reset_steps = ast.literal_eval(reset_steps_raw)
                print(f"[LAYER_RESET_DEBUG] Parsed reset_steps from string '{reset_steps_raw}' to list {self.reset_steps}")
            except (ValueError, SyntaxError) as e:
                print(f"[LAYER_RESET_DEBUG] Error parsing reset_steps string '{reset_steps_raw}': {e}")
                print(f"[LAYER_RESET_DEBUG] Using empty list for reset_steps")
                self.reset_steps = []
        else:
            self.reset_steps = reset_steps_raw
        
        # Model-specific settings (both actor and critic use same k values)
        self.reset_actor = self.config.get('reset_actor', True)
        self.reset_critic = self.config.get('reset_critic', True)
        
        # Always reset optimizer states when layers are reset
        self.reset_optimizer_states = True
        
        # Enhanced debug logging
        print(f"[LAYER_RESET_DEBUG] LayerResetManager initialized:")
        print(f"[LAYER_RESET_DEBUG]   enable_reset={self.enable_reset}")
        print(f"[LAYER_RESET_DEBUG]   reset_k_first={self.reset_k_first}")
        print(f"[LAYER_RESET_DEBUG]   reset_k_last={self.reset_k_last}")
        print(f"[LAYER_RESET_DEBUG]   reset_steps={self.reset_steps}")
        print(f"[LAYER_RESET_DEBUG]   reset_actor={self.reset_actor}")
        print(f"[LAYER_RESET_DEBUG]   reset_critic={self.reset_critic}")
        
        logger.info(f"LayerResetManager initialized: enable_reset={self.enable_reset}, "
                   f"reset_k_first={self.reset_k_first}, reset_k_last={self.reset_k_last}, "
                   f"reset_steps={self.reset_steps}")
    
    def should_reset(self, global_step: int) -> bool:
        """
        Check if a reset should be performed at the given global step.
        
        Args:
            global_step: Current global training step
            
        Returns:
            True if reset should be performed, False otherwise
        """
        should_reset = self.enable_reset and global_step in self.reset_steps
        
        # Debug logging for every check
        if global_step % 10 == 0:  # Log every 10 steps to avoid spam
            print(f"[LAYER_RESET_DEBUG] Step {global_step}: enable_reset={self.enable_reset}, "
                  f"in_reset_steps={global_step in self.reset_steps}, should_reset={should_reset}")
        
        if should_reset:
            print(f"[LAYER_RESET_DEBUG] *** RESET TRIGGERED at global_step={global_step} ***")
            logger.info(f"Layer reset triggered at global step {global_step}")
        
        return should_reset
    
    def get_transformer_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """
        Extract transformer layers from a model.
        
        Args:
            model: The model to extract layers from
            
        Returns:
            List of transformer layer modules
        """
        # Handle FSDP wrapped models
        if isinstance(model, FSDP):
            base_model = model._fsdp_wrapped_module
        else:
            base_model = model
            
        # Common patterns for transformer layers in different architectures
        layer_patterns = [
            'model.layers',      # Llama, Qwen
            'transformer.h',     # GPT-2 style
            'transformer.layers', # Some other architectures
            'layers',            # Direct layers attribute
        ]
        
        for pattern in layer_patterns:
            try:
                layers = base_model
                for attr in pattern.split('.'):
                    layers = getattr(layers, attr)
                if isinstance(layers, (list, torch.nn.ModuleList)):
                    logger.info(f"Found {len(layers)} transformer layers using pattern '{pattern}'")
                    return list(layers)
            except AttributeError:
                continue
        
        # If no pattern matches, try to find layers by iterating through modules
        logger.warning("Could not find transformer layers using standard patterns, "
                      "attempting to find layers by module inspection")
        
        # Look for modules that contain multiple similar submodules (likely transformer layers)
        for name, module in base_model.named_modules():
            if hasattr(module, '__len__') and len(module) > 1:
                # Check if all children have similar structure (likely transformer layers)
                children = list(module.children())
                if len(children) > 1 and all(type(child) == type(children[0]) for child in children):
                    logger.info(f"Found {len(children)} potential transformer layers in '{name}'")
                    return children
        
        raise ValueError("Could not identify transformer layers in the model")
    
    def get_layer_indices_to_reset(self, total_layers: int) -> List[int]:
        """
        Get the indices of layers that should be reset, with boundary checking.
        
        Args:
            total_layers: Total number of transformer layers in the model
            
        Returns:
            List of layer indices to reset
        """
        indices_to_reset = []
        
        # Cap reset_k_first to total_layers if it exceeds
        reset_k_first_capped = min(self.reset_k_first, total_layers)
        reset_k_last_capped = min(self.reset_k_last, total_layers)
        
        # Add first k layers
        if reset_k_first_capped > 0:
            indices_to_reset.extend(range(reset_k_first_capped))
        
        # Add last k layers (avoid duplicates if first k and last k overlap)
        if reset_k_last_capped > 0:
            last_indices = range(total_layers - reset_k_last_capped, total_layers)
            indices_to_reset.extend(last_indices)
        
        # Remove duplicates and sort
        indices_to_reset = sorted(list(set(indices_to_reset)))
        
        if reset_k_first_capped != self.reset_k_first:
            logger.warning(f"reset_k_first capped from {self.reset_k_first} to {reset_k_first_capped} "
                          f"(total layers: {total_layers})")
        
        if reset_k_last_capped != self.reset_k_last:
            logger.warning(f"reset_k_last capped from {self.reset_k_last} to {reset_k_last_capped} "
                          f"(total layers: {total_layers})")
        
        logger.info(f"Will reset {len(indices_to_reset)} layers: {indices_to_reset}")
        return indices_to_reset
    
    def get_layer_parameter_names(self, model: torch.nn.Module, layer_indices: List[int]) -> Set[str]:
        """
        Get the parameter names for specified transformer layers.
        
        Args:
            model: The model to get parameter names from
            layer_indices: List of layer indices to get parameters for
            
        Returns:
            Set of parameter names for the specified layers
        """
        layers = self.get_transformer_layers(model)
        param_names = set()
        
        # Handle FSDP wrapped models
        if isinstance(model, FSDP):
            base_model = model._fsdp_wrapped_module
        else:
            base_model = model
        
        # Get the path to the layers container
        layer_container_path = None
        layer_patterns = ['model.layers', 'transformer.h', 'transformer.layers', 'layers']
        
        for pattern in layer_patterns:
            try:
                container = base_model
                for attr in pattern.split('.'):
                    container = getattr(container, attr)
                if isinstance(container, (list, torch.nn.ModuleList)) and len(container) == len(layers):
                    layer_container_path = pattern
                    break
            except AttributeError:
                continue
        
        if layer_container_path is None:
            raise ValueError("Could not determine layer container path")
        
        # Collect parameter names for specified layers
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                layer_prefix = f"{layer_container_path}.{layer_idx}."
                for name, _ in layers[layer_idx].named_parameters():
                    param_names.add(f"{layer_prefix}{name}")
        
        logger.info(f"Found {len(param_names)} parameters to reset in {len(layer_indices)} layers")
        return param_names
    
    def get_reference_layer_state_dict(self, ref_worker, layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Get state dict for specific layers from reference worker in a memory-efficient way.
        
        Args:
            ref_worker: Reference policy worker containing the reference model
            layer_indices: List of layer indices to extract
            
        Returns:
            Dictionary containing state dict for specified layers
        """
        print(f"[LAYER_RESET_DEBUG] Extracting layers {layer_indices} from reference worker")
        
        # Call the reference worker to extract specific layers
        # Use the proper Ray worker group interface
        layer_state_dict = ref_worker.extract_layers_for_reset(layer_indices)
        return layer_state_dict
    
    def reset_model_layers_from_ref(self, model: torch.nn.Module, ref_layer_state_dict: Dict[str, torch.Tensor],
                                   reset_k_first: int, reset_k_last: int,
                                   layer_indices: Optional[List[int]] = None) -> Set[str]:
        """
        Reset specified transformer layers of a model using a reference layer state dict.
        This is a memory-efficient version that uses pre-extracted reference layers.
        
        Args:
            model: Target model to reset layers for
            ref_layer_state_dict: Pre-extracted reference layer state dict (on CPU)
            reset_k_first: Number of first layers to reset
            reset_k_last: Number of last layers to reset
            
        Returns:
            Set of parameter names that were reset
        """
        reset_param_names = set()
        
        # Get transformer layers
        transformer_layers = self.get_transformer_layers(model)
        if not transformer_layers:
            print(f"[LAYER_RESET_DEBUG] No transformer layers found in model")
            return reset_param_names
            
        total_layers = len(transformer_layers)
        # Prefer explicitly provided indices, fallback to legacy k-first/last computation
        if layer_indices is not None:
            layers_to_reset = sorted({idx for idx in layer_indices if 0 <= idx < total_layers})
        else:
            layers_to_reset = self.get_layer_indices_to_reset(total_layers)
        
        print(f"[LAYER_RESET_DEBUG] Resetting layers {layers_to_reset} out of {total_layers} total layers")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        # Check if model is FSDP wrapped
        if hasattr(model, '_fsdp_wrapped_module') or isinstance(model, FSDP):
            print(f"[LAYER_RESET_DEBUG] Model is FSDP wrapped, using FSDP state dict context")
            
            # Use FSDP context to get unflattened parameters that match reference shapes
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
                # Get the full state dict with unflattened parameters
                target_state_dict = model.state_dict()
                
                # Copy parameters from reference to target state dict
                for ref_param_name, ref_param in ref_layer_state_dict.items():
                    if ref_param_name in target_state_dict:
                        print(f"[LAYER_RESET_DEBUG] Copying {ref_param_name} (ref: {ref_param.shape} -> target: {target_state_dict[ref_param_name].shape})")
                        
                        # Check shape compatibility
                        if target_state_dict[ref_param_name].shape != ref_param.shape:
                            print(f"[LAYER_RESET_DEBUG] ERROR: Shape mismatch for {ref_param_name}! Target: {target_state_dict[ref_param_name].shape}, Reference: {ref_param.shape}")
                            continue
                            
                        # Copy the parameter
                        target_state_dict[ref_param_name].data.copy_(ref_param.to(target_state_dict[ref_param_name].device, non_blocking=True))
                        reset_param_names.add(ref_param_name)
                        print(f"[LAYER_RESET_DEBUG] Successfully copied {ref_param_name}")
                    else:
                        print(f"[LAYER_RESET_DEBUG] WARNING: Reference parameter {ref_param_name} not found in target model")
                
                # Load the modified state dict back into the model
                model.load_state_dict(target_state_dict, strict=False)
                print(f"[LAYER_RESET_DEBUG] Loaded modified state dict back into FSDP model")
        else:
            print(f"[LAYER_RESET_DEBUG] Model is not FSDP wrapped, using direct parameter access")
            # Reset each specified layer using the reference state dict (original method)
            for layer_idx in layers_to_reset:
                target_layer = transformer_layers[layer_idx]
                layer_param_names = self._copy_layer_weights_from_state_dict(
                    ref_layer_state_dict, target_layer, layer_idx)
                reset_param_names.update(layer_param_names)
        
        print(f"[LAYER_RESET_DEBUG] Reset {len(reset_param_names)} parameters across {len(layers_to_reset)} layers")
        return reset_param_names
        ## Use FSDP state dict context to handle parameter flattening properly
        #from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        
        ## Check if model is FSDP wrapped
        #if hasattr(model, '_fsdp_wrapped_module') or isinstance(model, FSDP):
        #    print(f"[LAYER_RESET_DEBUG] Model is FSDP wrapped, using FSDP state dict context")
        #    
        #    # Use FSDP context to get unflattened parameters that match reference shapes
        #    # Add timeout and better error handling for collective operations
        #    print(f"[LAYER_RESET_DEBUG] Starting FSDP state_dict collective operation...")
        #    
        #    try:
        #        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
        #                                 FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        #            # Get the full state dict with unflattened parameters
        #            print(f"[LAYER_RESET_DEBUG] Calling model.state_dict()...")
        #            target_state_dict = model.state_dict()
        #            print(f"[LAYER_RESET_DEBUG] Successfully got target_state_dict with {len(target_state_dict)} parameters")
        #    except Exception as e:
        #        print(f"[LAYER_RESET_ERROR] FSDP state_dict failed: {e}")
        #        import traceback
        #        print(f"[LAYER_RESET_ERROR] Traceback: {traceback.format_exc()}")
        #        raise
        #    
        #    # All ranks should now have the same full reference parameters
        #    import torch.distributed as dist
        #    if dist.is_initialized():
        #        current_rank = dist.get_rank()
        #        print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Processing reset with reinitialization fallback")
        #        
        #        # Check if we have None values (reinitialization markers)
        #        has_reinit_markers = any(param is None for param in ref_layer_state_dict.values())
        #        
        #        if has_reinit_markers:
        #            print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Using reinitialization approach for marked layers")
        #            
        #            # Reinitialize the specified layers using their original initialization
        #            params_reset = 0
        #            for param_name, ref_param in ref_layer_state_dict.items():
        #                if ref_param is None:  # Reinitialization marker
        #                    # Find the actual parameter in the model
        #                    param_found = False
        #                    for name, param in model.named_parameters():
        #                        if name == param_name:
        #                            # Reinitialize using the same method as original initialization
        #                            if 'weight' in name and param.dim() >= 2:
        #                                torch.nn.init.xavier_uniform_(param)
        #                            elif 'bias' in name:
        #                                torch.nn.init.zeros_(param)
        #                            else:
        #                                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        #                            
        #                            params_reset += 1
        #                            param_found = True
        #                            print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Reinitialized parameter {param_name}")
        #                            break
        #                    
        #                    if not param_found:
        #                        print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Parameter {param_name} not found for reinitialization")
        #            
        #            print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Successfully reinitialized {params_reset} parameters")
        #            
        #        else:
        #            print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Starting FSDP collective state_dict call")
        #            
        #            # Use FSDP context for proper parameter handling
        #            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
        #                                     FullStateDictConfig(offload_to_cpu=False, rank0_only=False)):
        #                target_state_dict = model.state_dict()
        #                
        #                print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: FSDP collective state_dict call completed")
        #                print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Target state dict has {len(target_state_dict)} parameters")
        #                
        #                # Copy reference parameters to target state dict
        #                params_copied = 0
        #                for param_name, ref_param in ref_layer_state_dict.items():
        #                    if param_name in target_state_dict:
        #                        target_state_dict[param_name].copy_(ref_param)
        #                        params_copied += 1
        #                        print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Copied parameter {param_name}")
        #                    else:
        #                        print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Parameter {param_name} not found in target model")
        #                
        #                print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Starting model.load_state_dict call")
        #                
        #                # Load the updated state dict back to the model
        #                model.load_state_dict(target_state_dict)
        #                
        #                print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: model.load_state_dict completed")
        #                print(f"[LAYER_RESET_DEBUG] Rank {current_rank}: Successfully reset {params_copied} parameters")
        #        
        #    else:
        #        print(f"[LAYER_RESET_DEBUG] Processing {len(ref_layer_state_dict)} reference parameters (non-distributed)")
        #        
        #        # Copy parameters from reference to target state dict
        #        for ref_param_name, ref_param in ref_layer_state_dict.items():
        #            if ref_param_name in target_state_dict:
        #                if ref_param is None:
        #                    # None indicates reset to initialization
        #                    print(f"[LAYER_RESET_DEBUG] Reinitializing {ref_param_name} to default initialization")
        #                    target_param = target_state_dict[ref_param_name]
        #                    
        #                    # Reinitialize using common initialization schemes
        #                    if 'weight' in ref_param_name:
        #                        if len(target_param.shape) >= 2:
        #                            # Linear layer or similar - use Xavier/Glorot initialization
        #                            torch.nn.init.xavier_uniform_(target_param)
        #                        else:
        #                            # 1D parameter - use normal initialization
        #                            torch.nn.init.normal_(target_param, mean=0.0, std=0.02)
        #                    elif 'bias' in ref_param_name:
        #                        # Bias parameters - initialize to zero
        #                        torch.nn.init.zeros_(target_param)
        #                    else:
        #                        # Other parameters - use normal initialization
        #                        torch.nn.init.normal_(target_param, mean=0.0, std=0.02)
        #                    
        #                    reset_param_names.add(ref_param_name)
        #                    print(f"[LAYER_RESET_DEBUG] Successfully reinitialized {ref_param_name}")
        #                elif ref_param == "BROADCAST_FROM_RANK0":
        #                    # Skip broadcast markers - they were already handled above
        #                    continue
        #                else:
        #                    # Normal parameter copy
        #                    print(f"[LAYER_RESET_DEBUG] Copying {ref_param_name} (ref: {ref_param.shape} -> target: {target_state_dict[ref_param_name].shape})")
        #                    
        #                    # Check shape compatibility
        #                    if target_state_dict[ref_param_name].shape != ref_param.shape:
        #                        print(f"[LAYER_RESET_DEBUG] ERROR: Shape mismatch for {ref_param_name}! Target: {target_state_dict[ref_param_name].shape}, Reference: {ref_param.shape}")
        #                        continue
        #                        
        #                    # Copy the parameter
        #                    target_state_dict[ref_param_name].data.copy_(ref_param.to(target_state_dict[ref_param_name].device, non_blocking=True))
        #                    reset_param_names.add(ref_param_name)
        #                    print(f"[LAYER_RESET_DEBUG] Successfully copied {ref_param_name}")
        #            else:
        #                print(f"[LAYER_RESET_DEBUG] WARNING: Reference parameter {ref_param_name} not found in target model")
        #        
        #        # Load the modified state dict back into the model
        #        model.load_state_dict(target_state_dict, strict=False)
        #        print(f"[LAYER_RESET_DEBUG] Loaded modified state dict back into FSDP model")
        #else:
        #    print(f"[LAYER_RESET_DEBUG] Model is not FSDP wrapped, using direct parameter access")
        #    # Reset each specified layer using the reference state dict (original method)
        #    for layer_idx in layers_to_reset:
        #        target_layer = transformer_layers[layer_idx]
        #        layer_param_names = self._copy_layer_weights_from_state_dict(
        #            ref_layer_state_dict, target_layer, layer_idx)
        #        reset_param_names.update(layer_param_names)
        
        #print(f"[LAYER_RESET_DEBUG] Reset {len(reset_param_names)} parameters across {len(layers_to_reset)} layers")
        #return reset_param_names
    
    def reset_model_layers(self, 
                          current_model: torch.nn.Module,
                          reference_model: torch.nn.Module,
                          optimizer: Optional[torch.optim.Optimizer] = None,
                          model_name: str = "model") -> None:
        """
        Reset specified layers of current_model to match reference_model.
        
        Args:
            current_model: The model to reset layers in
            reference_model: The reference model to copy layers from
            optimizer: Optional optimizer to reset states for
            model_name: Name of the model for logging
        """
        if not self.enable_reset:
            return
        
        logger.info(f"Starting layer reset for {model_name}")
        
        # Get transformer layers from both models
        current_layers = self.get_transformer_layers(current_model)
        reference_layers = self.get_transformer_layers(reference_model)
        
        if len(current_layers) != len(reference_layers):
            raise ValueError(f"Layer count mismatch: current={len(current_layers)}, "
                           f"reference={len(reference_layers)}")
        
        # Get indices of layers to reset
        layer_indices = self.get_layer_indices_to_reset(len(current_layers))
        
        if not layer_indices:
            logger.info(f"No layers to reset for {model_name}")
            return
        
        # Handle FSDP models - need to use full state dict context
        if isinstance(current_model, FSDP) or isinstance(reference_model, FSDP):
            self._reset_fsdp_model_layers(current_model, reference_model, layer_indices, 
                                        optimizer, model_name)
        else:
            self._reset_regular_model_layers(current_model, reference_model, layer_indices,
                                           optimizer, model_name)
        
        logger.info(f"Successfully reset {len(layer_indices)} layers for {model_name}")
    
    def _reset_fsdp_model_layers(self,
                                current_model: FSDP,
                                reference_model: FSDP, 
                                layer_indices: List[int],
                                optimizer: Optional[torch.optim.Optimizer],
                                model_name: str) -> None:
        """Reset layers for FSDP wrapped models."""
        
        # Get parameter names to reset
        param_names_to_reset = self.get_layer_parameter_names(current_model, layer_indices)
        
        # Get full state dicts from both models
        with FSDP.state_dict_type(current_model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            current_state_dict = current_model.state_dict()
        
        with FSDP.state_dict_type(reference_model, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            reference_state_dict = reference_model.state_dict()
        
        # Copy specified layer parameters
        reset_count = 0
        for param_name in param_names_to_reset:
            if param_name in current_state_dict and param_name in reference_state_dict:
                current_state_dict[param_name].copy_(reference_state_dict[param_name])
                reset_count += 1
            else:
                logger.warning(f"Parameter {param_name} not found in one of the models")
        
        # Load the updated state dict back
        current_model.load_state_dict(current_state_dict)
        
        logger.info(f"Reset {reset_count} parameters for {model_name}")
        
        # Reset optimizer states if provided
        if optimizer is not None and self.reset_optimizer_states:
            self._reset_optimizer_states(optimizer, param_names_to_reset, model_name)
    
    def _reset_regular_model_layers(self,
                                   current_model: torch.nn.Module,
                                   reference_model: torch.nn.Module,
                                   layer_indices: List[int],
                                   optimizer: Optional[torch.optim.Optimizer],
                                   model_name: str) -> None:
        """Reset layers for regular (non-FSDP) models."""
        
        current_layers = self.get_transformer_layers(current_model)
        reference_layers = self.get_transformer_layers(reference_model)
        
        # Get parameter names to reset (for optimizer state reset)
        param_names_to_reset = self.get_layer_parameter_names(current_model, layer_indices)
        
        # Copy layer parameters
        reset_count = 0
        for layer_idx in layer_indices:
            current_layer = current_layers[layer_idx]
            reference_layer = reference_layers[layer_idx]
            
            for (curr_name, curr_param), (ref_name, ref_param) in zip(
                current_layer.named_parameters(), reference_layer.named_parameters()):
                
                if curr_name != ref_name:
                    logger.warning(f"Parameter name mismatch: {curr_name} vs {ref_name}")
                    continue
                
                curr_param.data.copy_(ref_param.data)
                reset_count += 1
        
        logger.info(f"Reset {reset_count} parameters for {model_name}")
        
        # Reset optimizer states if provided
        if optimizer is not None and self.reset_optimizer_states:
            self._reset_optimizer_states(optimizer, param_names_to_reset, model_name)
    
    def _copy_layer_weights_from_state_dict(self, ref_state_dict: Dict[str, torch.Tensor], 
                                           target_layer: torch.nn.Module, layer_idx: int) -> Set[str]:
        """
        Copy weights from reference state dict to target layer.
        Efficiently handles CPU-to-GPU transfer during layer reset.
        
        Args:
            ref_state_dict: Reference state dict containing layer weights (on CPU)
            target_layer: Target layer to copy weights to (on GPU)
            layer_idx: Index of the layer being reset
            
        Returns:
            Set of parameter names that were copied
        """
        copied_params = set()
        
        # Get the base name pattern for this layer
        layer_patterns = [f'model.layers.{layer_idx}', f'transformer.h.{layer_idx}', 
                         f'transformer.layers.{layer_idx}', f'layers.{layer_idx}']
        
        print(f"[LAYER_RESET_DEBUG] Copying weights for layer {layer_idx} from CPU to GPU...")
        
        # Debug: Show what keys are actually in the reference state dict
        ref_keys = list(ref_state_dict.keys())
        print(f"[LAYER_RESET_DEBUG] Reference state dict has {len(ref_keys)} keys")
        if ref_keys:
            print(f"[LAYER_RESET_DEBUG] Sample reference keys: {ref_keys[:5]}")
        else:
            print(f"[LAYER_RESET_DEBUG] WARNING: Reference state dict is empty!")
        
        for target_name, target_param in target_layer.named_parameters():
            param_found = False
            
            # Strip FSDP wrapper prefix from target parameter name
            clean_target_name = target_name
            if target_name.startswith('_fsdp_wrapped_module.'):
                clean_target_name = target_name[len('_fsdp_wrapped_module.'):]
                print(f"[LAYER_RESET_DEBUG] Stripped FSDP prefix: {target_name} -> {clean_target_name}")
            
            # Try different layer naming patterns
            for pattern in layer_patterns:
                ref_param_name = f"{pattern}.{clean_target_name}"
                print(f"[LAYER_RESET_DEBUG] Looking for reference parameter: {ref_param_name}")
                
                if ref_param_name in ref_state_dict:
                    print(f"[LAYER_RESET_DEBUG] Found reference parameter: {ref_param_name}")
                    ref_param = ref_state_dict[ref_param_name]
                    
                    # Efficiently transfer from CPU to GPU and copy
                    if ref_param.device != target_param.device:
                        # Transfer to target device (GPU) and copy in one operation
                        with torch.no_grad():
                            target_param.data.copy_(ref_param.to(target_param.device, non_blocking=True))
                    else:
                        # Same device, direct copy
                        with torch.no_grad():
                            target_param.data.copy_(ref_param.data)
                    
                    # Multi-GPU: Ensure all ranks have consistent weights after copy
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()
                        print(f"[LAYER_RESET_DEBUG] Multi-GPU: Synchronized parameter {target_name} across ranks")
                    
                    copied_params.add(ref_param_name)
                    param_found = True
                    print(f"[LAYER_RESET_DEBUG] Copied {target_name} ({ref_param.shape}) from CPU to GPU")
                    break
            
            if not param_found:
                logger.warning(f"Could not find reference parameter for {target_name} in layer {layer_idx}")
                print(f"[LAYER_RESET_DEBUG] Warning: Missing parameter {target_name} for layer {layer_idx}")
        
        # Clean up GPU memory after transfer
        torch.cuda.empty_cache()
        
        print(f"[LAYER_RESET_DEBUG] Copied {len(copied_params)} parameters for layer {layer_idx}")
        return copied_params
    
    def _reset_optimizer_states(self,
                               optimizer: torch.optim.Optimizer,
                               param_names_to_reset: Set[str],
                               model_name: str) -> None:
        """Reset optimizer states for specified parameters."""
        
        if not hasattr(optimizer, 'state'):
            logger.warning(f"Optimizer for {model_name} has no state attribute")
            return
        
        # Get parameter objects that correspond to the names we want to reset
        # This is tricky because optimizer.state keys are parameter objects, not names
        # We'll need to match by parameter identity
        
        reset_state_count = 0
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param in optimizer.state:
                    # Clear the state for this parameter
                    # This will reset momentum, variance estimates, etc.
                    del optimizer.state[param]
                    reset_state_count += 1
        
        logger.info(f"Reset optimizer states for {reset_state_count} parameters in {model_name}")


def create_layer_reset_manager(config: Optional[Dict[str, Any]] = None) -> LayerResetManager:
    """
    Factory function to create a LayerResetManager instance.
    
    Args:
        config: Configuration dictionary with reset parameters
        
    Returns:
        LayerResetManager instance
    """
    return LayerResetManager(config)
