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
C_K-Based Reset Manager for Intelligent Layer Reset

This module provides utilities to reset transformer layers based on Fisher Information
C_K values, supporting multiple reset strategies for experimental comparison.
"""

import logging
import torch
import numpy as np
import random
from typing import List, Optional, Dict, Any, Set, Tuple
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils.redo_utils.layer_reset import LayerResetManager

logger = logging.getLogger(__name__)


class CKBasedResetManager(LayerResetManager):
    """
    Manages layer reset operations based on Fisher Information C_K values.
    
    Supports multiple reset strategies:
    - ck_guided: Reset layers with highest C_K weighted values
    - random: Reset randomly selected layers
    - first_k: Reset first K layers (traditional approach)
    - last_k: Reset last K layers (traditional approach)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CKBasedResetManager.
        
        Args:
            config: Configuration dictionary with reset parameters
        """
        super().__init__(config)
        
        # C_K-based reset settings
        self.reset_strategy = self.config.get('reset_strategy', 'ck_guided')  # ck_guided, random, first_k, last_k
        self.reset_k_layers = self.config.get('reset_k_layers', 4)  # Number of layers to reset
        self.ck_history_window = self.config.get('ck_history_window', 20)  # Steps to average C_K over (sliding window)
        
        # Logging and tracking
        self.reset_history = []  # Track which layers were reset at each step
        self.ck_layer_rankings = {}  # Store C_K rankings for analysis
        
        # Random seed for reproducible random reset
        self.random_seed = self.config.get('random_seed', 42)
        random.seed(self.random_seed)
        
        logger.info(f"CKBasedResetManager initialized: strategy={self.reset_strategy}, "
                   f"reset_k_layers={self.reset_k_layers}, history_window={self.ck_history_window}")
        
        print(f"[CK_RESET_DEBUG] CKBasedResetManager initialized:")
        print(f"[CK_RESET_DEBUG]   reset_strategy={self.reset_strategy}")
        print(f"[CK_RESET_DEBUG]   reset_k_layers={self.reset_k_layers}")
        print(f"[CK_RESET_DEBUG]   ck_history_window={self.ck_history_window}")
        print(f"[CK_RESET_DEBUG]   random_seed={self.random_seed}")
    
    def calculate_layer_ck_weights(self, fisher_stats: Dict[str, Any], 
                                  original_param_shapes: Dict[str, torch.Size]) -> Dict[int, float]:
        """
        Calculate C_K weighted values for each transformer layer.
        
        Uses the same weighting formula as Fisher analyzer:
        layer_c_k_weighted = sum(param_c_k * param_count) / layer_total_params
        
        Args:
            fisher_stats: Fisher analysis results from analyzer
            original_param_shapes: Original parameter shapes before FSDP
            
        Returns:
            Dictionary mapping layer_index -> weighted_c_k_value
        """
        layer_weights = {}
        
        # Debug: Check if original_param_shapes is provided and populated
        print(f"[CK_WEIGHT_DEBUG] original_param_shapes provided: {original_param_shapes is not None}")
        if original_param_shapes:
            print(f"[CK_WEIGHT_DEBUG] original_param_shapes contains {len(original_param_shapes)} parameters")
            # Show first few parameter names as examples
            example_params = list(original_param_shapes.keys())[:3]
            print(f"[CK_WEIGHT_DEBUG] Example parameter names: {example_params}")
        else:
            print(f"[CK_WEIGHT_DEBUG] original_param_shapes is empty or None!")
        
        # Get Fisher stats - handle both old and new formats
        print(f"[CK_WEIGHT_DEBUG] Fisher stats keys: {list(fisher_stats.keys())}")
        print(f"[CK_WEIGHT_DEBUG] Fisher stats structure:")
        for key, value in fisher_stats.items():
            if isinstance(value, dict):
                print(f"[CK_WEIGHT_DEBUG]   {key}: dict with {len(value)} keys - {list(value.keys())[:3]}...")
                # Show first level structure
                for subkey, subvalue in list(value.items())[:2]:
                    if isinstance(subvalue, dict):
                        print(f"[CK_WEIGHT_DEBUG]     {subkey}: dict with {len(subvalue)} keys - {list(subvalue.keys())[:3]}...")
                    else:
                        print(f"[CK_WEIGHT_DEBUG]     {subkey}: {type(subvalue).__name__}")
            else:
                print(f"[CK_WEIGHT_DEBUG]   {key}: {type(value).__name__} = {value}")
        
        # Try new format first: fisher_stats['components']['layer_X']['params']
        components_fisher = fisher_stats.get('components', {})
        if components_fisher:
            print(f"[CK_WEIGHT_DEBUG] Using new format - Components found: {len(components_fisher)} - {list(components_fisher.keys())[:5]}...")
        else:
            # Try old format: fisher_stats['params']['component_name']
            params_data = fisher_stats.get('params', {})
            if params_data:
                print(f"[CK_WEIGHT_DEBUG] Using old format - Params found: {len(params_data)} - {list(params_data.keys())[:5]}...")
                # Convert old format to new format
                components_fisher = {}
                for component_name, param_dict in params_data.items():
                    if component_name.startswith('layer_'):
                        components_fisher[component_name] = {'params': param_dict}
            else:
                print(f"[CK_WEIGHT_DEBUG] No Fisher stats found in either format")
                return layer_weights
        
        # Group parameters by transformer layer
        layer_param_groups = {}  # layer_idx -> [(param_name, c_k, param_count), ...]
        
        for component_name, comp_stats in components_fisher.items():
            print(f"[CK_WEIGHT_DEBUG] Processing component: {component_name}")
            # Extract layer index from component name (e.g., "layer_0" -> 0)
            if component_name.startswith('layer_'):
                try:
                    layer_idx = int(component_name.split('_')[1])
                    layer_param_groups.setdefault(layer_idx, [])
                    
                    params_stats = comp_stats.get('params', comp_stats)  # Handle both formats
                    print(f"[CK_WEIGHT_DEBUG] Layer {layer_idx}: found {len(params_stats)} params")
                    
                    for param_name, param_metrics in params_stats.items():
                        c_k = param_metrics.get('c_k', 0.0)
                        
                        # Get parameter count from original shapes
                        param_shape = original_param_shapes.get(param_name)
                        if param_shape:
                            param_count = param_shape.numel()
                            layer_param_groups[layer_idx].append((param_name, c_k, param_count))
                            print(f"[CK_WEIGHT_DEBUG] Layer {layer_idx}: {param_name} c_k={c_k:.6f} count={param_count}")
                        else:
                            # Fallback: estimate from parameter name patterns
                            print(f"[CK_WEIGHT_DEBUG] No original shape found for {param_name}, using fallback estimation")
                            param_count = self._estimate_param_count_from_name(param_name)
                            if param_count > 0:
                                layer_param_groups[layer_idx].append((param_name, c_k, param_count))
                                print(f"[CK_WEIGHT_DEBUG] Layer {layer_idx}: {param_name} c_k={c_k:.6f} count={param_count} (estimated)")
                            else:
                                print(f"[CK_WEIGHT_DEBUG] Could not estimate parameter count for {param_name}, skipping")
                        
                except (ValueError, IndexError):
                    print(f"[CK_WEIGHT_DEBUG] Could not parse layer index from component: {component_name}")
                    continue
            else:
                print(f"[CK_WEIGHT_DEBUG] Skipping non-layer component: {component_name}")
        
        # Calculate weighted C_K for each layer
        for layer_idx, param_list in layer_param_groups.items():
            if not param_list:
                continue
                
            total_weighted_ck = 0.0
            total_param_count = 0
            
            for param_name, c_k, param_count in param_list:
                total_weighted_ck += c_k * param_count
                total_param_count += param_count
            
            if total_param_count > 0:
                layer_c_k_weighted = total_weighted_ck / total_param_count
                layer_weights[layer_idx] = layer_c_k_weighted
                
                print(f"[CK_WEIGHT_DEBUG] Layer {layer_idx}: weighted_c_k={layer_c_k_weighted:.4f} "
                      f"(total_params={total_param_count}, weighted_sum={total_weighted_ck:.2f})")
        
        return layer_weights
    
    def _estimate_param_count_from_name(self, param_name: str) -> int:
        """
        Estimate parameter count from parameter name patterns.
        This is a fallback when original shapes are not available.
        """
        # Common parameter size patterns for transformer models
        # These are rough estimates based on typical model architectures
        
        if 'embed_tokens.weight' in param_name:
            # Embedding: vocab_size * hidden_size (e.g., 32000 * 3072 for Qwen2.5-0.5B)
            return 32000 * 3072
        elif 'lm_head' in param_name:
            # Language model head: hidden_size * vocab_size
            return 3072 * 32000
        elif 'q_proj.weight' in param_name or 'k_proj.weight' in param_name or 'v_proj.weight' in param_name:
            # Attention projections: hidden_size * (hidden_size or head_dim * num_heads)
            if 'k_proj' in param_name or 'v_proj' in param_name:
                return 3072 * 1024  # For key/value projections (often smaller)
            else:
                return 3072 * 3072  # For query projections
        elif 'o_proj.weight' in param_name:
            # Output projection: hidden_size * hidden_size
            return 3072 * 3072
        elif 'gate_proj.weight' in param_name or 'up_proj.weight' in param_name:
            # MLP gate/up projections: hidden_size * intermediate_size
            return 3072 * 8192  # 8192 is typical intermediate size
        elif 'down_proj.weight' in param_name:
            # MLP down projection: intermediate_size * hidden_size
            return 8192 * 3072
        elif 'layernorm.weight' in param_name or 'input_layernorm.weight' in param_name or 'post_attention_layernorm.weight' in param_name:
            # Layer norm weights: hidden_size
            return 3072
        else:
            # Unknown parameter type, return 0 to skip
            print(f"[CK_WEIGHT_DEBUG] Unknown parameter type for estimation: {param_name}")
            return 0

    def _select_random_layers(self, total_layers: int, global_step: int) -> List[int]:
        """
        Select random layers for reset using deterministic random selection.
        Uses global_step as seed for reproducibility across actor/critic.
        """
        import random
        random.seed(global_step)  # Use global step as seed for reproducibility
        available_layers = list(range(total_layers))
        selected_layers = random.sample(available_layers, min(self.reset_k_layers, total_layers))
        return sorted(selected_layers)
        # Select k random layers
        k_layers = min(self.reset_k_layers, total_layers)
        selected_layers = rng.sample(range(total_layers), k_layers)
        
        print(f"[CK_RESET_SELECTION] Random selection (seed={seed}): layers {selected_layers}")
        return selected_layers
    
    def select_layers_to_reset(self, total_layers: int, layer_ck_weights: Dict[int, float], global_step: int, 
                              force_random_for_sync: bool = False) -> List[int]:
        """
        Select layers to reset based on strategy and C_K weights.
        
        Args:
            total_layers: Total number of transformer layers
            layer_ck_weights: Dictionary mapping layer_index -> weighted_c_k_value
            global_step: Current global training step
            force_random_for_sync: If True, force random selection for actor-critic sync
            
        Returns:
            List of layer indices to reset
        """
        if self.reset_strategy == 'ck_guided':
            if layer_ck_weights and not force_random_for_sync:
                # Sort layers by C_K weights (descending - highest C_K first)
                sorted_layers = sorted(layer_ck_weights.items(), key=lambda x: x[1], reverse=True)
                selected_layers = [layer_idx for layer_idx, _ in sorted_layers[:self.reset_k_layers]]
                
                print(f"[CK_RESET_SELECTION] C_K guided selection:")
                for i, (layer_idx, weight) in enumerate(sorted_layers[:self.reset_k_layers]):
                    print(f"[CK_RESET_SELECTION]   Layer {layer_idx}: C_K weight = {weight:.6f}")
                    
                return selected_layers
            else:
                if force_random_for_sync:
                    print(f"[CK_RESET_SELECTION] Using synchronized random selection for actor-critic consistency")
                else:
                    print(f"[CK_RESET_SELECTION] No C_K weights available, falling back to random selection")
                # Use synchronized random selection
                return self._select_random_layers(total_layers, global_step)
        
        elif self.reset_strategy == 'random':
            return self._select_random_layers(total_layers, global_step)
        elif self.reset_strategy == 'first_k':
            return list(range(min(self.reset_k_layers, total_layers)))
        elif self.reset_strategy == 'last_k':
            start_idx = max(0, total_layers - self.reset_k_layers)
            return list(range(start_idx, total_layers))
        
        else:
            raise ValueError(f"Unknown reset strategy: {self.reset_strategy}")
    
    def get_layer_indices_to_reset(self, total_layers: int, 
                                  layer_ck_weights: Dict[int, float] = None,
                                  global_step: int = None) -> List[int]:
        """
        Override parent method to use strategy-based layer selection.
        
        Args:
            total_layers: Total number of transformer layers
            layer_ck_weights: C_K weighted values for each layer
            global_step: Current global step
            
        Returns:
            List of layer indices to reset
        """
        return self.select_layers_to_reset(total_layers, layer_ck_weights, global_step)
    
    def should_reset_with_ck_analysis(self, global_step: int, 
                                     fisher_stats: Dict[str, Any] = None) -> Tuple[bool, Dict[int, float]]:
        """
        Enhanced reset decision that considers both global step and C_K analysis availability.
        
        Args:
            global_step: Current global training step
            fisher_stats: Fisher analysis results (optional)
            
        Returns:
            Tuple of (should_reset, layer_ck_weights)
        """
        should_reset = self.should_reset(global_step)
        layer_ck_weights = {}
        
        # Debug: Print detailed reset decision info
        print(f"[CK_RESET_DEBUG] Step {global_step}: Reset decision details:")
        print(f"[CK_RESET_DEBUG]   enable_reset={self.enable_reset}")
        print(f"[CK_RESET_DEBUG]   reset_steps={self.reset_steps}")
        print(f"[CK_RESET_DEBUG]   step_in_reset_steps={global_step in self.reset_steps}")
        print(f"[CK_RESET_DEBUG]   should_reset={should_reset}")
        print(f"[CK_RESET_DEBUG]   reset_strategy={self.reset_strategy}")
        print(f"[CK_RESET_DEBUG]   has_fisher_stats={fisher_stats is not None}")
        
        if should_reset and fisher_stats and self.reset_strategy == 'ck_guided':
            # Only calculate C_K weights if we're using ck_guided strategy
            layer_ck_weights = self.calculate_layer_ck_weights(fisher_stats, {})
            
            if not layer_ck_weights:
                logger.warning(f"Step {global_step}: No C_K weights calculated, but reset is scheduled. "
                              f"Will proceed with fallback strategy.")
        
        return should_reset, layer_ck_weights
    
    def reset_model_layers_with_ck_guidance(self,
                                           current_model: torch.nn.Module,
                                           ref_layer_state_dict: Dict[str, torch.Tensor],
                                           layer_ck_weights: Dict[int, float],
                                           global_step: int,
                                           optimizer: Optional[torch.optim.Optimizer] = None,
                                           model_name: str = "model") -> Set[str]:
        """
        Reset model layers using C_K guidance or other strategies.
        
        Args:
            current_model: The model to reset layers in
            ref_layer_state_dict: Pre-extracted reference layer state dict
            layer_ck_weights: C_K weighted values for each layer
            global_step: Current global step
            optimizer: Optional optimizer to reset states for
            model_name: Name of the model for logging
            
        Returns:
            Set of parameter names that were reset
        """
        if not self.enable_reset:
            return set()
        
        logger.info(f"Starting C_K-guided layer reset for {model_name} at step {global_step}")
        
        # Get transformer layers
        transformer_layers = self.get_transformer_layers(current_model)
        total_layers = len(transformer_layers)
        
        # Select layers to reset based on strategy
        layer_indices = self.select_layers_to_reset(total_layers, layer_ck_weights, global_step)
        
        if not layer_indices:
            logger.info(f"No layers selected for reset in {model_name}")
            return set()
        
        # Override the get_layer_indices_to_reset method temporarily for dynamic selection
        original_method = self.get_layer_indices_to_reset
        self.selected_layer_indices = layer_indices
        
        def dynamic_get_layer_indices(total_layers_arg):
            return self.selected_layer_indices
        
        self.get_layer_indices_to_reset = dynamic_get_layer_indices
        
        try:
            # Perform the actual reset using parent class method
            reset_param_names = self.reset_model_layers_from_ref(
                current_model, ref_layer_state_dict, 
                reset_k_first=0, reset_k_last=0  # Not used in dynamic selection
            )
        finally:
            # Restore original method
            self.get_layer_indices_to_reset = original_method
            if hasattr(self, 'selected_layer_indices'):
                delattr(self, 'selected_layer_indices')
        
        # Log reset summary
        logger.info(f"Successfully reset {len(layer_indices)} layers in {model_name}: {layer_indices}")
        print(f"[CK_RESET_SUMMARY] Step {global_step}: Reset {len(layer_indices)} layers using "
              f"'{self.reset_strategy}' strategy")
        print(f"[CK_RESET_SUMMARY] Reset layers: {layer_indices}")
        print(f"[CK_RESET_SUMMARY] Reset {len(reset_param_names)} parameters")
        
        return reset_param_names
    
    def get_reset_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete reset history for analysis.
        
        Returns:
            List of reset records with step, strategy, and layer information
        """
        return self.reset_history.copy()
    
    def get_ck_rankings_history(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get the C_K layer rankings history for analysis.
        
        Returns:
            Dictionary mapping global_step -> [(layer_idx, c_k_weight), ...]
        """
        return self.ck_layer_rankings.copy()
    
    def log_reset_statistics(self, global_step: int = None):
        """
        Log comprehensive reset statistics for analysis.
        
        Args:
            global_step: Current global step (optional)
        """
        if not self.reset_history:
            logger.info("No reset history available")
            return
        
        logger.info(f"=== Reset Statistics Summary ===")
        logger.info(f"Total resets performed: {len(self.reset_history)}")
        logger.info(f"Reset strategy: {self.reset_strategy}")
        logger.info(f"Layers per reset: {self.reset_k_layers}")
        
        # Analyze layer reset frequency
        layer_reset_counts = {}
        for record in self.reset_history:
            for layer_idx in record['reset_layers']:
                layer_reset_counts[layer_idx] = layer_reset_counts.get(layer_idx, 0) + 1
        
        if layer_reset_counts:
            logger.info(f"Layer reset frequency:")
            for layer_idx in sorted(layer_reset_counts.keys()):
                count = layer_reset_counts[layer_idx]
                logger.info(f"  Layer {layer_idx}: {count} times")
        
        # Log recent resets
        recent_resets = self.reset_history[-3:] if len(self.reset_history) >= 3 else self.reset_history
        logger.info(f"Recent resets:")
        for record in recent_resets:
            step = record['global_step']
            layers = record['reset_layers']
            logger.info(f"  Step {step}: {layers}")
        
        logger.info(f"=== End Reset Statistics ===")


def create_ck_based_reset_manager(config: Optional[Dict[str, Any]] = None) -> CKBasedResetManager:
    """
    Factory function to create a CKBasedResetManager instance.
    
    Args:
        config: Configuration dictionary with reset parameters
        
    Returns:
        CKBasedResetManager instance
    """
    return CKBasedResetManager(config)
