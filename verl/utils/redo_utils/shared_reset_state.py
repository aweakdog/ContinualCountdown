"""
Shared reset state management using Ray for actor-critic synchronization.
"""

import ray
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@ray.remote
class SharedResetState:
    """Ray actor for sharing reset layer information between actor and critic workers."""
    
    def __init__(self):
        self.reset_layers = {}  # step -> layers mapping
        self.reset_metadata = {}  # step -> metadata mapping
        
    def set_reset_layers(self, global_step: int, selected_layers: List[int], 
                        strategy: str = None, metadata: Dict = None):
        """
        Store reset layers for a given global step.
        
        Args:
            global_step: Training step number
            selected_layers: List of layer indices to reset
            strategy: Reset strategy used
            metadata: Additional metadata about the reset
        """
        self.reset_layers[global_step] = selected_layers
        self.reset_metadata[global_step] = {
            'strategy': strategy,
            'metadata': metadata or {},
            'timestamp': ray.util.get_current_time_ms()
        }
        
        logger.info(f"[SharedResetState] Stored reset layers for step {global_step}: {selected_layers}")
        
    def get_reset_layers(self, global_step: int) -> Optional[List[int]]:
        """
        Retrieve reset layers for a given global step.
        
        Args:
            global_step: Training step number
            
        Returns:
            List of layer indices or None if not found
        """
        layers = self.reset_layers.get(global_step)
        if layers:
            logger.info(f"[SharedResetState] Retrieved reset layers for step {global_step}: {layers}")
        else:
            logger.warning(f"[SharedResetState] No reset layers found for step {global_step}")
        return layers
        
    def get_reset_metadata(self, global_step: int) -> Optional[Dict]:
        """Get metadata for a reset at given step."""
        return self.reset_metadata.get(global_step)
        
    def cleanup_old_entries(self, current_step: int, keep_last_n: int = 10):
        """Clean up old entries to prevent memory growth."""
        steps_to_remove = []
        all_steps = sorted(self.reset_layers.keys())
        
        if len(all_steps) > keep_last_n:
            steps_to_remove = all_steps[:-keep_last_n]
            
        for step in steps_to_remove:
            if step < current_step - keep_last_n:  # Only remove old steps
                self.reset_layers.pop(step, None)
                self.reset_metadata.pop(step, None)
                
        if steps_to_remove:
            logger.info(f"[SharedResetState] Cleaned up {len(steps_to_remove)} old entries")
            
    def get_status(self) -> Dict:
        """Get current status of shared state."""
        return {
            'total_entries': len(self.reset_layers),
            'latest_steps': sorted(self.reset_layers.keys())[-5:] if self.reset_layers else [],
            'memory_usage_estimate': len(str(self.reset_layers)) + len(str(self.reset_metadata))
        }


class SharedResetStateManager:
    """Manager for shared reset state with singleton pattern."""
    
    _instance = None
    _shared_state = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._shared_state = None
            
    def get_shared_state(self):
        """Get or create the shared reset state Ray actor."""
        if self._shared_state is None:
            try:
                # Try to get existing actor first
                self._shared_state = ray.get_actor("shared_reset_state")
                logger.info("[SharedResetStateManager] Connected to existing shared reset state")
            except ValueError:
                # Create new actor if doesn't exist
                self._shared_state = SharedResetState.options(
                    name="shared_reset_state",
                    lifetime="detached"  # Survive driver restarts
                ).remote()
                logger.info("[SharedResetStateManager] Created new shared reset state")
                
        return self._shared_state
    
    def save_actor_reset_layers(self, global_step: int, selected_layers: List[int], 
                               strategy: str = None, metadata: Dict = None):
        """Save actor's reset layers to shared state."""
        try:
            shared_state = self.get_shared_state()
            ray.get(shared_state.set_reset_layers.remote(
                global_step, selected_layers, strategy, metadata
            ))
            logger.info(f"[SharedResetStateManager] Saved reset layers for step {global_step}")
        except Exception as e:
            logger.error(f"[SharedResetStateManager] Failed to save reset layers: {e}")
            
    def load_actor_reset_layers(self, global_step: int) -> Optional[List[int]]:
        """Load actor's reset layers from shared state."""
        try:
            shared_state = self.get_shared_state()
            layers = ray.get(shared_state.get_reset_layers.remote(global_step))
            return layers
        except Exception as e:
            logger.error(f"[SharedResetStateManager] Failed to load reset layers: {e}")
            return None
            
    def cleanup_old_entries(self, current_step: int):
        """Clean up old entries."""
        try:
            shared_state = self.get_shared_state()
            ray.get(shared_state.cleanup_old_entries.remote(current_step))
        except Exception as e:
            logger.error(f"[SharedResetStateManager] Failed to cleanup: {e}")
