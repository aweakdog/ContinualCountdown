#!/usr/bin/env python3
"""
Configuration for unified analyzer metrics storage.
This script helps set up the storage directories and provides utilities.
"""

import os
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "base_dir": "./analyzer_metrics",
    "enable_wandb_detailed": True,
    "enable_csv_export": True,
    "auto_save_interval": 10,  # Save summary every N steps
    "max_file_size_mb": 100,   # Rotate files when they exceed this size
}

def setup_analyzer_storage(experiment_name: str = None, base_dir: str = None):
    """
    Set up analyzer storage directories and return configuration.
    
    Args:
        experiment_name: Name of the experiment (auto-generated if None)
        base_dir: Base directory for storage (uses default if None)
    
    Returns:
        dict: Configuration for analyzer storage
    """
    if experiment_name is None:
        from datetime import datetime
        experiment_name = f"continual_countdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if base_dir is None:
        base_dir = DEFAULT_CONFIG["base_dir"]
    
    # Create directories
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    (base_path / "gradients").mkdir(exist_ok=True)
    (base_path / "fisher").mkdir(exist_ok=True)
    (base_path / "exports").mkdir(exist_ok=True)
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "experiment_name": experiment_name,
        "base_dir": str(base_path),
        "gradient_dir": str(base_path / "gradients"),
        "fisher_dir": str(base_path / "fisher"),
        "export_dir": str(base_path / "exports"),
    })
    
    # Save config
    config_file = base_path / "storage_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[AnalyzerStorage] Setup complete:")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Base directory: {base_path}")
    print(f"  - Config saved to: {config_file}")
    
    return config

def get_wandb_storage_location():
    """Find WandB storage location."""
    wandb_dirs = []
    
    # Check common locations
    common_locations = ["./wandb", "~/wandb", "./logs/wandb"]
    
    for location in common_locations:
        path = Path(location).expanduser()
        if path.exists():
            wandb_dirs.append(str(path))
    
    return wandb_dirs

if __name__ == "__main__":
    print("=== Analyzer Storage Setup ===")
    
    # Setup storage
    config = setup_analyzer_storage()
    
    # Check WandB locations
    wandb_locations = get_wandb_storage_location()
    if wandb_locations:
        print(f"\n[WandB] Found WandB directories:")
        for loc in wandb_locations:
            print(f"  - {loc}")
    else:
        print(f"\n[WandB] No existing WandB directories found")
        print(f"[WandB] WandB will create new directories when logging starts")
    
    print(f"\n=== Summary ===")
    print(f"✅ Analyzer metrics will be stored in: {config['base_dir']}")
    print(f"✅ Detailed component/matrix analysis will be saved to JSON")
    print(f"✅ WandB integration enabled for hierarchical metrics")
    print(f"✅ CSV export available for post-processing")
