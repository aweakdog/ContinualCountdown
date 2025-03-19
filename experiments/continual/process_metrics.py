"""Process and aggregate metrics from training runs."""

import os
import json
import glob
import re
import pandas as pd
from typing import Dict, List, Tuple
from rich import print as rprint


def parse_experiment_name(name: str) -> Tuple[int, str]:
    """Parse round number and group from experiment name."""
    round_match = re.search(r'round(\d+)', name)
    group_match = re.search(r'_(plus(?:_minus)?(?:_mul)?(?:_div)?)$', name)
    
    round_num = int(round_match.group(1)) if round_match else 0
    group = group_match.group(1) if group_match else "unknown"
    
    return round_num, group


def load_wandb_metrics(run_dir: str) -> Dict:
    """Load metrics from a WandB run directory."""
    metrics = {
        "success_rate": [],
        "weight_change": [],
        "loss": [],
        "gradient_norm": [],
        "response_length": []
    }
    
    try:
        # Load WandB metrics files
        metrics_files = glob.glob(os.path.join(run_dir, "*.jsonl"))
        if not metrics_files:
            return metrics
            
        # Read all metrics files
        all_metrics = []
        for file in metrics_files:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "_step" in data:
                            # Extract round and group from run name
                            if "wandb" in data and "run" in data["wandb"]:
                                round_num, group = parse_experiment_name(data["wandb"]["run"]["name"])
                                data["round"] = round_num
                                data["group"] = group
                            all_metrics.append(data)
                    except json.JSONDecodeError:
                        continue
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_metrics)
        if df.empty:
            return metrics
            
        # Add round and group if not present
        if "round" not in df.columns:
            df["round"] = 0
        if "group" not in df.columns:
            df["group"] = "unknown"
        
        # Group by round and group, then aggregate
        grouped = df.groupby(["round", "group"])
        
        # Extract relevant metrics
        for key in metrics:
            if key in df.columns:
                metrics[key] = grouped[key].mean().to_dict()
            
    except Exception as e:
        rprint(f"[red]Error loading metrics: {e}[/red]")
    
    return metrics


def process_metrics(base_dir: str = "/data/continual") -> None:
    """Process and aggregate metrics from all training runs."""
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Initialize metrics storage with round and group structure
    aggregated_metrics = {
        "success_rate": {},
        "weight_change": {},
        "loss": {},
        "gradient_norm": {},
        "response_length": {}
    }
    
    # Load metrics from all WandB runs
    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
        for run_dir in run_dirs:
            metrics = load_wandb_metrics(run_dir)
            for key in aggregated_metrics:
                if metrics[key]:
                    # Merge dictionaries, preserving round and group structure
                    aggregated_metrics[key].update(metrics[key])
    
    # Save aggregated metrics
    output_path = os.path.join(metrics_dir, "summary.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    rprint(f"[green]Saved aggregated metrics to {output_path}[/green]")


if __name__ == "__main__":
    process_metrics()
