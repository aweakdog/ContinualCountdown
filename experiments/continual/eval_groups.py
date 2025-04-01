"""Evaluation script for tracking per-group success rates in Countdown task."""

import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import glob
import argparse

# Disable WandB logging
os.environ['WANDB_MODE'] = 'disabled'

def extract_group_metrics_from_log(log_file: str) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """Extract metrics from a training log file, tracking per-group training and test success rates."""
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return {}
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Remove ANSI color codes
    content = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', content)
    
    # Split content by training start markers to get separate runs
    runs = re.split(r'Starting curriculum training at.*?\n', content)
    if len(runs) > 1:
        print(f"Found {len(runs)-1} training runs in log file")
        # Use the last run (most recent)
        content = runs[-1]
    
    # Track metrics for each group
    groups = ['plus_minus_mul', 'plus_minus_div', 'minus_mul_div', 'plus_mul_div']
    group_metrics = {group: {'train': [], 'test': []} for group in groups}
    
    # Extract training and test success rates for each group
    for group in groups:
        # Look for test metrics
        test_pattern = rf'test/{group}/success_rate:([0-9.]+)'
        test_matches = re.finditer(test_pattern, content)
        
        for match in test_matches:
            try:
                success_rate = float(match.group(1))
                group_metrics[group]['test'].append({
                    'step': len(group_metrics[group]['test']) + 1,
                    'success_rate': success_rate
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse test success rate for group {group}: {e}")
                continue
                
        # Look for training metrics
        train_pattern = rf'train/{group}/success_rate:([0-9.]+)'
        train_matches = re.finditer(train_pattern, content)
        
        for match in train_matches:
            try:
                success_rate = float(match.group(1))
                group_metrics[group]['train'].append({
                    'step': len(group_metrics[group]['train']) + 1,
                    'success_rate': success_rate
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse training success rate for group {group}: {e}")
                continue
    
    return group_metrics


class GroupEvaluator:
    def __init__(self, model_size: str = "0.5b", metrics_dir: str = "./metrics", plots_dir: str = "./plots", logs_dir: str = "./logs"):
        self.groups = ["plus_minus_mul", "plus_minus_div", "minus_mul_div", "plus_mul_div"]
        self.metrics_dir = os.getenv("METRICS_DIR", metrics_dir)
        self.plots_dir = os.getenv("PLOTS_DIR", plots_dir)
        self.logs_dir = os.getenv("LOGS_DIR", logs_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.model_size = model_size.lower()
        self.group_metrics = {}

    def plot_group_metrics(self, group_metrics: Dict[str, Dict[str, List[Dict]]], save_path: Optional[str] = None):
        """Create plots for per-group training and test success rates."""
        if not group_metrics:
            print("No metrics to plot")
            return
            
        plt.figure(figsize=(15, 10))
        plt.title(f'Training and Test Success Rates by Group - {self.model_size.upper()} Model', fontsize=14)
        
        colors = ['b', 'g', 'r', 'm']  # Colors for each group
        
        for group, color in zip(self.groups, colors):
            if group in group_metrics:
                # Plot training data with dashed lines
                if group_metrics[group]['train']:
                    steps = [m['step'] for m in group_metrics[group]['train']]
                    rates = [m['success_rate'] for m in group_metrics[group]['train']]
                    plt.plot(steps, rates, f'{color}--', label=f'{group} (train)',
                            linewidth=2, alpha=0.5)
                
                # Plot test data with solid lines and markers
                if group_metrics[group]['test']:
                    steps = [m['step'] for m in group_metrics[group]['test']]
                    rates = [m['success_rate'] for m in group_metrics[group]['test']]
                    plt.plot(steps, rates, f'{color}-o', label=f'{group} (test)',
                            linewidth=2, markersize=6, alpha=0.7)
        
        plt.xlabel('Evaluation Step', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, f"group_success_rates_{self.model_size}.png"), 
                       dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self):
        """Save the group metrics to a JSON file."""
        consolidated_path = os.path.join(self.metrics_dir, f"group_metrics_{self.model_size}.json")
        with open(consolidated_path, 'w') as f:
            json.dump(self.group_metrics, f, indent=2)

    def evaluate_model(self, model_size: str):
        """Evaluate test success rates for each group."""
        print(f"\nEvaluating Qwen {model_size} model group success rates...")
        
        # Updated log file paths - use absolute paths
        log_patterns = [
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_curriculum_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_R*_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_SingleRun.log")
        ]
        
        # Try each log pattern
        found_logs = []
        for pattern in log_patterns:
            found_logs.extend(glob.glob(pattern))
        
        if not found_logs:
            print(f"Error: No log files found matching patterns: {log_patterns}")
            print(f"Searched in directory: {self.logs_dir}")
            print("Available log files:")
            try:
                for f in os.listdir(self.logs_dir):
                    if f.endswith('.log'):
                        print(f"  - {f}")
            except Exception as e:
                print(f"  Error listing directory: {e}")
            return
            
        # Use the most recent log file
        log_file = max(found_logs, key=os.path.getmtime)
        print(f"Reading metrics from {log_file}")
        self.group_metrics = extract_group_metrics_from_log(log_file)
        
        if not self.group_metrics:
            print("Error: No group metrics found to evaluate")
            print("Please check that training is outputting test metrics in the format:")
            print("  test/<group_name>/success_rate:<number>")
            return
        
        # Save metrics
        self.save_metrics()
        
        # Create plots
        plot_path = os.path.join(self.plots_dir, f"group_success_rates_{model_size}.png")
        self.plot_group_metrics(self.group_metrics, plot_path)
        
        # Print summary of final success rates
        print("\nFinal success rates for each group:")
        for group in self.groups:
            if group in self.group_metrics:
                train_data = self.group_metrics[group]['train']
                test_data = self.group_metrics[group]['test']
                
                train_rate = train_data[-1]['success_rate'] if train_data else None
                test_rate = test_data[-1]['success_rate'] if test_data else None
                
                print(f"  {group}:")
                print(f"    Train: {train_rate:.4f if train_rate is not None else 'No data'}") 
                print(f"    Test:  {test_rate:.4f if test_rate is not None else 'No data'}")
            else:
                print(f"  {group}: No data available")
        
        print(f"\n{model_size.upper()} group evaluation completed successfully!")
        print("Results can be found in:")
        print(f"  - Plots: {plot_path}")
        print(f"  - Metrics: {os.path.join(self.metrics_dir, f'group_metrics_{model_size}.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate per-group success rates")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b", help="Model size to evaluate")
    parser.add_argument("--metrics-dir", default="./metrics", help="Directory containing metrics files")
    parser.add_argument("--plots-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--logs-dir", default="./logs", help="Directory containing training logs")
    
    args = parser.parse_args()
    
    evaluator = GroupEvaluator(
        model_size=args.model_size,
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
        logs_dir=args.logs_dir
    )
    evaluator.evaluate_model(args.model_size)
