"""Combined evaluation script for tracking both overall and per-group metrics in Countdown task."""

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

def extract_metrics_from_log(log_file: str) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, List[Dict[str, float]]]]]:
    """Extract both overall metrics and per-group success rates from a training log file."""
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return [], {}
    
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
    
    # Track overall metrics
    step_metrics = []
    training_step = 0
    
    # Track per-group metrics
    groups = ['plus_minus_mul', 'plus_minus_div', 'minus_mul_div', 'plus_mul_div']
    group_metrics = {group: {'train': [], 'test': []} for group in groups}
    
    # Extract overall metrics from training steps
    step_lines = re.findall(r'\(main_task pid=\d+\) step:\d+.*?(?:\r\n|\n)', content)
    
    for line in step_lines:
        # Skip validation/test steps
        if any(x in line for x in ['val/', 'test/', 'eval/']):
            continue
            
        # Extract metrics
        step_match = re.search(r'step:(\d+)', line)
        score_match = re.search(r'critic/score/mean:([0-9.]+)', line)
        pg_loss_match = re.search(r'actor/pg_loss:([-.0-9]+)', line)
        grad_norm_match = re.search(r'actor/grad_norm:([0-9.]+)', line)
        response_length_match = re.search(r'response_length/mean:([0-9.]+)', line)
        
        if all([score_match, pg_loss_match, grad_norm_match, response_length_match]):
            training_step += 1
            metrics = {
                'step': training_step,
                'score': float(score_match.group(1)),
                'pg_loss': float(pg_loss_match.group(1)),
                'grad_norm': float(grad_norm_match.group(1)),
                'response_length': float(response_length_match.group(1))
            }
            step_metrics.append(metrics)
    
    # Extract per-group metrics
    for group in groups:
        # Extract test metrics
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
        
        # Extract training metrics
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
    
    return step_metrics, group_metrics


class CombinedEvaluator:
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
        self.step_metrics = []
        self.group_metrics = {}

    def plot_overall_metrics(self, metrics: List[Dict], save_path: Optional[str] = None):
        """Create plots for overall training metrics."""
        if not metrics:
            print("No overall metrics to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Overall Training Metrics - {self.model_size.upper()} Model', fontsize=16, y=0.98)
        
        steps = list(range(1, len(metrics) + 1))
        
        # Success Rate
        values = [m.get("score", 0) for m in metrics]
        ax1.plot(steps, values, "-b", linewidth=0.5, marker='o', markersize=0.1)
        ax1.set_title("Overall Success Rate")
        ax1.set_ylim(0, max(0.15, max(values) * 1.1))
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Success Rate")
        ax1.grid(True, alpha=0.3)
        
        # Policy Gradient Loss
        values = [m.get("pg_loss", 0) for m in metrics]
        ax2.plot(steps, values, "-r", linewidth=0.5, marker='o', markersize=0.1)
        ax2.set_title("Policy Gradient Loss")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(min(values) * 1.1, max(values) * 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Gradient Norm
        values = [m.get("grad_norm", 0) for m in metrics]
        ax3.plot(steps, values, "-g", linewidth=0.5, marker='o', markersize=0.1)
        ax3.set_title("Gradient Norm")
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Norm")
        ax3.set_ylim(0, max(values) * 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Response Length
        values = [m.get("response_length", 0) for m in metrics]
        ax4.plot(steps, values, "-m", linewidth=0.5, marker='o', markersize=0.1)
        ax4.set_title("Response Length")
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Length")
        ax4.set_ylim(0, max(values) * 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.5, w_pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_group_metrics(self, group_metrics: Dict[str, Dict[str, List[Dict]]], save_path: Optional[str] = None):
        """Create plots for per-group training and test success rates."""
        if not group_metrics:
            print("No group metrics to plot")
            return
            
        plt.figure(figsize=(15, 10))
        plt.title(f'Per-Group Success Rates - {self.model_size.upper()} Model', fontsize=14)
        
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
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self):
        """Save both overall and group metrics."""
        # Save overall metrics
        overall_path = os.path.join(self.metrics_dir, f"overall_metrics_{self.model_size}.json")
        with open(overall_path, 'w') as f:
            json.dump(self.step_metrics, f, indent=2)
            
        # Save group metrics
        group_path = os.path.join(self.metrics_dir, f"group_metrics_{self.model_size}.json")
        with open(group_path, 'w') as f:
            json.dump(self.group_metrics, f, indent=2)

    def evaluate_model(self, model_size: str):
        """Evaluate both overall and per-group metrics."""
        print(f"\nEvaluating Qwen {model_size} model...")
        
        # Find log files
        log_patterns = [
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_curriculum_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_R*_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size.upper()}_SingleRun.log")
        ]
        
        found_logs = []
        for pattern in log_patterns:
            found_logs.extend(glob.glob(pattern))
        
        if not found_logs:
            print(f"Error: No log files found matching patterns: {log_patterns}")
            return
            
        # Use the most recent log file
        log_file = max(found_logs, key=os.path.getmtime)
        print(f"Reading metrics from {log_file}")
        
        # Extract both types of metrics
        self.step_metrics, self.group_metrics = extract_metrics_from_log(log_file)
        
        if not self.step_metrics and not self.group_metrics:
            print("Error: No metrics found to evaluate")
            return
        
        # Save all metrics
        self.save_metrics()
        
        # Create plots
        overall_plot = os.path.join(self.plots_dir, f"overall_metrics_{model_size}.png")
        group_plot = os.path.join(self.plots_dir, f"group_success_rates_{model_size}.png")
        
        self.plot_overall_metrics(self.step_metrics, overall_plot)
        self.plot_group_metrics(self.group_metrics, group_plot)
        
        # Print summary
        print("\nFinal overall metrics:")
        if self.step_metrics:
            final_metrics = self.step_metrics[-1]
            print(f"  Success Rate: {final_metrics['score']:.4f}")
            print(f"  PG Loss: {final_metrics['pg_loss']:.4f}")
            print(f"  Gradient Norm: {final_metrics['grad_norm']:.4f}")
            print(f"  Response Length: {final_metrics['response_length']:.1f}")
        
        print("\nFinal per-group success rates:")
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
        
        print(f"\n{model_size.upper()} evaluation completed successfully!")
        print("Results can be found in:")
        print(f"  - Overall Plots: {overall_plot}")
        print(f"  - Group Plots: {group_plot}")
        print(f"  - Overall Metrics: {os.path.join(self.metrics_dir, f'overall_metrics_{model_size}.json')}")
        print(f"  - Group Metrics: {os.path.join(self.metrics_dir, f'group_metrics_{model_size}.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate both overall and per-group metrics")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b", help="Model size to evaluate")
    parser.add_argument("--metrics-dir", default="./metrics", help="Directory containing metrics files")
    parser.add_argument("--plots-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--logs-dir", default="./logs", help="Directory containing training logs")
    
    args = parser.parse_args()
    
    evaluator = CombinedEvaluator(
        model_size=args.model_size,
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
        logs_dir=args.logs_dir
    )
    evaluator.evaluate_model(args.model_size)
