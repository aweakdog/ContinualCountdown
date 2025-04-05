"""Evaluation script for continual learning on Countdown task."""

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

def extract_metrics_from_log(log_file: str) -> List[Dict[str, float]]:
    """Extract metrics from a training log file after each step."""
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return []
    
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
    
    # Extract all step metrics
    step_metrics = []
    
    # Extract lines containing step and metrics, but only training steps
    step_lines = re.findall(r'\(main_task pid=\d+\) step:\d+.*?(?:\r\n|\n)', content)
    print(f"Raw content length: {len(content)} chars")
    print(f"Found {len(step_lines)} lines with step metrics")
    
    training_step = 0  # Counter for actual training steps
    
    for i, line in enumerate(step_lines):
        # Extract metrics one by one
        step_match = re.search(r'step:(\d+)', line)
        score_match = re.search(r'critic/score/mean:([0-9.]+)', line)
        pg_loss_match = re.search(r'actor/pg_loss:([-.0-9]+)', line)
        grad_norm_match = re.search(r'actor/grad_norm:([0-9.]+)', line)
        response_length_match = re.search(r'response_length/mean:([0-9.]+)', line)
        val_test_score_match = re.search(r'val/test_score/countdown_continual:([0-9.]+)', line)
        
        # For training steps, require all training metrics
        if not val_test_score_match and not all([score_match, pg_loss_match, grad_norm_match, response_length_match]):
            print(f"Skipping line {i} due to missing training metrics")
            continue
        
        # For validation steps, only require val/test_score
        if val_test_score_match and not any([score_match, pg_loss_match, grad_norm_match, response_length_match]):
            metrics = {
                'step': len(step_metrics) + 1,
                'score': 0.0,
                'pg_loss': 0.0,
                'grad_norm': 0.0,
                'response_length': 0.0,
                'val/test_score/countdown_continual': float(val_test_score_match.group(1))
            }
            step_metrics.append(metrics)
            print(f"Added validation metrics for step {len(step_metrics)}")
            continue
            
        training_step += 1  # Increment training step counter

        
        try:
            metrics = {
                'step': training_step,  # Use our training step counter
                'score': float(score_match.group(1)),
                'pg_loss': float(pg_loss_match.group(1)),
                'grad_norm': float(grad_norm_match.group(1)),
                'response_length': float(response_length_match.group(1)),
                'val/test_score/countdown_continual': float(val_test_score_match.group(1)) if val_test_score_match else 0.0
            }
            step_metrics.append(metrics)
            print(f"Successfully parsed metrics for training step {training_step}")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse metrics from line {i}: {e}")
            training_step -= 1  # Revert the step counter if parsing failed
            continue
    
    if not step_metrics:
        print(f"Warning: No metrics found in {log_file}")
        print("Please check that training is outputting metrics in the expected format:")
        print("  - step:<number>")
        print("  - critic/score/mean:<number>")
        print("  - actor/pg_loss:<number>")
        print("  - actor/grad_norm:<number>")
        print("  - response_length/mean:<number>")
        
    return step_metrics


class ContinualEvaluator:
    def __init__(self, model_size: str = "0.5b", metrics_dir: str = "./metrics", plots_dir: str = "./plots", logs_dir: str = "./logs"):
        self.groups = ["plus", "plus_minus", "plus_minus_mul", "plus_minus_mul_div"]
        self.metrics_dir = os.getenv("METRICS_DIR", metrics_dir)
        self.plots_dir = os.getenv("PLOTS_DIR", plots_dir)
        self.logs_dir = os.getenv("LOGS_DIR", logs_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.model_size = model_size.lower()
        
        # Configuration
        self.rounds = 3
        self.metrics = []
        
        # Model paths based on size
        self.base_model = "/app/models/qwen" if model_size == "0.5b" else "/app/models/countdown_continual_1.5b"
        self.project_name = "ContinualCountdown" if model_size == "0.5b" else "ContinualCountdown1.5B"
        self.run_name = f"{self.project_name}_SingleRun"

    def compute_weight_change(self, round_num: int, group: str) -> float:
        """Compute normalized weight change between initial and final model states."""
        initial_path = os.path.join(self.metrics_dir, f"initial_{round_num}_{group}/model.safetensors")
        final_path = os.path.join(self.metrics_dir, f"final_{round_num}_{group}/model.safetensors")
        
        if not os.path.exists(initial_path) or not os.path.exists(final_path):
            print(f"Warning: Missing model states for round {round_num}, group {group}")
            return 0.0
        
        initial_state = torch.load(initial_path)
        final_state = torch.load(final_path)
        
        total_change = 0.0
        total_params = 0
        
        for key in initial_state:
            if key in final_state:
                param_change = torch.norm(final_state[key] - initial_state[key])
                param_magnitude = torch.norm(initial_state[key])
                if param_magnitude > 0:
                    total_change += (param_change / param_magnitude).item()
                total_params += 1
        
        return total_change / total_params if total_params > 0 else 0.0

    def plot_metrics(self, metrics: List[Dict], save_path: Optional[str] = None):
        """Create plots for all metrics."""
        if not metrics:
            print("No metrics to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.model_size.upper()} Model', fontsize=16, y=0.98)
        
        # Get continuous step numbers
        steps = list(range(1, len(metrics) + 1))
        
        # Success Rate and Val/Test Score
        train_scores = [m.get("score", 0) for m in metrics]
        
        # Apply smoothing to training scores
        smoothed_scores = []
        smooth_factor = 0.9  # Configurable smoothing factor
        if train_scores:
            smoothed_scores.append(train_scores[0])  # First value unchanged
            for score in train_scores[1:]:
                smoothed_scores.append(smooth_factor * smoothed_scores[-1] + (1 - smooth_factor) * score)
        
        # Get validation scores only where they exist (no zeros)
        val_test_scores = [(i+1, m["val/test_score/countdown_continual"]) 
                          for i, m in enumerate(metrics) 
                          if "val/test_score/countdown_continual" in m]
        
        # Plot both raw and smoothed training scores
        ax1.plot(steps, train_scores, "-b", linewidth=0.2, alpha=0.3, label='Raw Training Score')
        ax1.plot(steps, smoothed_scores, "-b", linewidth=1.0, label=f'Smoothed Training Score')
        
        # Plot validation scores with line chart
        if val_test_scores:
            val_steps, val_scores = zip(*val_test_scores)
            print("Validation steps:", val_steps)
            print("Validation scores:", val_scores)
            ax1.plot(val_steps, val_scores, "-r", linewidth=1.0, label='Val/Test Score')
        
        ax1.set_title("Score")
        max_score = max(max(train_scores), max(s for _, s in val_test_scores) if val_test_scores else 0)
        ax1.set_ylim(0, max(0.15, max_score * 1.1))  # Cap at 0.15 or 10% above max
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Score")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
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
        
        # Add some padding between subplots
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.5, w_pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.plots_dir, f"training_metrics_{self.model_size}.png"), dpi=300, bbox_inches='tight')

    def save_metrics(self):
        consolidated_path = os.path.join(self.metrics_dir, f"consolidated_metrics_{self.model_size}.json")
        with open(consolidated_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def evaluate_model(self, model_size: str):
        """Evaluate a specific model size."""
        print(f"\nEvaluating Qwen {model_size} model...")
        
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
        metrics = extract_metrics_from_log(log_file)
        
        if not metrics:
            print("Error: No metrics found to evaluate")
            print("Please check that training is outputting metrics in the expected format:")
            print("  - step:<number>")
            print("  - critic/score/mean:<number>")
            print("  - actor/pg_loss:<number>")
            print("  - actor/grad_norm:<number>")
            print("  - response_length/mean:<number>")
            return
        
        # Save metrics
        self.metrics = metrics
        self.save_metrics()
        
        # Create plots
        plot_path = os.path.join(self.plots_dir, f"training_metrics_{model_size}.png")
        self.plot_metrics(metrics, plot_path)
        
        print(f"\n{model_size.upper()} evaluation completed successfully!")
        print("Results can be found in:")
        print(f"  - Plots: {plot_path}")
        print(f"  - Metrics: {os.path.join(self.metrics_dir, f'consolidated_metrics_{model_size}.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate continual learning results")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b"], default="0.5b", help="Model size to evaluate")
    parser.add_argument("--metrics-dir", default="./metrics", help="Directory containing metrics files")
    parser.add_argument("--plots-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--logs-dir", default="./logs", help="Directory containing training logs")
    
    args = parser.parse_args()
    
    evaluator = ContinualEvaluator(
        model_size=args.model_size,
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
        logs_dir=args.logs_dir
    )
    evaluator.evaluate_model(args.model_size)
