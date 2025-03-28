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
        
    # Extract all step metrics
    step_metrics = []
    # Pattern to match all relevant metrics
    step_pattern = r'step:(\d+).*?critic/score/mean:([0-9.]+).*?actor/pg_loss:([-.0-9]+).*?actor/grad_norm:([0-9.]+).*?response_length/mean:([0-9.]+)'
    
    matches = re.finditer(step_pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        metrics = {
            'step': int(match.group(1)),
            'score': float(match.group(2)),
            'pg_loss': float(match.group(3)),
            'grad_norm': float(match.group(4)),
            'response_length': float(match.group(5))
        }
        step_metrics.append(metrics)
    
    if not step_metrics:
        print(f"Warning: No metrics found in {log_file}")
        
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
        # For initial state:
        # - For round 1 plus: use base model
        # - For round 1 other groups: use previous group's checkpoint
        # - For round 2+ all groups: use same group's checkpoint from previous round
        if round_num == 1 and group == "plus":
            initial_path = Path(self.base_model) / "model.safetensors"
        elif round_num == 1:
            prev_group = self.groups[self.groups.index(group) - 1]
            initial_path = Path(f"checkpoints/{self.project_name}") / f"{self.run_name}/actor/group_{prev_group}/model.safetensors"
        else:
            initial_path = Path(f"checkpoints/{self.project_name}") / f"{self.run_name}/actor/round_{round_num-1}_group_{group}/model.safetensors"
        
        # For final state, use current group's checkpoint
        final_path = Path(f"checkpoints/{self.project_name}") / f"{self.run_name}/actor/round_{round_num}_group_{group}/model.safetensors"
        
        print(f"Comparing models for round {round_num}, group {group}:")
        print(f"  - Initial: {initial_path}")
        print(f"  - Final: {final_path}")
        
        if not (initial_path.exists() and final_path.exists()):
            print(f"Warning: Missing model files for round {round_num}, group {group}")
            print(f"  - Initial exists: {initial_path.exists()}")
            print(f"  - Final exists: {final_path.exists()}")
            return 0.0
        
        # Load models and compute weight change
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

    def process_metrics(self, log_file: Path) -> List[Dict]:
        """Process metrics from a log file"""
        metrics = []
        step_metrics = extract_metrics_from_log(str(log_file))
        
        # For curriculum learning single run, we just process metrics sequentially
        current_step_metrics = []
        window_size = 100  # Average over last 100 steps
        
        for step in step_metrics:
            current_step_metrics.append(step)
            if len(current_step_metrics) >= window_size:
                # Calculate moving averages
                avg_metrics = {
                    'step': step['step'],
                    'score': np.mean([m['score'] for m in current_step_metrics[-window_size:]]),
                    'pg_loss': np.mean([m['pg_loss'] for m in current_step_metrics[-window_size:]]),
                    'grad_norm': np.mean([m['grad_norm'] for m in current_step_metrics[-window_size:]]),
                    'response_length': np.mean([m['response_length'] for m in current_step_metrics[-window_size:]])
                }
                metrics.append(avg_metrics)
        
        return metrics

    def plot_metrics(self, metrics: List[Dict], save_path: Optional[str] = None):
        """Create plots for all metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        steps = [m["step"] for m in metrics]
        
        # Success Rate
        values = [m.get("score", 0) for m in metrics]
        ax1.plot(steps, values, "-")
        ax1.set_title("Score")
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("Step")
        
        # Loss
        values = [m.get("pg_loss", 0) for m in metrics]
        ax2.plot(steps, values, "-")
        ax2.set_title("Policy Gradient Loss")
        ax2.set_xlabel("Step")
        
        # Gradient Norm
        values = [m.get("grad_norm", 0) for m in metrics]
        ax3.plot(steps, values, "-")
        ax3.set_title("Average Gradient Norm")
        ax3.set_xlabel("Step")
        
        # Response Length
        values = [m.get("response_length", 0) for m in metrics]
        ax4.plot(steps, values, "-")
        ax4.set_title("Average Response Length")
        ax4.set_xlabel("Step")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.plots_dir / "training_metrics_1.5b.png")

    def save_metrics(self):
        consolidated_path = self.metrics_dir / f"consolidated_metrics_{self.model_size}.json"
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
        metrics = self.process_metrics(log_file)
        
        if not metrics:
            print("Error: No metrics found to evaluate")
            print("Please check that training is outputting metrics in the expected format:")
            print("  - step:<number>")
            print("  - critic/score/mean:<number>")
            print("  - actor/pg_loss:<number>")
            print("  - actor/grad_norm:<number>")
            print("  - response_length/mean:<number>")
            return

        # Plot metrics
        self.plot_metrics(metrics)
        
        # Save consolidated metrics
        self.save_metrics()
        
        print(f"\nEvaluation complete. Results saved to:")
        print(f"  - Plots: {self.plots_dir}/training_metrics_{self.model_size}.png")
        print(f"  - Metrics: {self.metrics_dir}/consolidated_metrics_{self.model_size}.json")

    def print_summary(self, metrics: List[Dict]):
        """Print summary statistics for each round."""
        print("\nSummary Statistics:")
        print("-" * 50)
        
        for round_num in range(1, self.rounds + 1):
            round_metrics = [m for m in metrics if m['round'] == round_num]
            if not round_metrics:
                continue
            
            print(f"\nRound {round_num}:")
            print("-" * 20)
            
            # Average success rate per group
            for group in self.groups:
                group_metrics = next((m for m in round_metrics if m['group'] == group), None)
                if group_metrics:
                    print(f"{group:20}: {group_metrics['score']:.2%} score")


if __name__ == "__main__":
    import argparse
    
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
