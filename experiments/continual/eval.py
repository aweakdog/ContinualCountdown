"""Evaluation script for continual learning on Countdown task."""

import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Disable WandB logging
os.environ['WANDB_MODE'] = 'disabled'

def extract_metrics_from_log(log_file: str) -> Dict[str, float]:
    """Extract metrics from a training log file.
    
    Args:
        log_file: Path to the log file, e.g. 'logs/ContinualCountdown_R1_plus_minus.log'
        
    Returns:
        Dictionary containing metrics
    """
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return {
            'success_rate': 0.0,
            'avg_loss': 0.0,
            'avg_grad_norm': 0.0,
            'avg_response_length': 0.0
        }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    def extract_metric(pattern: str) -> float:
        matches = re.findall(pattern, content)
        return float(matches[-1]) if matches else 0.0
    
    # Get final validation metrics
    success_rate = 0.0
    final_metrics_match = re.search(r'Final validation metrics.*?val/test_score/countdown_continual.*?step:4 - val/test_score/countdown_continual:([0-9.]+)', content, re.DOTALL)
    if final_metrics_match:
        success_rate = float(final_metrics_match.group(1))
    
    # Get final step metrics
    final_step_match = re.search(r'step:([0-9]+).*?critic/returns/mean:([0-9.]+).*?actor/grad_norm:([0-9.]+).*?response_length/mean:([0-9.]+)', content, re.DOTALL)
    
    avg_loss = 0.0
    avg_grad_norm = 0.0
    avg_response_length = 0.0
    
    if final_step_match:
        avg_loss = float(final_step_match.group(2))
        avg_grad_norm = float(final_step_match.group(3))
        avg_response_length = float(final_step_match.group(4))
    
    return {
        'success_rate': success_rate,
        'avg_loss': avg_loss,
        'avg_grad_norm': avg_grad_norm,
        'avg_response_length': avg_response_length
    }


class ContinualEvaluator:
    def __init__(self, metrics_dir: str = "./metrics", plots_dir: str = "./plots", logs_dir: str = "./logs"):
        self.metrics_dir = Path(metrics_dir)
        self.plots_dir = Path(plots_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.groups = ["plus", "plus_minus", "plus_minus_mul", "plus_minus_mul_div"]
        self.rounds = 3
        self.metrics = []
        
    def compute_weight_change(self, round_num: int, group: str) -> float:
        """Compute normalized weight change between initial and final model states."""
        # For initial state:
        # - For round 1 plus: use base model
        # - For round 1 other groups: use previous group's final state
        # - For round 2+ all groups: use same group's final state from previous round
        if round_num == 1 and group == "plus":
            initial_path = Path("/app/models/qwen") / "model.safetensors"
        elif round_num == 1:
            prev_group = self.groups[self.groups.index(group) - 1]
            initial_path = Path("checkpoints/ContinualCountdown") / f"ContinualCountdown_R{round_num}_{prev_group}/actor/global_step_0/model.safetensors"
        else:
            initial_path = Path("checkpoints/ContinualCountdown") / f"ContinualCountdown_R{round_num-1}_{group}/actor/global_step_0/model.safetensors"
        
        # For final state, use current group's final state
        final_path = Path("checkpoints/ContinualCountdown") / f"ContinualCountdown_R{round_num}_{group}/actor/global_step_0/model.safetensors"
        
        print(f"Comparing models for round {round_num}, group {group}:")
        print(f"  - Initial: {initial_path}")
        print(f"  - Final: {final_path}")
        
        if not (initial_path.exists() and final_path.exists()):
            print(f"Warning: Missing model files for round {round_num}, group {group}")
            print(f"  - Initial exists: {initial_path.exists()}")
            print(f"  - Final exists: {final_path.exists()}")
            return 0.0
        
        try:
            from safetensors import safe_open
            initial = {}
            final = {}
            
            with safe_open(str(initial_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    initial[key] = f.get_tensor(key)
                    
            with safe_open(str(final_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    final[key] = f.get_tensor(key)
            
            total_change = 0.0
            total_weights = 0.0
            
            # Only compare weights that exist in both models
            common_keys = set(initial.keys()) & set(final.keys())
            if not common_keys:
                print(f"Warning: No common weights found between initial and final models")
                return 0.0
                
            for key in common_keys:
                if initial[key].shape != final[key].shape:
                    print(f"Warning: Shape mismatch for {key}: {initial[key].shape} vs {final[key].shape}")
                    continue
                    
                change = torch.norm(final[key] - initial[key])
                weights = torch.norm(initial[key])
                total_change += change.item()
                total_weights += weights.item()
            
            if total_weights == 0:
                print("Warning: Total weights is zero")
                return 0.0
                
            normalized_change = total_change / total_weights
            print(f"Weight change for round {round_num}, group {group}:")
            print(f"  - Total change: {total_change:.4f}")
            print(f"  - Total weights: {total_weights:.4f}")
            print(f"  - Normalized change: {normalized_change:.4f}")
            return normalized_change
            
        except Exception as e:
            print(f"Error loading model files for round {round_num}, group {group}: {e}")
            return 0.0

    def process_metrics(self) -> List[Dict]:
        """Process metrics from logs and model checkpoints."""
        metrics = []
        
        for round_num in range(1, self.rounds + 1):
            for group in self.groups:
                # Get log file path
                log_file = self.logs_dir / f"ContinualCountdown_R{round_num}_{group}.log"
                print(f"Processing log file: {log_file}")
                
                if not log_file.exists():
                    print(f"Warning: Missing log file for round {round_num}, group {group}")
                    continue
                
                # Extract metrics from log
                data = extract_metrics_from_log(str(log_file))
                
                # Add weight change
                data['weight_change'] = self.compute_weight_change(round_num, group)
                
                # Add metadata
                data.update({
                    'round': round_num,
                    'group': group
                })
                
                metrics.append(data)
                
                # Save individual metrics file
                metrics_file = self.metrics_dir / f"metrics_{round_num}_{group}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(data, f, indent=2)
        
        return metrics

    def plot_metrics(self, metrics: List[Dict], save_path: Optional[str] = None):
        """Create plots for all metrics."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
        
        # Sort metrics by round and group order
        metrics.sort(key=lambda x: (x["round"], self.groups.index(x["group"])))
        
        # Success Rate
        values = [m.get("success_rate", 0) for m in metrics]
        ax1.plot(values, "-o")
        ax1.set_title("Success Rate")
        ax1.set_ylim(0, 1)
        ax1.grid(True)
        
        # Normalized Weight Change
        values = [m.get("weight_change", 0) for m in metrics]
        ax2.plot(values, "-o")
        ax2.set_title("Normalized Weight Change")
        ax2.grid(True)
        
        # Loss Function
        values = [m.get("avg_loss", 0) for m in metrics]
        ax3.plot(values, "-o")
        ax3.set_title("Average Loss")
        ax3.grid(True)
        
        # Normalized Gradient
        values = [m.get("avg_grad_norm", 0) for m in metrics]
        ax4.plot(values, "-o")
        ax4.set_title("Average Gradient Norm")
        ax4.grid(True)
        
        # Response Length
        values = [m.get("avg_response_length", 0) for m in metrics]
        ax5.plot(values, "-o")
        ax5.set_title("Average Response Length")
        ax5.grid(True)
        
        # Add legend
        labels = [f"R{m['round']}_{m['group']}" for m in metrics]
        ax6.axis("off")
        ax6.legend([f"{i+1}: {label}" for i, label in enumerate(labels)], loc="center")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def evaluate(self):
        """Run full evaluation and generate plots."""
        # Process all metrics
        metrics = self.process_metrics()
        
        if not metrics:
            print("Error: No metrics found to evaluate")
            return
        
        # Generate plots
        plot_path = self.plots_dir / "training_metrics.png"
        self.plot_metrics(metrics, str(plot_path))
        
        # Save consolidated metrics
        consolidated_path = self.metrics_dir / "consolidated_metrics.json"
        with open(consolidated_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Evaluation complete. Results saved to:")
        print(f"  - Plots: {plot_path}")
        print(f"  - Metrics: {consolidated_path}")
        
        # Print summary statistics
        self.print_summary(metrics)


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
                    print(f"{group:20}: {group_metrics['success_rate']:.2%} success rate")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate continual learning results")
    parser.add_argument("--metrics-dir", default="./metrics", help="Directory containing metrics files")
    parser.add_argument("--plots-dir", default="./plots", help="Directory to save plots")
    parser.add_argument("--logs-dir", default="./logs", help="Directory containing training logs")
    
    args = parser.parse_args()
    
    evaluator = ContinualEvaluator(
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
        logs_dir=args.logs_dir
    )
    evaluator.evaluate()
