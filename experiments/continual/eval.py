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

        
        # Extract entropy if available
        entropy_match = re.search(r'actor/entropy_loss:([-.0-9]+)', line)
        # Extract nullspace and zero grad space ratios if available
        nullspace_match = re.search(r'actor/nullspace_ratio:([0-9.]+)', line)
        zero_gradspace_match = re.search(r'actor/zero_gradspace_ratio:([0-9.]+)', line)
        try:
            metrics = {
                'step': training_step,  # Use our training step counter
                'score': float(score_match.group(1)),
                'pg_loss': float(pg_loss_match.group(1)),
                'grad_norm': float(grad_norm_match.group(1)),
                'response_length': float(response_length_match.group(1)),
                'entropy': float(entropy_match.group(1)) if entropy_match else 0.0,
                'val/test_score/countdown_continual': float(val_test_score_match.group(1)) if val_test_score_match else 0.0,
                'nullspace_ratio': float(nullspace_match.group(1)) if nullspace_match else 0.0,
                'zero_gradspace_ratio': float(zero_gradspace_match.group(1)) if zero_gradspace_match else 0.0
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
        print("  - actor/entropy_loss:<number>")
        print("  - response_length/mean:<number>")
        
    return step_metrics


def extract_group_metrics_from_log(log_file: str) -> Dict[int, Dict[int, List[Dict[str, float]]]]:
    """Extract metrics organized by round and group from a training log file.
    
    Returns:
        A dictionary where:
        - Keys are group IDs
        - Values are dictionaries where:
          - Keys are round IDs
          - Values are lists of metrics dictionaries for that round and group
    """
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
    
    # Initialize the group metrics dictionary
    # Structure: {group_id: {round_id: [metrics]}}
    group_metrics = {}
    
    # Extract lines containing Round and Group information
    round_group_pattern = r'\(main_task pid=\d+\) Round (\d+), Group (\d+), Epoch (\d+), Step (\d+)'
    round_group_matches = re.finditer(round_group_pattern, content)
    
    # Extract lines containing step metrics
    step_pattern = r'\(main_task pid=\d+\) step:(\d+) - .*?critic/score/mean:([0-9.]+).*?(?:\r\n|\n)'
    step_matches = re.finditer(step_pattern, content)
    
    # Create a mapping of step numbers to metrics
    step_to_metrics = {}
    for match in step_matches:
        step_num = int(match.group(1))
        step_line = match.group(0)
        
        score_match = re.search(r'critic/score/mean:([0-9.]+)', step_line)
        if score_match:
            score = float(score_match.group(1))
            step_to_metrics[step_num] = {'score': score, 'step': step_num}
    
    # Process each Round, Group, Epoch, Step entry
    current_round = None
    current_group = None
    current_step = None
    
    for match in round_group_matches:
        round_id = int(match.group(1))
        group_id = int(match.group(2))
        epoch_id = int(match.group(3))
        step_id = int(match.group(4))
        
        # If we have metrics for this step
        if step_id in step_to_metrics:
            # Initialize group dictionary if needed
            if group_id not in group_metrics:
                group_metrics[group_id] = {}
            
            # Initialize round list if needed
            if round_id not in group_metrics[group_id]:
                group_metrics[group_id][round_id] = []
            
            # Add metrics to the appropriate round and group
            metrics = step_to_metrics[step_id].copy()
            metrics['round'] = round_id
            metrics['group'] = group_id
            metrics['epoch'] = epoch_id
            group_metrics[group_id][round_id].append(metrics)
    
    # Check if we found any metrics
    if not group_metrics:
        print(f"Warning: No group metrics found in {log_file}")
        print("Please check that training is outputting metrics in the expected format:")
        print("  - Round X, Group Y, Epoch Z, Step W")
        print("  - step:N - critic/score/mean:M")
    
    return group_metrics


class ContinualEvaluator:
    def __init__(self, model_size: str = "0.5b", metrics_dir: str = "./metrics", plots_dir: str = "./plots", logs_dir: str = "./logs"):
        self.groups = ["0", "1", "2", "3"]  # Group names are now numbers
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
        if model_size == "0.5b":
            self.base_model = "/app/models/qwen"
            self.project_name = "ContinualCountdown"
        elif model_size == "1.5b":
            self.base_model = "/app/models/countdown_continual_1.5b"
            self.project_name = "ContinualCountdown1.5B"
        elif model_size == "3b":
            self.base_model = "/app/models/countdown_continual_3b"
            self.project_name = "ContinualCountdown3B"
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
            
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
            
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
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
        
        # Entropy
        values = [m.get("entropy", 0) for m in metrics]
        ax5.plot(steps, values, "-c", linewidth=0.5, marker='o', markersize=0.1)
        ax5.set_title("Entropy Loss")
        ax5.set_xlabel("Training Step")
        ax5.set_ylabel("Entropy Loss")
        if any(values):  # Only set ylim if we have non-zero values
            ax5.set_ylim(min(values) * 0.9, max(values) * 1.1)
        ax5.grid(True, alpha=0.3)
        
        # Plot nullspace and zero grad space ratios
        nullspace_vals = [m.get("nullspace_ratio", 0) for m in metrics]
        zero_grad_vals = [m.get("zero_gradspace_ratio", 0) for m in metrics]
        ax6.set_visible(True)
        ax6.plot(steps, nullspace_vals, "-b", label="Nullspace Ratio")
        ax6.plot(steps, zero_grad_vals, "-r", label="Zero Grad Space Ratio")
        ax6.set_title("Nullspace & Zero Grad Ratios")
        ax6.set_xlabel("Training Step")
        ax6.set_ylabel("Ratio")
        ax6.set_ylim(0, max(max(nullspace_vals), max(zero_grad_vals), 0.15) * 1.1)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
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

    def plot_by_group(self, group_metrics: Dict[int, Dict[int, List[Dict[str, float]]]], save_path_prefix: Optional[str] = None, smooth_strength: float = 0.9):
        """Create separate plots for each group showing concatenated scores across all rounds.
        Also overlays the single-group baseline as a background for comparison.
        Args:
            group_metrics: Dict mapping group_id to a dict of round_id to list of score dicts.
            save_path_prefix: Optional prefix for saving plots.
            smooth_strength: Smoothing strength from 0 (no smoothing) to 1 (max smoothing, EMA). Default 0.5.
        """
        import os
        if not group_metrics:
            print("No group metrics to plot")
            return

        def smooth_ema(data, smooth_strength):
            if smooth_strength <= 0:
                return data
            alpha = 1 - min(max(smooth_strength, 0), 1)  # invert so higher = smoother
            smoothed = []
            for i, x in enumerate(data):
                if i == 0:
                    smoothed.append(x)
                else:
                    smoothed.append(alpha * x + (1 - alpha) * smoothed[-1])
            return smoothed

        for group_id, rounds_data in sorted(group_metrics.items()):
            plt.figure(figsize=(10, 6))
            group_name = f"Group {group_id}"

            # --- BASELINE: Load and plot single-group log as background ---
            baseline_log = os.path.join(self.logs_dir, f"ContinualCountdown3B_Group{group_id}.log")
            if os.path.exists(baseline_log):
                baseline_metrics = extract_metrics_from_log(baseline_log)
                baseline_scores = [m.get('score', 0) for m in baseline_metrics]
                if baseline_scores:
                    # Repeat the baseline for each round
                    num_rounds = len(rounds_data)
                    baseline_scores_tiled = baseline_scores * num_rounds
                    # If curriculum steps are not an integer multiple, trim or pad
                    total_steps = sum(len(metrics_list) for metrics_list in rounds_data.values())
                    if len(baseline_scores_tiled) > total_steps:
                        baseline_scores_tiled = baseline_scores_tiled[:total_steps]
                    elif len(baseline_scores_tiled) < total_steps:
                        baseline_scores_tiled += [baseline_scores[-1]] * (total_steps - len(baseline_scores_tiled))
                    baseline_steps = list(range(1, len(baseline_scores_tiled) + 1))
                    # Smooth the repeated baseline using EMA
                    smoothed_baseline = smooth_ema(baseline_scores_tiled, smooth_strength)
                    plt.plot(
                        baseline_steps,
                        smoothed_baseline,
                        color="gray",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="Single-group Baseline (smoothed, repeated)"
                    )

            # --- CURRICULUM: Plot curriculum (across-group) results ---
            all_scores = []
            for round_id, metrics_list in sorted(rounds_data.items()):
                round_scores = [m.get('score', 0) for m in metrics_list]
                all_scores.extend(round_scores)

            if all_scores:
                steps = list(range(1, len(all_scores) + 1))
                smoothed_scores = smooth_ema(all_scores, smooth_strength)

                plt.plot(steps, all_scores, color='lightblue', linewidth=0.8, alpha=0.5, label="Curriculum (raw)")
                plt.plot(steps, smoothed_scores, color='blue', linewidth=2, label="Curriculum (smoothed)")

                # Add round boundaries as before
                round_boundaries = [0]
                current_position = 0
                for round_id, metrics_list in sorted(rounds_data.items()):
                    current_position += len(metrics_list)
                    round_boundaries.append(current_position)
                for boundary in round_boundaries[1:-1]:
                    plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
                    plt.text(boundary, 0.02, f"R{round_boundaries.index(boundary)}", color='red', ha='center', va='bottom', alpha=0.7)

            plt.title(f'Training Scores for {group_name} - {self.model_size.upper()} Model', fontsize=16)
            plt.xlabel('Steps (Concatenated Across Rounds)', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.0)
            plt.legend()

            if save_path_prefix:
                save_path = f"{save_path_prefix}_group{group_id}.png"
            else:
                save_path = os.path.join(self.plots_dir, f"plot_group{group_id}_{self.model_size}.png")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created plot for {group_name}: {save_path}")

    def evaluate_model(self, model_size: str):
        """Evaluate a specific model size."""
        print(f"\nEvaluating Qwen {model_size} model...")
        
        # Updated log file paths - use absolute paths
        # Handle special case for 3b model which might use different naming convention
        if model_size == "3b":
            model_size_prefix = "3B"
        else:
            model_size_prefix = model_size.upper()
            
        log_patterns = [
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_curriculum_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_R*_*.log"),
            os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_SingleRun.log")
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
        
        # Extract group metrics for the group-based plot
        group_metrics = extract_group_metrics_from_log(log_file)
        
        if not metrics:
            print("Error: No metrics found to evaluate")
            print("Please check that training is outputting metrics in the expected format:")
            print("  - step:<number>")
            print("  - critic/score/mean:<number>")
            print("  - actor/pg_loss:<number>")
            print("  - actor/grad_norm:<number>")
            print("  - actor/entropy_loss:<number>")
            print("  - response_length/mean:<number>")
            return
        
        # Save metrics
        self.metrics = metrics
        self.save_metrics()
        
        # Create standard plots
        plot_path = os.path.join(self.plots_dir, f"training_metrics_{model_size}.png")
        self.plot_metrics(metrics, plot_path)
        
        # Create group-based plots if we have group metrics
        if group_metrics:
            group_plot_path_prefix = os.path.join(self.plots_dir, f"plot_{model_size}")
            self.plot_by_group(group_metrics, group_plot_path_prefix)
            print(f"Group-based plots saved to: {self.plots_dir}")
        else:
            print("No group metrics found for group-based plots")
        
        print(f"\n{model_size.upper()} evaluation completed successfully!")
        print("Results can be found in:")
        print(f"  - Plots: {plot_path}")
        print(f"  - Metrics: {os.path.join(self.metrics_dir, f'consolidated_metrics_{model_size}.json')}")


def extract_token_probabilities(log_file: str, model_size: str) -> List[Dict]:
    """Extract token probabilities from training log file for visualization.
    
    Args:
        log_file: Path to the log file
        model_size: Size of the model ("0.5b", "1.5b", "3b")
        
    Returns:
        List of dictionaries containing token probability data
    """
    if not os.path.exists(log_file):
        print(f"Warning: Log file not found: {log_file}")
        return []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Remove ANSI color codes from each line
    cleaned_lines = [re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line) for line in lines]
    
    # Extract problem information from the cleaned content
    content = ''.join(cleaned_lines)
    problem_pattern = r'\(main_task pid=\d+\) Target: (\d+) \| Numbers: \[([^\]]+)\]\s*\n.*?\(main_task pid=\d+\) Extracted equation: ([^\n]+)'
    problems = re.findall(problem_pattern, content)
    
    # Process lines to extract token probability data
    results = []
    current_tokens = []
    current_round = None
    current_group = None
    current_epoch = None
    current_step = None
    current_target = "Unknown"
    current_numbers = "Unknown"
    current_equation = "Unknown"
    in_token_analysis = False
    problem_index = 0
    
    for i, line in enumerate(cleaned_lines):
        # Check for token analysis section start
        if "Token-by-token probability analysis:" in line:
            in_token_analysis = True
            current_tokens = []
            
            # Try to find matching problem information
            if problem_index < len(problems):
                current_target = problems[problem_index][0]
                current_numbers = problems[problem_index][1]
                current_equation = problems[problem_index][2]
                problem_index += 1
            else:
                current_target = "Unknown"
                current_numbers = "Unknown"
                current_equation = "Unknown"
            
            continue
        
        # If we're in a token analysis section, process token lines
        if in_token_analysis:
            # Check for token line pattern
            token_match = re.search(r'Round: (\d+), Group: (\d+), Epoch: (\d+), Step: (\d+), Token (\d+): \'(.*?)\' - Probability: ([0-9.]+) \(log prob: ([-.0-9]+)\)', line)
            
            if token_match:
                # Extract token information
                round_num = token_match.group(1)
                group_num = token_match.group(2)
                epoch_num = token_match.group(3)
                step_num = token_match.group(4)
                token_num = token_match.group(5)
                token_text = token_match.group(6).strip()
                probability = float(token_match.group(7))
                
                # Update current section information
                current_round = round_num
                current_group = group_num
                current_epoch = epoch_num
                current_step = step_num
                
                # Add token to current collection
                current_tokens.append((token_num, token_text, probability))
            
            # Check for end of token analysis section (empty line or new section start)
            elif line.strip() == "" or "Starting" in line or "Finished" in line:
                # If we have tokens, save the results
                if current_tokens and current_round is not None:
                    # Extract tokens and probabilities in order
                    tokens = [t[1] for t in sorted(current_tokens, key=lambda x: int(x[0]))]
                    probabilities = [p[2] for p in sorted(current_tokens, key=lambda x: int(x[0]))]
                    
                    results.append({
                        'round': current_round,
                        'group': current_group,
                        'epoch': current_epoch,
                        'step': current_step,
                        'target': current_target,
                        'numbers': current_numbers,
                        'equation': current_equation,
                        'tokens': tokens,
                        'probabilities': probabilities
                    })
                
                # Reset for next section
                in_token_analysis = False
                current_tokens = []
                current_round = None
                current_group = None
                current_epoch = None
                current_step = None
            
            # Handle continuation lines (broken token text)
            elif "(main_task pid=" in line and not "Round:" in line:
                # This is a continuation line, try to extract any content after the process ID
                content_match = re.search(r'\(main_task pid=\d+\)\s*(.*)', line)
                if content_match and content_match.group(1).strip() and current_tokens:
                    # Append the content to the last token's text
                    last_token_num, last_token_text, last_prob = current_tokens[-1]
                    current_tokens[-1] = (last_token_num, last_token_text + " " + content_match.group(1).strip(), last_prob)
    
    # Handle any remaining tokens at the end of the file
    if in_token_analysis and current_tokens and current_round is not None:
        tokens = [t[1] for t in sorted(current_tokens, key=lambda x: int(x[0]))]
        probabilities = [p[2] for p in sorted(current_tokens, key=lambda x: int(x[0]))]
        
        results.append({
            'round': current_round,
            'group': current_group,
            'epoch': current_epoch,
            'step': current_step,
            'target': current_target,
            'numbers': current_numbers,
            'equation': current_equation,
            'tokens': tokens,
            'probabilities': probabilities
        })
    
    return results

def visualize_token_probabilities(token_data: List[Dict], model_size: str, plots_dir: str):
    """Create visualizations of token probabilities.
    
    Args:
        token_data: List of dictionaries containing token probability data
        model_size: Size of the model ("0.5b", "1.5b", "3b")
        plots_dir: Directory to save the plots
    """
    if not token_data:
        print("No token probability data to visualize")
        return
    
    # Create directory for case study plots
    case_study_dir = os.path.join(plots_dir, f"case_study_{model_size}")
    os.makedirs(case_study_dir, exist_ok=True)
    
    for i, data in enumerate(token_data):
        try:
            tokens = data['tokens']
            probs = data['probabilities']
            target = data['target']
            numbers = data['numbers']
            equation = data['equation']
            round_num = data['round']
            group_num = data['group']
            epoch_num = data['epoch']
            step_num = data['step']
            
            # Skip if we have no tokens or probabilities
            if not tokens or not probs:
                continue
                
            # Create a figure for this example with a proper axes object
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Remove all tokens after the 'special-token' token
            if '<|endoftext|>' in tokens:
                special_token_idx = tokens.index('<|endoftext|>')
                # Keep the special token as the last token
                tokens = tokens[:special_token_idx + 1]
                probs = probs[:special_token_idx + 1]
                print(f"Truncated tokens at 'special-token' (position {special_token_idx})")
                
            # Limit the number of tokens to prevent image size issues
            max_tokens = 1024  # Maximum tokens to display
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                probs = probs[:max_tokens]
                print(f"Warning: Limited visualization to {max_tokens} tokens to prevent image size issues")
            
            # Create a custom colormap from red (low prob) to green (high prob)
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = [(0.9, 0.2, 0.2), (0.9, 0.5, 0.2), (0.6, 0.7, 0.2), (0.2, 0.8, 0.2)]  # Red to green
            cmap = LinearSegmentedColormap.from_list('RedToGreen', colors_list)
            
            # Normalize probabilities for coloring
            norm = plt.Normalize(0, max(probs))
            colors = [cmap(norm(p)) for p in probs]
            
            # Calculate a reasonable font size based on token count
            font_size = max(8, min(12, 1000 / len(tokens)))
            
            # Set up parameters for token layout - treating tokens as a whole sentence
            max_width = 0.7  # Reduced width to avoid tiling the entire image
            token_padding = 0.0  # No padding between tokens for a completely cohesive sentence
            # Fixed line height in figure fraction units for consistent spacing
            line_height = 0.005  # Smaller value for tighter line spacing
            
            # Start position
            current_x = 0.05  # Left margin
            current_y = 0.95  # Top margin
            
            # Track token positions for drawing
            token_positions = []
            
            # Join tokens into a continuous sentence with proper spacing
            # First pass: calculate positions and line breaks
            
            # Use a consistent width for all characters since we're using a monospace font
            # Fixed character width in figure fraction units
            char_width = 0.006
            
            for j, token in enumerate(tokens):
                # With monospace font, all characters (including spaces) have the same width
                token_width = len(token) * char_width
                
                # Special handling for token spacing
                # Only consider the previous token when determining spacing
                needs_space = True
                
                if j > 0:
                    prev_token = tokens[j-1]
                    
                    # Check if previous token is punctuation or number-only
                    is_prev_punctuation = all(c in '.,!?:;()[]{}+-*/=<>"\'' for c in prev_token)
                    is_prev_number = all(c.isdigit() or c in '.-+' for c in prev_token)
                    
                    # Don't add space if:
                    # - Previous token is punctuation or number
                    # - Previous token ends with a space
                    if is_prev_punctuation or is_prev_number or prev_token.endswith((' ', '\n')):
                        needs_space = False
                else:
                    needs_space = False  # First token doesn't need leading space
                
                # Add space between tokens if needed
                if needs_space:
                    current_x += char_width  # Use the same width for spaces with monospace font
                
                # Check if we need to move to the next line
                if current_x + token_width > max_width:
                    current_x = 0.05  # Reset to left margin
                    current_y -= line_height  # Move down
                
                # Store the position for this token
                token_positions.append((current_x, current_y, token, probs[j], colors[j]))
                
                # Move to position for next token
                current_x += token_width
            
            # Second pass: draw the tokens with colored text as a cohesive sentence
            # Use a monospace font where all characters have the same width
            from matplotlib.font_manager import FontProperties
            mono_font = FontProperties(family='monospace', weight='normal')
            
            # Create a single text string with precise positioning to preserve spaces
            # Group tokens by line position (y_pos) to handle line breaks
            lines = {}
            for x_pos, y_pos, token, prob, color in token_positions:
                if y_pos not in lines:
                    lines[y_pos] = []
                lines[y_pos].append((x_pos, token, color))
            
            # Sort lines by y position (top to bottom)
            sorted_y_positions = sorted(lines.keys(), reverse=True)
            
            # Draw each line as a series of individual characters to preserve exact spacing
            for y_pos in sorted_y_positions:
                tokens_in_line = sorted(lines[y_pos], key=lambda x: x[0])  # Sort by x position
                
                for x_pos, token, color in tokens_in_line:
                    # Draw each character individually with precise positioning
                    for i, char in enumerate(token):
                        # Calculate exact position for each character
                        char_x = x_pos + (i * char_width)
                        ax.text(char_x, y_pos, char,
                                fontsize=font_size,
                                color=color,  # Color the text directly
                                fontproperties=mono_font,  # Use monospace font
                                ha='center', va='top')  # Center each character in its position
            
            # Set plot properties
            ax.set_xlim(0, 1)
            # Adjust y-axis limit based on the lowest token position to ensure all tokens are visible
            min_y = min([pos[1] for pos in token_positions]) - line_height if token_positions else 0
            ax.set_ylim(max(0, min_y - 0.05), 1)
            ax.axis('off')
            
            # Add colorbar for reference with explicit axes
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
            cbar.set_label('Token Probability')
            
            # Add title with problem information
            title = f"Target: {target} | Numbers: [{numbers}] | Equation: {equation}\n"
            title += f"Round: {round_num}, Group: {group_num}, Epoch: {epoch_num}, Step: {step_num}"
            ax.set_title(title, fontsize=12)
            
            # Only show token count in statistics
            stats_text = f"Total Tokens: {len(tokens)}"
            fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10)
            
            # Save the figure
            save_path = os.path.join(case_study_dir, f"prob_viz_R{round_num}_G{group_num}_E{epoch_num}_S{step_num}_T{target}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created token probability visualization: {save_path}")
        except Exception as e:
            print(f"Error creating visualization for example {i}: {e}")

# Add token probability visualization to the evaluator class
def add_token_probability_analysis(self, log_file: str):
    """Extract and visualize token probabilities from the log file."""
    print(f"\nExtracting token probabilities from {log_file}...")
    token_data = extract_token_probabilities(log_file, self.model_size)
    
    if token_data:
        print(f"Found {len(token_data)} examples with token probability data")
        visualize_token_probabilities(token_data, self.model_size, self.plots_dir)
        print(f"Token probability visualizations saved to: {os.path.join(self.plots_dir, f'case_study_{self.model_size}')}")
    else:
        print("No token probability data found in the log file")

# Add the method to the ContinualEvaluator class
ContinualEvaluator.add_token_probability_analysis = add_token_probability_analysis

# Update the evaluate_model method to include token probability analysis
def updated_evaluate_model(self, model_size: str):
    """Evaluate a specific model size."""
    print(f"\nEvaluating Qwen {model_size} model...")
    
    # Updated log file paths - use absolute paths
    # Handle special case for 3b model which might use different naming convention
    if model_size == "3b":
        model_size_prefix = "3B"
    else:
        model_size_prefix = model_size.upper()
        
    log_patterns = [
        os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_curriculum_*.log"),
        os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_R*_*.log"),
        os.path.join(self.logs_dir, f"ContinualCountdown{model_size_prefix}_SingleRun.log")
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
    
    # Extract group metrics for the group-based plot
    group_metrics = extract_group_metrics_from_log(log_file)
    
    if not metrics:
        print("Error: No metrics found to evaluate")
        print("Please check that training is outputting metrics in the expected format:")
        print("  - step:<number>")
        print("  - critic/score/mean:<number>")
        print("  - actor/pg_loss:<number>")
        print("  - actor/grad_norm:<number>")
        print("  - actor/entropy_loss:<number>")
        print("  - response_length/mean:<number>")
        return
    
    # Save metrics
    self.metrics = metrics
    self.save_metrics()
    
    # Create standard plots
    plot_path = os.path.join(self.plots_dir, f"training_metrics_{model_size}.png")
    self.plot_metrics(metrics, plot_path)
    
    # Create group-based plots if we have group metrics
    if group_metrics:
        group_plot_path_prefix = os.path.join(self.plots_dir, f"plot_{model_size}")
        self.plot_by_group(group_metrics, group_plot_path_prefix)
        print(f"Group-based plots saved to: {self.plots_dir}")
    else:
        print("No group metrics found for group-based plots")
    
    # Add token probability analysis
    self.add_token_probability_analysis(log_file)
    
    print(f"\n{model_size.upper()} evaluation completed successfully!")
    print("Results can be found in:")
    print(f"  - Plots: {plot_path}")
    print(f"  - Token Probability Visualizations: {os.path.join(self.plots_dir, f'case_study_{model_size}')}")
    print(f"  - Metrics: {os.path.join(self.metrics_dir, f'consolidated_metrics_{model_size}.json')}")

# Replace the original evaluate_model method
ContinualEvaluator.evaluate_model = updated_evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate continual learning results")
    parser.add_argument("--model-size", choices=["0.5b", "1.5b", "3b"], default="0.5b", help="Model size to evaluate")
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
