#!/usr/bin/env python3
"""
Script to plot critic/score/mean from experimental logs.
Supports both llama_logs and qwen_logs with parameter extraction.
Creates separate plots for each experiment folder.
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import seaborn as sns

def extract_parameters_from_dirname(dirname):
    """
    Extract parameters from directory name.
    Expected format: {prefix}_step_{SFT_STEP}_reset_k{reset_first_num}f_k{reset_last_num}l
    """
    pattern = r'.*_step_(\d+)_reset_k(\d+)f_k(\d+)l'
    match = re.search(pattern, dirname)
    if match:
        sft_step = int(match.group(1))
        reset_first_num = int(match.group(2))
        reset_last_num = int(match.group(3))
        return sft_step, reset_first_num, reset_last_num
    return None, None, None

def parse_log_line(line):
    """
    Parse a log line to extract step number and critic/score/mean value.
    """
    # Extract step number
    step_match = re.search(r'step:(\d+)', line)
    if not step_match:
        return None, None
    
    step = int(step_match.group(1))
    
    # Extract critic/score/mean value
    score_match = re.search(r'critic/score/mean:([\d\.-]+)', line)
    if not score_match:
        return None, None
    
    score = float(score_match.group(1))
    return step, score

def read_experiment_logs(log_dir):
    """
    Read all log files in an experiment directory and extract critic/score/mean data.
    Returns dict: {step: [list of scores from different log files]}
    """
    step_scores = defaultdict(list)
    
    # Find all .log files (excluding experiment_master.log)
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    log_files = [f for f in log_files if not f.endswith("experiment_master.log")]
    
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'critic/score/mean:' in line:
                        step, score = parse_log_line(line)
                        if step is not None and score is not None:
                            step_scores[step].append(score)
        except Exception as e:
            print(f"Warning: Could not read {log_file}: {e}")
    
    return step_scores

def aggregate_scores(step_scores):
    """
    Aggregate scores across multiple log files for each step.
    Returns steps, means, stds
    """
    steps = sorted(step_scores.keys())
    means = []
    stds = []
    
    for step in steps:
        scores = step_scores[step]
        if scores:
            means.append(np.mean(scores))
            stds.append(np.std(scores) if len(scores) > 1 else 0.0)
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return np.array(steps), np.array(means), np.array(stds)

def plot_single_experiment(exp_dir, exp_path, model_type, output_dir):
    """
    Plot critic/score/mean for a single experiment.
    """
    # Extract parameters from directory name
    sft_step, reset_first, reset_last = extract_parameters_from_dirname(exp_dir)
    
    # Read log data
    step_scores = read_experiment_logs(exp_path)
    if not step_scores:
        print(f"Warning: No data found in {exp_dir}")
        return
    
    # Aggregate scores
    steps, means, stds = aggregate_scores(step_scores)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(means)
    steps_valid = steps[valid_mask]
    means_valid = means[valid_mask]
    stds_valid = stds[valid_mask]
    
    if len(steps_valid) == 0:
        print(f"Warning: No valid data points in {exp_dir}")
        return
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line with error bars (shadow)
    color = '#1f77b4'  # Default blue color
    ax.plot(steps_valid, means_valid, color=color, linewidth=2, label='Critic Score Mean')
    ax.fill_between(steps_valid, 
                   means_valid - stds_valid, 
                   means_valid + stds_valid, 
                   color=color, alpha=0.2, label='±1 std')
    
    # Add reset operation markers at steps 40, 80, and 120 only if reset is actually performed
    if reset_first is not None and reset_last is not None and (reset_first > 0 or reset_last > 0):
        reset_steps = [40, 80, 120]
        for reset_step in reset_steps:
            if reset_step >= steps_valid.min() and reset_step <= steps_valid.max():
                ax.axvline(x=reset_step, color='red', linestyle='--', alpha=0.7, linewidth=2)
                ax.text(reset_step, ax.get_ylim()[1] * 0.95, f'Reset\n{reset_step}', 
                       horizontalalignment='center', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                       fontsize=10, color='red', weight='bold')
    
    # Create title with parameters
    if sft_step is not None:
        title = f'{model_type.upper()} - SFT Step: {sft_step}, Reset First: {reset_first}, Reset Last: {reset_last}'
    else:
        title = f'{model_type.upper()} - {exp_dir}'
    
    # Customize plot
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Critic Score Mean', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Steps: {len(steps_valid)}\nMean: {np.mean(means_valid):.3f}\nMax: {np.max(means_valid):.3f}\nMin: {np.min(means_valid):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_exp_name = re.sub(r'[^\w\-_]', '_', exp_dir)
    output_file = os.path.join(output_dir, f'{safe_exp_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()  # Close the figure to free memory

def plot_experiments(base_dir, model_type, output_dir):
    """
    Plot critic/score/mean for each experiment in separate plots.
    """
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    if not exp_dirs:
        print(f"No experiment directories found in {base_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    for exp_dir in sorted(exp_dirs):
        exp_path = os.path.join(base_dir, exp_dir)
        print(f"\nProcessing experiment: {exp_dir}")
        plot_single_experiment(exp_dir, exp_path, model_type, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Plot critic/score/mean from experimental logs')
    parser.add_argument('--model_type', choices=['llama', 'qwen'], required=True,
                       help='Model type: llama or qwen')
    parser.add_argument('--log_dir', type=str, 
                       help='Custom log directory (default: {model_type}_logs)')
    parser.add_argument('--output_dir', type=str,
                       help='Custom output directory (default: ./{model_type}_plots/reset_{model_type})')
    
    args = parser.parse_args()
    
    # Set default directories
    if args.log_dir is None:
        args.log_dir = f'{args.model_type}_logs'
    
    if args.output_dir is None:
        args.output_dir = f'./{args.model_type}_plots/reset_{args.model_type}'
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    
    log_dir = os.path.join(project_root, args.log_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    print(f"Looking for logs in: {log_dir}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    # Plot experiments
    plot_experiments(log_dir, args.model_type, output_dir)

if __name__ == '__main__':
    main()
