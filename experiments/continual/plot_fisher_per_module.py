"""
This script analyzes and plots experiment results from log files for a continual learning experiment.
It performs three main tasks:
1.  Plots RLHF performance curves (critic/score/mean) for different SFT steps.
2.  Plots per-module Fisher information metrics (C_K, L_K, sigma_max, sigma_min) trends over training.
3.  Plots case studies of how Fisher metrics change across layers at specific training steps.
"""

import os
import re
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

# Directory configurations will be set based on backend parameter
LOGS_DIR = './logs'
PLOTS_DIR = './plots/plot_fisher_per_module'
PARTIAL_PLOTS_DIR = os.path.join(PLOTS_DIR, 'partial')

# Regex to capture relevant log lines
PERFORMANCE_RE = re.compile(r'step:(\d+).*critic/score/mean:([\d.e+-]+)')
FISHER_RE = re.compile(r"\[FisherInfo\] Param '([^']+)': .*C_K=([\d.e+-]+), L_K=([\d.e+-]+), sigma_max=([\d.e+-]+), sigma_min=([\d.e+-]+)")
# Regex to capture normalized Fisher metrics from training logs
# Separate patterns for uppercase and lowercase C_K as they represent different metrics
FISHER_NORMALIZED_UPPERCASE_RE = re.compile(r'step:(\d+).*fisher/C_K_normalized:([\d.e+-]+).*fisher/sigma_max_normalized:([\d.e+-]+).*fisher/sigma_min_normalized:([\d.e+-]+)')
FISHER_NORMALIZED_LOWERCASE_RE = re.compile(r'step:(\d+).*fisher/c_k_normalized:([\d.e+-]+).*fisher/sigma_max_normalized:([\d.e+-]+).*fisher/sigma_min_normalized:([\d.e+-]+)')
RESPONSE_LENGTH_RE = re.compile(r'step:(\d+).*response_length/mean:([\d.e+-]+)')
ACTOR_ENTROPY_LOSS_RE = re.compile(r'step:(\d+).*actor/entropy_loss:([\d.e+-]+)')
PARAM_MODULE_RE = re.compile(r'model\.layers\.(\d+)\.(.*)')


def handle_outliers(y_values, method='percentile', percentile=95):
    """Handle outliers in data to improve visualization.
    
    Args:
        y_values: Array of y-values to process
        method: Method to handle outliers ('percentile', 'iqr', 'robust')
        percentile: Percentile threshold for clipping (default: 95)
    
    Returns:
        Processed y-values with outliers handled
    """
    if len(y_values) < 3:
        return y_values
    
    y_array = np.array(y_values)
    
    if method == 'percentile':
        # Clip values above the specified percentile
        upper_bound = np.percentile(y_array, percentile)
        return np.clip(y_array, None, upper_bound)
    
    elif method == 'iqr':
        # Use IQR method to identify and clip outliers
        Q1 = np.percentile(y_array, 25)
        Q3 = np.percentile(y_array, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        return np.clip(y_array, None, upper_bound)
    
    elif method == 'robust':
        # Use median + 3*MAD (Median Absolute Deviation) as upper bound
        median = np.median(y_array)
        mad = np.median(np.abs(y_array - median))
        upper_bound = median + 3 * mad
        return np.clip(y_array, None, upper_bound)
    
    return y_array


def smooth_curve(y_values, sigma=1.5):
    """Apply Gaussian smoothing to a curve.
    
    Args:
        y_values: Array of y-values to smooth
        sigma: Standard deviation for Gaussian kernel (higher = more smoothing)
    
    Returns:
        Smoothed y-values
    """
    if len(y_values) < 3:
        return y_values
    return gaussian_filter1d(y_values, sigma=sigma)


def compute_derived_C_K(c_k_values):
    """Compute C_K from c_k values using cumulative mean formula.
    
    C_K[j] = sum(c_k[i] for i in 0..j) / (j+1)
    
    Args:
        c_k_values: Array of c_k values
    
    Returns:
        Array of derived C_K values
    """
    if len(c_k_values) == 0:
        return np.array([])
    
    c_k_array = np.array(c_k_values)
    # Compute cumulative sum and divide by position+1 to get cumulative mean
    cumsum = np.cumsum(c_k_array)
    positions = np.arange(1, len(c_k_array) + 1)
    C_K_derived = cumsum / positions
    
    return C_K_derived


def parse_log_file(log_path, sft_step, exp_dir):
    """Parses a single log file to extract performance and Fisher info.

    Args:
        log_path (str): Path to the log file.
        sft_step (int): The SFT step for this experiment run.
        exp_dir (str): The source experiment directory path.

    Returns:
        list: A list of dictionaries, each containing parsed data for a step.
    """
    group_id = int(re.search(r'Group(\d+)', os.path.basename(log_path)).group(1))
    data = []
    current_step = -1
    
    with open(log_path, 'r') as f:
        for line in f:
            perf_match = PERFORMANCE_RE.search(line)
            if perf_match:
                current_step = int(perf_match.group(1))
                score = float(perf_match.group(2))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': current_step,
                    'score': score,
                    'param_name': 'performance_metric',
                    'C_K': np.nan,
                    'L_K': np.nan,
                    'sigma_max': np.nan,
                    'sigma_min': np.nan,
                    'C_K_normalized': np.nan,
                    'c_k_normalized': np.nan,
                    'sigma_max_normalized': np.nan,
                    'sigma_min_normalized': np.nan,
                    'response_length_mean': np.nan,
                    'actor_entropy_loss': np.nan,
                })

            # Parse response_length/mean metrics
            response_length_match = RESPONSE_LENGTH_RE.search(line)
            if response_length_match:
                step = int(response_length_match.group(1))
                response_length_mean = float(response_length_match.group(2))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': step,
                    'score': np.nan,
                    'param_name': 'response_length_mean',
                    'C_K': np.nan,
                    'L_K': np.nan,
                    'sigma_max': np.nan,
                    'sigma_min': np.nan,
                    'C_K_normalized': np.nan,
                    'c_k_normalized': np.nan,
                    'sigma_max_normalized': np.nan,
                    'sigma_min_normalized': np.nan,
                    'response_length_mean': response_length_mean,
                    'actor_entropy_loss': np.nan,
                })

            # Parse actor/entropy_loss metrics
            actor_entropy_loss_match = ACTOR_ENTROPY_LOSS_RE.search(line)
            if actor_entropy_loss_match:
                step = int(actor_entropy_loss_match.group(1))
                actor_entropy_loss = float(actor_entropy_loss_match.group(2))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': step,
                    'score': np.nan,
                    'param_name': 'actor_entropy_loss',
                    'C_K': np.nan,
                    'L_K': np.nan,
                    'sigma_max': np.nan,
                    'sigma_min': np.nan,
                    'C_K_normalized': np.nan,
                    'c_k_normalized': np.nan,
                    'sigma_max_normalized': np.nan,
                    'sigma_min_normalized': np.nan,
                    'response_length_mean': np.nan,
                    'actor_entropy_loss': actor_entropy_loss,
                })

            # Parse uppercase C_K_normalized Fisher metrics from training logs
            fisher_normalized_uppercase_match = FISHER_NORMALIZED_UPPERCASE_RE.search(line)
            if fisher_normalized_uppercase_match:
                step = int(fisher_normalized_uppercase_match.group(1))
                c_k_norm = float(fisher_normalized_uppercase_match.group(2))
                sigma_max_norm = float(fisher_normalized_uppercase_match.group(3))
                sigma_min_norm = float(fisher_normalized_uppercase_match.group(4))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': step,
                    'score': np.nan,
                    'param_name': 'normalized_fisher_metrics_uppercase',
                    'C_K': np.nan,
                    'L_K': np.nan,
                    'sigma_max': np.nan,
                    'sigma_min': np.nan,
                    'C_K_normalized': c_k_norm,
                    'c_k_normalized': np.nan,
                    'sigma_max_normalized': sigma_max_norm,
                    'sigma_min_normalized': sigma_min_norm,
                    'response_length_mean': np.nan,
                    'actor_entropy_loss': np.nan,
                })

            # Parse lowercase c_k_normalized Fisher metrics from training logs
            fisher_normalized_lowercase_match = FISHER_NORMALIZED_LOWERCASE_RE.search(line)
            if fisher_normalized_lowercase_match:
                step = int(fisher_normalized_lowercase_match.group(1))
                c_k_norm = float(fisher_normalized_lowercase_match.group(2))
                sigma_max_norm = float(fisher_normalized_lowercase_match.group(3))
                sigma_min_norm = float(fisher_normalized_lowercase_match.group(4))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': step,
                    'score': np.nan,
                    'param_name': 'normalized_fisher_metrics_lowercase',
                    'C_K': np.nan,
                    'L_K': np.nan,
                    'sigma_max': np.nan,
                    'sigma_min': np.nan,
                    'C_K_normalized': np.nan,
                    'c_k_normalized': c_k_norm,
                    'sigma_max_normalized': sigma_max_norm,
                    'sigma_min_normalized': sigma_min_norm,
                    'response_length_mean': np.nan,
                    'actor_entropy_loss': np.nan,
                })

            fisher_match = FISHER_RE.search(line)
            if fisher_match and current_step != -1:
                param_name = fisher_match.group(1)
                c_k = float(fisher_match.group(2))
                l_k = float(fisher_match.group(3))
                sigma_max = float(fisher_match.group(4))
                sigma_min = float(fisher_match.group(5))
                data.append({
                    'exp_dir': exp_dir,
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': current_step,
                    'score': np.nan,
                    'param_name': param_name,
                    'C_K': c_k,
                    'L_K': l_k,
                    'sigma_max': sigma_max,
                    'sigma_min': sigma_min,
                    'C_K_normalized': np.nan,
                    'c_k_normalized': np.nan,
                    'sigma_max_normalized': np.nan,
                    'sigma_min_normalized': np.nan,
                    'response_length_mean': np.nan,
                    'actor_entropy_loss': np.nan,
                })
    return data

def extract_module_info(param_name):
    """Extracts layer number and module name from a parameter name."""
    if 'embed_tokens' in param_name:
        return 0, 'embed_tokens'
    if 'lm_head' in param_name:
        return -1, 'lm_head' # Use -1 for lm_head to place it last

    match = PARAM_MODULE_RE.search(param_name)
    if not match:
        return None, None

    layer = int(match.group(1))
    
    # Simplify module name
    module_part = match.group(2)
    if 'self_attn' in module_part:
        sub_module = module_part.split('.')[1] # q_proj, k_proj, etc.
        module_name = f'self_attn.{sub_module}'
    elif 'mlp' in module_part:
        sub_module = module_part.split('.')[1] # gate_proj, etc.
        module_name = f'mlp.{sub_module}'
    elif 'layernorm' in module_part:
        module_name = module_part.split('.')[0]
    else:
        module_name = module_part

    return layer, module_name

def plot_performance_curves(df, plots_dir):
    """Plots RLHF performance curves for Known and Unknown groups."""
    perf_df = df[df['score'].notna()].copy()
    if perf_df.empty:
        print("No performance data found to plot.")
        return

    for group_type in ['Known', 'Unknown']:
        plt.figure(figsize=(12, 8))
        group_df = perf_df[perf_df['group_type'] == group_type]

        for sft_step in sorted(group_df['sft_step'].unique()):
            sft_df = group_df[group_df['sft_step'] == sft_step]
            
            # Group by training step and calculate mean and std for variance
            agg_df = sft_df.groupby('training_step')['score'].agg(['mean', 'std']).reset_index()
            agg_df['std'] = agg_df['std'].fillna(0)

            plt.plot(agg_df['training_step'], agg_df['mean'], label=f'SFT Step {sft_step}')
            # Clip the lower bound of the shadow at 0
            lower_bound = np.maximum(0, agg_df['mean'] - agg_df['std'])
            plt.fill_between(
                agg_df['training_step'],
                lower_bound,
                agg_df['mean'] + agg_df['std'],
                alpha=0.2
            )

        plt.title(f'RLHF Performance (critic/score/mean) - {group_type} Groups')
        plt.xlabel('Training Step')
        plt.ylabel('Critic Score Mean')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'performance_{group_type.lower()}.png'))
        plt.close()
        print(f"Saved performance plot for {group_type} groups.")

def plot_additional_metrics(df, plots_dir):
    """Plots additional metrics: response_length/mean and actor/entropy_loss."""
    # Define the metrics to plot
    additional_metrics = ['response_length_mean', 'actor_entropy_loss']
    
    for metric in additional_metrics:
        # Filter data for this metric
        metric_df = df[df['param_name'] == metric].copy()
        if metric_df.empty:
            print(f"No data found for {metric}.")
            continue
            
        for group_type in ['Known', 'Unknown']:
            plt.figure(figsize=(12, 8))
            group_df = metric_df[metric_df['group_type'] == group_type]
            
            if group_df.empty:
                print(f"No {metric} data found for {group_type} groups.")
                plt.close()
                continue
            
            for sft_step in sorted(group_df['sft_step'].unique()):
                sft_df = group_df[group_df['sft_step'] == sft_step]
                
                # Group by training step and calculate mean and std
                if metric == 'response_length_mean':
                    agg_df = sft_df.groupby('training_step')['response_length_mean'].agg(['mean', 'std']).reset_index()
                else:  # actor_entropy_loss
                    agg_df = sft_df.groupby('training_step')['actor_entropy_loss'].agg(['mean', 'std']).reset_index()
                
                agg_df['std'] = agg_df['std'].fillna(0)
                
                # Use performance plot style (clean lines, no markers)
                plt.plot(agg_df['training_step'], agg_df['mean'], 
                        label=f'SFT Step {sft_step}', linewidth=2)
                
                # Add confidence interval
                plt.fill_between(
                    agg_df['training_step'],
                    agg_df['mean'] - agg_df['std'],
                    agg_df['mean'] + agg_df['std'],
                    alpha=0.2
                )
            
            # Format metric name for display
            display_name = metric.replace('_', ' ').replace('mean', 'Mean').replace('loss', 'Loss').title()
            if 'response' in metric.lower():
                display_name = 'Response Length Mean'
            elif 'entropy' in metric.lower():
                display_name = 'Actor Entropy Loss'
                
            plt.title(f'{display_name} - {group_type} Groups', fontsize=14)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel(display_name, fontsize=12)
            plt.legend()
            plt.grid(True)
            
            # Save plot
            filename = f'{metric}_{group_type.lower()}.png'
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()
            print(f"Saved {display_name} plot for {group_type} groups.")

def plot_fisher_trends(df, plots_dir):
    """Plots per-module C_K, L_K, sigma_max, and sigma_min trends across training steps."""
    fisher_df = df[(df['C_K'].notna()) & (df['module'].notna())].copy()
    if fisher_df.empty:
        print("No Fisher information data found to plot trends.")
        return
        
    modules_to_plot = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
        'input_layernorm', 'post_attention_layernorm'
    ]
    fisher_df = fisher_df[fisher_df['module'].isin(modules_to_plot)]

    metrics_to_plot = ['C_K', 'L_K', 'sigma_max', 'sigma_min']

    for group_type in ['Known', 'Unknown']:
        group_df = fisher_df[fisher_df['group_type'] == group_type]
        
        # Plot 1: Per-module trends for each SFT step
        for sft_step in sorted(group_df['sft_step'].unique()):
            sft_df = group_df[group_df['sft_step'] == sft_step]
            agg_df = sft_df.groupby(['training_step', 'module'])[metrics_to_plot].mean().reset_index()

            for metric in metrics_to_plot:
                plt.figure(figsize=(15, 10))
                for module in modules_to_plot:
                    module_df = agg_df[agg_df['module'] == module]
                    if not module_df.empty:
                        plt.plot(module_df['training_step'], module_df[metric], label=module, marker='o', linestyle='-', markersize=2)
                
                plt.title(f'{metric} Trend - SFT {sft_step} - {group_type} Group')
                plt.xlabel('Training Step')
                plt.ylabel(f'Average {metric}')
                plt.yscale('log' if metric == 'L_K' else 'linear')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.grid(True, which="both", ls="--")
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(os.path.join(plots_dir, f'{metric}_trend_sft{sft_step}_{group_type.lower()}.png'))
                plt.close()
                print(f"Saved {metric} trend plot for SFT {sft_step}, {group_type} group.")

        # Plot 2: Average trend across all modules, comparing SFT steps
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 8))
            for sft_step in sorted(group_df['sft_step'].unique()):
                sft_df = group_df[group_df['sft_step'] == sft_step]
                
                # Average across all modules, layers, and runs for each training step
                avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()

                if not avg_metric_df.empty:
                    plt.plot(avg_metric_df['training_step'], avg_metric_df[metric], label=f'SFT Step {sft_step}', marker='o', linestyle='-', markersize=4)

            plt.title(f'Average {metric} Trend Comparison - {group_type} Group')
            plt.xlabel('Training Step')
            plt.ylabel(f'Average {metric} (all modules)')
            plt.yscale('log' if metric == 'L_K' else 'linear')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{metric}_trend_average_{group_type.lower()}.png'))
            plt.close()
            print(f"Saved average {metric} trend plot for {group_type} group.")

def plot_normalized_fisher_comparison(df, plots_dir):
    """Plots comparison of normalized Fisher metrics across different SFT steps."""
    # Filter for both uppercase and lowercase normalized Fisher metrics data
    uppercase_df = df[df['param_name'] == 'normalized_fisher_metrics_uppercase'].copy()
    lowercase_df = df[df['param_name'] == 'normalized_fisher_metrics_lowercase'].copy()
    
    if uppercase_df.empty and lowercase_df.empty:
        print("No normalized Fisher metrics data found to plot.")
        return
    
    # Process uppercase metrics (C_K_normalized)
    if not uppercase_df.empty:
        uppercase_df['group_type'] = uppercase_df['group_id'].apply(lambda x: 'Known' if x == 0 else 'Unknown')
        plot_fisher_metric_set(uppercase_df, plots_dir, 'uppercase', ['C_K_normalized', 'sigma_max_normalized', 'sigma_min_normalized'])
    
    # Process lowercase metrics (c_k_normalized)
    if not lowercase_df.empty:
        lowercase_df['group_type'] = lowercase_df['group_id'].apply(lambda x: 'Known' if x == 0 else 'Unknown')
        plot_fisher_metric_set(lowercase_df, plots_dir, 'lowercase', ['c_k_normalized', 'sigma_max_normalized', 'sigma_min_normalized'])

def plot_fisher_metric_set(normalized_df, plots_dir, metric_type, metrics_to_plot):
    """Plot a set of Fisher metrics (either uppercase or lowercase variants)."""
    
    for group_type in ['Known', 'Unknown']:
        group_df = normalized_df[normalized_df['group_type'] == group_type]
        if group_df.empty:
            continue
            
        # Create subplots for the three metrics
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'Normalized Fisher Metrics Comparison ({metric_type.title()}) - {group_type} Group', fontsize=16)
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Plot each SFT step
            for sft_step in sorted(group_df['sft_step'].unique()):
                sft_df = group_df[group_df['sft_step'] == sft_step]
                
                # Average across all runs for each training step
                avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                
                if not avg_metric_df.empty:
                    y_values = avg_metric_df[metric].values
                    # Apply outlier handling only for c_k_normalized metrics
                    if 'c_k_normalized' in metric and metric != 'C_K_normalized':
                        y_values = handle_outliers(y_values, method='percentile', percentile=90)
                    # Use performance plot style for c_k_normalized metrics
                    if 'c_k_normalized' in metric and metric != 'C_K_normalized':
                        ax.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                    else:
                        ax.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', marker='o', linestyle='-', markersize=3)
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'normalized_fisher_comparison_{group_type.lower()}_{"_".join(metrics_to_plot)}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined normalized Fisher comparison plot for {group_type} group.")
        
        # Create additional smoothed version for c_k_normalized metrics
        c_k_metrics = [m for m in metrics_to_plot if 'c_k_normalized' in m and m != 'C_K_normalized']
        if c_k_metrics:
            n_metrics = len(c_k_metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6*n_metrics))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(c_k_metrics):
                ax = axes[i]
                
                for sft_step in sorted(group_df['sft_step'].unique()):
                    sft_df = group_df[group_df['sft_step'] == sft_step]
                    
                    # Average across all runs for each training step
                    avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                    
                    if not avg_metric_df.empty:
                        y_values = avg_metric_df[metric].values
                        # Apply both outlier handling and smoothing
                        y_values = handle_outliers(y_values, method='percentile', percentile=90)
                        y_values = smooth_curve(y_values, sigma=1.5)
                        ax.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                
                ax.set_title(f'{metric.replace("_", " ").title()} (Smoothed) - {group_type} Group', fontsize=12)
                ax.set_xlabel('Training Step', fontsize=10)
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'normalized_fisher_comparison_smoothed_{group_type.lower()}_{"_".join(c_k_metrics)}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined smoothed normalized Fisher comparison plot for {group_type} group.")
            
            # Create additional derived C_K version from processed c_k_normalized metrics
            n_metrics = len(c_k_metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6*n_metrics))
            if n_metrics == 1:
                axes = [axes]
            
            for i, metric in enumerate(c_k_metrics):
                ax = axes[i]
                
                for sft_step in sorted(group_df['sft_step'].unique()):
                    sft_df = group_df[group_df['sft_step'] == sft_step]
                    
                    # Average across all runs for each training step
                    avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                    
                    if not avg_metric_df.empty:
                        c_k_values = avg_metric_df[metric].values
                        # Apply outlier handling to c_k values
                        c_k_processed = handle_outliers(c_k_values, method='percentile', percentile=90)
                        # Compute derived C_K from processed c_k values
                        C_K_derived = compute_derived_C_K(c_k_processed)
                        ax.plot(avg_metric_df['training_step'], C_K_derived, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                
                derived_metric_name = metric.replace('c_k_normalized', 'C_K_derived_normalized')
                ax.set_title(f'{derived_metric_name.replace("_", " ").title()} (Derived from Processed c_k) - {group_type} Group', fontsize=12)
                ax.set_xlabel('Training Step', fontsize=10)
                ax.set_ylabel(derived_metric_name.replace('_', ' ').title(), fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            
            derived_metric_names = [m.replace('c_k_normalized', 'C_K_derived_normalized') for m in c_k_metrics]
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'normalized_fisher_comparison_derived_{group_type.lower()}_{"_".join(derived_metric_names)}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined derived C_K normalized Fisher comparison plot for {group_type} group.")
        
        # Also create individual plots for each metric
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 8))
            
            for sft_step in sorted(group_df['sft_step'].unique()):
                sft_df = group_df[group_df['sft_step'] == sft_step]
                
                # Average across all runs for each training step
                avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                
                if not avg_metric_df.empty:
                    y_values = avg_metric_df[metric].values
                    # Apply outlier handling only for c_k_normalized metrics
                    if 'c_k_normalized' in metric and metric != 'C_K_normalized':
                        y_values = handle_outliers(y_values, method='percentile', percentile=90)
                    # Use performance plot style for c_k_normalized metrics
                    if 'c_k_normalized' in metric and metric != 'C_K_normalized':
                        plt.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                    else:
                        plt.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', marker='o', linestyle='-', markersize=4, linewidth=2)
            
            plt.title(f'{metric.replace("_", " ").title()} Comparison Across SFT Steps - {group_type} Group', fontsize=14)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{metric}_comparison_{group_type.lower()}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {metric} comparison plot for {group_type} group.")
            
            # Create additional smoothed version for c_k_normalized metrics
            if 'c_k_normalized' in metric and metric != 'C_K_normalized':
                plt.figure(figsize=(12, 8))
                
                for sft_step in sorted(group_df['sft_step'].unique()):
                    sft_df = group_df[group_df['sft_step'] == sft_step]
                    
                    # Average across all runs for each training step
                    avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                    
                    if not avg_metric_df.empty:
                        y_values = avg_metric_df[metric].values
                        # Apply both outlier handling and smoothing
                        y_values = handle_outliers(y_values, method='percentile', percentile=90)
                        y_values = smooth_curve(y_values, sigma=1.5)
                        plt.plot(avg_metric_df['training_step'], y_values, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                
                plt.title(f'{metric.replace("_", " ").title()} Comparison (Smoothed) - {group_type} Group', fontsize=14)
                plt.xlabel('Training Step', fontsize=12)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{metric}_comparison_smoothed_{group_type.lower()}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {metric} smoothed comparison plot for {group_type} group.")
                
                # Create additional derived C_K plot from processed c_k_normalized
                plt.figure(figsize=(12, 8))
                
                for sft_step in sorted(group_df['sft_step'].unique()):
                    sft_df = group_df[group_df['sft_step'] == sft_step]
                    
                    # Average across all runs for each training step
                    avg_metric_df = sft_df.groupby('training_step')[metric].mean().reset_index()
                    
                    if not avg_metric_df.empty:
                        c_k_values = avg_metric_df[metric].values
                        # Apply outlier handling to c_k values
                        c_k_processed = handle_outliers(c_k_values, method='percentile', percentile=90)
                        # Compute derived C_K from processed c_k values
                        C_K_derived = compute_derived_C_K(c_k_processed)
                        plt.plot(avg_metric_df['training_step'], C_K_derived, 
                               label=f'SFT Step {sft_step}', linewidth=2)
                
                derived_metric_name = metric.replace('c_k_normalized', 'C_K_derived_normalized')
                plt.title(f'{derived_metric_name.replace("_", " ").title()} (Derived from Processed c_k) - {group_type} Group', fontsize=14)
                plt.xlabel('Training Step', fontsize=12)
                plt.ylabel(derived_metric_name.replace('_', ' ').title(), fontsize=12)
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{derived_metric_name}_comparison_{group_type.lower()}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {derived_metric_name} derived comparison plot for {group_type} group.")

def plot_fisher_case_study(df, plots_dir):
    """Plots C_K/L_K/sigma_max/sigma_min across layers for specific training steps."""
    fisher_df = df[(df['C_K'].notna()) & (df['layer'] >= 0)].copy() # Exclude lm_head
    if fisher_df.empty:
        print("No Fisher information data found for case study.")
        return

    # Dynamically determine case study steps based on the max step in the data
    max_step = fisher_df['training_step'].max()
    case_study_steps = [1, 25, 50, 100]
    if pd.notna(max_step):
        case_study_steps.append(int(max_step))
    case_study_steps = sorted(list(set(case_study_steps)))

    modules_to_plot = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
        'input_layernorm', 'post_attention_layernorm'
    ]
    fisher_df = fisher_df[fisher_df['module'].isin(modules_to_plot)]
    metrics_to_plot = ['C_K', 'L_K', 'sigma_max', 'sigma_min']

    for sft_step in sorted(fisher_df['sft_step'].unique()):
        for group_type in ['Known', 'Unknown']:
            for train_step in case_study_steps:
                # Use data from step 149, but display it as 150
                display_step = 150 if train_step == 149 else train_step
                
                step_df = fisher_df[
                    (fisher_df['sft_step'] == sft_step) &
                    (fisher_df['group_type'] == group_type) &
                    (fisher_df['training_step'] == train_step)
                ]
                if step_df.empty:
                    continue

                # Average across runs for each layer and module
                agg_df = step_df.groupby(['layer', 'module'])[metrics_to_plot].mean().reset_index()

                for metric in metrics_to_plot:
                    plt.figure(figsize=(15, 10))
                    for module in modules_to_plot:
                        module_df = agg_df[agg_df['module'] == module].sort_values('layer')
                        if not module_df.empty:
                            plt.plot(module_df['layer'], module_df[metric], label=module, marker='o', linestyle='-')

                    plt.title(f'{metric} vs Layer - SFT {sft_step} - Step {display_step} - {group_type} Group')
                    plt.xlabel('Layer Number')
                    plt.ylabel(f'Average {metric}')
                    plt.yscale('log' if metric == 'L_K' else 'linear')
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.grid(True, which="both", ls="--")
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig(os.path.join(plots_dir, f'case_study_{metric}_sft{sft_step}_step{display_step}_{group_type.lower()}.png'))
                    plt.close()
    print("Saved all case study plots.")

def get_max_step_from_log(log_path):
    """Extracts the maximum training step from a log file."""
    max_step = -1
    with open(log_path, 'r') as f:
        for line in f:
            match = PERFORMANCE_RE.search(line)
            if match:
                max_step = max(max_step, int(match.group(1)))
    return max_step

def get_directories(backend):
    """Get logs and plots directories based on backend."""
    if backend == 'llama':
        logs_dir = './llama_logs'
        plots_dir = './llama_plots/plot_fisher_per_module'
    elif backend == 'qwen':
        logs_dir = './qwen_logs'
        plots_dir = './qwen_plots/plot_fisher_per_module'
    else:  # default
        logs_dir = './logs'
        plots_dir = './plots/plot_fisher_per_module'
    
    partial_plots_dir = os.path.join(plots_dir, 'partial')
    return logs_dir, plots_dir, partial_plots_dir

def process_backend(backend):
    """Process a single backend (llama, qwen, or default)."""
    logs_dir, plots_dir, partial_plots_dir = get_directories(backend)
    
    print(f"\n=== Processing {backend.upper()} Backend ===")
    print(f"Logs directory: {logs_dir}")
    print(f"Plots directory: {plots_dir}")
    
    # Clean and create directories
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)
    os.makedirs(plots_dir)
    os.makedirs(partial_plots_dir, exist_ok=True)

    exp_dirs = glob.glob(os.path.join(logs_dir, '*sft_global_step_*'))
    if not exp_dirs:
        print(f"Error: No experiment directories found in {logs_dir}")
        return

    complete_data = []
    partial_data = []

    for exp_dir in exp_dirs:
        # --- Classify Experiment ---
        all_log_files = glob.glob(os.path.join(exp_dir, 'Group*.log'))

        # Process even if there's only one log file
        if len(all_log_files) == 0:
            print(f"No log files found in experiment: {exp_dir}")
            continue
        
        if len(all_log_files) == 1:
            print(f"Processing experiment with single group log: {exp_dir}")

        group0_log = next((f for f in all_log_files if 'Group0' in os.path.basename(f)), None)

        # 1. Group0 must exist and have steps.
        if not group0_log:
            print(f"Ignoring experiment (missing Group0 log): {exp_dir}")
            continue
        
        ref_step = get_max_step_from_log(group0_log)
        if ref_step == -1:
            print(f"Ignoring experiment (Group0 has no steps): {exp_dir}")
            continue

        # 2. Check for step consistency against Group0. (DISABLED - treat all as complete)
        is_consistent = True
        max_steps = {0: ref_step}
        for log_file in all_log_files:
            if log_file == group0_log:
                continue
            match = re.search(r'Group(\d+)', os.path.basename(log_file))
            if match:
                gid = int(match.group(1))
                g_step = get_max_step_from_log(log_file)
                max_steps[gid] = g_step
                # if g_step != ref_step:
                #     is_consistent = False
        
        # is_run_complete = is_consistent and ref_step >= 149
        is_run_complete = True  # Treat all experiments as complete

        # --- Parse and Assign Data ---
        try:
            sft_step = int(re.search(r'global_step_(\d+)', exp_dir).group(1))
        except (AttributeError, ValueError):
            print(f"Could not parse SFT step from directory name: {exp_dir}")
            continue

        current_exp_data = []
        for log_file in all_log_files:
            current_exp_data.extend(parse_log_file(log_file, sft_step, exp_dir))

        if not current_exp_data:
            continue

        if is_run_complete:
            print(f"Found complete experiment: {exp_dir}")
            complete_data.extend(current_exp_data)
        else:
            print(f"Found partial experiment: {exp_dir} (Steps: {max_steps})")
            partial_data.extend(current_exp_data)

    # --- Helper function for processing and plotting ---
    def _process_and_plot(data, data_type, plots_dir):
        if not data:
            print(f"\nNo data for {data_type} experiments to process.")
            return

        print(f"\n--- Processing {data_type} Data ---")
        df = pd.DataFrame(data)

        # Truncate all data to maximum 150 steps
        df = df[df['training_step'] <= 150]
        
        # For partial data, truncate to the minimum common step for each experiment
        if data_type == "Partial":
            max_steps = df.groupby(['exp_dir', 'group_id'])['training_step'].max()
            truncation_steps = max_steps.groupby('exp_dir').min().rename('truncation_step')
            df = df.merge(truncation_steps, on='exp_dir')
            df = df[df['training_step'] <= df['truncation_step']].drop(columns=['truncation_step'])

        # Pre-processing
        df['group_type'] = df['group_id'].apply(lambda x: 'Known' if x == 0 else 'Unknown')
        fisher_mask = df['C_K'].notna()
        if fisher_mask.any():
            parsed_tuples = df.loc[fisher_mask, 'param_name'].apply(extract_module_info)
            df.loc[fisher_mask, 'layer'] = parsed_tuples.str[0]
            df.loc[fisher_mask, 'module'] = parsed_tuples.str[1]
        
        print(f"Total records: {len(df)}")
        print(f"Performance records: {df['score'].notna().sum()}")
        print(f"Fisher records: {df['C_K'].notna().sum()}")
        print(f"Normalized Fisher records: {(df['param_name'] == 'normalized_fisher_metrics').sum()}")

        # Plotting
        plot_performance_curves(df, plots_dir)
        plot_additional_metrics(df, plots_dir)  # New additional metrics: response_length/mean and actor/entropy_loss
        plot_fisher_trends(df, plots_dir)
        plot_normalized_fisher_comparison(df, plots_dir)  # New normalized Fisher metrics plots
        plot_fisher_case_study(df, plots_dir)
        print(f"\nAll {data_type} plots saved to {plots_dir}")

    # --- Process and plot COMPLETE data ---
    _process_and_plot(complete_data, "Complete", plots_dir)

    # --- Process and plot PARTIAL data --- (DISABLED)
    # if partial_data:
    #     _process_and_plot(partial_data, "Partial", partial_plots_dir)

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Plot Fisher information metrics from experiment logs')
    parser.add_argument('--backend', type=str, default='default', 
                       choices=['llama', 'qwen', 'default', 'all'],
                       help='Backend to process: llama (llama_logs->llama_plots), qwen (qwen_logs->qwen_plots), default (logs->plots), or all (process all three)')
    
    args = parser.parse_args()
    
    if args.backend == 'all':
        # Process all backends
        backends = ['default', 'llama', 'qwen']
        for backend in backends:
            try:
                process_backend(backend)
            except Exception as e:
                print(f"Error processing {backend} backend: {e}")
                continue
    else:
        # Process single backend
        process_backend(args.backend)
    
    print("\n=== Processing Complete ===")

if __name__ == '__main__':
    main()
