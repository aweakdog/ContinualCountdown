#!/usr/bin/env python3
"""
Continual Learning Experiment Analysis and Plotting Script

This script analyzes and plots experiment results from log files for continual learning experiments
on countdown problems. It supports both Llama and Qwen models with phase-based experimental design.
"""

import os
import re
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Regex patterns for parsing log files
# Pattern for LLAMA logs (with SFT_global_step)
LOG_FILENAME_RE_LLAMA = re.compile(r'Phase(\d+)_Group([0-9and]+)_Iter(\d+)_SFT_global_step_(\d+)_(\d{8}_\d{6})\.log')
# Pattern for QWEN logs (without SFT_global_step)
LOG_FILENAME_RE_QWEN = re.compile(r'Phase(\d+)_Group([0-9and]+)_Iter(\d+)_(\d{8}_\d{6})\.log')
# Pattern for QWEN directory names
QWEN_DIR_RE = re.compile(r'.*_sft_global_step_(\d+)(?:_\d{8}_\d{6})?/?$')
STEP_RE = re.compile(r'step:(\d+)')

# Metric extraction patterns
METRICS_PATTERNS = {
    'critic_score_mean': re.compile(r'step:(\d+).*critic/score/mean:([\d.e+-]+)'),
    'val_test_score': re.compile(r'step:(\d+).*val/test_score/countdown_continual_simpler:([\d.e+-]+)'),
    'response_length_mean': re.compile(r'step:(\d+).*response_length/mean:([\d.e+-]+)'),
    'fisher_C_K_normalized': re.compile(r'step:(\d+).*fisher/C_K_normalized:([\d.e+-]+)'),
    'fisher_c_k_normalized': re.compile(r'step:(\d+).*fisher/c_k_normalized:([\d.e+-]+)'),
    'fisher_L_K_normalized': re.compile(r'step:(\d+).*fisher/L_K_normalized:([\d.e+-]+)'),
    'fisher_l_k_normalized': re.compile(r'step:(\d+).*fisher/l_k_normalized:([\d.e+-]+)'),
    'fisher_sigma_max_normalized': re.compile(r'step:(\d+).*fisher/sigma_max_normalized:([\d.e+-]+)'),
    'fisher_sigma_min_normalized': re.compile(r'step:(\d+).*fisher/sigma_min_normalized:([\d.e+-]+)'),
    'fisher_c_k_running_avg': re.compile(r'step:(\d+).*fisher/c_k_running_avg:([\d.e+-]+)'),
    'actor_zero_gradspace_ratio': re.compile(r'step:(\d+).*actor/zero_gradspace_ratio:([\d.e+-]+)'),
    'critic_grad_norm': re.compile(r'step:(\d+).*critic/grad_norm:([\d.e+-]+)'),
    'actor_pg_loss': re.compile(r'step:(\d+).*actor/pg_loss:([\d.e+-]+)'),
    'critic_vf_loss': re.compile(r'step:(\d+).*critic/vf_loss:([\d.e+-]+)'),
    'actor_entropy_loss': re.compile(r'step:(\d+).*actor/entropy_loss:([\d.e+-]+)'),
    'actor_ppo_kl': re.compile(r'step:(\d+).*actor/ppo_kl:([\d.e+-]+)'),
    'critic_kl': re.compile(r'step:(\d+).*critic/kl:([\d.e+-]+)'),
    'actor_pg_clipfrac': re.compile(r'step:(\d+).*actor/pg_clipfrac:([\d.e+-]+)'),
    'critic_vf_clipfrac': re.compile(r'step:(\d+).*critic/vf_clipfrac:([\d.e+-]+)'),
}


def smooth_curve(y_values, sigma=1.5):
    """Apply Gaussian smoothing to a curve."""
    if len(y_values) < 3:
        return y_values
    return gaussian_filter1d(y_values, sigma=sigma)


def handle_outliers(y_values, method='percentile', percentile=95):
    """Handle outliers in data to improve visualization.
    
    Detects outliers using the specified method and replaces them with the mean of 
    the previous 3 values (temporal local averaging).
    """
    if len(y_values) < 3:
        return y_values
    
    y_array = np.array(y_values).copy()  # Make a copy to avoid modifying original
    
    if method == 'percentile':
        # Calculate bounds for outlier detection
        upper_bound = np.percentile(y_array, percentile)
        lower_bound = np.percentile(y_array, 100 - percentile)
        
        # Process each point sequentially
        for i in range(len(y_array)):
            # Check if current point is an outlier
            if y_array[i] < lower_bound or y_array[i] > upper_bound:
                # Only replace outliers if we have at least 3 previous values
                if i >= 3:
                    # Use previous 3 values: i-1, i-2, i-3
                    replacement_value = np.mean(y_array[i-3:i])
                    y_array[i] = replacement_value
                # For early points (i < 3), don't replace - keep original values
        
        return y_array
    
    return y_array


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


def parse_log_filename(filename, parent_dir=None):
    """Parse log filename to extract experiment metadata.
    
    Args:
        filename: The log filename
        parent_dir: Parent directory name (used for QWEN logs to extract SFT step)
    """
    # Try LLAMA pattern first (with SFT_global_step)
    match = LOG_FILENAME_RE_LLAMA.match(filename)
    if match:
        return {
            'phase': int(match.group(1)),
            'groups': match.group(2),
            'iteration': int(match.group(3)),
            'sft_global_step': int(match.group(4)),
            'timestamp': match.group(5)
        }
    
    # Try QWEN pattern (without SFT_global_step)
    match = LOG_FILENAME_RE_QWEN.match(filename)
    if match:
        # For QWEN logs, extract SFT global step from parent directory name
        sft_global_step = 0
        if parent_dir:
            dir_match = QWEN_DIR_RE.match(parent_dir)
            if dir_match:
                sft_global_step = int(dir_match.group(1))
        
        return {
            'phase': int(match.group(1)),
            'groups': match.group(2),
            'iteration': int(match.group(3)),
            'sft_global_step': sft_global_step,
            'timestamp': match.group(4)
        }
    
    return None


def parse_log_file(log_path):
    """Parse a single log file to extract all metrics."""
    filename = os.path.basename(log_path)
    parent_dir = os.path.basename(os.path.dirname(log_path))
    metadata = parse_log_filename(filename, parent_dir)
    if not metadata:
        print(f"Warning: Could not parse filename {filename}")
        return []
    
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            
            step = int(step_match.group(1))
            line_data = {
                'filename': filename,
                'phase': metadata['phase'],
                'groups': metadata['groups'],
                'iteration': metadata['iteration'],
                'sft_global_step': metadata['sft_global_step'],
                'timestamp': metadata['timestamp'],
                'step': step
            }
            
            # Extract each metric
            for metric_name, pattern in METRICS_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    line_data[metric_name] = float(match.group(2))
            
            # Only add data if at least one metric was found
            if any(key in line_data for key in METRICS_PATTERNS.keys()):
                data.append(line_data)
    
    return data


def load_experiment_data(logs_dir):
    """Load and parse all log files from the logs directory and subdirectories."""
    # Search for log files in the main directory and all subdirectories
    log_files = []
    
    # Search in main directory
    main_logs = glob.glob(os.path.join(logs_dir, "*.log"))
    log_files.extend([f for f in main_logs if not f.endswith('master.log')])
    
    # Search in subdirectories
    for subdir in os.listdir(logs_dir):
        subdir_path = os.path.join(logs_dir, subdir)
        if os.path.isdir(subdir_path):
            sub_logs = glob.glob(os.path.join(subdir_path, "*.log"))
            log_files.extend([f for f in sub_logs if not f.endswith('master.log')])
    
    print(f"Found {len(log_files)} log files in {logs_dir} (including subdirectories)")
    if log_files:
        print("Log files found:")
        for log_file in log_files[:5]:  # Show first 5 files
            print(f"  - {os.path.relpath(log_file, logs_dir)}")
        if len(log_files) > 5:
            print(f"  ... and {len(log_files) - 5} more files")
    
    all_data = []
    for log_file in log_files:
        print(f"Parsing {os.path.basename(log_file)}...")
        file_data = parse_log_file(log_file)
        all_data.extend(file_data)
    
    if not all_data:
        print("No data found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} data points")
    return df


def plot_sft_comparison_by_phase(df, plots_dir, model_name):
    """Plot SFT step comparison for each phase separately."""
    
    # For each phase, create comparison plots across SFT steps
    for phase in sorted(df['phase'].unique()):
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            continue
            
        print(f"Plotting Phase {phase} SFT comparisons...")
        
        # 1. Critic Score Comparison
        plot_phase_critic_score(phase_df, plots_dir, model_name, phase)
        
        # 2. Derived C_K Comparison
        plot_phase_derived_ck(phase_df, plots_dir, model_name, phase)
        
        # 3. GRAMA Comparison
        plot_phase_grama(phase_df, plots_dir, model_name, phase)


def plot_phase_critic_score(phase_df, plots_dir, model_name, phase):
    """Plot critic score comparison across SFT steps for a specific phase."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Phase {phase} Critic Score Comparison Across SFT Steps', 
                 fontsize=14, fontweight='bold')
    
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['critic_score_mean'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['critic_score_mean'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(grouped['mean'])
            ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=3)
            
            # Add confidence interval
            std_smooth = smooth_curve(grouped['std'].fillna(0))
            ax.fill_between(grouped['step'], 
                          y_smooth - std_smooth, 
                          y_smooth + std_smooth, 
                          alpha=0.2)
    
    ax.set_title('Critic Score Mean', fontsize=16)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Critic Score Mean', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'phase{phase}_critic_score_sft_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase {phase} critic score comparison")


def plot_phase_derived_ck(phase_df, plots_dir, model_name, phase):
    """Plot derived C_K comparison across SFT steps for a specific phase."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Phase {phase} Derived C_K Comparison Across SFT Steps', 
                 fontsize=14, fontweight='bold')
    
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        # Compute derived C_K from c_k_normalized values
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['fisher_c_k_normalized'])
        if not sft_df.empty:
            # Group by step and compute derived C_K for each step
            grouped_data = []
            for step in sorted(sft_df['step'].unique()):
                step_df = sft_df[sft_df['step'] == step]
                if not step_df.empty:
                    # Get c_k values for this step
                    c_k_values = step_df['fisher_c_k_normalized'].values
                    if len(c_k_values) > 0:
                        # Use the mean c_k value for this step
                        mean_c_k = np.mean(c_k_values)
                        grouped_data.append({'step': step, 'c_k_mean': mean_c_k})
            
            if grouped_data:
                grouped_df = pd.DataFrame(grouped_data)
                # Compute derived C_K as cumulative mean of c_k values
                derived_C_K = compute_derived_C_K(grouped_df['c_k_mean'].values)
                
                # Apply outlier handling and smoothing
                y_smooth = smooth_curve(handle_outliers(derived_C_K))
                ax.plot(grouped_df['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=3)
    
    ax.set_title('Derived C_K (Cumulative Mean of c_k)', fontsize=16)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Derived C_K', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'phase{phase}_derived_ck_sft_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase {phase} derived C_K comparison")


def plot_phase_grama(phase_df, plots_dir, model_name, phase):
    """Plot GRAMA (zero gradient space ratio) comparison across SFT steps for a specific phase."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Phase {phase} GRAMA Comparison Across SFT Steps', 
                 fontsize=14, fontweight='bold')
    
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['actor_zero_gradspace_ratio'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['actor_zero_gradspace_ratio'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(grouped['mean'])
            ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=3)
            
            # Add confidence interval
            std_smooth = smooth_curve(grouped['std'].fillna(0))
            ax.fill_between(grouped['step'], 
                          y_smooth - std_smooth, 
                          y_smooth + std_smooth, 
                          alpha=0.2)
    
    ax.set_title('Zero Gradient Space Ratio (GRAMA)', fontsize=16)
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Zero Gradient Space Ratio', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'phase{phase}_grama_sft_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase {phase} GRAMA comparison")


def plot_phase_fisher_comparison(phase_df, plots_dir, model_name, phase):
    """Plot Fisher info comparison across SFT steps for a specific phase."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{model_name} - Phase {phase} Fisher Information Comparison Across SFT Steps', 
                 fontsize=14, fontweight='bold')
    
    # Define metrics - we'll compute C_K from c_k, and use others directly
    fisher_metrics = [
        ('derived_C_K', 'C_K Derived (Cumulative Mean)'),
        ('fisher_c_k_normalized', 'c_k Normalized'),
        ('fisher_L_K_normalized', 'L_K Normalized'),
        ('fisher_l_k_normalized', 'l_k Normalized'),
        ('fisher_sigma_max_normalized', 'Sigma Max Normalized'),
        ('fisher_sigma_min_normalized', 'Sigma Min Normalized')
    ]
    
    for idx, (metric, title) in enumerate(fisher_metrics):
        ax = axes[idx // 3, idx % 3]
        
        for sft_step in sorted(phase_df['sft_global_step'].unique()):
            if metric == 'derived_C_K':
                # Compute derived C_K from c_k_normalized values
                sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['fisher_c_k_normalized'])
                if not sft_df.empty:
                    # Group by step and compute derived C_K for each step
                    grouped_data = []
                    for step in sorted(sft_df['step'].unique()):
                        step_df = sft_df[sft_df['step'] == step]
                        if not step_df.empty:
                            # Get c_k values for this step and compute derived C_K
                            c_k_values = step_df['fisher_c_k_normalized'].values
                            if len(c_k_values) > 0:
                                # Use the mean c_k value for this step, then compute cumulative mean
                                mean_c_k = np.mean(c_k_values)
                                grouped_data.append({'step': step, 'c_k_mean': mean_c_k})
                    
                    if grouped_data:
                        grouped_df = pd.DataFrame(grouped_data)
                        # Compute derived C_K as cumulative mean of c_k values
                        derived_C_K = compute_derived_C_K(grouped_df['c_k_mean'].values)
                        
                        # Apply outlier handling and smoothing
                        y_smooth = smooth_curve(handle_outliers(derived_C_K))
                        ax.plot(grouped_df['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=2)
            else:
                # Use the metric directly from the dataframe
                sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=[metric])
                if not sft_df.empty:
                    grouped = sft_df.groupby('step')[metric].agg(['mean', 'std']).reset_index()
                    y_smooth = smooth_curve(handle_outliers(grouped['mean']))
                    ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=2)
                    
                    std_smooth = smooth_curve(grouped['std'].fillna(0))
                    ax.fill_between(grouped['step'], 
                                  y_smooth - std_smooth, 
                                  y_smooth + std_smooth, 
                                  alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'phase{phase}_fisher_sft_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase {phase} Fisher comparison")


def plot_phase_grama_comparison(phase_df, plots_dir, model_name, phase):
    """Plot GRAMA/plasticity comparison across SFT steps for a specific phase."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Phase {phase} GRAMA/Plasticity Comparison Across SFT Steps', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Zero Gradient Space Ratio
    ax = axes[0]
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['actor_zero_gradspace_ratio'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['actor_zero_gradspace_ratio'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(grouped['mean'])
            ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=2)
            
            std_smooth = smooth_curve(grouped['std'].fillna(0))
            ax.fill_between(grouped['step'], 
                          y_smooth - std_smooth, 
                          y_smooth + std_smooth, 
                          alpha=0.2)
    
    ax.set_title('Zero Gradient Space Ratio')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Zero Gradient Space Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Critic Gradient Norm
    ax = axes[1]
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['critic_grad_norm'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['critic_grad_norm'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(handle_outliers(grouped['mean']))
            ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=2)
    
    ax.set_title('Critic Gradient Norm')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fisher C_K Running Average (also plasticity-related)
    ax = axes[2]
    for sft_step in sorted(phase_df['sft_global_step'].unique()):
        sft_df = phase_df[phase_df['sft_global_step'] == sft_step].dropna(subset=['fisher_c_k_running_avg'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['fisher_c_k_running_avg'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(handle_outliers(grouped['mean']))
            ax.plot(grouped['step'], y_smooth, label=f'SFT Step {sft_step}', linewidth=2)
    
    ax.set_title('Fisher c_k Running Average')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fisher c_k Running Avg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'phase{phase}_grama_sft_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Phase {phase} GRAMA comparison")


def plot_fisher_analysis(df, plots_dir, model_name):
    """Plot Fisher Information analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{model_name} - Fisher Information Analysis', fontsize=16, fontweight='bold')
    
    fisher_metrics = [
        ('fisher_C_K_normalized', 'C_K Normalized'),
        ('fisher_c_k_normalized', 'c_k Normalized'),
        ('fisher_L_K_normalized', 'L_K Normalized'),
        ('fisher_l_k_normalized', 'l_k Normalized'),
        ('fisher_sigma_max_normalized', 'Sigma Max Normalized'),
        ('fisher_sigma_min_normalized', 'Sigma Min Normalized')
    ]
    
    for idx, (metric, title) in enumerate(fisher_metrics):
        ax = axes[idx // 3, idx % 3]
        
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase].dropna(subset=[metric])
            if not phase_df.empty:
                grouped = phase_df.groupby('step')[metric].agg(['mean', 'std']).reset_index()
                y_smooth = smooth_curve(handle_outliers(grouped['mean']))
                ax.plot(grouped['step'], y_smooth, label=f'Phase {phase}', linewidth=2)
                
                std_smooth = smooth_curve(grouped['std'].fillna(0))
                ax.fill_between(grouped['step'], 
                              y_smooth - std_smooth, 
                              y_smooth + std_smooth, 
                              alpha=0.2)
        
        ax.set_title(title)
        ax.set_xlabel('Training Step')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'fisher_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Fisher analysis")


def plot_plasticity_analysis(df, plots_dir, model_name):
    """Plot plasticity analysis including zero gradient space ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Plasticity Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Zero Gradient Space Ratio
    ax = axes[0]
    for phase in sorted(df['phase'].unique()):
        phase_df = df[df['phase'] == phase].dropna(subset=['actor_zero_gradspace_ratio'])
        if not phase_df.empty:
            grouped = phase_df.groupby('step')['actor_zero_gradspace_ratio'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(grouped['mean'])
            ax.plot(grouped['step'], y_smooth, label=f'Phase {phase}', linewidth=2)
            ax.fill_between(grouped['step'], 
                          y_smooth - grouped['std'].fillna(0), 
                          y_smooth + grouped['std'].fillna(0), 
                          alpha=0.2)
    ax.set_title('Zero Gradient Space Ratio')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Zero Gradient Space Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Critic Gradient Norm
    ax = axes[1]
    for phase in sorted(df['phase'].unique()):
        phase_df = df[df['phase'] == phase].dropna(subset=['critic_grad_norm'])
        if not phase_df.empty:
            grouped = phase_df.groupby('step')['critic_grad_norm'].agg(['mean', 'std']).reset_index()
            y_smooth = smooth_curve(handle_outliers(grouped['mean']))
            ax.plot(grouped['step'], y_smooth, label=f'Phase {phase}', linewidth=2)
    ax.set_title('Critic Gradient Norm')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Combined Plasticity View
    ax = axes[2]
    for phase in sorted(df['phase'].unique()):
        phase_df = df[df['phase'] == phase]
        
        zero_grad_df = phase_df.dropna(subset=['actor_zero_gradspace_ratio'])
        if not zero_grad_df.empty:
            grouped = zero_grad_df.groupby('step')['actor_zero_gradspace_ratio'].mean().reset_index()
            ax.plot(grouped['step'], grouped['actor_zero_gradspace_ratio'], 
                   label=f'Phase {phase} - Zero Grad Ratio', linewidth=2)
        
        fisher_df = phase_df.dropna(subset=['fisher_C_K_normalized'])
        if not fisher_df.empty:
            ax2 = ax.twinx()
            grouped = fisher_df.groupby('step')['fisher_C_K_normalized'].mean().reset_index()
            ax2.plot(grouped['step'], grouped['fisher_C_K_normalized'], 
                    label=f'Phase {phase} - Fisher C_K', linewidth=2, linestyle='--')
            ax2.set_ylabel('Fisher C_K Normalized', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('Combined Plasticity Metrics')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Zero Gradient Space Ratio')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'plasticity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved plasticity analysis")


def plot_training_dynamics(df, plots_dir, model_name):
    """Plot training dynamics including losses and PPO metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{model_name} - Training Dynamics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss Curves
    ax = axes[0]
    loss_metrics = ['actor_pg_loss', 'critic_vf_loss', 'actor_entropy_loss']
    colors = ['blue', 'red', 'green']
    
    for metric, color in zip(loss_metrics, colors):
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase].dropna(subset=[metric])
            if not phase_df.empty:
                grouped = phase_df.groupby('step')[metric].mean().reset_index()
                y_smooth = smooth_curve(handle_outliers(grouped[metric]))
                label = f'Phase {phase} - {metric.replace("_", " ").title()}'
                ax.plot(grouped['step'], y_smooth, label=label, color=color, 
                       alpha=0.7, linewidth=2, linestyle='-' if phase == 1 else '--')
    
    ax.set_title('Loss Curves')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: PPO KL Divergence
    ax = axes[1]
    kl_metrics = ['actor_ppo_kl', 'critic_kl']
    
    for metric in kl_metrics:
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase].dropna(subset=[metric])
            if not phase_df.empty:
                grouped = phase_df.groupby('step')[metric].mean().reset_index()
                y_smooth = smooth_curve(grouped[metric])
                label = f'Phase {phase} - {metric.replace("_", " ").title()}'
                ax.plot(grouped['step'], y_smooth, label=label, linewidth=2)
    
    ax.set_title('KL Divergence')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Clipping Fractions
    ax = axes[2]
    clip_metrics = ['actor_pg_clipfrac', 'critic_vf_clipfrac']
    
    for metric in clip_metrics:
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase].dropna(subset=[metric])
            if not phase_df.empty:
                grouped = phase_df.groupby('step')[metric].mean().reset_index()
                y_smooth = smooth_curve(grouped[metric])
                label = f'Phase {phase} - {metric.replace("_", " ").title()}'
                ax.plot(grouped['step'], y_smooth, label=label, linewidth=2)
    
    ax.set_title('Clipping Fractions')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Clipping Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved training dynamics")


def plot_phase_transitions(df, plots_dir, model_name):
    """Plot phase transition analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Phase Transition Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance transitions
    ax = axes[0, 0]
    key_metrics = ['critic_score_mean', 'val_test_score']
    
    for metric in key_metrics:
        phase_data = {}
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase].dropna(subset=[metric])
            if not phase_df.empty:
                grouped = phase_df.groupby('step')[metric].mean().reset_index()
                phase_data[phase] = grouped
        
        for phase, data in phase_data.items():
            ax.plot(data['step'], data[metric], 
                   label=f'Phase {phase} - {metric.replace("_", " ").title()}', 
                   linewidth=2, alpha=0.8)
    
    ax.set_title('Performance Across Phase Transitions')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Performance Metric')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Fisher metrics transitions
    ax = axes[0, 1]
    fisher_key = 'fisher_C_K_normalized'
    
    for phase in sorted(df['phase'].unique()):
        phase_df = df[df['phase'] == phase].dropna(subset=[fisher_key])
        if not phase_df.empty:
            grouped = phase_df.groupby('step')[fisher_key].mean().reset_index()
            y_smooth = smooth_curve(handle_outliers(grouped[fisher_key]))
            ax.plot(grouped['step'], y_smooth, 
                   label=f'Phase {phase}', linewidth=2)
    
    ax.set_title('Fisher C_K Across Phases')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Fisher C_K Normalized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Iteration comparison within phases
    ax = axes[1, 0]
    for phase in sorted(df['phase'].unique()):
        for iteration in sorted(df[df['phase'] == phase]['iteration'].unique()):
            iter_df = df[(df['phase'] == phase) & (df['iteration'] == iteration)]
            iter_df = iter_df.dropna(subset=['critic_score_mean'])
            if not iter_df.empty:
                grouped = iter_df.groupby('step')['critic_score_mean'].mean().reset_index()
                ax.plot(grouped['step'], grouped['critic_score_mean'], 
                       label=f'Phase {phase} - Iter {iteration}', linewidth=2)
    
    ax.set_title('Iteration Comparison Within Phases')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Critic Score Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: SFT Global Step comparison
    ax = axes[1, 1]
    for sft_step in sorted(df['sft_global_step'].unique()):
        sft_df = df[df['sft_global_step'] == sft_step].dropna(subset=['critic_score_mean'])
        if not sft_df.empty:
            grouped = sft_df.groupby('step')['critic_score_mean'].mean().reset_index()
            ax.plot(grouped['step'], grouped['critic_score_mean'], 
                   label=f'SFT Global Step {sft_step}', linewidth=2)
    
    ax.set_title('SFT Global Step Comparison')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Critic Score Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'phase_transitions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved phase transitions analysis")


def generate_summary_report(df, plots_dir, model_name):
    """Generate a summary report of the experiment."""
    report_path = os.path.join(plots_dir, 'experiment_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"Continual Learning Experiment Summary - {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Experiment Overview:\n")
        f.write(f"- Total data points: {len(df)}\n")
        f.write(f"- Phases: {sorted(df['phase'].unique())}\n")
        f.write(f"- SFT Global Steps: {sorted(df['sft_global_step'].unique())}\n")
        f.write(f"- Iterations: {sorted(df['iteration'].unique())}\n")
        f.write(f"- Training steps range: {df['step'].min()} - {df['step'].max()}\n\n")
        
        f.write("Data Availability by Metric:\n")
        for metric in METRICS_PATTERNS.keys():
            count = df[metric].notna().sum()
            percentage = (count / len(df)) * 100
            f.write(f"- {metric}: {count} points ({percentage:.1f}%)\n")
        
        f.write("\nPhase-wise Statistics:\n")
        for phase in sorted(df['phase'].unique()):
            phase_df = df[df['phase'] == phase]
            f.write(f"\nPhase {phase}:\n")
            f.write(f"  - Data points: {len(phase_df)}\n")
            f.write(f"  - Groups: {phase_df['groups'].unique()}\n")
            f.write(f"  - Step range: {phase_df['step'].min()} - {phase_df['step'].max()}\n")
            
            if 'critic_score_mean' in phase_df.columns:
                scores = phase_df['critic_score_mean'].dropna()
                if not scores.empty:
                    f.write(f"  - Critic score mean: {scores.mean():.3f} ± {scores.std():.3f}\n")
    
    print(f"Saved experiment summary to {report_path}")


def process_model_logs(model_name, logs_dir, plots_dir):
    """Process logs for a specific model."""
    print(f"\n{'='*60}")
    print(f"Processing {model_name.upper()} logs from {logs_dir}")
    print(f"{'='*60}")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    df = load_experiment_data(logs_dir)
    if df.empty:
        print(f"No data found for {model_name}")
        return
    
    print(f"Loaded {len(df)} data points for {model_name}")
    
    # Generate SFT comparison plots by phase
    plot_sft_comparison_by_phase(df, plots_dir, model_name)
    
    generate_summary_report(df, plots_dir, model_name)
    
    print(f"Completed analysis for {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Experiment Analysis')
    parser.add_argument('--model', choices=['llama', 'qwen', 'both'], default='both',
                       help='Which model logs to process')
    parser.add_argument('--base_dir', default='.', 
                       help='Base directory containing logs and plots folders')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    
    if args.model in ['llama', 'both']:
        llama_logs_dir = os.path.join(base_dir, 'llama_logs')
        llama_plots_dir = os.path.join(base_dir, 'llama_plots')
        if os.path.exists(llama_logs_dir):
            process_model_logs('Llama', llama_logs_dir, llama_plots_dir)
        else:
            print(f"Warning: {llama_logs_dir} not found")
    
    if args.model in ['qwen', 'both']:
        qwen_logs_dir = os.path.join(base_dir, 'qwen_logs')
        qwen_plots_dir = os.path.join(base_dir, 'qwen_plots')
        if os.path.exists(qwen_logs_dir):
            process_model_logs('Qwen', qwen_logs_dir, qwen_plots_dir)
        else:
            print(f"Warning: {qwen_logs_dir} not found")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
