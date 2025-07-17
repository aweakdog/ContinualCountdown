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
from collections import defaultdict

LOGS_DIR = './logs'
PLOTS_DIR = './plots/plot_fisher_per_module'
PARTIAL_PLOTS_DIR = os.path.join(PLOTS_DIR, 'partial')

# Regex to capture relevant log lines
PERFORMANCE_RE = re.compile(r'step:(\d+).*critic/score/mean:([\d.e+-]+)')
FISHER_RE = re.compile(r"\[FisherInfo\] Param '([^']+)': .*C_K=([\d.e+-]+), L_K=([\d.e+-]+), sigma_max=([\d.e+-]+), sigma_min=([\d.e+-]+)")
PARAM_MODULE_RE = re.compile(r'model\.layers\.(\d+)\.(.*)')


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

def main():
    """Main function to orchestrate parsing and plotting for complete and partial runs."""
    # Clean and create directories
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    os.makedirs(PLOTS_DIR)
    os.makedirs(PARTIAL_PLOTS_DIR, exist_ok=True)

    exp_dirs = glob.glob(os.path.join(LOGS_DIR, '*sft_global_step_*'))
    if not exp_dirs:
        print(f"Error: No experiment directories found in {LOGS_DIR}")
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

        # Plotting
        plot_performance_curves(df, plots_dir)
        plot_fisher_trends(df, plots_dir)
        plot_fisher_case_study(df, plots_dir)
        print(f"\nAll {data_type} plots saved to {plots_dir}")

    # --- Process and plot COMPLETE data ---
    _process_and_plot(complete_data, "Complete", PLOTS_DIR)

    # --- Process and plot PARTIAL data --- (DISABLED)
    # if partial_data:
    #     _process_and_plot(partial_data, "Partial", PARTIAL_PLOTS_DIR)

if __name__ == '__main__':
    main()
