"""
This script analyzes and plots experiment results from log files for a continual learning experiment.
It performs three main tasks:
1.  Plots RLHF performance curves (critic/score/mean) for different SFT steps.
2.  Plots per-module Fisher information metrics (C_K, L_K) trends over training.
3.  Plots case studies of how Fisher metrics change across layers at specific training steps.
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOGS_DIR = './logs'
PLOTS_DIR = './plots/plot_fisher_per_module'

# Regex to capture relevant log lines
PERFORMANCE_RE = re.compile(r'step:(\d+).*critic/score/mean:([\d.e+-]+)')
FISHER_RE = re.compile(r"\[FisherInfo\] Param '([^']+)': .*C_K=([\d.e+-]+), L_K=([\d.e+-]+)")
PARAM_MODULE_RE = re.compile(r'model\.layers\.(\d+)\.(.*)')


def parse_log_file(log_path, sft_step):
    """Parses a single log file to extract performance and Fisher info.

    Args:
        log_path (str): Path to the log file.
        sft_step (int): The SFT step for this experiment run.

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
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': current_step,
                    'score': score,
                    'param_name': 'performance_metric',
                    'C_K': np.nan,
                    'L_K': np.nan,
                })

            fisher_match = FISHER_RE.search(line)
            if fisher_match and current_step != -1:
                param_name = fisher_match.group(1)
                c_k = float(fisher_match.group(2))
                l_k = float(fisher_match.group(3))
                data.append({
                    'sft_step': sft_step,
                    'group_id': group_id,
                    'training_step': current_step,
                    'score': np.nan,
                    'param_name': param_name,
                    'C_K': c_k,
                    'L_K': l_k,
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

def plot_performance_curves(df):
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
        plt.savefig(os.path.join(PLOTS_DIR, f'performance_{group_type.lower()}.png'))
        plt.close()
        print(f"Saved performance plot for {group_type} groups.")

def plot_fisher_trends(df):
    """Plots per-module C_K and L_K trends across training steps."""
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

    for group_type in ['Known', 'Unknown']:
        group_df = fisher_df[fisher_df['group_type'] == group_type]
        for sft_step in sorted(group_df['sft_step'].unique()):
            sft_df = group_df[group_df['sft_step'] == sft_step]

            # Average across layers and runs for each module and training step
            agg_df = sft_df.groupby(['training_step', 'module'])[['C_K', 'L_K']].mean().reset_index()

            for metric in ['C_K', 'L_K']:
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
                plt.savefig(os.path.join(PLOTS_DIR, f'{metric}_trend_sft{sft_step}_{group_type.lower()}.png'))
                plt.close()
                print(f"Saved {metric} trend plot for SFT {sft_step}, {group_type} group.")

def plot_fisher_case_study(df):
    """Plots C_K/L_K across layers for specific training steps."""
    fisher_df = df[(df['C_K'].notna()) & (df['layer'] >= 0)].copy() # Exclude lm_head
    if fisher_df.empty:
        print("No Fisher information data found for case study.")
        return

    case_study_steps = [1, 25, 50, 100, 149] # Use 149 as it's the max step with data
    modules_to_plot = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
        'input_layernorm', 'post_attention_layernorm'
    ]
    fisher_df = fisher_df[fisher_df['module'].isin(modules_to_plot)]

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
                agg_df = step_df.groupby(['layer', 'module'])[['C_K', 'L_K']].mean().reset_index()

                for metric in ['C_K', 'L_K']:
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
                    plt.savefig(os.path.join(PLOTS_DIR, f'case_study_{metric}_sft{sft_step}_step{display_step}_{group_type.lower()}.png'))
                    plt.close()
    print("Saved all case study plots.")

def main():
    """Main function to orchestrate parsing and plotting."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    exp_dirs = glob.glob(os.path.join(LOGS_DIR, '*sft_global_step_*'))
    if not exp_dirs:
        print(f"Error: No experiment directories found in {LOGS_DIR}")
        return

    all_data = []
    for exp_dir in exp_dirs:
        try:
            sft_step = int(re.search(r'global_step_(\d+)', exp_dir).group(1))
        except (AttributeError, ValueError):
            print(f"Could not parse SFT step from directory name: {exp_dir}")
            continue
        
        log_files = glob.glob(os.path.join(exp_dir, 'Group*.log'))
        print(f"Found {len(log_files)} log files in {exp_dir} for SFT step {sft_step}")
        for log_file in log_files:
            all_data.extend(parse_log_file(log_file, sft_step))

    if not all_data:
        print("Error: No data could be parsed from any log files.")
        return

    df = pd.DataFrame(all_data)

    # --- Data Pre-processing ---
    df['group_type'] = df['group_id'].apply(lambda x: 'Known' if x == 0 else 'Unknown')
    
    # Extract module and layer info for Fisher data
    fisher_mask = df['C_K'].notna()
    if fisher_mask.any():
        parsed_tuples = df.loc[fisher_mask, 'param_name'].apply(extract_module_info)
        df.loc[fisher_mask, 'layer'] = parsed_tuples.str[0]
        df.loc[fisher_mask, 'module'] = parsed_tuples.str[1]
    
    print("Data parsing and pre-processing complete.")
    print(f"Total records: {len(df)}")
    print(f"Performance records: {df['score'].notna().sum()}")
    print(f"Fisher records: {df['C_K'].notna().sum()}")
    if fisher_mask.any():
        print(f"Fisher records with parsed module: {df[df['C_K'].notna()]['module'].notna().sum()}")

    # --- Plotting ---
    plot_performance_curves(df)
    plot_fisher_trends(df)
    plot_fisher_case_study(df)

    print(f"\nAll plots saved to {PLOTS_DIR}")

if __name__ == '__main__':
    main()
