"""
This script analyzes and plots experiment results from RLHF training logs after different SFT steps.

It performs the following analyses:
A. Performance and 'grama ratio' analysis
  1. Plots RLHF performance (critic/score/mean) curves for different SFT steps.
  2. Plots RLHF 'grama ratio' (actor/zero_gradspace_ratio) curves for different SFT steps.

B. Per-module 'grama ratio' analysis
  1. Plots the training trend of per-module grama ratios, averaged across layers.
  2. Creates case study plots showing how grama ratio changes across layers at specific training steps.

The script is designed to parse log files with a specific structure and handles interleaved output from multiple processes.
"""

import argparse
import re
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Individual regexes for robust, order-independent parsing
STEP_RE = re.compile(r"step:(\d+)")
SCORE_RE = re.compile(r"critic/score/mean:([\d\.]+)")
GRAMA_RE = re.compile(r"actor/zero_gradspace_ratio:([\d\.]+)")

# Regexes for stateful, multi-line parsing of per-module analysis
PARAM_LINE_RE = re.compile(r"\s*\[Param: model\.layers\.(\d+)\.(?P<param_name>.*?)\]")
ANALYSIS_LINE_RE = re.compile(r"\s*-> Analysis: .*? dormant \((?P<dormant_pct>[\d\.]+)%\)")

# Define modules for case study
SELF_ATTN_MODULES = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight']
MLP_MODULES = ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
LN_MODULES = ['input_layernorm.weight', 'post_attention_layernorm.weight']
ALL_MODULES_FOR_CASE_STUDY = SELF_ATTN_MODULES + MLP_MODULES + LN_MODULES


def parse_log_file(log_path: Path):
    """Parse a single log file to extract performance and per-module grama data."""
    perf_data = []
    module_analysis_steps = []
    current_step_module_analysis = defaultdict(list)
    last_param_info = None

    ansi_re = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')
    prefix_re = re.compile(r'^\s*\(.*?\)\s*')

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = ansi_re.sub('', line)
            cleaned_line = prefix_re.sub('', cleaned_line).strip()

            # Performance metrics
            if 'step:' in cleaned_line and 'critic/score/mean:' in cleaned_line:
                step_match = STEP_RE.search(cleaned_line)
                score_match = SCORE_RE.search(cleaned_line)
                grama_match = GRAMA_RE.search(cleaned_line)
                if step_match and score_match and grama_match:
                    perf_data.append({
                        'step': int(step_match.group(1)),
                        'score': float(score_match.group(1)),
                        'grama': float(grama_match.group(1))
                    })

            # Per-module analysis
            if "Analyzing component 'layer_0'" in cleaned_line and current_step_module_analysis:
                processed_step = {param: [v[1] for v in sorted(values)] for param, values in current_step_module_analysis.items()}
                module_analysis_steps.append(processed_step)
                current_step_module_analysis = defaultdict(list)
                last_param_info = None

            param_match = PARAM_LINE_RE.search(cleaned_line)
            if param_match:
                layer = int(param_match.group(1))
                param_name = param_match.group('param_name').strip()
                last_param_info = {'layer': layer, 'param_name': param_name}

            analysis_match = ANALYSIS_LINE_RE.search(cleaned_line)
            if analysis_match and last_param_info:
                dormant_pct = float(analysis_match.group('dormant_pct'))
                current_step_module_analysis[last_param_info['param_name']].append((last_param_info['layer'], dormant_pct))
                last_param_info = None

    if current_step_module_analysis:
        processed_step = {param: [v[1] for v in sorted(values)] for param, values in current_step_module_analysis.items()}
        module_analysis_steps.append(processed_step)

    perf_df = pd.DataFrame(perf_data)

    # Align module analysis with performance steps
    final_module_data = []
    if not perf_df.empty and module_analysis_steps:
        num_perf_steps = len(perf_df)
        num_module_blocks = len(module_analysis_steps)
        print(f"  ... found {num_perf_steps} perf steps and {num_module_blocks} module blocks. Aligning to {num_perf_steps} steps.")
        if num_module_blocks >= num_perf_steps:
            final_module_data = module_analysis_steps[:num_perf_steps]
        else:
            final_module_data = module_analysis_steps + [{} for _ in range(num_perf_steps - num_module_blocks)]

    return perf_df, final_module_data

def _plot_mean_std(df, ax, label):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    
    # Smooth the curves for better readability
    smooth_mean = gaussian_filter1d(mean, sigma=2)
    smooth_std = gaussian_filter1d(std, sigma=2)
    
    ax.plot(mean.index, smooth_mean, label=label)
    ax.fill_between(mean.index, smooth_mean - smooth_std, smooth_mean + smooth_std, alpha=0.2)

def _finalize_plot(fig, ax, title, xlabel, ylabel, path):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved plot: {path}")

def plot_performance_curves(data, output_dir: Path):
    """Plots performance and grama ratio curves (Task A1, A2)."""
    for group_type in ['known', 'unknown']:
        fig_score, ax_score = plt.subplots(figsize=(12, 8))
        fig_grama, ax_grama = plt.subplots(figsize=(12, 8))

        sft_steps = sorted(data.keys(), key=int)
        for sft_step in sft_steps:
            perf_runs = data[sft_step][group_type]['perf']
            if not perf_runs:
                continue

            aligned_scores = pd.concat([df.set_index('step')['score'] for df in perf_runs], axis=1).sort_index()
            aligned_gramas = pd.concat([df.set_index('step')['grama'] for df in perf_runs], axis=1).sort_index()

            _plot_mean_std(aligned_scores, ax_score, f'SFT Step {sft_step}')
            _plot_mean_std(aligned_gramas, ax_grama, f'SFT Step {sft_step}')

        _finalize_plot(fig_score, ax_score, f'Performance (Score) by SFT Step ({group_type.capitalize()} Groups)', 'RLHF Training Step', 'Critic Score', output_dir / f'score_by_sft_step_{group_type}.png')
        _finalize_plot(fig_grama, ax_grama, f'Grama Ratio by SFT Step ({group_type.capitalize()} Groups)', 'RLHF Training Step', 'Grama Ratio', output_dir / f'grama_by_sft_step_{group_type}.png')

def plot_per_module_grama_trends(data, output_dir: Path):
    """Plots per-module grama ratio trends averaged across layers for each SFT step (Task B1)."""
    sft_steps = sorted(data.keys(), key=int)

    for sft_step in sft_steps:
        for group_type in ['known', 'unknown']:
            fig, ax = plt.subplots(figsize=(15, 10))
            plot_created = False

            sft_step_module_data = defaultdict(list)
            runs_data = data[sft_step][group_type]['modules']
            if not runs_data:
                plt.close(fig)
                continue

            for run in runs_data:
                run_module_trends = defaultdict(list)
                for step_data in run:
                    for module, layers in step_data.items():
                        if layers:
                            avg_over_layers = np.mean(layers)
                            run_module_trends[module].append(avg_over_layers)
                
                for module, trend in run_module_trends.items():
                    sft_step_module_data[module].append(trend)

            for module, all_runs_trends in sft_step_module_data.items():
                if module not in ALL_MODULES_FOR_CASE_STUDY:
                    continue
                
                min_len = min(len(r) for r in all_runs_trends) if all_runs_trends else 0
                if min_len > 1:
                    runs_padded = [r[:min_len] for r in all_runs_trends]
                    mean_trend = np.mean(runs_padded, axis=0)
                    smooth_mean_trend = gaussian_filter1d(mean_trend, sigma=2)
                    ax.plot(range(1, len(smooth_mean_trend) + 1), smooth_mean_trend, label=module)
                    plot_created = True

            if plot_created:
                _finalize_plot(fig, ax, f'Per-Module Grama Ratio Trend\nSFT Step: {sft_step} ({group_type.capitalize()} Groups)', 
                               'RLHF Training Step', 'Grama Ratio (Dormant %) Averaged Across Layers', 
                               output_dir / f'per_module_grama_trend_sft{sft_step}_{group_type}.png')
            else:
                plt.close(fig)

def plot_layerwise_grama_case_study(data, output_dir: Path):
    """Plots layer-wise grama ratio at specific steps (Task B2)."""
    case_study_steps = [1, 25, 50, 100, 150]

    for sft_step, group_data in data.items():
        for group_type, runs in group_data.items():
            runs_data = runs['modules']
            if not runs_data:
                continue

            avg_run_data = defaultdict(list)
            for module in ALL_MODULES_FOR_CASE_STUDY:
                all_runs_for_module = []
                for run in runs_data:
                    module_data_for_run = [step.get(module, []) for step in run]
                    if any(module_data_for_run):
                        all_runs_for_module.append(module_data_for_run)

                if not all_runs_for_module:
                    continue

                max_layers = max(len(layer_data) for run in all_runs_for_module for layer_data in run if layer_data)
                min_steps = min(len(r) for r in all_runs_for_module)
                if not max_layers or not min_steps:
                    continue

                padded_runs = []
                for run in all_runs_for_module:
                    padded_run_steps = []
                    for step_data in run[:min_steps]:
                        padding = [np.nan] * (max_layers - len(step_data))
                        padded_run_steps.append(step_data + padding)
                    padded_runs.append(padded_run_steps)
                
                if padded_runs:
                    mean_over_runs = np.nanmean(np.array(padded_runs), axis=0)
                    avg_run_data[module] = mean_over_runs

            for step in case_study_steps:
                step_idx = step - 1
                fig, ax = plt.subplots(figsize=(15, 10))
                plot_created = False

                for module in ALL_MODULES_FOR_CASE_STUDY:
                    if module in avg_run_data and step_idx < len(avg_run_data[module]):
                        layer_data = avg_run_data[module][step_idx]
                        valid_indices = ~np.isnan(layer_data)
                        if np.any(valid_indices):
                            ax.plot(np.arange(len(layer_data))[valid_indices], layer_data[valid_indices], label=module, marker='o', linestyle='-')
                            plot_created = True

                if plot_created:
                    _finalize_plot(fig, ax, f'Layer-wise Grama Ratio Case Study\nSFT Step: {sft_step}, RLHF Step: {step} ({group_type.capitalize()} Groups)',
                                   'Layer Number', 'Grama Ratio (Dormant %)',
                                   output_dir / f'case_study_sft{sft_step}_rlhf{step}_{group_type}.png')
                else:
                    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot experiment results from logs.")
    parser.add_argument("--log_dir", type=Path, default="logs", help="Directory containing log files.")
    parser.add_argument("--output_dir", type=Path, default="plots/plot_grama_ratio_per_module", help="Directory to save plots.")
    parser.add_argument("--sft_steps", type=str, default="2,8,15,20", help="Comma-separated list of SFT steps to process.")
    args = parser.parse_args()

    sft_steps_to_process = [int(s) for s in args.sft_steps.split(',')]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for sft_step in sft_steps_to_process:
        sft_step_dirs = list(args.log_dir.glob(f"*sft_global_step_{sft_step}_*"))
        if not sft_step_dirs:
            print(f"Warning: No directories found for SFT step {sft_step}")
            continue

        for sft_dir in sft_step_dirs:
            log_files = sorted(sft_dir.glob("Group*.log")) # Process only group logs
            for log_file in log_files:
                print(f"Parsing {log_file}...")
                group_type = 'known' # Assuming all are known for simplicity, adjust if needed
                perf_df, module_data = parse_log_file(log_file)
                if not perf_df.empty:
                    all_data[sft_step][group_type]['perf'].append(perf_df)
                if module_data:
                    all_data[sft_step][group_type]['modules'].append(module_data)

    if not all_data:
        print("No data parsed. Exiting.")
        return

    print("\n--- Generating Plots ---")
    plot_performance_curves(all_data, args.output_dir)
    plot_per_module_grama_trends(all_data, args.output_dir)
    plot_layerwise_grama_case_study(all_data, args.output_dir)

if __name__ == "__main__":
    main()
