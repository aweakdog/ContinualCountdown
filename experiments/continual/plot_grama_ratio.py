import os
import re
import glob
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # For Savitzky-Golay smoothing, often better than simple moving average

SMOOTHING_WINDOW_SIZE = 5
SMOOTHING_POLYORDER = 2 # Relevant for Savitzky-Golay

def smooth_curve(y_values, window_size=SMOOTHING_WINDOW_SIZE, polyorder=SMOOTHING_POLYORDER):
    """Smooths a curve using Savitzky-Golay filter."""
    if len(y_values) < window_size:
        return y_values # Not enough data to smooth
    # Ensure window_size is odd for Savitzky-Golay
    if window_size % 2 == 0:
        window_size += 1
    if len(y_values) < window_size: # Check again after potential increment
        return y_values
    return savgol_filter(y_values, window_size, polyorder)

import matplotlib.colors as mcolors
from collections import defaultdict
from scipy.signal import savgol_filter # Ensure import is at the top if not already there due to other chunks

# Configure Matplotlib to use 'Agg' backend for non-GUI environments
plt.switch_backend('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
PLOT_OUTPUT_DIR = BASE_DIR / "experiments" / "continual" / "plots"

# Constants
TARGET_PARAMS = [
    "q_proj.weight",
    "k_proj.weight",
    "o_proj.weight",
    "v_proj.weight"
]
CASE_STUDY_PPO_STEPS = [1, 25, 50, 100, 200, 300]
NUM_LAYERS = 28  # 0 to 27

# Regex patterns
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*[mK]')
SFT_DIR_PATTERN = re.compile(r"continual_countdown3b_sft_global_step_(\d+)_\d{8}_\d{6}")
GROUP_LOG_PATTERN = re.compile(r"Group(\d+)_\d{8}_\d{6}\.log")

GENERAL_METRICS_PATTERN = re.compile(
    r"step:(\d+) .* critic/score/mean:([-\d.]+) .* actor/zero_gradspace_ratio:([-\d.]+)"
)
LAYER_WISE_DETAIL_PATTERN = re.compile(
    r"Layer: model\.layers\.(\d+)\.self_attn\.({target_params_regex})\s*\|"
    r".*?\(\s*([\d\.]+)%\s*\)"
    r".*?B/H_calc:\s*([-\d\.eE]+)".format(
        target_params_regex="|".join(p.replace('.', '\\.') for p in TARGET_PARAMS)
    )
)


def clean_ansi_codes(text):
    return ANSI_ESCAPE_PATTERN.sub('', text)

def parse_general_metrics(log_file_path):
    metrics = defaultdict(lambda: {'score': [], 'grama': []})
    steps = [] 
    scores = []
    gramas = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = GENERAL_METRICS_PATTERN.search(line)
                if match:
                    step = int(match.group(1))
                    score = float(match.group(2))
                    grama = float(match.group(3))
                    steps.append(step)
                    scores.append(score)
                    gramas.append(grama)
    except FileNotFoundError:
        print(f"Warning: Log file not found {log_file_path}")
        return {}, [], [], []
    except Exception as e:
        print(f"Error parsing general metrics from {log_file_path}: {e}")
        return {}, [], [], []
    
    # Convert to dict for easier lookup by step, though not strictly needed for simple plotting
    for i, step in enumerate(steps):
        metrics[step]['score'].append(scores[i]) # Use append in case of duplicate steps, though unlikely
        metrics[step]['grama'].append(gramas[i])

    return metrics, sorted(list(set(steps))), scores, gramas # Return unique sorted steps and original lists

def parse_layer_wise_metrics(log_file_path, target_ppo_steps):
    layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    # Structure: layer_data[ppo_step][param_name][layer_id] = {'grama_ratio': val, 'bh_calc': val}
    
    current_ppo_step = -1
    collected_for_current_step = []

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = clean_ansi_codes(line)
                
                # Check for general step line first to associate buffered layer lines
                step_match = GENERAL_METRICS_PATTERN.search(cleaned_line)
                if step_match:
                    new_ppo_step = int(step_match.group(1))
                    if current_ppo_step != -1 and current_ppo_step in target_ppo_steps:
                        for entry in collected_for_current_step:
                            param, layer_id, grama_r, bh_c = entry
                            layer_data[current_ppo_step][param][layer_id]['grama_ratio'] = grama_r
                            layer_data[current_ppo_step][param][layer_id]['bh_calc'] = bh_c
                    
                    current_ppo_step = new_ppo_step
                    collected_for_current_step = []

                # Then check for layer detail line
                if current_ppo_step in target_ppo_steps:
                    layer_match = LAYER_WISE_DETAIL_PATTERN.search(cleaned_line)
                    if layer_match:
                        layer_id = int(layer_match.group(1))
                        param_name = layer_match.group(2)
                        grama_ratio = float(layer_match.group(3)) / 100.0  # Convert percentage to 0-1 scale
                        bh_calc = float(layer_match.group(4))
                        collected_for_current_step.append((param_name, layer_id, grama_ratio, bh_calc))
            
            # Process any remaining buffered lines for the last step
            if current_ppo_step != -1 and current_ppo_step in target_ppo_steps:
                for entry in collected_for_current_step:
                    param, layer_id, grama_r, bh_c = entry
                    layer_data[current_ppo_step][param][layer_id]['grama_ratio'] = grama_r
                    layer_data[current_ppo_step][param][layer_id]['bh_calc'] = bh_c

    except FileNotFoundError:
        print(f"Warning: Log file not found {log_file_path}")
    except Exception as e:
        print(f"Error parsing layer-wise metrics from {log_file_path}: {e}")
    return layer_data

def plot_performance_curves(sft_step_num_str, group_data, group_label, metric_key, y_label, output_dir):
    plt.figure(figsize=(10, 6))
    steps = sorted(group_data.keys())
    values = [np.mean(group_data[step][metric_key]) for step in steps]
    if len(values) > 1: # Only smooth if there's more than one point
        smoothed_values = smooth_curve(values)
    else:
        smoothed_values = values
    
    if not steps or not values:
        print(f"No data for {group_label} - {y_label} in SFT step {sft_step_num_str}")
        plt.close()
        return

    plt.plot(steps, smoothed_values, marker='o', linestyle='-', label=f"{group_label} - {y_label} (Smoothed)")
    # Optionally, plot original data lightly
    # plt.plot(steps, values, marker='.', linestyle='--', alpha=0.4, label=f"{group_label} - {y_label} (Raw)")
    plt.xlabel("PPO Step")
    plt.ylabel(y_label)
    plt.title(f"{y_label} for {group_label} (SFT Step {sft_step_num_str})")
    plt.legend()
    plt.grid(True)
    plot_path = output_dir / f"{sft_step_num_str}_{group_label.lower().replace(' ', '_')}_{metric_key.replace('/', '_')}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

def plot_heatmap(data_matrix, title, output_path, avg_bh_calc):
    if data_matrix.size == 0:
        print(f"Skipping heatmap due to empty data: {title}")
        return
    plt.figure(figsize=(12, 8))
    # Use a sequential colormap (e.g., 'viridis', 'plasma', 'magma', 'cividis')
    # Or a diverging one if data can be positive/negative around a central point
    # For 0-1 ratio, 'viridis' or 'YlGnBu' are good choices.
    cmap = plt.cm.get_cmap('YlGnBu').copy() # Or 'viridis', 'plasma'
    cmap.set_bad(color='lightgrey') # Color for NaN values
    
    plt.imshow(data_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(label="Grama Ratio")
    plt.xlabel("Layer ID (0-27)")
    plt.ylabel("Parameter (Implicit Index)") # This needs adjustment if plotting single param
    # If data_matrix is (1, NUM_LAYERS) for a single parameter:
    plt.yticks([]) # No y-ticks if it's just one row for one parameter
    plt.ylabel(title.split(' - ')[1].split(' (')[0]) # Extract param name for y-label

    plt.xticks(np.arange(NUM_LAYERS))
    plt.title(f"{title}\nAvg B/H_calc: {avg_bh_calc:.4e}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap: {output_path}")

def plot_bh_calc_curves(sft_step_num_str, group_label, data_dict, output_dir):
    # data_dict: {param_name: {ppo_step: avg_bh_calc_for_param_over_layers}}
    plt.figure(figsize=(12, 7))
    
    plotted_anything = False
    for param_name in TARGET_PARAMS:
        if param_name in data_dict:
            param_data = data_dict[param_name]
            steps = sorted(param_data.keys())
            avg_bh_calcs = [param_data[step] for step in steps]
            if len(avg_bh_calcs) > 1:
                smoothed_avg_bh_calcs = smooth_curve(avg_bh_calcs)
            else:
                smoothed_avg_bh_calcs = avg_bh_calcs
            if steps and avg_bh_calcs:
                plt.plot(steps, smoothed_avg_bh_calcs, marker='o', linestyle='-', label=f"{param_name} (Smoothed)")
                # Optionally, plot original data lightly
                # plt.plot(steps, avg_bh_calcs, marker='.', linestyle='--', alpha=0.4, label=f"{param_name} (Raw)")
                plotted_anything = True
    
    if not plotted_anything:
        print(f"No B/H_calc data to plot for {group_label} in SFT step {sft_step_num_str}")
        plt.close()
        return

    plt.xlabel("PPO Step")
    plt.ylabel("Average B/H_calc (across layers 0-27)")
    plt.title(f"Avg B/H_calc vs. PPO Step for {group_label} (SFT Step {sft_step_num_str})")
    plt.legend(loc='best')
    plt.grid(True)
    plt.yscale('log') # B/H_calc can vary a lot, log scale might be useful
    plot_path = output_dir / f"{sft_step_num_str}_{group_label.lower().replace(' ', '_')}_avg_bh_calc_vs_step.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot: {plot_path}")

def aggregate_unknown_group_data(log_files, parse_func, *args):
    all_group_data = []
    for log_file in log_files:
        data = parse_func(log_file, *args)
        if data:
            all_group_data.append(data)
    
    if not all_group_data:
        return None

    # For general metrics (dict of dicts: {step: {'score': [val], 'grama': [val]}})
    if parse_func == parse_general_metrics:
        # Data from parse_general_metrics is (metrics_dict, steps_list, scores_list, gramas_list)
        # We need to average scores and gramas per step across groups.
        aggregated_metrics = defaultdict(lambda: {'score': [], 'grama': []})
        all_steps = set()
        
        processed_data = []
        for data_tuple in all_group_data:
            metrics_dict, _, _, _ = data_tuple
            processed_data.append(metrics_dict)
            for step in metrics_dict:
                all_steps.add(step)
        
        for step in sorted(list(all_steps)):
            step_scores = []
            step_gramas = []
            for group_metrics in processed_data:
                if step in group_metrics:
                    step_scores.extend(group_metrics[step]['score'])
                    step_gramas.extend(group_metrics[step]['grama'])
            if step_scores:
                aggregated_metrics[step]['score'] = [np.mean(step_scores)]
            if step_gramas:
                aggregated_metrics[step]['grama'] = [np.mean(step_gramas)]
        return aggregated_metrics

    # For layer-wise metrics (dict of dicts: {ppo_step: {param: {layer: {'grama_ratio', 'bh_calc'}}}})
    elif parse_func == parse_layer_wise_metrics:
        # all_group_data is a list of dicts like: [ {ppo_step: {param: {layer: data}}} , ... ]
        aggregated_layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'grama_ratio': [], 'bh_calc': []})))
        
        all_ppo_steps = set()
        for group_data in all_group_data:
            for ppo_step in group_data:
                all_ppo_steps.add(ppo_step)

        for ppo_step in sorted(list(all_ppo_steps)):
            for group_data in all_group_data:
                if ppo_step in group_data:
                    for param_name, layers in group_data[ppo_step].items():
                        for layer_id, values in layers.items():
                            if 'grama_ratio' in values:
                                aggregated_layer_data[ppo_step][param_name][layer_id]['grama_ratio'].append(values['grama_ratio'])
                            if 'bh_calc' in values:
                                aggregated_layer_data[ppo_step][param_name][layer_id]['bh_calc'].append(values['bh_calc'])
        
        # Now average the collected lists
        final_avg_layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for ppo_step, params_data in aggregated_layer_data.items():
            for param_name, layers_data in params_data.items():
                for layer_id, values_dict in layers_data.items():
                    if values_dict['grama_ratio']:
                        final_avg_layer_data[ppo_step][param_name][layer_id]['grama_ratio'] = np.mean(values_dict['grama_ratio'])
                    if values_dict['bh_calc']:
                        final_avg_layer_data[ppo_step][param_name][layer_id]['bh_calc'] = np.mean(values_dict['bh_calc'])
        return final_avg_layer_data
    return None

def main():
    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        return

    # Clear and recreate plot output directory
    if PLOT_OUTPUT_DIR.exists():
        shutil.rmtree(PLOT_OUTPUT_DIR)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_step_dirs = [d for d in LOG_DIR.iterdir() if d.is_dir() and SFT_DIR_PATTERN.match(d.name)]

    for sft_dir_path in sft_step_dirs:
        sft_match = SFT_DIR_PATTERN.match(sft_dir_path.name)
        if not sft_match:
            continue
        sft_step_num_str = sft_match.group(1)
        print(f"\nProcessing SFT Step: {sft_step_num_str} (from {sft_dir_path.name})")

        sft_plot_output_dir = PLOT_OUTPUT_DIR / sft_step_num_str
        sft_plot_output_dir.mkdir(parents=True, exist_ok=True)

        group_log_files = list(sft_dir_path.glob("Group*.log"))
        known_group_files = [f for f in group_log_files if GROUP_LOG_PATTERN.match(f.name) and GROUP_LOG_PATTERN.match(f.name).group(1) == '0']
        unknown_group_files = [f for f in group_log_files if GROUP_LOG_PATTERN.match(f.name) and GROUP_LOG_PATTERN.match(f.name).group(1) in ['1', '2', '3']]

        # --- 1. Performance Curves ---
        print("  Plotting performance curves...")
        if known_group_files:
            # For simplicity, assuming one Group0 log file. If multiple, this would need averaging or selection.
            known_data, _, _, _ = parse_general_metrics(known_group_files[0]) 
            if known_data:
                plot_performance_curves(sft_step_num_str, known_data, "Known Group", 'score', "Score (critic/score/mean)", sft_plot_output_dir)
                plot_performance_curves(sft_step_num_str, known_data, "Known Group", 'grama', "Grama (actor/zero_gradspace_ratio)", sft_plot_output_dir)
        
        if unknown_group_files:
            avg_unknown_data_general = aggregate_unknown_group_data(unknown_group_files, parse_general_metrics)
            if avg_unknown_data_general:
                plot_performance_curves(sft_step_num_str, avg_unknown_data_general, "Unknown Group Avg", 'score', "Score (critic/score/mean)", sft_plot_output_dir)
                plot_performance_curves(sft_step_num_str, avg_unknown_data_general, "Unknown Group Avg", 'grama', "Grama (actor/zero_gradspace_ratio)", sft_plot_output_dir)

        # --- 2. Case Study --- 
        print("  Processing case study data...")
        # Known Group Case Study
        known_layer_data = None
        if known_group_files:
            known_layer_data = parse_layer_wise_metrics(known_group_files[0], CASE_STUDY_PPO_STEPS)

        # Unknown Group Average Case Study
        avg_unknown_layer_data = None
        if unknown_group_files:
            avg_unknown_layer_data = aggregate_unknown_group_data(unknown_group_files, parse_layer_wise_metrics, CASE_STUDY_PPO_STEPS)

        for group_data, group_label_prefix in [(known_layer_data, "Known_Group"), (avg_unknown_layer_data, "Unknown_Group_Avg")]:
            if not group_data: continue

            # Heatmaps
            heatmap_dir = sft_plot_output_dir / f"{group_label_prefix}_heatmaps"
            heatmap_dir.mkdir(exist_ok=True)
            
            bh_calc_for_curves = defaultdict(lambda: defaultdict(float)) # {param_name: {ppo_step: avg_bh_calc}}

            for ppo_step in CASE_STUDY_PPO_STEPS:
                if ppo_step not in group_data: continue
                
                for param_name in TARGET_PARAMS:
                    if param_name not in group_data[ppo_step]: continue
                    
                    grama_ratios_for_heatmap = np.full(NUM_LAYERS, np.nan) # Initialize with NaN
                    bh_calcs_for_avg = []

                    for layer_id in range(NUM_LAYERS):
                        if layer_id in group_data[ppo_step][param_name]:
                            layer_metrics = group_data[ppo_step][param_name][layer_id]
                            if 'grama_ratio' in layer_metrics:
                                grama_ratios_for_heatmap[layer_id] = layer_metrics['grama_ratio']
                            if 'bh_calc' in layer_metrics:
                                bh_calcs_for_avg.append(layer_metrics['bh_calc'])
                    
                    avg_bh_calc_for_step_param = np.mean(bh_calcs_for_avg) if bh_calcs_for_avg else 0.0
                    bh_calc_for_curves[param_name][ppo_step] = avg_bh_calc_for_step_param
                    
                    # Reshape for single-parameter heatmap (1 row, NUM_LAYERS columns)
                    heatmap_matrix = grama_ratios_for_heatmap.reshape(1, NUM_LAYERS)
                    
                    heatmap_title = f"{group_label_prefix} - {param_name} (PPO Step {ppo_step})"
                    heatmap_filename = f"{sft_step_num_str}_{group_label_prefix}_{param_name.replace('.', '_')}_step{ppo_step}_heatmap.png"
                    plot_heatmap(heatmap_matrix, heatmap_title, heatmap_dir / heatmap_filename, avg_bh_calc_for_step_param)
            
            # B/H_calc Curves
            plot_bh_calc_curves(sft_step_num_str, group_label_prefix, bh_calc_for_curves, sft_plot_output_dir)

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()
