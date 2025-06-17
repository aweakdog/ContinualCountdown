import os
import re
import glob
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter # For Savitzky-Golay smoothing, often better than simple moving average

SMOOTHING_WINDOW_SIZE = 23
SMOOTHING_POLYORDER = 2 # Relevant for Savitzky-Golay

# Helper function to transform parsed general metrics to step-centric format
def transform_to_step_centric(general_metrics_dict):
    step_centric_data = defaultdict(dict)
    if not general_metrics_dict or not general_metrics_dict.get('ppo_steps'):
        return step_centric_data
    for i, step in enumerate(general_metrics_dict['ppo_steps']):
        if general_metrics_dict.get('scores') and i < len(general_metrics_dict['scores']):
            step_centric_data[step]['scores'] = general_metrics_dict['scores'][i]
        if general_metrics_dict.get('grama_ratios') and i < len(general_metrics_dict['grama_ratios']):
            step_centric_data[step]['grama_ratios'] = general_metrics_dict['grama_ratios'][i]
    return step_centric_data

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

# Configure Matplotlib to use 'Agg' backend for non-GUI environments
plt.switch_backend('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
PLOT_OUTPUT_DIR = BASE_DIR / "plots" / "plot_grama_ratio"

# Constants
TARGET_PARAMS = [
    "q_proj.weight",
    "k_proj.weight",
    "o_proj.weight",
    "v_proj.weight"
]
CASE_STUDY_PPO_STEPS = [1, 25, 50, 100, 200, 300] # PPO steps for detailed layer-wise case study (e.g., heatmaps)
BH_CALC_PPO_STEPS_RANGE = list(range(1, 301)) # PPO steps for bh_calc line plots
NUM_LAYERS = 28  # 0 to 27

# Regex patterns
ANSI_ESCAPE_PATTERN = re.compile(r'\x1b\[[0-9;]*[mK]')
SFT_DIR_PATTERN = re.compile(r"continual_countdown3b_sft_global_step_(\d+)_\d{8}_\d{6}")
GROUP_LOG_PATTERN = re.compile(r"Group(\d+)_\d{8}_\d{6}\.log")

GENERAL_METRICS_PATTERN = re.compile(
    r"step:(\d+) .*? actor/zero_gradspace_ratio:([-\d.]+) .*? critic/score/mean:([-\d.]+)"
)
# LAYER_WISE_DETAIL_PATTERN is now defined locally in parse_layer_wise_metrics
# to handle multiple formats.


def clean_ansi_codes(text):
    return ANSI_ESCAPE_PATTERN.sub('', text)

def parse_general_metrics(log_file_path):
    general_metrics = {'ppo_steps': [], 'scores': [], 'grama_ratios': []}
    print(f"  Parsing general metrics from: {log_file_path}")
    lines_parsed_general = 0
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = clean_ansi_codes(line)
                match = GENERAL_METRICS_PATTERN.search(cleaned_line)
                if match:
                    ppo_step = int(match.group(1))
                    grama_ratio = float(match.group(2))  # Group 2 is now actor/zero_gradspace_ratio
                    score_mean = float(match.group(3))   # Group 3 is now critic/score/mean
                    general_metrics['ppo_steps'].append(ppo_step)
                    general_metrics['scores'].append(score_mean)
                    general_metrics['grama_ratios'].append(grama_ratio)
                    lines_parsed_general += 1
    except FileNotFoundError:
        print(f"Warning: Log file not found {log_file_path}")
    except Exception as e:
        print(f"Error parsing general metrics from {log_file_path}: {e}")
    print(f"  Finished parsing general metrics from {log_file_path}. Found {lines_parsed_general} PPO step entries.")
    return general_metrics

def parse_layer_wise_metrics(log_file_path, target_ppo_steps):
    layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    # Structure: layer_data[ppo_step][param_name][layer_id] = {'grama_ratio': val, 'bh_calc': val}
    print(f"  Parsing layer-wise metrics from: {log_file_path} for PPO steps: {target_ppo_steps}")
    found_layer_lines = 0
    associated_layer_data_count = 0
    temp_layer_details_buffer = []
    target_params_str = "|".join(p.replace('.', '\\.') for p in TARGET_PARAMS)

    # Pattern for [ZeroGradV2] format
    LAYER_WISE_DETAIL_PATTERN_V2 = re.compile(
        r"\[ZeroGradV2\] model\.layers\.(\d+)\.self_attn\.({params_regex}):"
        r"\s*\d+/\d+\s*\(([\d\.]+)\),"  # Group 3: grama_ratio (decimal)
        r"\s*B/H:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)".format(params_regex=target_params_str) # Group 4: B/H_calc
    )

    # Pattern for the older format
    LAYER_WISE_DETAIL_PATTERN_OLD = re.compile(
        r"Layer: model\.layers\.(\d+)\.self_attn\.({params_regex})\s*\|"
        r".*?\(\s*([\d\.]+)%\s*\)"  # Group 3: grama_ratio (percentage)
        r".*?B/H_calc:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)".format(params_regex=target_params_str) # Group 4: B/H_calc
    )

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned_line = clean_ansi_codes(line)
                
                layer_id, param_name, grama_ratio, bh_calc = None, None, None, None

                # Try matching [ZeroGradV2] format first
                match_v2 = LAYER_WISE_DETAIL_PATTERN_V2.search(cleaned_line)
                if match_v2:
                    layer_id = int(match_v2.group(1))
                    param_name = match_v2.group(2)
                    grama_ratio = float(match_v2.group(3))  # Already decimal
                    bh_calc = float(match_v2.group(4))
                else:
                    # Try matching the older format
                    match_old = LAYER_WISE_DETAIL_PATTERN_OLD.search(cleaned_line)
                    if match_old:
                        layer_id = int(match_old.group(1))
                        param_name = match_old.group(2)
                        grama_ratio = float(match_old.group(3)) / 100.0  # Convert percentage
                        bh_calc = float(match_old.group(4))
                
                if param_name is not None: # If either pattern matched
                    temp_layer_details_buffer.append({
                        'layer_id': layer_id, 'param_name': param_name,
                        'grama_ratio': grama_ratio, 'bh_calc': bh_calc
                    })
                    found_layer_lines +=1
                    continue # Successfully processed as a layer line, move to next line

                # If not a layer detail line, check if it's a general step summary line
                step_match = GENERAL_METRICS_PATTERN.search(cleaned_line)
                if step_match:
                    current_ppo_step = int(step_match.group(1))
                    
                    # Layer details in buffer belong to this current_ppo_step
                    if current_ppo_step in target_ppo_steps and temp_layer_details_buffer:
                        for entry in temp_layer_details_buffer:
                            layer_data[current_ppo_step][entry['param_name']][entry['layer_id']]['grama_ratio'] = entry['grama_ratio']
                            layer_data[current_ppo_step][entry['param_name']][entry['layer_id']]['bh_calc'] = entry['bh_calc']
                            associated_layer_data_count +=1
                    
                    temp_layer_details_buffer = [] # Clear buffer, these details have been assigned or step not targeted

    except FileNotFoundError:
        print(f"Warning: Log file not found {log_file_path}")
    except Exception as e:
        print(f"Error parsing layer-wise metrics from {log_file_path}: {e}")
    print(f"  Finished parsing layer-wise metrics from {log_file_path}. Found {found_layer_lines} layer detail lines. Associated {associated_layer_data_count} data points to PPO steps.")
    # print(f"  Layer data for {log_file_path}: {json.dumps(layer_data, indent=2)}") # Potentially very verbose
    return layer_data

def plot_performance_curves(sft_step_num_str, group_data, group_label, metric_key, y_label, output_dir):
    plt.figure(figsize=(10, 6))
    steps_from_data = sorted(group_data.keys())
    values = []
    valid_steps_for_plot = []

    for step_val in steps_from_data:
        step_metrics = group_data.get(step_val)
        if not isinstance(step_metrics, dict):
            # print(f"Warning: Data for PPO step {step_val} in {group_label} is not a dictionary. Skipping.")
            continue

        metric_val = step_metrics.get(metric_key)
        if metric_val is None:
            # print(f"Warning: Metric '{metric_key}' not found for PPO step {step_val} in {group_label}. Skipping.")
            continue
        
        current_val_to_plot = None
        if isinstance(metric_val, list):
            if not metric_val: 
                # print(f"Warning: Metric '{metric_key}' for PPO step {step_val} in {group_label} is an empty list. Skipping.")
                continue
            current_val_to_plot = np.mean(metric_val)
        elif isinstance(metric_val, (int, float)):
            current_val_to_plot = metric_val # Use directly if already a number
        else:
            # print(f"Warning: Metric '{metric_key}' for PPO step {step_val} in {group_label} is of unexpected type: {type(metric_val)}. Skipping.")
            continue
        
        values.append(current_val_to_plot)
        valid_steps_for_plot.append(step_val)

    if not values:
        print(f"No valid data points to plot for {group_label} - {y_label} in SFT step {sft_step_num_str}")
        plt.close()
        return

    steps_to_plot = valid_steps_for_plot
    
    if len(values) > SMOOTHING_WINDOW_SIZE:
        smoothed_values = smooth_curve(values)
        plt.plot(steps_to_plot, smoothed_values, linestyle='-', label=f"{group_label} - {y_label} (Smoothed)") # No marker for smoothed
    else:
        # Plot raw data if not enough points to smooth or if smoothing is not desired for short series
        plt.plot(steps_to_plot, values, marker='o', markersize=0.1, linestyle='-', label=f"{group_label} - {y_label}")
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

def plot_heatmaps(sft_step_num_str, group_label_prefix, heatmap_data_group, output_dir, ppo_steps_for_heatmap):
    # heatmap_data_group is: {param_name: {(ppo_step, layer_id): grama_ratio}}
    # ppo_steps_for_heatmap is CASE_STUDY_PPO_STEPS (a list of PPO step numbers)

    for param_name in TARGET_PARAMS: # Iterate through TARGET_PARAMS to ensure consistent order and inclusion
        if param_name not in heatmap_data_group:
            # This check is useful if heatmap_data_group might not contain all TARGET_PARAMS
            # print(f"    Skipping heatmap for {param_name} in {group_label_prefix} (SFT {sft_step_num_str}): Param not in collected heatmap data.")
            continue

        param_specific_data = heatmap_data_group[param_name] # This is {(ppo_step, layer_id): grama_ratio}
        
        num_ppo_steps = len(ppo_steps_for_heatmap)
        data_matrix = np.full((num_ppo_steps, NUM_LAYERS), np.nan)

        for i, ppo_step in enumerate(ppo_steps_for_heatmap):
            for j in range(NUM_LAYERS): # layer_id
                if (ppo_step, j) in param_specific_data:
                    data_matrix[i, j] = param_specific_data[(ppo_step, j)]
        
        if np.all(np.isnan(data_matrix)):
            print(f"    Skipping heatmap for {param_name} in {group_label_prefix} (SFT {sft_step_num_str}): All NaN data matrix.")
            continue

        heatmap_title = f"Grama Ratio Heatmap: {group_label_prefix} - {param_name}\n(SFT {sft_step_num_str})"
        heatmap_filename = f"{sft_step_num_str}_{group_label_prefix.lower().replace(' ', '_')}_{param_name.replace('.', '_')}_grama_heatmap.png"
        heatmap_path = output_dir / heatmap_filename
        
        # Call the singular plot_heatmap function, passing the PPO step labels for the y-axis
        ytick_labels_to_use = ppo_steps_for_heatmap if ppo_steps_for_heatmap is not None else True
        xtick_labels_to_use = [str(i) for i in range(NUM_LAYERS)] if NUM_LAYERS <= 10 else (True if NUM_LAYERS <=28 else False) # Auto if too many, or show some
        if NUM_LAYERS > 10 and NUM_LAYERS <=28:
            # Show every few layers if there are many, e.g., 0, 4, 8 ...
            xtick_positions = np.arange(0, NUM_LAYERS, 4)
            xtick_labels_actual = [str(pos) for pos in xtick_positions]
        elif NUM_LAYERS <= 10:
            xtick_positions = np.arange(NUM_LAYERS)
            xtick_labels_actual = [str(i) for i in range(NUM_LAYERS)]
        else: # Too many layers, let heatmap decide or hide
            xtick_positions = None 
            xtick_labels_actual = True # let heatmap decide

        plot_heatmap(data_matrix, heatmap_title, heatmap_path, None, ppo_steps_for_heatmap_labels=ytick_labels_to_use, xtick_labels=xtick_labels_actual, xtick_positions=xtick_positions)

def plot_heatmap(data_matrix, title, output_path, avg_bh_calc, ppo_steps_for_heatmap_labels=None, xtick_labels=True, xtick_positions=None):
    if data_matrix.size == 0:
        print(f"Skipping heatmap due to empty data: {title}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap('Blues') # White for low values, dark blue for high values
    cmap.set_bad(color='lightgrey') # Color for NaN values

    # Determine y-tick labels for PPO steps
    yticklabels_to_use = ppo_steps_for_heatmap_labels if ppo_steps_for_heatmap_labels is not None else True

    # Determine x-tick labels for Layers
    if xtick_positions is not None and xtick_labels is not None:
        # Use provided positions and labels
        actual_xticklabels = xtick_labels
        actual_xticks = xtick_positions
    elif isinstance(xtick_labels, list):
        # Use provided list of labels, assume positions are range(len(xtick_labels))
        actual_xticklabels = xtick_labels
        actual_xticks = np.arange(len(xtick_labels))
    else: # Default behavior or True
        actual_xticklabels = True # Let seaborn decide default layer ticks
        actual_xticks = np.arange(NUM_LAYERS) # Default positions

    sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                cbar_kws={'label': 'Grama Ratio'}, 
                yticklabels=yticklabels_to_use, 
                xticklabels=actual_xticklabels,
                vmin=0, vmax=1) # Explicitly set vmin/vmax for grama ratio
    
    ax.set_xlabel(f"Layer ID (0 to {NUM_LAYERS-1})")
    ax.set_ylabel("PPO Step")
    ax.set_title(title)

    # Adjust x-ticks if custom positions were used to center them
    if xtick_positions is not None:
        ax.set_xticks([pos + 0.5 for pos in actual_xticks]) # Center custom ticks
        ax.set_xticklabels(actual_xticklabels) # Ensure custom labels are applied
    elif isinstance(xtick_labels, list) and len(xtick_labels) < NUM_LAYERS: # e.g. showing every Nth layer
        ax.set_xticks([pos + 0.5 for pos in actual_xticks])
        ax.set_xticklabels(actual_xticklabels)
    # Else, seaborn's default tick handling for xticklabels=True should be fine

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap: {output_path}")

def plot_grama_vs_layer_curves(sft_step_num_str, group_label_prefix, ppo_step, params_data_for_step, output_dir):
    """Plots grama ratio vs. layer ID for q, k, v, o parameters on one graph for a specific PPO step."""
    plt.figure(figsize=(12, 7))
    layer_ids = np.arange(NUM_LAYERS)
    plotted_anything = False

    for param_name in TARGET_PARAMS:
        if param_name in params_data_for_step:
            grama_ratios_for_layers = np.full(NUM_LAYERS, np.nan)
            for layer_id in range(NUM_LAYERS):
                if layer_id in params_data_for_step[param_name] and 'grama_ratio' in params_data_for_step[param_name][layer_id]:
                    grama_ratios_for_layers[layer_id] = params_data_for_step[param_name][layer_id]['grama_ratio']
            
            # Only plot if there's some non-NaN data for this parameter
            if not np.all(np.isnan(grama_ratios_for_layers)):
                if len(grama_ratios_for_layers[~np.isnan(grama_ratios_for_layers)]) > SMOOTHING_WINDOW_SIZE:
                    # Smooth only non-NaN parts if possible, or handle NaNs carefully
                    # For simplicity, if NaNs are present, we might plot raw or skip smoothing for that line
                    # This example assumes smooth_curve can handle NaNs or we filter them before
                    # A more robust approach would be to smooth contiguous non-NaN segments
                    valid_indices = ~np.isnan(grama_ratios_for_layers)
                    if np.any(valid_indices):
                        smoothed_segment = smooth_curve(grama_ratios_for_layers[valid_indices])
                        plt.plot(layer_ids[valid_indices], smoothed_segment, linestyle='-', label=f"{param_name} (s)") # No marker for smoothed
                    else: # All NaNs, plot nothing or raw (which will be nothing)
                        plt.plot(layer_ids, grama_ratios_for_layers, marker='o', markersize=0.2, linestyle='-', label=param_name)
                else:
                    plt.plot(layer_ids, grama_ratios_for_layers, marker='o', markersize=0.2, linestyle='-', label=param_name)
                plotted_anything = True

    if not plotted_anything:
        print(f"    No grama ratio vs layer data to plot for {group_label_prefix}, PPO step {ppo_step} in SFT step {sft_step_num_str}")
        plt.close()
        return

    plt.xlabel("Layer ID")
    plt.ylabel("Grama Ratio")
    plt.title(f"Grama Ratio vs. Layer ID for {group_label_prefix} (SFT {sft_step_num_str}, PPO Step {ppo_step})")
    plt.xticks(np.arange(0, NUM_LAYERS, 2)) # Show ticks every 2 layers
    plt.ylim(0, 1.05) # Grama ratio is 0-1
    plt.legend(loc='best')
    plt.grid(True)
    plot_filename = f"{sft_step_num_str}_{group_label_prefix.lower().replace(' ', '_')}_ppo_step_{ppo_step}_grama_vs_layer.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path)
    plt.close()
    print(f"    Saved grama_vs_layer plot: {plot_path}")
def plot_bh_calc_curves(sft_step_num_str, group_label_prefix, bh_calc_data_step_centric, output_dir):
    # bh_calc_data_step_centric is {param_name: {layer_id: {ppo_step: bh_calc_value}}}
    # Creates one plot per SFT_STEP & GROUP, showing all TARGET_PARAMS, with lines for each layer.
    plt.figure(figsize=(14, 8))
    plotted_anything_on_this_figure = False

    for param_name in TARGET_PARAMS: # Iterate through defined TARGET_PARAMS to maintain order if possible
        if param_name not in bh_calc_data_step_centric:
            continue
        
        layers_data = bh_calc_data_step_centric[param_name]
        if not layers_data:
            continue

        for layer_id, ppo_step_bh_values in sorted(layers_data.items()): # Sort by layer_id for consistent legend order
            if not ppo_step_bh_values:
                continue

            # Filter for steps within BH_CALC_PPO_STEPS_RANGE, though parsing should already handle this
            sorted_ppo_steps = sorted([step for step in ppo_step_bh_values.keys() if step in BH_CALC_PPO_STEPS_RANGE])
            if not sorted_ppo_steps:
                continue
            
            bh_values_for_plot = [ppo_step_bh_values[step] for step in sorted_ppo_steps]
            
            if not bh_values_for_plot or len(bh_values_for_plot) < 1:
                continue

            if len(bh_values_for_plot) > SMOOTHING_WINDOW_SIZE:
                smoothed_bh_values = smooth_curve(bh_values_for_plot)
                plt.plot(sorted_ppo_steps, smoothed_bh_values, linestyle='-', label=f"{param_name} L{layer_id} (s)") # No marker for smoothed
            else:
                plt.plot(sorted_ppo_steps, bh_values_for_plot, marker='o', markersize=0.2, linestyle='-', label=f"{param_name} L{layer_id}")
            plotted_anything_on_this_figure = True

    if not plotted_anything_on_this_figure:
        print(f"    No B/H_calc data to plot for {group_label_prefix} in SFT step {sft_step_num_str} (PPO Steps 1-{BH_CALC_PPO_STEPS_RANGE[-1] if BH_CALC_PPO_STEPS_RANGE else 'N/A'})")
        plt.close()
        return

    plt.xlabel("PPO Step")
    plt.ylabel("B/H_calc (Log Scale)")
    plt.title(f"B/H_calc vs. PPO Step for {group_label_prefix} (SFT {sft_step_num_str})")
    plt.legend(title="Param & Layer", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for wider legend
    plot_filename = f"{sft_step_num_str}_{group_label_prefix.lower().replace(' ', '_')}_all_params_layers_bh_calc_vs_step.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path)
    plt.close()
    print(f"    Saved B/H_calc plot: {plot_path}")

def plot_sft_comparison_curves(all_sfts_data, group_key_in_sft_data, metric_key, plot_title, y_axis_label, output_dir):
    plt.figure(figsize=(12, 7))
    sft_steps_plotted = 0
    # Using matplotlib's default color cycle. To use a specific one:
    # num_sft_steps = len([k for k, v in all_sfts_data.items() if v.get(group_key_in_sft_data)])
    # colors = plt.cm.viridis(np.linspace(0, 1, num_sft_steps if num_sft_steps > 0 else 1))

    sorted_sft_keys = sorted(all_sfts_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for i, sft_step_num in enumerate(sorted_sft_keys):
        sft_data = all_sfts_data.get(sft_step_num)
        if not sft_data:
            continue
        group_specific_data = sft_data.get(group_key_in_sft_data)
        if not group_specific_data:
            continue

        steps_from_data = sorted(group_specific_data.keys())
        mean_values = []
        std_values = [] # To store standard deviations
        valid_steps_for_plot = []

        metric_key_mean = f"{metric_key}_mean"
        metric_key_std = f"{metric_key}_std"

        for ppo_step_val in steps_from_data:
            step_metrics = group_specific_data.get(ppo_step_val)
            if not isinstance(step_metrics, dict):
                continue
            
            mean_val = step_metrics.get(metric_key_mean)
            std_val = step_metrics.get(metric_key_std)

            if mean_val is None: # If mean is None, skip this point
                # print(f"Warning: Mean for {metric_key} at PPO step {ppo_step_val} for SFT {sft_step_num} is None. Skipping.")
                continue
            
            # std_val can be None or 0 if not available or single data point, default to 0.0
            std_val = std_val if std_val is not None else 0.0
            
            if not (isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float))):
                print(f"Warning: Mean or Std for {metric_key} at PPO step {ppo_step_val} for SFT {sft_step_num} are not numbers. Mean: {mean_val} (type: {type(mean_val)}), Std: {std_val} (type: {type(std_val)}). Skipping.")
                continue
            
            mean_values.append(mean_val)
            std_values.append(std_val)
            valid_steps_for_plot.append(ppo_step_val)

        if not mean_values:
            continue

        mean_values_np = np.array(mean_values)
        std_values_np = np.array(std_values)

        if len(mean_values_np) > 1:
            smoothed_means = smooth_curve(mean_values_np)
            smoothed_stds = smooth_curve(std_values_np) # Smooth std deviations as well
        else:
            smoothed_means = mean_values_np
            smoothed_stds = std_values_np
        
        # current_color = colors[i % len(colors)] if specific colors are used, else None for default cycle
        line, = plt.plot(valid_steps_for_plot, smoothed_means, marker='o', markersize=0.1, linestyle='-', label=f"SFT {sft_step_num}") # color=current_color
        plt.fill_between(valid_steps_for_plot, 
                         smoothed_means - smoothed_stds, 
                         smoothed_means + smoothed_stds, 
                         color=line.get_color(), alpha=0.2)
        sft_steps_plotted += 1

    if sft_steps_plotted == 0:
        print(f"No data found to plot for: {plot_title}")
        plt.close()
        return

    plt.xlabel("PPO Step")
    plt.ylabel(y_axis_label)
    plt.title(plot_title)
    plt.legend(title="SFT Step", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    # Use the original metric_key for the filename, not the suffixed one
    filename_metric_cleaned = metric_key.replace('/', '_').replace(' ', '_').lower()
    filename_group = group_key_in_sft_data.replace(' ', '_').lower()
    plot_path = output_dir / f"comparison_{filename_group}_{filename_metric_cleaned}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved SFT comparison plot: {plot_path}")

def aggregate_unknown_group_data(log_files, parse_func, *args):
    all_group_data = []
    for log_file in log_files:
        data = parse_func(log_file, *args)
        if data:
            all_group_data.append(data)
    
    if not all_group_data:
        return None

    # For general metrics (dict: {'ppo_steps': [], 'scores': [], 'grama_ratios': []})
    if parse_func == parse_general_metrics:
        all_ppo_steps = []
        all_scores = []
        all_grama_ratios = []
        
        # all_group_data already contains the dictionaries from parse_general_metrics
        for parsed_data_dict in all_group_data:
            if parsed_data_dict and parsed_data_dict.get('ppo_steps'):
                all_ppo_steps.extend(parsed_data_dict['ppo_steps'])
                all_scores.extend(parsed_data_dict['scores'])
                all_grama_ratios.extend(parsed_data_dict['grama_ratios'])
        
        if not all_ppo_steps:
            return None

        # Create a DataFrame for easier averaging per PPO step
        df = pd.DataFrame({
            'ppo_step': all_ppo_steps,
            'score': all_scores,
            'grama_ratio': all_grama_ratios
        })
        
        # Ensure ppo_step is not empty before groupby
        if df.empty or 'ppo_step' not in df.columns or df['ppo_step'].isnull().all():
             print("DEBUG: DataFrame for aggregation is empty or ppo_step column is missing/all NaN.")
             return None

        # Calculate mean and standard deviation
        avg_df = df.groupby('ppo_step').mean()
        std_df = df.groupby('ppo_step').std()
        
        # Transform averaged data to step-centric format
        aggregated_step_centric = defaultdict(dict)
        for step in avg_df.index: # Iterate over PPO steps (which are the index of avg_df)
            if 'score' in avg_df.columns:
                aggregated_step_centric[step]['scores_mean'] = avg_df.loc[step, 'score']
                # std_df might not have entries for all steps if only one data point exists for that step (std is NaN)
                # Or if a column is missing (e.g. all NaNs for that group), pandas std() might drop it.
                aggregated_step_centric[step]['scores_std'] = std_df.loc[step, 'score'] if step in std_df.index and 'score' in std_df.columns and not pd.isna(std_df.loc[step, 'score']) else 0.0
            if 'grama_ratio' in avg_df.columns:
                aggregated_step_centric[step]['grama_ratios_mean'] = avg_df.loc[step, 'grama_ratio']
                aggregated_step_centric[step]['grama_ratios_std'] = std_df.loc[step, 'grama_ratio'] if step in std_df.index and 'grama_ratio' in std_df.columns and not pd.isna(std_df.loc[step, 'grama_ratio']) else 0.0
        return aggregated_step_centric

    # For layer-wise metrics (dict of dicts: {ppo_step: {param: {layer: {'grama_ratio', 'bh_calc'}}}})
    elif parse_func == parse_layer_wise_metrics:
        # all_group_data is a list of dicts like: [ {ppo_step: {param: {layer: data}}} , ... ]
        aggregated_layer_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        # aggregated_layer_data[ppo_step][param_name][layer_id] = {'grama_ratio': val, 'bh_calc': val}
        
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

def generate_case_study_plots(sft_step_num_str, group_label_prefix, group_layer_data, output_dir):
    print(f"  Generating case study plots for SFT {sft_step_num_str}, Group: {group_label_prefix}")
    if not group_layer_data:
        print(f"    DEBUG: No layer data provided for {group_label_prefix}. Skipping case study plots.")
        return

    # bh_calc_for_curves is used here to collect avg B/H for heatmap annotation, specific to these case study steps
    bh_calc_for_heatmap_annotation = defaultdict(lambda: defaultdict(float)) # {param_name: {ppo_step: avg_bh_calc}}

    for ppo_step in CASE_STUDY_PPO_STEPS:
        if ppo_step not in group_layer_data:
            # print(f"    DEBUG: PPO step {ppo_step} not in group_layer_data for {group_label_prefix}. Skipping this step.")
            continue
        
        # Ensure data for this PPO step is not empty
        if not group_layer_data[ppo_step]:
            # print(f"    DEBUG: Data for PPO step {ppo_step} is empty for {group_label_prefix}. Skipping this step.")
            continue

        # Generate individual heatmaps (one per param, per PPO step)
        for param_name in TARGET_PARAMS:
            if param_name not in group_layer_data[ppo_step]:
                # print(f"      DEBUG: Param {param_name} not in PPO step {ppo_step} data for {group_label_prefix}. Skipping this param.")
                continue
            
            grama_ratios_for_heatmap = np.full(NUM_LAYERS, np.nan)
            bh_calcs_for_avg = []

            for layer_id in range(NUM_LAYERS):
                if layer_id in group_layer_data[ppo_step][param_name]:
                    layer_metrics = group_layer_data[ppo_step][param_name][layer_id]
                    if 'grama_ratio' in layer_metrics:
                        grama_ratios_for_heatmap[layer_id] = layer_metrics['grama_ratio']
                    if 'bh_calc' in layer_metrics:
                        bh_calcs_for_avg.append(layer_metrics['bh_calc'])
            
            avg_bh_calc_for_step_param = np.mean(bh_calcs_for_avg) if bh_calcs_for_avg else 0.0
            bh_calc_for_heatmap_annotation[param_name][ppo_step] = avg_bh_calc_for_step_param
            
            heatmap_matrix = grama_ratios_for_heatmap.reshape(1, NUM_LAYERS)
            heatmap_title = f"{group_label_prefix} - {param_name} (PPO Step {ppo_step})"
            heatmap_filename = f"{sft_step_num_str}_{group_label_prefix.replace(' ', '_')}_{param_name.replace('.', '_')}_step{ppo_step}_heatmap.png"
            
            # print(f"      DEBUG: Plotting heatmap for SFT {sft_step_num_str}, Group {group_label_prefix}, PPO {ppo_step}, Param {param_name}. Matrix shape: {heatmap_matrix.shape}, Avg B/H: {avg_bh_calc_for_step_param}")
            if not np.isnan(heatmap_matrix).all():
                plot_heatmap(heatmap_matrix, heatmap_title, output_dir / heatmap_filename, avg_bh_calc_for_step_param)
            # else:
                # print(f"        DEBUG: Heatmap data for {param_name} at PPO {ppo_step} is all NaN. Skipping plot.")

        # Plot grama ratio vs layer for q,k,v,o for this PPO step (after all params for this step's heatmaps)
        params_data_for_this_step = group_layer_data[ppo_step]
        # print(f"      DEBUG: Plotting grama_vs_layer for SFT {sft_step_num_str}, Group {group_label_prefix}, PPO {ppo_step}. Data keys: {list(params_data_for_this_step.keys()) if params_data_for_this_step else 'No data'}")
        if params_data_for_this_step:
             plot_grama_vs_layer_curves(sft_step_num_str, group_label_prefix, ppo_step, params_data_for_this_step, output_dir)
        # else:
            # print(f"      DEBUG: No param data for PPO step {ppo_step} in {group_label_prefix} for grama_vs_layer plot. Skipping.")

def main():
    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        return

    # Clear and recreate plot output directory
    if PLOT_OUTPUT_DIR.exists():
        shutil.rmtree(PLOT_OUTPUT_DIR)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Output directory set to: {PLOT_OUTPUT_DIR}")

    all_sfts_performance_data = defaultdict(lambda: defaultdict(dict))

    sft_dirs = [d for d in LOG_DIR.iterdir() if d.is_dir() and SFT_DIR_PATTERN.match(d.name)]

    for sft_dir_path in sft_dirs:
        sft_match = SFT_DIR_PATTERN.match(sft_dir_path.name)
        if not sft_match:
            continue
        sft_step_num_str = sft_match.group(1)
        print(f"\nProcessing SFT Step: {sft_step_num_str} (from {sft_dir_path.name})")

        group_log_files = list(sft_dir_path.glob("Group*.log"))
        known_group_files = [f for f in group_log_files if GROUP_LOG_PATTERN.match(f.name) and GROUP_LOG_PATTERN.match(f.name).group(1) == '0']
        unknown_group_files = [f for f in group_log_files if GROUP_LOG_PATTERN.match(f.name) and GROUP_LOG_PATTERN.match(f.name).group(1) in ['1', '2', '3']]

        # --- 1. Performance Curves ---
        print("  Plotting performance curves...")
        if known_group_files:
            print(f"    Processing Known Group (Group0) from: {known_group_files[0].name}")
            # Parse general metrics for performance curves
            parsed_known_data = parse_general_metrics(known_group_files[0])
            known_data_step_centric = transform_to_step_centric(parsed_known_data)
            
            if known_data_step_centric:
                plot_performance_curves(sft_step_num_str, known_data_step_centric, "Known Group", 'scores', "Critic Score (Mean)", PLOT_OUTPUT_DIR)
                plot_performance_curves(sft_step_num_str, known_data_step_centric, "Known Group", 'grama_ratios', "Grama Ratio (actor/zero_gradspace_ratio)", PLOT_OUTPUT_DIR)
                all_sfts_performance_data[sft_step_num_str]['known_group'] = known_data_step_centric # Store for SFT comparison
                print(f"      DEBUG: Known Group general metrics (SFT {sft_step_num_str}) parsed. PPO steps: {len(known_data_step_centric)}")
            else:
                print(f"      DEBUG: Known Group general metrics (SFT {sft_step_num_str}): No data or no PPO steps parsed.")

            # Parse layer-wise metrics for heatmaps and case study plots
            known_layer_data = parse_layer_wise_metrics(known_group_files[0], CASE_STUDY_PPO_STEPS + BH_CALC_PPO_STEPS_RANGE)
            if known_layer_data:
                # Prepare data for consolidated heatmap (PPO steps on y-axis)
                heatmap_data_known = defaultdict(dict) # {(ppo_step, layer_id): grama_ratio}
                for ppo_step_cs in CASE_STUDY_PPO_STEPS:
                    if ppo_step_cs in known_layer_data:
                        for param_name in TARGET_PARAMS:
                            if param_name in known_layer_data[ppo_step_cs]:
                                for layer_id, values in known_layer_data[ppo_step_cs][param_name].items():
                                    if 'grama_ratio' in values:
                                        heatmap_data_known[param_name][(ppo_step_cs, layer_id)] = values['grama_ratio']
                
                if heatmap_data_known:
                    plot_heatmaps(sft_step_num_str, "Known Group", heatmap_data_known, PLOT_OUTPUT_DIR, CASE_STUDY_PPO_STEPS)
                else:
                    print(f"      DEBUG: No data prepared for consolidated heatmap for Known Group (SFT {sft_step_num_str}).")

                # Generate individual PPO step heatmaps and grama_vs_layer plots for Known Group case studies
                generate_case_study_plots(sft_step_num_str, "Known Group", known_layer_data, PLOT_OUTPUT_DIR)
                
                # Prepare and plot B/H Calc curves for Known Group
                bh_calc_data_known_step_centric = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                for ppo_step, params_data in known_layer_data.items():
                    if ppo_step in BH_CALC_PPO_STEPS_RANGE:
                        for param_name, layers_data in params_data.items():
                            for layer_id, metrics in layers_data.items():
                                if 'bh_calc' in metrics:
                                    bh_calc_data_known_step_centric[param_name][layer_id][ppo_step].append(metrics['bh_calc'])
                # Average B/H calc if multiple values exist (should not happen with current parsing, but good practice)
                for param_name in bh_calc_data_known_step_centric:
                    for layer_id in bh_calc_data_known_step_centric[param_name]:
                        for ppo_step in bh_calc_data_known_step_centric[param_name][layer_id]:
                            bh_calc_data_known_step_centric[param_name][layer_id][ppo_step] = np.mean(bh_calc_data_known_step_centric[param_name][layer_id][ppo_step])
                
                if bh_calc_data_known_step_centric:
                    plot_bh_calc_curves(sft_step_num_str, "Known Group", bh_calc_data_known_step_centric, PLOT_OUTPUT_DIR)
                else:
                    print(f"      DEBUG: No B/H calc data for Known Group (SFT {sft_step_num_str}).")
            else:
                print(f"      DEBUG: Known Group layer-wise data (SFT {sft_step_num_str}): No data parsed for heatmaps/case studies/B_H calc.")
        else:
            print(f"    No Known Group (Group0) log file found for SFT step {sft_step_num_str}.")
        
        if unknown_group_files:
            avg_unknown_data_step_centric = aggregate_unknown_group_data(unknown_group_files, parse_general_metrics)
            if avg_unknown_data_step_centric: # Check if data exists
                num_steps = len(avg_unknown_data_step_centric)
                num_scores = sum(1 for step_data in avg_unknown_data_step_centric.values() if 'scores' in step_data)
                num_gramas = sum(1 for step_data in avg_unknown_data_step_centric.values() if 'grama_ratios' in step_data)
                print(f"    DEBUG: Avg Unknown Group general metrics (SFT {sft_step_num_str}): PPO steps count: {num_steps}, Scores count: {num_scores}, Grama Ratios count: {num_gramas}")
                all_sfts_performance_data[sft_step_num_str]['unknown_group_avg'] = avg_unknown_data_step_centric
            else:
                print(f"    DEBUG: Avg Unknown Group general metrics (SFT {sft_step_num_str}): No data or no PPO steps parsed.")

        # --- 2. Case Study --- 
        print("  Processing case study data...")
        # Known Group Case Study
        known_layer_data = None
        if known_group_files:
            known_layer_data = parse_layer_wise_metrics(known_group_files[0], CASE_STUDY_PPO_STEPS)
            if known_layer_data:
                print(f"    DEBUG: Known Group layer data (SFT {sft_step_num_str}): Parsed for PPO steps: {sorted(list(known_layer_data.keys()))}")
                for p_step_debug in CASE_STUDY_PPO_STEPS:
                    if p_step_debug in known_layer_data:
                        print(f"      DEBUG: PPO {p_step_debug} (Known): Params found: {list(known_layer_data[p_step_debug].keys())}")
                        # for param_k_debug in known_layer_data[p_step_debug]:
                        #      print(f"        DEBUG: Param {param_k_debug}: Layers found: {list(known_layer_data[p_step_debug][param_k_debug].keys())}")
            else:
                print(f"    DEBUG: Known Group layer data (SFT {sft_step_num_str}): No data parsed.")

        # Unknown Group Average Case Study
        avg_unknown_layer_data = None
        if unknown_group_files:
            group_label_prefix = "Unknown Group Avg" # Define for all unknown group processing
            avg_unknown_layer_data = aggregate_unknown_group_data(unknown_group_files, parse_layer_wise_metrics, CASE_STUDY_PPO_STEPS)
            if avg_unknown_layer_data:
                print(f"    DEBUG: Avg Unknown Group layer data (SFT {sft_step_num_str}): Parsed for PPO steps: {sorted(list(avg_unknown_layer_data.keys()))}")
                # Heatmap for Avg Unknown Group
                heatmap_data_unknown_avg = defaultdict(lambda: defaultdict(lambda: np.nan))
                for ppo_step_cs in CASE_STUDY_PPO_STEPS:
                    if ppo_step_cs in avg_unknown_layer_data:
                        for param_name, layers in avg_unknown_layer_data[ppo_step_cs].items():
                            for layer_id, values in layers.items():
                                if 'grama_ratio' in values:
                                    heatmap_data_unknown_avg[param_name][(ppo_step_cs, layer_id)] = values['grama_ratio']
                plot_heatmaps(sft_step_num_str, group_label_prefix, heatmap_data_unknown_avg, PLOT_OUTPUT_DIR, CASE_STUDY_PPO_STEPS)
                # Generate individual PPO step heatmaps and grama_vs_layer plots for Unknown Group Avg
                generate_case_study_plots(sft_step_num_str, group_label_prefix, avg_unknown_layer_data, PLOT_OUTPUT_DIR)

                # Prepare and plot B/H Calc curves for Unknown Group Avg
                bh_calc_data_unknown_avg_step_centric = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                if avg_unknown_layer_data: # Ensure there is data to process
                    for ppo_step, params_data in avg_unknown_layer_data.items():
                        if ppo_step in BH_CALC_PPO_STEPS_RANGE:
                            for param_name, layers_data in params_data.items():
                                for layer_id, metrics in layers_data.items():
                                    if 'bh_calc' in metrics:
                                        # avg_unknown_layer_data already contains averaged values, so just assign
                                        bh_calc_data_unknown_avg_step_centric[param_name][layer_id][ppo_step] = metrics['bh_calc'] 
                
                if bh_calc_data_unknown_avg_step_centric:
                    print(f"      DEBUG: For {group_label_prefix} (SFT {sft_step_num_str}), B/H Calc data points per param before plotting:")
                    for param_name_dbg, layer_data_dbg in bh_calc_data_unknown_avg_step_centric.items():
                        # Count PPO steps that have data for this param (across any layer)
                        ppo_steps_with_data_count = set()
                        for layer_id_dbg, step_data_dbg in layer_data_dbg.items():
                            ppo_steps_with_data_count.update(step_data_dbg.keys()) # .keys() are PPO steps
                        print(f"        Param {param_name_dbg}: {len(ppo_steps_with_data_count)} PPO steps with B/H data (out of potential {len(BH_CALC_PPO_STEPS_RANGE)}). Example steps: {sorted(list(ppo_steps_with_data_count))[:5]}...")
                    
                    plot_bh_calc_curves(sft_step_num_str, group_label_prefix, bh_calc_data_unknown_avg_step_centric, PLOT_OUTPUT_DIR)
                else:
                    print(f"      DEBUG: No B/H calc data for {group_label_prefix} (SFT {sft_step_num_str}).")


    # --- Plot SFT Comparison Curves ---
    if all_sfts_performance_data:
        print("\nPlotting SFT comparison curves...")
        plot_sft_comparison_curves(all_sfts_performance_data, 'known_group', 'scores', 
                                   'Known Group Scores vs PPO Step (Across SFTs)', 
                                   'Score (critic/score/mean)', PLOT_OUTPUT_DIR)
        plot_sft_comparison_curves(all_sfts_performance_data, 'known_group', 'grama_ratios', 
                                   'Known Group Grama Ratios vs PPO Step (Across SFTs)', 
                                   'Grama Ratio (actor/zero_gradspace_ratio)', PLOT_OUTPUT_DIR)
        plot_sft_comparison_curves(all_sfts_performance_data, 'unknown_group_avg', 'scores', 
                                   'Avg Unknown Group Scores vs PPO Step (Across SFTs)', 
                                   'Score (critic/score/mean)', PLOT_OUTPUT_DIR)
        plot_sft_comparison_curves(all_sfts_performance_data, 'unknown_group_avg', 'grama_ratios', 
                                   'Avg Unknown Group Grama Ratios vs PPO Step (Across SFTs)', 
                                   'Grama Ratio (actor/zero_gradspace_ratio)', PLOT_OUTPUT_DIR)
    else:
        print("\nNo data collected across SFT steps to plot comparison curves.")

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()
