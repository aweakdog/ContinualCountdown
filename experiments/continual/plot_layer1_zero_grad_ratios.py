import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Specific layer and weights to track
TARGET_LAYER_PREFIX = "model.layers.1."
TARGET_WEIGHT_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    # "self_attn.k_proj.bias", # Removed
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    # "self_attn.q_proj.bias", # Removed
    "self_attn.q_proj.weight",
    # "self_attn.v_proj.bias", # Removed
    "self_attn.v_proj.weight"
]

# Regex to strip ANSI escape codes
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Regex to capture param name and zero-grad ratio from lines like:
# "(WorkerDict pid=1025363) [ZeroGradV2] model.layers.0.self_attn.k_proj.weight: 83/256 (0.3242)"
# or "[ZeroGradV2] model.layers.0.self_attn.k_proj.weight: 83/256 (0.3242)" (if prefix is missing)
# It also handles the "Global: ..." lines by not matching them for param_name capture if they don't have a ':' before ratio
ZERO_GRAD_LINE_PATTERN = re.compile(r"(?:\(WorkerDict pid=\d+\)\s*)?\[ZeroGradV2\]\s*([a-zA-Z0-9_\.]+?):\s*\d+/\d+\s*\(([\d.eE+-]+)\)")

# Regex to capture step from lines like "INFO worker.py:1575 -- update_policy() step:0 ..." or just "step:0"
STEP_PATTERN = re.compile(r"step:(\d+)")

def smooth_data(values, window_size=5):
    """Smooths data using a rolling mean."""
    if window_size <= 1 or len(values) < window_size:
        return values
    return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean().tolist()

def parse_log_file_for_layer_weights(filepath):
    # Data: {weight_suffix: [(step, ratio), ...]}
    layer_weight_data = {suffix: [] for suffix in TARGET_WEIGHT_SUFFIXES}
    current_step = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_with_ansi in f:
                line = ANSI_ESCAPE_PATTERN.sub('', line_with_ansi) # Strip ANSI codes

                step_match = STEP_PATTERN.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                
                if current_step is None: # Only process if we have a current step context
                    continue

                # Check for lines starting with the specific prefix or just [ZeroGradV2]
                # This helps to avoid matching other random lines that might have similar patterns
                if not (line.strip().startswith("(WorkerDict pid=") or line.strip().startswith("[ZeroGradV2]")):
                    if not "[ZeroGradV2]" in line: # double check if it's a relevant line at all
                        continue

                zero_grad_match = ZERO_GRAD_LINE_PATTERN.search(line)
                if zero_grad_match:
                    param_name = zero_grad_match.group(1).strip()
                    ratio_str = zero_grad_match.group(2).strip()
                    try:
                        ratio = float(ratio_str)
                    except ValueError:
                        print(f"Warning: Could not parse ratio '{ratio_str}' for param '{param_name}' in line: {line.strip()}")
                        continue
                    
                    if param_name.startswith(TARGET_LAYER_PREFIX):
                        suffix = param_name[len(TARGET_LAYER_PREFIX):]
                        if suffix in TARGET_WEIGHT_SUFFIXES:
                            layer_weight_data[suffix].append((current_step, ratio))
    except Exception as e:
        print(f"Error parsing file {filepath}: {e}")
    
    # Deduplicate based on step, keeping the last seen for that step
    for suffix in layer_weight_data:
        if layer_weight_data[suffix]:
            # Sort by step, then use a dict to overwrite duplicates, keeping last
            sorted_points = sorted(layer_weight_data[suffix], key=lambda x: x[0])
            deduped_points = list(dict(sorted_points).items())
            layer_weight_data[suffix] = deduped_points
            
    return layer_weight_data

def main():
    parser = argparse.ArgumentParser(description='Plot layer 1 zero-gradient ratios from a single log file.')
    parser.add_argument('log_file', type=str, help='Path to the log file to process.')
    args = parser.parse_args()

    log_file_path = args.log_file

    if not os.path.isfile(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    print(f"Processing log file: {log_file_path}...")
    
    experiment_data = parse_log_file_for_layer_weights(log_file_path)

    output_dir = os.path.join('plots', 'layer1_zero_grad_ratios_single_file')
    os.makedirs(output_dir, exist_ok=True)

    # Derive a label/name from the log file
    log_file_basename = os.path.splitext(os.path.basename(log_file_path))[0]
    experiment_label = re.sub(r'[^a-zA-Z0-9_-]', '-', log_file_basename) # Sanitize for filename

    plt.figure(figsize=(18, 10))
    has_data_to_plot = False
    for weight_suffix in TARGET_WEIGHT_SUFFIXES: # Plot in defined order
        data_points = experiment_data.get(weight_suffix, [])
        if data_points:
            has_data_to_plot = True
            steps, ratios = zip(*data_points)
            if not ratios: # Should not happen if data_points is not empty
                continue
            smoothed_ratios = smooth_data(list(ratios))
            plt.plot(steps, smoothed_ratios, label=weight_suffix, marker='.', linestyle='-')
    
    if not has_data_to_plot:
        print(f"No data to plot from log file {log_file_path}. Skipping plot generation.")
        plt.close()
        return

    plt.title(f'Layer 1 Zero-Gradient Ratios - Log: {os.path.basename(log_file_path)}')
    plt.xlabel('Step')
    plt.ylabel('Zero-Gradient Ratio (Smoothed)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plot_filename = os.path.join(output_dir, f'layer1_zero_grad_{experiment_label}.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")
    print(f'Plot saved to {output_dir}')

if __name__ == '__main__':
    main()
