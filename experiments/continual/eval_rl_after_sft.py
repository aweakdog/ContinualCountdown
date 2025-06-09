import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
EXPER_PATTERN = re.compile(r'continual_countdown3b_sft_global_step_(\d+)_\d+')
GROUPS = ['Group0', 'Group1', 'Group2', 'Group3']

METRICS = [
    ('critic/score/mean', r'critic/score/mean:([\d.eE+-]+)'),
    ('zero_gradspace_ratio', r'zero_gradspace_ratio:([\d.eE+-]+)'),
    ('actor/zero_gradspace_ratio', r'actor/zero_gradspace_ratio:([\d.eE+-]+)'),
]

def smooth_data(values, window_size=5):
    """Smooths data using a rolling mean."""
    if window_size <= 1 or len(values) < window_size:
        return values
    return pd.Series(values).rolling(window=window_size, center=True, min_periods=1).mean().tolist()

# Helper to extract float value from log line

def extract_metric(line, metric_regex):
    match = re.search(metric_regex, line)
    if match:
        return float(match.group(1))
    return None

def parse_log_file(filepath, metrics_regex):
    metric_data = {k: [] for k, _ in metrics_regex}
    steps = []
    with open(filepath, 'r') as f:
        for line in f:
            # Try to extract step if present
            step_match = re.search(r'step:(\d+)', line)
            step = int(step_match.group(1)) if step_match else None
            for metric, regex in metrics_regex:
                val = extract_metric(line, regex)
                if val is not None:
                    metric_data[metric].append(val)
            if step is not None:
                steps.append(step)
    # Align steps and metrics if possible
    for k in metric_data:
        if len(steps) == len(metric_data[k]):
            metric_data[k] = list(zip(steps, metric_data[k]))
        else:
            metric_data[k] = list(enumerate(metric_data[k]))
    return metric_data

def get_experiment_folders(logs_dir):
    folders = []
    for entry in os.listdir(logs_dir):
        path = os.path.join(logs_dir, entry)
        if os.path.isdir(path) and EXPER_PATTERN.match(entry):
            folders.append((entry, path))
    return sorted(folders, key=lambda x: int(EXPER_PATTERN.match(x[0]).group(1)))

def main():
    experiment_folders = get_experiment_folders(LOGS_DIR)
    print(f"Found {len(experiment_folders)} experiment folders.")

    # Data structure: {global_step: {group: {metric: [(step, value), ...]}}}
    all_data = {}
    for folder_name, folder_path in experiment_folders:
        global_step = int(EXPER_PATTERN.match(folder_name).group(1))
        all_data[global_step] = {}
        for group in GROUPS:
            group_files = glob.glob(os.path.join(folder_path, f"{group}_*.log"))
            if not group_files:
                continue
            # Use the latest file if multiple
            group_file = sorted(group_files)[-1]
            metric_data = parse_log_file(group_file, METRICS)
            all_data[global_step][group] = metric_data

    # --- Plotting ---
    output_dir = os.path.join('plots', 'rl_after_sft')
    os.makedirs(output_dir, exist_ok=True)

    # a. For metrics critic/score/mean, plot the group0's results as known in a figure, each line is a global_step
    plt.figure(figsize=(10,6))
    for global_step in sorted(all_data.keys()):
        group0 = all_data[global_step].get('Group0')
        if group0 and 'critic/score/mean' in group0:
            steps, values = zip(*group0['critic/score/mean'])
            smoothed_values = smooth_data(list(values))
            plt.plot(steps, smoothed_values, label=f'global_step_{global_step}')
    plt.title('Group0 (Known) - critic/score/mean')
    plt.xlabel('Step')
    plt.ylabel('critic/score/mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'known_group0_critic_score_mean.png'))
    plt.close()

    # b. For metrics critic/score/mean, plot the average of [group1,group2,group3] as unknown
    plt.figure(figsize=(10,6))
    for global_step in sorted(all_data.keys()):
        group_vals = []
        for group in ['Group1','Group2','Group3']:
            g = all_data[global_step].get(group)
            if g and 'critic/score/mean' in g:
                steps, values = zip(*g['critic/score/mean'])
                group_vals.append(values)
        if group_vals:
            # Pad to min length
            min_len = min(map(len, group_vals))
            vals = np.mean([v[:min_len] for v in group_vals], axis=0)
            steps = range(min_len)
            smoothed_vals = smooth_data(list(vals))
            plt.plot(steps, smoothed_vals, label=f'global_step_{global_step}')
    plt.title('Unknown Groups (avg 1-3) - critic/score/mean')
    plt.xlabel('Step')
    plt.ylabel('critic/score/mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unknown_group123_critic_score_mean.png'))
    plt.close()

    # c. For metrics zero_gradspace_ratio, plot the group0's results as known in a figure, each line is a global_step
    plt.figure(figsize=(10,6))
    for global_step in sorted(all_data.keys()):
        group0 = all_data[global_step].get('Group0')
        values = None
        if group0:
            if 'zero_gradspace_ratio' in group0 and group0['zero_gradspace_ratio']:
                steps, values = zip(*group0['zero_gradspace_ratio'])
            elif 'actor/zero_gradspace_ratio' in group0 and group0['actor/zero_gradspace_ratio']:
                steps, values = zip(*group0['actor/zero_gradspace_ratio'])
        if values:
            smoothed_values = smooth_data(list(values))
            plt.plot(steps, smoothed_values, label=f'global_step_{global_step}')
    plt.title('Group0 (Known) - zero_gradspace_ratio')
    plt.xlabel('Step')
    plt.ylabel('zero_gradspace_ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'known_group0_zero_gradspace_ratio.png'))
    plt.close()

    # d. Duplicate for second plot if needed (can be customized as needed)
    plt.figure(figsize=(10,6))
    for global_step in sorted(all_data.keys()):
        group_vals = []
        for group in ['Group1','Group2','Group3']:
            g = all_data[global_step].get(group)
            values = None
            if g:
                if 'zero_gradspace_ratio' in g and g['zero_gradspace_ratio']:
                    _, values = zip(*g['zero_gradspace_ratio'])
                elif 'actor/zero_gradspace_ratio' in g and g['actor/zero_gradspace_ratio']:
                    _, values = zip(*g['actor/zero_gradspace_ratio'])
            if values:
                group_vals.append(values)
        if group_vals:
            min_len = min(map(len, group_vals))
            vals = np.mean([v[:min_len] for v in group_vals], axis=0)
            steps = range(min_len)
            smoothed_vals = smooth_data(list(vals))
            plt.plot(steps, smoothed_vals, label=f'global_step_{global_step}')
    plt.title('Unknown Groups (avg 1-3) - zero_gradspace_ratio')
    plt.xlabel('Step')
    plt.ylabel('zero_gradspace_ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unknown_group123_zero_gradspace_ratio_2.png'))
    plt.close()

    print(f'Plots saved to {output_dir}')

if __name__ == '__main__':
    main()
