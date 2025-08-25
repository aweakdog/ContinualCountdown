#!/usr/bin/env python3
"""
Plot Qwen DeepMath performance comparison across different global steps.
Compares critic/score/mean and fisher/C_K_normalized metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import glob
from pathlib import Path
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QwenDeepMathPlotter:
    def __init__(self, base_dir="/cpfs04/user/liyuanhang.p/src/ContinualCountdown/qwen_logs", 
                 output_dir="./plots/qwen_deepmath"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the directories for different global steps
        self.step_dirs = {
            0: "debug_countdown3b_qwen_sft_global_step_0_reset_k0f_k20l",
            5: "develop_countdown3b_qwen_sft_global_step_5_reset_k20f_k0l", 
            15: "train_countdown3b_qwen_sft_global_step_15_reset_k0f_k0l",
            # 15: "train_countdown3b_qwen_sft_global_step_15_reset_k0f_k0l"  # Temporarily disabled
        }
    
    def parse_log_line(self, line):
        """Parse a single log line to extract metrics"""
        metrics = {}
        
        # Remove ANSI color codes
        line = re.sub(r'\^?\[\[\d+m', '', line)
        
        # Extract step number
        step_match = re.search(r'step:(\d+)', line)
        if step_match:
            metrics['step'] = int(step_match.group(1))
        
        # Extract critic/score/mean
        score_match = re.search(r'critic/score/mean:([\d.-]+)', line)
        if score_match:
            metrics['critic_score_mean'] = float(score_match.group(1))
        
        # Extract fisher/C_K_normalized
        fisher_match = re.search(r'fisher/C_K_normalized:([\d.-]+)', line)
        if fisher_match:
            metrics['fisher_C_K_normalized'] = float(fisher_match.group(1))
        
        # Extract fisher/c_k_normalized (lowercase)
        fisher_lowercase_match = re.search(r'fisher/c_k_normalized:([\d.-]+)', line)
        if fisher_lowercase_match:
            metrics['fisher_c_k_normalized'] = float(fisher_lowercase_match.group(1))
        
        return metrics
    
    def parse_log_file(self, log_file):
        """Parse entire log file and extract metrics"""
        data = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'step:' in line and ('critic/score/mean:' in line or 'fisher/C_K_normalized:' in line or 'fisher/c_k_normalized:' in line):
                        metrics = self.parse_log_line(line.strip())
                        if metrics:
                            data.append(metrics)
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
        
        return data
    
    def collect_data_for_global_step(self, global_step):
        """Collect all data for a specific global step"""
        step_dir = self.base_dir / self.step_dirs[global_step]
        if not step_dir.exists():
            print(f"Directory not found: {step_dir}")
            return []
        
        # Find all log files matching the pattern
        log_pattern = f"Phase*_Group*_Iter*_SFT_global_step_{global_step}_*.log"
        log_files = list(step_dir.glob(log_pattern))
        
        all_data = []
        for log_file in log_files:
            data = self.parse_log_file(log_file)
            for entry in data:
                entry['global_step'] = global_step
                entry['log_file'] = log_file.name
            all_data.extend(data)
        
        print(f"Global step {global_step}: Found {len(log_files)} log files, {len(all_data)} data points")
        return all_data
    
    def plot_critic_score_comparison(self, smooth=True, sigma=1.5):
        """Plot critic/score/mean comparison across global steps with optional smoothing"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for critic score comparison")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df = df.dropna(subset=['critic_score_mean'])
        
        if df.empty:
            print("No critic score data found")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot for each global step
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].sort_values('step')
            if not step_data.empty:
                x_data = step_data['step'].values
                y_data = step_data['critic_score_mean'].values
                
                if smooth and len(y_data) > 3:
                    # Apply Gaussian smoothing
                    y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                    
                    # Calculate confidence band using rolling standard deviation
                    window_size = min(5, len(y_data) // 3)
                    if window_size >= 2:
                        df_temp = pd.DataFrame({'x': x_data, 'y': y_data})
                        rolling_std = df_temp['y'].rolling(window=window_size, center=True, min_periods=1).std()
                        y_upper = y_smooth + rolling_std
                        y_lower = y_smooth - rolling_std
                        
                        # Plot confidence band as shadow
                        ax.fill_between(x_data, y_lower, y_upper, color=color, alpha=0.2)
                    
                    # Plot smoothed line
                    ax.plot(x_data, y_smooth, label=f'Global Step {global_step}', 
                           color=color, linewidth=2.5, alpha=0.9)
                else:
                    # Plot without smoothing
                    ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                           color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Critic Score Mean', fontsize=12)
        title = 'Qwen DeepMath Performance: Critic Score Mean Comparison'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_critic_score_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Critic score plot saved to: {output_path}")
    
    def plot_fisher_ck_comparison(self, smooth=True, sigma=1.5):
        """Plot fisher/C_K_normalized comparison across global steps with smoothing"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for Fisher C_K comparison")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df = df.dropna(subset=['fisher_C_K_normalized'])
        
        if df.empty:
            print("No Fisher C_K data found")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot for each global step
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].sort_values('step')
            if not step_data.empty:
                x_data = step_data['step'].values
                y_data = step_data['fisher_C_K_normalized'].values
                
                if smooth and len(y_data) > 3:
                    # Apply Gaussian smoothing
                    y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                    # Plot smoothed line only
                    ax.plot(x_data, y_smooth, label=f'Global Step {global_step}', 
                           color=color, linewidth=2.5, alpha=0.9)
                else:
                    # Plot without smoothing
                    ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                           color=color, linewidth=2, marker='s', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Fisher C_K Normalized', fontsize=12)
        title = 'Qwen DeepMath Performance: Fisher C_K Normalized Comparison'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_fisher_ck_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fisher C_K plot saved to: {output_path}")
    
    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """Remove outliers from data using IQR or Z-score method"""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return z_scores < threshold
        else:
            return np.ones(len(data), dtype=bool)

    def plot_fisher_ck_lowercase_comparison(self, smooth=True, sigma=1.5, remove_outliers=True):
        """Plot fisher/c_k_normalized (lowercase) comparison across global steps with outlier removal and smoothing"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for Fisher c_k (lowercase) comparison")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df = df.dropna(subset=['fisher_c_k_normalized'])
        
        if df.empty:
            print("No Fisher c_k (lowercase) data found")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot for each global step
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].sort_values('step')
            if not step_data.empty:
                x_data = step_data['step'].values
                y_data = step_data['fisher_c_k_normalized'].values
                
                # Remove outliers if requested
                if remove_outliers and len(y_data) > 5:
                    outlier_mask = self.remove_outliers(y_data, method='iqr', threshold=2.0)
                    x_data_clean = x_data[outlier_mask]
                    y_data_clean = y_data[outlier_mask]
                    
                    # Outliers are removed but not displayed
                    
                    x_data, y_data = x_data_clean, y_data_clean
                
                if len(y_data) == 0:
                    continue
                
                if smooth and len(y_data) > 3:
                    # Apply Gaussian smoothing
                    y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                    # Plot smoothed line only (no scatter points)
                    ax.plot(x_data, y_smooth, label=f'Global Step {global_step}', 
                           color=color, linewidth=2.5, alpha=0.9)
                else:
                    # Plot without smoothing
                    ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                           color=color, linewidth=2, marker='^', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Fisher c_k Normalized (lowercase)', fontsize=12)
        title = 'Qwen DeepMath Performance: Fisher c_k Normalized (lowercase) Comparison'
        #if remove_outliers:
        #    title += ' (Outliers Removed)'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_fisher_ck_lowercase_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fisher c_k (lowercase) plot saved to: {output_path}")
    
    def calculate_computed_ck(self, data, step_threshold=150):
        """Calculate computed C_K from c_k values with step threshold logic"""
        if len(data) == 0:
            return []
        
        # Sort data by step
        data_sorted = sorted(data, key=lambda x: x['step'])
        computed_data = []
        
        for i, entry in enumerate(data_sorted):
            current_step = entry['step']
            
            if current_step < step_threshold:
                # Before step 150: average of all previous c_k values (including current)
                relevant_data = data_sorted[:i+1]
                ck_values = [d['fisher_c_k_normalized'] for d in relevant_data if 'fisher_c_k_normalized' in d]
            else:
                # After step 150: average from step 150 to current
                relevant_data = [d for d in data_sorted[:i+1] if d['step'] >= step_threshold]
                ck_values = [d['fisher_c_k_normalized'] for d in relevant_data if 'fisher_c_k_normalized' in d]
            
            if ck_values:
                computed_ck = np.mean(ck_values)
                computed_entry = entry.copy()
                computed_entry['computed_C_K'] = computed_ck
                computed_data.append(computed_entry)
        
        return computed_data
    
    def plot_computed_ck_comparison(self, smooth=True, sigma=1.5, remove_outliers=True):
        """Plot computed C_K comparison across global steps using c_k averages"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for computed C_K comparison")
            return
        
        # Convert to DataFrame and filter for c_k data
        df = pd.DataFrame(all_data)
        df = df.dropna(subset=['fisher_c_k_normalized'])
        
        if df.empty:
            print("No fisher c_k data found for computed C_K calculation")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot for each global step
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].sort_values('step')
            if not step_data.empty:
                # Convert to list of dictionaries for processing
                step_data_list = step_data.to_dict('records')
                
                # Remove outliers if requested
                if remove_outliers and len(step_data_list) > 5:
                    y_values = np.array([d['fisher_c_k_normalized'] for d in step_data_list])
                    outlier_mask = self.remove_outliers(y_values, method='iqr', threshold=2.0)
                    step_data_clean = [d for j, d in enumerate(step_data_list) if outlier_mask[j]]
                else:
                    step_data_clean = step_data_list
                
                if len(step_data_clean) == 0:
                    continue
                
                # Calculate computed C_K values
                computed_data = self.calculate_computed_ck(step_data_clean)
                
                if len(computed_data) == 0:
                    continue
                
                x_data = np.array([d['step'] for d in computed_data])
                y_data = np.array([d['computed_C_K'] for d in computed_data])
                
                if smooth and len(y_data) > 3:
                    # Apply Gaussian smoothing
                    y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                    # Plot smoothed line
                    ax.plot(x_data, y_smooth, label=f'Global Step {global_step}', 
                           color=color, linewidth=2.5, alpha=0.9)
                else:
                    # Plot without smoothing
                    ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                           color=color, linewidth=2, marker='D', markersize=4, alpha=0.8)
        
        # Add vertical line at step 150 to show threshold
        ax.axvline(x=150, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(152, ax.get_ylim()[1] * 0.9, 'Step 150\n(Switch to deepmath)', fontsize=10, alpha=0.7)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Computed C_K (from c_k averages)', fontsize=12)
        title = 'Qwen DeepMath Performance: Computed C_K from c_k Averages'
        #if remove_outliers:
            #title += ' (Outliers Removed)'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_computed_ck_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Computed C_K plot saved to: {output_path}")
    
    def calculate_sliding_window_ck(self, data, window_size=20, task_boundary=150):
        """Calculate sliding window C_K from c_k values with maximum window size and task boundary"""
        if len(data) == 0:
            return []
        
        # Sort data by step
        data_sorted = sorted(data, key=lambda x: x['step'])
        sliding_window_data = []
        
        for i, entry in enumerate(data_sorted):
            current_step = entry['step']
            
            # Get the window of data points (up to window_size most recent)
            start_idx = max(0, i + 1 - window_size)
            
            # Apply task boundary constraint
            if current_step < task_boundary:
                # Before step 150 (countdown task): only use data from countdown task
                window_data = [d for d in data_sorted[start_idx:i+1] if d['step'] < task_boundary]
            else:
                # After step 150 (deepmath task): only use data from deepmath task (step >= 150)
                window_data = [d for d in data_sorted[start_idx:i+1] if d['step'] >= task_boundary]
            
            # Extract c_k values from the window
            ck_values = [d['fisher_c_k_normalized'] for d in window_data if 'fisher_c_k_normalized' in d]
            
            if ck_values:
                sliding_window_ck = np.mean(ck_values)
                sliding_entry = entry.copy()
                sliding_entry['sliding_window_C_K'] = sliding_window_ck
                sliding_entry['window_size_used'] = len(ck_values)
                sliding_window_data.append(sliding_entry)
        
        return sliding_window_data
    
    def plot_sliding_window_ck_comparison(self, smooth=True, sigma=1.5, remove_outliers=True, window_size=20, task_boundary=150):
        """Plot sliding window C_K comparison across global steps using c_k sliding averages"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for sliding window C_K comparison")
            return
        
        # Create DataFrame and remove outliers if requested
        df = pd.DataFrame(all_data)
        
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].copy()
            
            if remove_outliers and len(step_data) > 0:
                outlier_mask = self.remove_outliers(step_data['fisher_c_k_normalized'].values, method='iqr', threshold=2.0)
                step_data_clean = step_data[outlier_mask].reset_index(drop=True)
            else:
                step_data_clean = step_data
            
            if len(step_data_clean) == 0:
                continue
            
            # Calculate sliding window C_K with task boundary
            sliding_data = self.calculate_sliding_window_ck(step_data_clean.to_dict('records'), window_size, task_boundary)
            
            if not sliding_data:
                continue
            
            # Convert to arrays for plotting
            x_data = np.array([d['step'] for d in sliding_data])
            y_data = np.array([d['sliding_window_C_K'] for d in sliding_data])
            
            if smooth and len(y_data) > 1:
                y_smooth = gaussian_filter1d(y_data, sigma=sigma)
            else:
                y_smooth = y_data
            
            # Plot the line
            ax.plot(x_data, y_smooth, label=f'Global Step {global_step}', 
                   color=color, linewidth=2.5, alpha=0.9, marker='o', markersize=3)
        
        # Add vertical line at task boundary
        ax.axvline(x=task_boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(task_boundary + 2, ax.get_ylim()[1] * 0.9, f'Step {task_boundary}\n(Swtich to deepmath)', fontsize=10, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel(f'Sliding Window C_K (window≤{window_size})', fontsize=12)
        
        title = f"Sliding Window C_K Comparison (Max Window Size {window_size}, Task Boundary {task_boundary})"
        if smooth:
            title += " (Smoothed)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"qwen_sliding_window_ck_comparison_w{window_size}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sliding window C_K plot saved to: {output_path}")
    
    def plot_raw_sliding_window_ck_comparison(self, window_size=20, task_boundary=150):
        """Plot raw sliding window C_K comparison without outlier removal or smoothing"""
        print("Creating raw sliding window C_K comparison plot...")
        
        # Parse data for all global steps
        all_data = []
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No raw sliding window C_K data found")
            return
        
        # Create DataFrame without outlier removal
        df = pd.DataFrame(all_data)
        
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].copy()
            
            if len(step_data) == 0:
                continue
            
            # Calculate sliding window C_K with task boundary (no outlier removal)
            sliding_data = self.calculate_sliding_window_ck(step_data.to_dict('records'), window_size, task_boundary)
            
            if not sliding_data:
                continue
            
            # Convert to arrays for plotting (no smoothing)
            x_data = np.array([d['step'] for d in sliding_data])
            y_data = np.array([d['sliding_window_C_K'] for d in sliding_data])
            
            # Plot the raw line
            ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                   color=color, linewidth=2.5, alpha=0.9, marker='o', markersize=3)
        
        # Add vertical line at task boundary
        ax.axvline(x=task_boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(task_boundary + 2, ax.get_ylim()[1] * 0.9, f'Step {task_boundary}\n(Swtich to deepmath)', fontsize=10, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel(f'Raw Sliding Window C_K (window≤{window_size})', fontsize=12)
        
        title = f"Raw Sliding Window C_K Comparison (Max Window Size {window_size}, Task Boundary {task_boundary})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"qwen_raw_sliding_window_ck_comparison_w{window_size}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Raw sliding window C_K plot saved to: {output_path}")
    
    def calculate_sliding_window_ck_log(self, data, window_size=20, task_boundary=150):
        """Calculate sliding window C_K from log-transformed c_k values with maximum window size and task boundary"""
        if len(data) == 0:
            return []
        
        # Sort data by step
        data_sorted = sorted(data, key=lambda x: x['step'])
        sliding_window_data = []
        
        for i, entry in enumerate(data_sorted):
            current_step = entry['step']
            
            # Get the window of data points (up to window_size most recent)
            start_idx = max(0, i + 1 - window_size)
            
            # Apply task boundary constraint
            if current_step < task_boundary:
                # Before step 150 (countdown task): only use data from countdown task
                window_data = [d for d in data_sorted[start_idx:i+1] if d['step'] < task_boundary]
            else:
                # After step 150 (deepmath task): only use data from deepmath task (step >= 150)
                window_data = [d for d in data_sorted[start_idx:i+1] if d['step'] >= task_boundary]
            
            # Extract c_k values from the window (no log transformation yet)
            ck_values = []
            for d in window_data:
                if 'fisher_c_k_normalized' in d:
                    ck_val = d['fisher_c_k_normalized']
                    ck_values.append(ck_val)
            
            if ck_values:
                # Calculate C_K: sum then average c_k values to get final C_K
                ck_sum = np.sum(ck_values)
                ck_average = ck_sum / len(ck_values)  # This is the final C_K
                # Apply log transformation to the final C_K to smooth extreme values
                # Add small epsilon to avoid log(0)
                if ck_average > 0:
                    sliding_window_ck = np.log(ck_average + 1e-8)
                else:
                    # Handle negative or zero C_K
                    sliding_window_ck = np.log(abs(ck_average) + 1e-8)
            
                sliding_entry = entry.copy()
                sliding_entry['sliding_window_C_K_log'] = sliding_window_ck
                sliding_entry['window_size_used'] = len(ck_values)
                sliding_window_data.append(sliding_entry)
        
        return sliding_window_data
    
    def plot_log_sliding_window_ck_comparison(self, window_size=20, task_boundary=150):
        """Plot log-transformed sliding window C_K comparison to smooth extreme values"""
        print(f"Creating log-transformed sliding window C_K comparison plot (window size {window_size})...")
        
        # Parse data for all global steps
        all_data = []
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No log sliding window C_K data found")
            return
        
        # Create DataFrame without outlier removal
        df = pd.DataFrame(all_data)
        
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.step_dirs)))
        
        for i, (global_step, color) in enumerate(zip(sorted(self.step_dirs.keys()), colors)):
            step_data = df[df['global_step'] == global_step].copy()
            
            if len(step_data) == 0:
                continue
            
            # Calculate log-transformed sliding window C_K with task boundary
            sliding_data = self.calculate_sliding_window_ck_log(step_data.to_dict('records'), window_size, task_boundary)
            
            if not sliding_data:
                continue
            
            # Convert to arrays for plotting (no smoothing)
            x_data = np.array([d['step'] for d in sliding_data])
            y_data = np.array([d['sliding_window_C_K_log'] for d in sliding_data])
            
            # Plot the log-transformed line
            ax.plot(x_data, y_data, label=f'Global Step {global_step}', 
                   color=color, linewidth=2.5, alpha=0.9, marker='o', markersize=3)
        
        # Add vertical line at task boundary
        ax.axvline(x=task_boundary, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(task_boundary + 2, ax.get_ylim()[1] * 0.9, f'Step {task_boundary}\n(Swtich to deepmath)', fontsize=10, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel(f'Log Sliding Window C_K (window≤{window_size})', fontsize=12)
        
        title = f"Log-Transformed Sliding Window C_K Comparison (Max Window Size {window_size}, Task Boundary {task_boundary})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f"qwen_log_sliding_window_ck_comparison_w{window_size}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Log sliding window C_K plot saved to: {output_path}")
    
    def plot_control_group_critic_score(self, smooth=True, sigma=1.5):
        """Plot critic score for control group from train_deepmath_qwen3b_sft_global_step_0 with smoothing"""
        control_dir = self.base_dir / "train_deepmath_qwen3b_sft_global_step_0"
        
        if not control_dir.exists():
            print(f"Control group directory not found: {control_dir}")
            return
        
        # Find all log files in control group directory
        log_files = list(control_dir.glob("*.log"))
        
        if not log_files:
            print(f"No log files found in control group directory: {control_dir}")
            return
        
        all_data = []
        for log_file in log_files:
            data = self.parse_log_file(log_file)
            for entry in data:
                entry['log_file'] = log_file.name
            all_data.extend(data)
        
        if not all_data:
            print("No data found for control group")
            return
        
        # Convert to DataFrame and filter for critic score data
        df = pd.DataFrame(all_data)
        df = df.dropna(subset=['critic_score_mean'])
        
        if df.empty:
            print("No critic score data found in control group")
            return
        
        # Sort data by step
        df = df.sort_values('step')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot control group data with smoothing
        x_data = df['step'].values
        y_data = df['critic_score_mean'].values
        color = 'red'
        
        if smooth and len(y_data) > 3:
            # Apply Gaussian smoothing
            y_smooth = gaussian_filter1d(y_data, sigma=sigma)
            
            # Calculate confidence band using rolling standard deviation
            window_size = min(5, len(y_data) // 3)
            if window_size >= 2:
                df_temp = pd.DataFrame({'x': x_data, 'y': y_data})
                rolling_std = df_temp['y'].rolling(window=window_size, center=True, min_periods=1).std()
                y_upper = y_smooth + rolling_std
                y_lower = y_smooth - rolling_std
                
                # Plot confidence band as shadow
                ax.fill_between(x_data, y_lower, y_upper, color=color, alpha=0.2)
            
            # Plot smoothed line
            ax.plot(x_data, y_smooth, label='Control Group (DeepMath Global Step 0)', 
                   color=color, linewidth=2.5, alpha=0.9)
        else:
            # Plot without smoothing
            ax.plot(x_data, y_data, label='Control Group (DeepMath Global Step 0)', 
                   color=color, linewidth=2, marker='o', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Critic Score Mean', fontsize=12)
        title = 'Qwen DeepMath Control Group: Critic Score Performance'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_control_group_critic_score.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Control group critic score plot saved to: {output_path}")
        print(f"Found {len(log_files)} log files, {len(all_data)} data points")

    def plot_summary_statistics(self):
        """Plot summary statistics for both metrics"""
        all_data = []
        
        # Collect data from all global steps
        for global_step in self.step_dirs.keys():
            data = self.collect_data_for_global_step(global_step)
            all_data.extend(data)
        
        if not all_data:
            print("No data found for summary statistics")
            return
        
        df = pd.DataFrame(all_data)
        
        # Create summary statistics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Critic score summary
        if 'critic_score_mean' in df.columns:
            critic_summary = df.groupby('global_step')['critic_score_mean'].agg(['mean', 'std', 'count']).reset_index()
            
            ax1.bar(critic_summary['global_step'], critic_summary['mean'], 
                   yerr=critic_summary['std'], capsize=5, alpha=0.7)
            ax1.set_xlabel('Global Step')
            ax1.set_ylabel('Critic Score Mean (Average)')
            ax1.set_title('Average Critic Score by Global Step')
            ax1.grid(True, alpha=0.3)
        
        # Fisher C_K summary (uppercase)
        if 'fisher_C_K_normalized' in df.columns:
            fisher_summary = df.groupby('global_step')['fisher_C_K_normalized'].agg(['mean', 'std', 'count']).reset_index()
            
            ax2.bar(fisher_summary['global_step'], fisher_summary['mean'], 
                   yerr=fisher_summary['std'], capsize=5, alpha=0.7, color='orange')
            ax2.set_xlabel('Global Step')
            ax2.set_ylabel('Fisher C_K Normalized (Average)')
            ax2.set_title('Average Fisher C_K (uppercase) by Global Step')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "qwen_summary_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Summary statistics plot saved to: {output_path}")

def main():
    """Main function to generate all plots"""
    plotter = QwenDeepMathPlotter()
    
    print("Generating Qwen DeepMath performance comparison plots...")
    print("=" * 60)
    
    # Generate critic score comparison
    print("\n1. Generating critic score comparison plot...")
    plotter.plot_critic_score_comparison()
    
    # Generate Fisher C_K comparison (uppercase)
    print("\n2. Generating Fisher C_K comparison plot (uppercase)...")
    plotter.plot_fisher_ck_comparison()
    
    # Generate Fisher c_k comparison (lowercase)
    print("\n3. Generating Fisher c_k comparison plot (lowercase)...")
    plotter.plot_fisher_ck_lowercase_comparison()
    
    # Generate computed C_K comparison plot
    print("\n4. Generating computed C_K comparison plot...")
    plotter.plot_computed_ck_comparison()
    
    # Generate sliding window C_K comparison plot
    print("\n5. Generating sliding window C_K comparison plot...")
    plotter.plot_sliding_window_ck_comparison()
    
    # Generate raw sliding window C_K comparison plot
    print("\n6. Generating raw sliding window C_K comparison plot...")
    plotter.plot_raw_sliding_window_ck_comparison()
    
    # Generate sliding window C_K comparison plot with window size 5
    print("\n7. Generating sliding window C_K comparison plot (window size 5)...")
    plotter.plot_sliding_window_ck_comparison(window_size=5)
    
    # Generate raw sliding window C_K comparison plot with window size 5
    print("\n8. Generating raw sliding window C_K comparison plot (window size 5)...")
    plotter.plot_raw_sliding_window_ck_comparison(window_size=5)
    
    # Generate log-transformed sliding window C_K comparison plot with window size 20
    print("\n9. Generating log-transformed sliding window C_K comparison plot (window size 20)...")
    plotter.plot_log_sliding_window_ck_comparison(window_size=20)
    
    # Generate log-transformed sliding window C_K comparison plot with window size 5
    print("\n10. Generating log-transformed sliding window C_K comparison plot (window size 5)...")
    plotter.plot_log_sliding_window_ck_comparison(window_size=5)
    
    # Generate control group critic score plot
    print("\n11. Generating control group critic score plot...")
    plotter.plot_control_group_critic_score()
    
    # Generate summary statistics
    print("\n12. Generating summary statistics plot...")
    plotter.plot_summary_statistics()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Plots saved to: {plotter.output_dir}")

if __name__ == "__main__":
    main()
