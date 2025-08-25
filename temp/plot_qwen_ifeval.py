#!/usr/bin/env python3

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QwenIFEvalPlotter:
    def __init__(self, base_dir="/cpfs04/user/liyuanhang.p/src/ContinualCountdown/qwen_logs", 
                 output_dir="./plots/qwen_ifeval"):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the directories for IFEval (only global_step_0 available)
        self.step_dirs = {
            0: "train_ifeval_qwen3b_sft_global_step_0"
        }
    
    def parse_log_line(self, line):
        """Parse a single log line to extract metrics"""
        metrics = {}
        
        # Remove ANSI color codes
        line = re.sub(r'\^?\[\[\d+m', '', line)
        
        # Extract step number
        step_match = re.search(r'step:\s*(\d+)', line)
        if step_match:
            metrics['step'] = int(step_match.group(1))
        
        # Extract critic/score/mean
        critic_match = re.search(r'critic/score/mean:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if critic_match:
            metrics['critic_score_mean'] = float(critic_match.group(1))
        
        # Extract fisher/C_K_normalized
        ck_match = re.search(r'fisher/C_K_normalized:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if ck_match:
            metrics['fisher_C_K_normalized'] = float(ck_match.group(1))
        
        # Extract fisher/c_k_normalized (lowercase)
        ck_lower_match = re.search(r'fisher/c_k_normalized:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if ck_lower_match:
            metrics['fisher_c_k_normalized'] = float(ck_lower_match.group(1))
        
        return metrics
    
    def parse_log_file(self, log_file_path):
        """Parse a single log file and extract all metrics"""
        data = []
        
        if not log_file_path.exists():
            print(f"Log file not found: {log_file_path}")
            return data
        
        print(f"Parsing log file: {log_file_path}")
        
        with open(log_file_path, 'r') as f:
            for line in f:
                metrics = self.parse_log_line(line)
                if metrics and 'step' in metrics:
                    data.append(metrics)
        
        print(f"Found {len(data)} data points")
        return data
    
    def parse_all_data(self):
        """Parse all log files and return organized data"""
        all_data = {}
        
        for global_step, dir_name in self.step_dirs.items():
            log_dir = self.base_dir / dir_name
            
            if not log_dir.exists():
                print(f"Directory not found: {log_dir}")
                continue
            
            # Find log files
            log_files = list(log_dir.glob("*.log"))
            
            if not log_files:
                print(f"No log files found in {log_dir}")
                continue
            
            step_data = []
            for log_file in log_files:
                file_data = self.parse_log_file(log_file)
                step_data.extend(file_data)
            
            if step_data:
                all_data[global_step] = sorted(step_data, key=lambda x: x['step'])
                print(f"Global step {global_step}: {len(step_data)} total data points")
        
        return all_data
    
    def remove_outliers(self, data, key, z_threshold=3):
        """Remove outliers using z-score method"""
        if not data:
            return data, []
        
        values = [d[key] for d in data if key in d]
        if len(values) < 3:
            return data, []
        
        z_scores = np.abs(stats.zscore(values))
        outlier_indices = np.where(z_scores > z_threshold)[0]
        
        clean_data = []
        outliers = []
        
        value_idx = 0
        for d in data:
            if key in d:
                if value_idx not in outlier_indices:
                    clean_data.append(d)
                else:
                    outliers.append(d)
                value_idx += 1
            else:
                clean_data.append(d)
        
        return clean_data, outliers
    
    def plot_critic_score_comparison(self, smooth=True, sigma=2):
        """Plot critic score comparison (single line for IFEval)"""
        print("Creating critic score comparison plot...")
        
        all_data = self.parse_all_data()
        
        if not all_data:
            print("No data found for critic score comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1f77b4']  # Single color for single line
        
        for i, (global_step, data) in enumerate(all_data.items()):
            # Extract critic score data
            critic_data = [(d['step'], d['critic_score_mean']) for d in data if 'critic_score_mean' in d]
            
            if not critic_data:
                continue
            
            x_data, y_data = zip(*critic_data)
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            color = colors[i % len(colors)]
            
            if smooth and len(y_data) > 3:
                # Apply Gaussian smoothing
                y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                # Plot smoothed line with confidence band
                ax.plot(x_data, y_smooth, label=f'IFEval Global Step {global_step}', 
                       color=color, linewidth=2.5, alpha=0.9)
                
                # Add confidence band (using standard error)
                window_size = min(20, len(y_data) // 10)
                if window_size > 1:
                    y_std = []
                    for j in range(len(y_data)):
                        start_idx = max(0, j - window_size // 2)
                        end_idx = min(len(y_data), j + window_size // 2 + 1)
                        y_std.append(np.std(y_data[start_idx:end_idx]))
                    
                    y_std = gaussian_filter1d(y_std, sigma=sigma)
                    ax.fill_between(x_data, y_smooth - y_std, y_smooth + y_std, 
                                   color=color, alpha=0.2)
            else:
                # Plot without smoothing
                ax.plot(x_data, y_data, label=f'IFEval Global Step {global_step}', 
                       color=color, linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Critic Score Mean', fontsize=12)
        title = 'Qwen IFEval Performance: Critic Score Comparison'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "qwen_ifeval_critic_score_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Critic score comparison plot saved to: {output_path}")
    
    def plot_fisher_ck_comparison(self, smooth=True, sigma=2):
        """Plot fisher C_K_normalized comparison"""
        print("Creating fisher C_K_normalized comparison plot...")
        
        all_data = self.parse_all_data()
        
        if not all_data:
            print("No data found for fisher C_K comparison")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1f77b4']
        
        for i, (global_step, data) in enumerate(all_data.items()):
            # Extract fisher C_K data
            ck_data = [(d['step'], d['fisher_C_K_normalized']) for d in data if 'fisher_C_K_normalized' in d]
            
            if not ck_data:
                continue
            
            x_data, y_data = zip(*ck_data)
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            
            color = colors[i % len(colors)]
            
            if smooth and len(y_data) > 3:
                # Apply Gaussian smoothing
                y_smooth = gaussian_filter1d(y_data, sigma=sigma)
                ax.plot(x_data, y_smooth, label=f'IFEval Global Step {global_step}', 
                       color=color, linewidth=2.5, alpha=0.9)
            else:
                ax.plot(x_data, y_data, label=f'IFEval Global Step {global_step}', 
                       color=color, linewidth=2, marker='s', markersize=4, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Fisher C_K_normalized', fontsize=12)
        title = 'Qwen IFEval Performance: Fisher C_K_normalized Comparison'
        if smooth:
            title += ' (Smoothed)'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "qwen_ifeval_fisher_ck_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Fisher C_K comparison plot saved to: {output_path}")
    
    def generate_all_plots(self):
        """Generate all plots for IFEval dataset"""
        print("Generating all IFEval plots...")
        
        # 1. Critic score comparison
        self.plot_critic_score_comparison(smooth=True)
        
        # 2. Fisher C_K comparison  
        self.plot_fisher_ck_comparison(smooth=True)
        
        print("All IFEval plots generated successfully!")

def main():
    plotter = QwenIFEvalPlotter()
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()
