# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import threading

class AnalyzerMetricsStorage:
    """
    Unified storage system for analyzer metrics with support for:
    - JSON file storage
    - WandB integration
    - Component-level and matrix-level detailed metrics
    - Thread-safe operations
    """
    
    def __init__(self, 
                 base_dir: str = "./analyzer_metrics",
                 experiment_name: str = "default_experiment",
                 enable_wandb: bool = True):
        """
        Initialize the analyzer metrics storage system.
        
        Args:
            base_dir: Base directory for storing metrics files
            experiment_name: Name of the current experiment
            enable_wandb: Whether to also log to WandB
        """
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.enable_wandb = enable_wandb
        
        # Create storage directories
        os.makedirs(base_dir, exist_ok=True)
        self.metrics_file = os.path.join(base_dir, f"{experiment_name}_analyzer_metrics.jsonl")
        self.summary_file = os.path.join(base_dir, f"{experiment_name}_analyzer_summary.json")
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Initialize summary data
        self.summary_data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "total_steps": 0,
            "gradient_analysis_count": 0,
            "fisher_analysis_count": 0,
            "last_updated": None
        }
        
        print(f"[AnalyzerMetricsStorage] Initialized storage at: {base_dir}")
        print(f"[AnalyzerMetricsStorage] Metrics file: {self.metrics_file}")
        print(f"[AnalyzerMetricsStorage] Summary file: {self.summary_file}")
    
    def store_gradient_metrics(self, 
                             step: int, 
                             gradient_stats: Dict[str, Any], 
                             tau: float,
                             additional_info: Optional[Dict] = None):
        """
        Store detailed gradient analysis metrics.
        
        Args:
            step: Training step number
            gradient_stats: Complete gradient analysis results
            tau: Threshold value used for analysis
            additional_info: Additional metadata
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Prepare the metrics record
            metrics_record = {
                "timestamp": timestamp,
                "step": step,
                "analysis_type": "gradient",
                "tau": tau,
                "global_stats": gradient_stats.get('__global__', {}),
                "component_stats": gradient_stats.get('components', {}),
                "additional_info": additional_info or {}
            }
            
            # Write to JSONL file (one record per line)
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_record) + '\n')
            
            # Update summary
            self.summary_data["gradient_analysis_count"] += 1
            self.summary_data["total_steps"] = max(self.summary_data["total_steps"], step)
            self.summary_data["last_updated"] = timestamp
            
            # Log to WandB if enabled
            if self.enable_wandb:
                self._log_to_wandb(metrics_record, step)
            
            print(f"[AnalyzerMetricsStorage] Stored gradient metrics for step {step}")
    
    def store_fisher_metrics(self, 
                           step: int, 
                           fisher_stats: Dict[str, Any],
                           additional_info: Optional[Dict] = None):
        """
        Store detailed Fisher information analysis metrics.
        
        Args:
            step: Training step number
            fisher_stats: Complete Fisher analysis results
            additional_info: Additional metadata
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Prepare the metrics record
            metrics_record = {
                "timestamp": timestamp,
                "step": step,
                "analysis_type": "fisher",
                "fisher_stats": fisher_stats,
                "additional_info": additional_info or {}
            }
            
            # Write to JSONL file
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_record) + '\n')
            
            # Update summary
            self.summary_data["fisher_analysis_count"] += 1
            self.summary_data["total_steps"] = max(self.summary_data["total_steps"], step)
            self.summary_data["last_updated"] = timestamp
            
            # Log to WandB if enabled
            if self.enable_wandb:
                self._log_to_wandb(metrics_record, step)
            
            print(f"[AnalyzerMetricsStorage] Stored Fisher metrics for step {step}")
    
    def _log_to_wandb(self, metrics_record: Dict[str, Any], step: int):
        """Log detailed metrics to WandB with proper hierarchical structure."""
        try:
            import wandb
            
            if metrics_record["analysis_type"] == "gradient":
                # Log global gradient metrics
                global_stats = metrics_record.get("global_stats", {})
                if global_stats:
                    wandb.log({
                        "gradient_analysis/global_ratio": global_stats.get("ratio", 0.0),
                        "gradient_analysis/global_zero_count": global_stats.get("zero", 0),
                        "gradient_analysis/global_total_count": global_stats.get("total", 0),
                        "gradient_analysis/tau": metrics_record.get("tau", 0.0)
                    }, step=step)
                
                # Log component-level metrics
                component_stats = metrics_record.get("component_stats", {})
                for comp_name, comp_data in component_stats.items():
                    wandb.log({
                        f"gradient_analysis/components/{comp_name}/ratio": comp_data.get("ratio", 0.0),
                        f"gradient_analysis/components/{comp_name}/zero_count": comp_data.get("zero", 0),
                        f"gradient_analysis/components/{comp_name}/total_count": comp_data.get("total", 0)
                    }, step=step)
                    
                    # Log matrix-level metrics
                    matrix_stats = comp_data.get("matrices", {})
                    for matrix_name, matrix_data in matrix_stats.items():
                        short_name = '.'.join(matrix_name.split('.')[-2:])  # Keep last 2 parts
                        wandb.log({
                            f"gradient_analysis/matrices/{comp_name}/{short_name}/ratio": matrix_data.get("ratio", 0.0),
                            f"gradient_analysis/matrices/{comp_name}/{short_name}/min_norm": matrix_data.get("min_row_norm", 0.0),
                            f"gradient_analysis/matrices/{comp_name}/{short_name}/avg_norm": matrix_data.get("avg_row_norm", 0.0),
                            f"gradient_analysis/matrices/{comp_name}/{short_name}/max_norm": matrix_data.get("max_row_norm", 0.0)
                        }, step=step)
            
            elif metrics_record["analysis_type"] == "fisher":
                # Log Fisher information metrics
                fisher_stats = metrics_record.get("fisher_stats", {})
                for key, value in fisher_stats.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"fisher_analysis/{key}": value}, step=step)
        
        except Exception as e:
            print(f"[AnalyzerMetricsStorage] Warning: Failed to log to WandB: {e}")
    
    def save_summary(self):
        """Save the summary data to file."""
        with self._lock:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.summary_data, f, indent=2)
    
    def get_metrics_for_step(self, step: int) -> Dict[str, Any]:
        """Retrieve all metrics for a specific step."""
        metrics = {"gradient": None, "fisher": None}
        
        if not os.path.exists(self.metrics_file):
            return metrics
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record.get("step") == step:
                        analysis_type = record.get("analysis_type")
                        metrics[analysis_type] = record
                except json.JSONDecodeError:
                    continue
        
        return metrics
    
    def export_to_csv(self, output_file: str = None):
        """Export metrics to CSV format for analysis."""
        if output_file is None:
            output_file = os.path.join(self.base_dir, f"{self.experiment_name}_metrics.csv")
        
        import pandas as pd
        
        records = []
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # Flatten the record for CSV
                        flat_record = {
                            "timestamp": record["timestamp"],
                            "step": record["step"],
                            "analysis_type": record["analysis_type"]
                        }
                        
                        if record["analysis_type"] == "gradient":
                            global_stats = record.get("global_stats", {})
                            flat_record.update({
                                "global_ratio": global_stats.get("ratio", 0.0),
                                "global_zero_count": global_stats.get("zero", 0),
                                "global_total_count": global_stats.get("total", 0),
                                "tau": record.get("tau", 0.0)
                            })
                        
                        records.append(flat_record)
                    except json.JSONDecodeError:
                        continue
        
        if records:
            df = pd.DataFrame(records)
            df.to_csv(output_file, index=False)
            print(f"[AnalyzerMetricsStorage] Exported metrics to: {output_file}")
        else:
            print("[AnalyzerMetricsStorage] No metrics to export")
    
    def __del__(self):
        """Cleanup: save summary on destruction."""
        try:
            self.save_summary()
        except:
            pass
