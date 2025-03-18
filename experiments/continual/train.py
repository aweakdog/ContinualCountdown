"""Continual learning training script for Countdown task."""

import os
import json
import time
from typing import Dict, List
import torch
from torch.nn.utils import parameters_to_vector
import wandb
from rich import print as rprint
from rich.progress import track
from omegaconf import OmegaConf
from verl.trainer.main_ppo import main as train_ppo

class ContinualTrainer:
    def __init__(self, base_dir: str = "/data/countdown/continual"):
        self.base_dir = base_dir
        self.groups = [
            "plus",
            "plus_minus",
            "plus_minus_mul",
            "plus_minus_mul_div"
        ]
        self.metrics_dir = os.path.join(base_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "success_rate": [],
            "weight_change": [],
            "loss": [],
            "gradient_norm": [],
            "response_length": []
        }
        
        # Store initial model state
        self.initial_state = None
    
    def compute_weight_change(self, old_state: Dict, new_state: Dict) -> float:
        """Compute normalized weight change between two model states."""
        old_params = []
        new_params = []
        
        for key in old_state:
            if key.endswith(".weight") or key.endswith(".bias"):
                old_params.append(old_state[key].view(-1))
                new_params.append(new_state[key].view(-1))
        
        old_vec = torch.cat(old_params)
        new_vec = torch.cat(new_params)
        
        weight_diff = torch.norm(new_vec - old_vec)
        weight_magnitude = torch.norm(old_vec)
        
        return (weight_diff / (weight_magnitude + 1e-7)).item()

    def compute_gradient_norm(self, model) -> float:
        """Compute normalized gradient norm."""
        total_norm = 0
        param_norm = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm += p.data.norm(2).item() ** 2
                total_norm += p.grad.data.norm(2).item() ** 2
        
        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        return (total_norm / (param_norm + 1e-7)).item()

    def update_metrics(self, round_idx: int, group_idx: int, metrics: Dict):
        """Update metrics for current training round and group."""
        for key in self.metrics:
            if len(self.metrics[key]) <= round_idx:
                self.metrics[key].append([])
            if len(self.metrics[key][round_idx]) <= group_idx:
                self.metrics[key][round_idx].append([])
            self.metrics[key][round_idx][group_idx].append(metrics[key])

    def save_metrics(self):
        """Save metrics to JSON file."""
        output_path = os.path.join(self.metrics_dir, f"training_metrics_{int(time.time())}.json")
        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        rprint(f"[green]Saved metrics to {output_path}[/green]")

    def train_group(self, round_idx: int, group_idx: int, cfg) -> None:
        """Train on a specific operator group."""
        group = self.groups[group_idx]
        rprint(f"[bold cyan]Training on group {group} (Round {round_idx + 1})[/bold cyan]")
        
        # Update data paths for current group
        cfg.data.train_files = os.path.join(self.base_dir, group, "train.parquet")
        cfg.data.val_files = os.path.join(self.base_dir, group, "test.parquet")
        
        # Update experiment name to include round and group
        cfg.trainer.experiment_name = f"{cfg.trainer.experiment_name}_round{round_idx+1}_{group}"
        
        # Get initial model state if first round and group
        if round_idx == 0 and group_idx == 0:
            self.initial_state = torch.load(cfg.actor_rollout_ref.model.path)
        
        # Train the model
        metrics = train_ppo(cfg)
        
        # Update and save metrics
        self.update_metrics(round_idx, group_idx, metrics)
        self.save_metrics()

    def train(self, cfg) -> None:
        """Run continual training for specified number of rounds."""
        num_rounds = 3
        
        for round_idx in track(range(num_rounds), description="Training rounds"):
            for group_idx in range(len(self.groups)):
                self.train_group(round_idx, group_idx, cfg)
            rprint(f"[bold green]✓ Completed round {round_idx + 1}/{num_rounds}[/bold green]")

def main():
    # Load base configuration
    cfg = OmegaConf.load("configs/train_continual.yaml")
    
    # Initialize trainer and start training
    trainer = ContinualTrainer()
    trainer.train(cfg)

if __name__ == "__main__":
    main()
