#!/bin/bash

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/path/to/base/model"}  # Set your base model path
export N_GPUS=${N_GPUS:-4}  # Number of GPUs to use
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Tensor parallel size
export DATA_DIR="/data/countdown/continual"

# Operator groups in order of complexity
GROUPS=("plus" "plus_minus" "plus_minus_mul" "plus_minus_mul_div")
ROUNDS=3

# Create metrics directory
mkdir -p metrics

# Function to compute weight change
compute_weight_change() {
    local old_model=$1
    local new_model=$2
    python3 -c "
import torch
old = torch.load('$old_model')
new = torch.load('$new_model')
total_change = 0
total_weights = 0
for key in old['model'].keys():
    if key in new['model']:
        change = torch.norm(new['model'][key] - old['model'][key])
        weights = torch.norm(old['model'][key])
        total_change += change.item()
        total_weights += weights.item()
print(f'{total_change/total_weights:.6f}')
"
}

# Function to extract metrics from WandB logs
extract_wandb_metrics() {
    local log_file=$1
    local metrics_file=$2
    python3 -c "
import re
import json

with open('$log_file', 'r') as f:
    content = f.read()

# Extract metrics
metrics = {
    'response_lengths': [],
    'rewards': [],
    'losses': []
}

for line in content.split('\n'):
    # Extract response length
    if 'decoded output:' in line:
        response = line.split('decoded output:')[1].strip()
        metrics['response_lengths'].append(len(response.split()))
    
    # Extract reward scores
    if 'Reward:' in line:
        try:
            reward = float(line.split('Reward:')[1].strip().split()[0])
            metrics['rewards'].append(reward)
        except:
            pass
    
    # Extract loss values
    if 'loss:' in line:
        try:
            loss = float(line.split('loss:')[1].strip().split()[0])
            metrics['losses'].append(loss)
        except:
            pass

# Compute statistics
stats = {}
if metrics['response_lengths']:
    stats.update({
        'avg_response_length': sum(metrics['response_lengths']) / len(metrics['response_lengths']),
        'max_response_length': max(metrics['response_lengths']),
        'min_response_length': min(metrics['response_lengths'])
    })

if metrics['rewards']:
    stats.update({
        'avg_reward': sum(metrics['rewards']) / len(metrics['rewards']),
        'success_rate': len([r for r in metrics['rewards'] if r > 0.9]) / len(metrics['rewards'])
    })

if metrics['losses']:
    stats.update({
        'avg_loss': sum(metrics['losses']) / len(metrics['losses']),
        'final_loss': metrics['losses'][-1] if metrics['losses'] else None
    })

# Write metrics
with open('$metrics_file', 'w') as f:
    json.dump(stats, f, indent=2)
"
}

# Training function for a specific group
train_group() {
    local round=$1
    local group=$2
    local experiment_name="ContinualCountdown_R${round}_${group}"
    local metrics_file="metrics/metrics_${round}_${group}.json"
    local log_file="logs/${experiment_name}.log"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Store initial model for weight change calculation
    cp "$BASE_MODEL" "metrics/initial_${round}_${group}.pt"
    
    # Run training
    python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/${group}/train.parquet \
        data.val_files=$DATA_DIR/${group}/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=1312 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=$BASE_MODEL \
        critic.ppo_micro_batch_size=8 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['wandb'] \
        +trainer.val_before_train=False \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.project_name=ContinualCountdown \
        trainer.experiment_name=$experiment_name \
        trainer.total_epochs=15 \
        2>&1 | tee "$log_file"
    
    # Update base model and compute metrics
    latest_checkpoint=$(find wandb/latest-run/files -name "*.pt" | sort -V | tail -n 1)
    if [ -n "$latest_checkpoint" ]; then
        # Extract metrics from logs
        extract_wandb_metrics "$log_file" "$metrics_file.tmp"
        
        # Compute normalized weight change
        weight_change=$(compute_weight_change "metrics/initial_${round}_${group}.pt" "$latest_checkpoint")
        
        # Combine metrics
        python3 -c "
import json
with open('$metrics_file.tmp') as f:
    metrics = json.load(f)
metrics['weight_change'] = $weight_change
with open('$metrics_file', 'w') as f:
    json.dump(metrics, f, indent=2)
"
        rm "$metrics_file.tmp"
        
        # Update base model for next training
        export BASE_MODEL=$latest_checkpoint
        echo "Updated BASE_MODEL to $BASE_MODEL"
        
        # Clean up initial model
        rm "metrics/initial_${round}_${group}.pt"
    fi
}

# Create necessary directories
mkdir -p logs metrics

# Main training loop
for ((round=1; round<=ROUNDS; round++)); do
    echo "\nStarting Round $round"
    for group in "${GROUPS[@]}"; do
        train_group $round $group
    done
    echo "Completed Round $round"
done

# Generate final metrics summary
python3 -c '
import json
import glob
import pandas as pd

# Load all metrics files
metrics = []
for f in glob.glob("metrics/metrics_*_*.json"):
    with open(f) as fp:
        data = json.load(fp)
        round_group = f.split("_")[-2:]
        data["round"] = round_group[0]
        data["group"] = round_group[1].split(".")[0]
        metrics.append(data)

# Create summary DataFrame
df = pd.DataFrame(metrics)

# Save overall summary
with open("metrics/summary.json", "w") as fp:
    json.dump({
        "per_round": df.groupby("round").mean().to_dict(),
        "per_group": df.groupby("group").mean().to_dict(),
        "overall": df.mean().to_dict()
    }, fp, indent=2)
'

echo "Training completed for all rounds and groups. See metrics/summary.json for final results."

