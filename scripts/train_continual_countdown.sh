#!/bin/bash

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/app/models/qwen"}  # Qwen 0.5B model mounted in container
export N_GPUS=8  # Using all 8 3090 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Default to 2 for Qwen-0.5B's 14 attention heads
export DATA_DIR="/data/countdown/continual"  # Match memory data structure

# Training configuration
ROUNDS=3
declare -a GROUPS=("plus" "plus_minus" "plus_minus_mul" "plus_minus_mul_div")

# Create necessary directories
mkdir -p metrics logs wandb

# Print configuration
echo "Starting training with configuration:"
echo "BASE_MODEL: $BASE_MODEL"
echo "DATA_DIR: $DATA_DIR"
echo "Number of GPUs: $N_GPUS"
echo "Tensor Parallel Size: $ROLLOUT_TP_SIZE"
echo "Training Rounds: $ROUNDS"
echo "Operator Groups: ${GROUPS[*]}"

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
    local round=$3
    local group=$4
    python3 -c "
import re
import json
import numpy as np

with open('$log_file', 'r') as f:
    content = f.read()

# Extract metrics
metrics = {
    'response_lengths': [],
    'rewards': [],
    'losses': [],
    'gradients': []
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
    
    # Extract gradient norms
    if 'grad_norm:' in line:
        try:
            grad = float(line.split('grad_norm:')[1].strip().split()[0])
            metrics['gradients'].append(grad)
        except:
            pass

# Compute statistics
stats = {}

# Response length metrics
if metrics['response_lengths']:
    stats.update({
        'avg_response_length': np.mean(metrics['response_lengths']),
        'max_response_length': max(metrics['response_lengths']),
        'min_response_length': min(metrics['response_lengths'])
    })

# Success rate and reward metrics
if metrics['rewards']:
    stats.update({
        'avg_reward': np.mean(metrics['rewards']),
        'success_rate': len([r for r in metrics['rewards'] if r > 0.9]) / len(metrics['rewards'])
    })

# Loss metrics
if metrics['losses']:
    stats.update({
        'avg_loss': np.mean(metrics['losses']),
        'final_loss': metrics['losses'][-1] if metrics['losses'] else None
    })

# Gradient metrics
if metrics['gradients']:
    stats.update({
        'avg_grad_norm': np.mean(metrics['gradients']),
        'max_grad_norm': max(metrics['gradients'])
    })

# Add round and group information
stats.update({
    'round': $round,
    'group': '$group'
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
    cp -r "$BASE_MODEL" "metrics/initial_${round}_${group}"
    
    # Run training with WandB configuration
    WANDB_RUN_NAME="ContinualCountdown_R${round}_${group}"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Configure environment for bfloat16 and vLLM
    export TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"  # For Ampere GPUs (RTX 3090)
    export VLLM_TP_SIZE=$ROLLOUT_TP_SIZE  # Explicitly set tensor parallel size
    export VLLM_DTYPE=bfloat16  # Force bfloat16 dtype
    export VLLM_USE_CUDA_GRAPH=0  # Disable CUDA graph due to cache engine
    export TRANSFORMERS_OFFLINE=1  # Prevent model downloads
    
    python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/${group}/train.parquet \
        data.val_files=$DATA_DIR/${group}/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=100 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        +actor_rollout_ref.model.trust_remote_code=true \
        +actor_rollout_ref.model.torch_dtype=bfloat16 \
        +actor_rollout_ref.model.low_cpu_mem_usage=true \
        +actor_rollout_ref.model.device_map=auto \
        +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
        +actor_rollout_ref.model.use_cache=false \
        +actor_rollout_ref.model.use_flash_attention_2=true \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enforce_eager=true \
        actor_rollout_ref.rollout.free_cache_engine=false \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
        critic.optim.lr=1e-5 \
        critic.model.path=$BASE_MODEL \
        +critic.model.trust_remote_code=true \
        +critic.model.model_type=qwen2 \
        +critic.model.architectures=["Qwen2ForCausalLM"] \
        +critic.model.config_overrides={"torch_dtype":"bfloat16"} \
        +critic.model.torch_dtype=bfloat16 \
        +critic.model.device_map=auto \
        +critic.model.attn_implementation=flash_attention_2 \
        +critic.model.use_cache=false \
        critic.ppo_micro_batch_size=8 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['wandb'] \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.project_name=ContinualCountdown \
        trainer.experiment_name=$WANDB_RUN_NAME \
        trainer.total_epochs=15 \
        2>&1 | tee "$log_file"
    
    # Update base model and compute metrics
    latest_checkpoint=$(find wandb/latest-run/files -name "*.pt" | sort -V | tail -n 1)
    if [ -n "$latest_checkpoint" ]; then
        # Extract metrics from logs
        extract_wandb_metrics "$log_file" "$metrics_file.tmp" "$round" "$group"
        
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



# Main training loop
for ((round=1; round<=ROUNDS; round++)); do
    echo "Starting Round $round"
    for group in plus plus_minus plus_minus_mul plus_minus_mul_div; do
        echo "Processing operator group: $group"
        echo "Checking data files in: $DATA_DIR/$group/"
        
        # Check if data exists
        if [ ! -f "$DATA_DIR/$group/train.parquet" ]; then
            echo "Error: Missing training data file: $DATA_DIR/$group/train.parquet"
            continue
        fi
        if [ ! -f "$DATA_DIR/$group/test.parquet" ]; then
            echo "Error: Missing test data file: $DATA_DIR/$group/test.parquet"
            continue
        fi
        
        echo "Found data files for group $group"
        train_group "$round" "$group"
        echo "Completed training for group: $group"
    done
    echo "Completed Round $round"
done

# Generate final metrics summary
python3 -c '
import json
import glob
import pandas as pd

# Load all metrics files
metrics_data = []
for f in glob.glob("metrics/metrics_*_*.json"):
    try:
        # Extract round and group from filename
        parts = f.split("_")
        round_num = int(parts[-2])
        group_name = parts[-1].replace(".json", "")
        
        with open(f) as fp:
            data = json.load(fp)
            # Add round and group info
            data["round"] = round_num
            data["group"] = group_name
            
            # Ensure all required metrics are present
            metrics_data.append({
                "round": round_num,
                "group": group_name,
                "success_rate": data.get("success_rate", 0),
                "weight_change": data.get("weight_change", 0),
                "avg_loss": data.get("avg_loss", 0),
                "avg_response_length": data.get("avg_response_length", 0),
                "avg_reward": data.get("avg_reward", 0),
                "avg_grad_norm": data.get("avg_grad_norm", 0),
                "max_grad_norm": data.get("max_grad_norm", 0)
            })
    except Exception as e:
        print(f"Error processing {f}: {e}")

if not metrics_data:
    print("No metrics data found")
    exit(0)

# Convert to DataFrame
df = pd.DataFrame(metrics_data)

# Calculate metrics by round and group
summary = {
    "overall": {
        metric: df[metric].mean()
        for metric in df.columns if metric not in ["round", "group"]
    },
    "by_round": df.groupby("round").mean().to_dict(),
    "by_group": df.groupby("group").mean().to_dict(),
    "by_round_and_group": df.set_index(["round", "group"]).to_dict()
}

# Save summary
with open("metrics/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Metrics summary saved to metrics/summary.json")
print("\nOverall metrics:")
for metric, value in summary["overall"].items():
    print(f"{metric}: {value:.4f}")
'

echo "Training completed for all rounds and groups. See metrics/summary.json for final results."

echo "Training completed for all rounds and groups. See metrics/summary.json for final results."

