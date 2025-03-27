#!/bin/bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate zero

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/home/yliog/model/qwen1.5b/snapshots/8faed761d45a263340a0528343f099c05c9a4323"}  # Qwen 1.5B model path
export N_GPUS=8  # Using all 8 3090 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Increased to 4 for Qwen-1.5B's larger model
export DATA_DIR="/data/countdown/continual"  # Match memory data structure
export TRAINED_MODEL="/app/models/countdown_continual_1.5b"  # Model being continually trained

# Training configuration
ROUNDS=3

# Create necessary directories
mkdir -p metrics logs wandb plots "$DATA_DIR/plus" "$DATA_DIR/plus_minus" "$DATA_DIR/plus_minus_mul" "$DATA_DIR/plus_minus_mul_div"

# Initialize model directory if it doesn't exist
if [ ! -d "$TRAINED_MODEL" ]; then
    echo "Creating trained model directory at $TRAINED_MODEL"
    mkdir -p "$TRAINED_MODEL"
    
    # Check if base model exists
    if [ ! -d "$BASE_MODEL" ]; then
        echo "Error: Base model directory $BASE_MODEL does not exist"
        exit 1
    fi
    
    echo "\nChecking base model config.json:"
    if [ ! -f "$BASE_MODEL/config.json" ]; then
        echo "Error: No config.json found in base model"
        exit 1
    fi
    echo "Base model config.json contents:"
    cat "$BASE_MODEL/config.json"
    
    # Remove empty configuration.json if it exists
    if [ -f "$BASE_MODEL/configuration.json" ]; then
        echo "\nRemoving empty configuration.json"
        rm "$BASE_MODEL/configuration.json"
    fi
    
    echo "\nCopying base model files from $BASE_MODEL to $TRAINED_MODEL"
    # First copy all files from base model, including hidden files
    cp -rv "$BASE_MODEL/." "$TRAINED_MODEL/"
    
    echo "\nVerifying copied files:"
    ls -la "$TRAINED_MODEL/"
    
    # Ensure config.json was copied correctly
    if [ ! -f "$TRAINED_MODEL/config.json" ]; then
        echo "Error: Failed to copy config.json from base model"
        exit 1
    fi
    
    echo "\nVerifying copied config.json contents:"
    cat "$TRAINED_MODEL/config.json"
    
    echo "\nSuccessfully initialized model from $BASE_MODEL"
fi

# Print configuration and check directories
echo "Starting training with configuration:"
echo "BASE_MODEL: $BASE_MODEL"
echo "DATA_DIR: $DATA_DIR"
echo "Number of GPUs: $N_GPUS"
echo "Tensor Parallel Size: $ROLLOUT_TP_SIZE"
echo "Training Rounds: $ROUNDS"
echo "Operator Groups: plus -> plus_minus -> plus_minus_mul -> plus_minus_mul_div"

# Check base model directory
echo "\nChecking base model directory:"
ls -la "$BASE_MODEL"
echo "\nBase model config.json contents:"
cat "$BASE_MODEL/config.json"

# Function to save model checkpoint
save_checkpoint() {
    local model_path=$1
    local round=$2
    local group=$3
    local checkpoint_type=$4  # 'initial' or 'final'
    
    # Create checkpoint directory
    local experiment_name="ContinualCountdown1.5B_R${round}_${group}"
    local checkpoint_dir="checkpoints/ContinualCountdown1.5B/$experiment_name/actor"
    
    # Check if source model exists
    if [ ! -d "$model_path" ]; then
        echo "Warning: Model path $model_path does not exist"
        return 1
    fi
    
    echo "Creating checkpoint directory at $checkpoint_dir/global_step_0"
    mkdir -p "$checkpoint_dir/global_step_0"
    
    echo "Copying model files from $model_path to checkpoint"
    # Copy all files from the model path
    cp -r "$model_path/." "$checkpoint_dir/global_step_0/"
    
    # Verify config.json was copied correctly
    if [ ! -f "$checkpoint_dir/global_step_0/config.json" ]; then
        echo "Error: Failed to copy config.json from model path"
        return 1
    fi
    
    echo "Successfully saved ${checkpoint_type} checkpoint for round $round, group $group"
}

# Create logs directory
mkdir -p "logs"

# Training function for a specific group
train_group() {
    local round=$1
    local group=$2
    local experiment_name="ContinualCountdown1.5B_R${round}_${group}"
    local log_file="logs/${experiment_name}.log"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Save initial model state
    save_checkpoint "$TRAINED_MODEL" "$round" "$group" "initial"
    
    # Run training with WandB configuration
    WANDB_RUN_NAME="ContinualCountdown1.5B_R${round}_${group}"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Prevent model downloads
    export TRANSFORMERS_OFFLINE=1
    
    echo "Training on data:"
    echo "  Train: $DATA_DIR/$group/train.parquet"
    echo "  Test:  $DATA_DIR/$group/test.parquet"
    
    python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/${group}/train.parquet \
        data.val_files=$DATA_DIR/${group}/test.parquet \
        data.train_batch_size=128 \
        data.val_batch_size=128 \
        data.max_prompt_length=256 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$TRAINED_MODEL \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        +actor_rollout_ref.model.trust_remote_code=true \
        +actor_rollout_ref.model.torch_dtype=bfloat16 \
        +actor_rollout_ref.model.low_cpu_mem_usage=true \
        +actor_rollout_ref.model.device_map=auto \
        +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
        +actor_rollout_ref.model.use_cache=false \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enforce_eager=true \
        actor_rollout_ref.rollout.free_cache_engine=false \
        actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
        critic.optim.lr=1e-5 \
        critic.model.path=$TRAINED_MODEL \
        +critic.model.trust_remote_code=true \
        +critic.model.torch_dtype=bfloat16 \
        +critic.model.device_map=auto \
        +critic.model.attn_implementation=flash_attention_2 \
        +critic.model.use_cache=false \
        critic.ppo_micro_batch_size=8 \
        critic.ppo_mini_batch_size=16 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['wandb','console'] \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=checkpoints/ContinualCountdown1.5B/$WANDB_RUN_NAME \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.project_name=ContinualCountdown1.5B \
        trainer.experiment_name=$WANDB_RUN_NAME \
        trainer.total_epochs=25 \
        +trainer.val_before_train=True \
        ++reward_model.enable=False \
        ++reward_model.model.path=$TRAINED_MODEL \
        2>&1 | tee "$log_file"
    
    # Check if training was successful by looking for validation metrics
    if grep -q "Final validation metrics" "$log_file"; then
        echo "Training successful for round $round, group $group"
        
        # Save final model checkpoint
        latest_checkpoint="metrics/final_${round}_${group}"
        mkdir -p "$latest_checkpoint"
        
        # Shutdown Ray and any lingering Python processes
        echo "Shutting down Ray and Python processes before copying checkpoint..."
        python3 -c 'import ray; ray.shutdown()'
        pkill -f "ray::" || true
        sleep 5  # Give time for processes to fully shut down
        
        # Copy final checkpoint for evaluation using rsync
        echo "Copying final checkpoint to $latest_checkpoint"
        rsync -av --delete "$TRAINED_MODEL/" "$latest_checkpoint/"
        
        return 0
    else
        echo "Training failed for round $round, group $group"
        return 1
    fi
}

# Main training loop
for ((round=1; round<=$ROUNDS; round++)); do
    echo "Starting Round $round"
    
    # Train on each group sequentially in specified order
    for group in plus plus_minus plus_minus_mul plus_minus_mul_div; do
        # Check if data exists
        if [ ! -f "$DATA_DIR/$group/train.parquet" ] || [ ! -f "$DATA_DIR/$group/test.parquet" ]; then
            echo "Error: Data files not found for group $group"
            echo "Expected files:"
            echo "  $DATA_DIR/$group/train.parquet"
            echo "  $DATA_DIR/$group/test.parquet"
            exit 1
        fi
        
        # Create metrics and logs directories
        mkdir -p "metrics" "logs"
        
        # Save initial model state
        save_checkpoint "$TRAINED_MODEL" "$round" "$group" "initial"
        
        # Train on current group
        train_group $round $group
        
        if [ $? -ne 0 ]; then
            echo "Training failed for Round $round, Group $group. Exiting..."
            exit 1
        fi
    done
done

echo "Training completed successfully!"
