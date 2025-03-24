#!/bin/bash

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/app/models/qwen"}  # Qwen 0.5B model mounted in container
export N_GPUS=8  # Using all 8 3090 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Default to 2 for Qwen-0.5B's 14 attention heads
export DATA_DIR="/data/countdown/continual"  # Match memory data structure
export TRAINED_MODEL="/app/models/countdown_continual"  # Model being continually trained

# Training configuration
ROUNDS=3
declare -a GROUPS=("plus" "plus_minus" "plus_minus_mul" "plus_minus_mul_div")

# Create necessary directories
mkdir -p metrics logs wandb plots

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
    local experiment_name="ContinualCountdown_R${round}_${group}"
    local checkpoint_dir="checkpoints/ContinualCountdown/$experiment_name/actor"
    
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
    local experiment_name="ContinualCountdown_R${round}_${group}"
    local log_file="logs/${experiment_name}.log"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Save initial model state
    save_checkpoint "$TRAINED_MODEL" "$round" "$group" "initial"
    
    # Run training with WandB configuration
    WANDB_RUN_NAME="ContinualCountdown_R${round}_${group}"
    
    echo "Starting training for Round ${round}, Group ${group}"
    
    # Prevent model downloads
    export TRANSFORMERS_OFFLINE=1
    
    echo "Training on data:"
    echo "  Train: $DATA_DIR/$group/train.parquet"
    echo "  Test:  $DATA_DIR/$group/test.parquet"
    
    python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/${group}/train.parquet \
        data.val_files=$DATA_DIR/${group}/test.parquet \
        data.train_batch_size=256 \
        data.val_batch_size=256 \
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
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['wandb','console'] \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=checkpoints/ContinualCountdown/$WANDB_RUN_NAME \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.project_name=ContinualCountdown \
        trainer.experiment_name=$WANDB_RUN_NAME \
        trainer.total_epochs=15 \
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
        if command -v rsync >/dev/null 2>&1; then
            rsync -av "$TRAINED_MODEL/" "$latest_checkpoint/" || {
                echo "Rsync failed, falling back to cp..."
                cp -rv "$TRAINED_MODEL"/* "$latest_checkpoint/"
            }
        else
            cp -rv "$TRAINED_MODEL"/* "$latest_checkpoint/"
        fi
        
        # Update base model for next training
        export BASE_MODEL=$latest_checkpoint
        echo "Updated BASE_MODEL to $BASE_MODEL"
    else
        echo "Error: Training failed for round $round, group $group - no validation metrics found"
        exit 1
    fi
}





# Main training loop
for ((round=1; round<=ROUNDS; round++)); do
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
        
        # Update trained model with latest weights if training was successful
        if [ -d "metrics/final_${round}_${group}" ]; then
            # Shutdown Ray and any lingering Python processes
            echo "Shutting down Ray and Python processes..."
            python3 -c 'import ray; ray.shutdown()'
            pkill -f "ray::" || true
            sleep 5  # Give time for processes to fully shut down
            
            # Get latest checkpoint from trainer
            latest_checkpoint="checkpoints/ContinualCountdown/ContinualCountdown_R${round}_${group}/actor"
            latest_step=$(ls "$latest_checkpoint" | grep -E '^global_step_[0-9]+$' | sort -V | tail -n 1)
            
            if [ -n "$latest_step" ]; then
                echo "Copying checkpoint from $latest_checkpoint/$latest_step to $TRAINED_MODEL"
                # First try rsync, fall back to cp if it fails
                if command -v rsync >/dev/null 2>&1; then
                    rsync -av --remove-source-files "$latest_checkpoint/$latest_step/" "$TRAINED_MODEL/" || {
                        echo "Rsync failed, falling back to cp..."
                        for file in "$latest_checkpoint/$latest_step"/*; do
                            if [ -f "$file" ]; then
                                cp -v "$file" "$TRAINED_MODEL/" || echo "Warning: Failed to copy $file"
                            fi
                        done
                    }
                else
                    for file in "$latest_checkpoint/$latest_step"/*; do
                        if [ -f "$file" ]; then
                            cp -v "$file" "$TRAINED_MODEL/" || echo "Warning: Failed to copy $file"
                        fi
                    done
                fi
            
                # Ensure config.json has model_type
                if [ -f "$TRAINED_MODEL/config.json" ]; then
                    if ! grep -q '"model_type"' "$TRAINED_MODEL/config.json"; then
                        sed -i 's/{/{"model_type":"qwen2",/' "$TRAINED_MODEL/config.json"
                    fi
                else
                    echo '{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}' > "$TRAINED_MODEL/config.json"
                fi
            else
                echo "Error: No checkpoint found in $latest_checkpoint"
                exit 1
            fi
        else
            echo "Error: Training failed for round $round, group $group"
            exit 1
        fi
    done
    
    echo "Completed Round $round"
done

echo "Training completed. Metrics plots saved in plots/training_metrics.png"
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

echo "Training completed for all rounds and groups."
