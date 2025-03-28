#!/bin/bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate zero

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/home/yliog/model/qwen1.5b/snapshots/8faed761d45a263340a0528343f099c05c9a4323"}  # Qwen 1.5B model path
export N_GPUS=8  # Using all 8 3090 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Tensor parallel size
export DATA_DIR="./data/continual"  # Match actual data location
export TRAINED_MODEL="/app/models/countdown_continual_1.5b"  # Model being continually trained

# Create necessary directories
mkdir -p /app/metrics /app/logs /app/wandb /app/plots "$DATA_DIR/plus" "$DATA_DIR/plus_minus" "$DATA_DIR/plus_minus_mul" "$DATA_DIR/plus_minus_mul_div"

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
    
    # Remove empty configuration.json if it exists
    if [ -f "$BASE_MODEL/configuration.json" ]; then
        echo "\nRemoving empty configuration.json"
        rm "$BASE_MODEL/configuration.json"
    fi
    
    echo "\nCopying base model files from $BASE_MODEL to $TRAINED_MODEL"
    cp -rv "$BASE_MODEL/." "$TRAINED_MODEL/"
fi

# Print configuration
echo "Starting training with configuration:"
echo "BASE_MODEL: $BASE_MODEL"
echo "DATA_DIR: $DATA_DIR"
echo "Number of GPUs: $N_GPUS"
echo "Tensor Parallel Size: $ROLLOUT_TP_SIZE"

# Prepare training and validation files
TRAIN_FILES_STR="[\"/app/data/continual/plus/train.parquet\",\"/app/data/continual/plus_minus/train.parquet\",\"/app/data/continual/plus_minus_mul/train.parquet\",\"/app/data/continual/plus_minus_mul_div/train.parquet\"]"

VAL_FILES_STR="[\"/app/data/continual/plus/test.parquet\",\"/app/data/continual/plus_minus/test.parquet\",\"/app/data/continual/plus_minus_mul/test.parquet\",\"/app/data/continual/plus_minus_mul_div/test.parquet\"]"

echo "\nFirst 100 chars of train files list:"
echo "${TRAIN_FILES_STR:0:100}..."
echo "\nFirst 100 chars of val files list:"
echo "${VAL_FILES_STR:0:100}..."

# Prevent model downloads
export TRANSFORMERS_OFFLINE=1

# Create logs directory if it doesn't exist
mkdir -p /app/logs
chmod -R 777 /app/logs
chmod -R 777 /app/data/continual

# Set environment variables
export WANDB_MODE="disabled"
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered
export PYTHONFAULTHANDLER=1  # Enable Python fault handler for better error reporting
export PYTHONPATH=/app:$PYTHONPATH

# Run single training process for all groups
WANDB_RUN_NAME="ContinualCountdown1.5B_SingleRun"
log_file="/app/logs/${WANDB_RUN_NAME}.log"

echo "Starting training for curriculum learning"

# Print debug info
echo "Starting curriculum training at $(date)" | tee -a "$log_file"
echo "Current directory: $(pwd)" | tee -a "$log_file"
echo "Python path: $(which python3)" | tee -a "$log_file"

echo "Training configuration:" | tee -a "$log_file"
echo "  Model: $TRAINED_MODEL" | tee -a "$log_file"
echo "  Data directory: $DATA_DIR" | tee -a "$log_file"
echo "  Groups: plus -> plus_minus -> plus_minus_mul -> plus_minus_mul_div" | tee -a "$log_file"
echo "  GPUs: $N_GPUS" | tee -a "$log_file"
echo "  Rollout TP size: $ROLLOUT_TP_SIZE" | tee -a "$log_file"

python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILES_STR" \
    data.val_files="$VAL_FILES_STR" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=256 \
    data.max_response_length=1024 \
    ++data.curriculum_learning=true \
    ++data.epochs_per_group=2 \
    ++data.total_rounds=3 \
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
    +logger.print_to_console=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/app/checkpoints/ContinualCountdown1.5B/$WANDB_RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=ContinualCountdown1.5B \
    trainer.experiment_name=$WANDB_RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$TRAINED_MODEL \
    2>&1 | tee -a "$log_file"

# Check if training was successful
if ! grep -q "Final validation metrics" "$log_file"; then
    echo "Training failed"
    exit 1
fi

# Save final model checkpoint
latest_checkpoint="/app/metrics/final_single_run"
mkdir -p "$latest_checkpoint"

# Install rsync if not present
apt-get update && apt-get install -y rsync

# Copy checkpoint
echo "Copying final checkpoint to $latest_checkpoint"
rsync -av --delete "$TRAINED_MODEL/" "$latest_checkpoint/"

# Shutdown Ray and any lingering Python processes
echo "Shutting down Ray and Python processes..."
python3 -c 'import ray; ray.shutdown()'
pkill -f "ray::" || true
sleep 5

# Copy final checkpoint
echo "Copying final checkpoint to $latest_checkpoint"
rsync -av --delete "$TRAINED_MODEL/" "$latest_checkpoint/"

echo "Training completed successfully on all groups"
exit 0
