#!/bin/bash

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate zero

# Configuration
export BASE_MODEL=${BASE_MODEL:-"/home/yliog/model/qwen1.5b/snapshots/8faed761d45a263340a0528343f099c05c9a4323"}  # Qwen 1.5B model path
export N_GPUS=8  # Using all 8 3090 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}  # Tensor parallel size
export DATA_DIR="/data/countdown/continual"  # Match memory data structure
export TRAINED_MODEL="/app/models/countdown_continual_1.5b"  # Model being continually trained

# Training configuration
ROUNDS=3
GROUPS=("plus" "plus_minus" "plus_minus_mul" "plus_minus_mul_div")

# Create necessary directories
mkdir -p metrics logs wandb plots
for group in "${GROUPS[@]}"; do
    mkdir -p "$DATA_DIR/$group"
done

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
echo "Training Rounds: $ROUNDS"
echo "Operator Groups: ${GROUPS[*]}"

# Prepare training and validation file lists in specific order to maintain optimizer state
TRAIN_FILES="$DATA_DIR/plus/train.parquet,$DATA_DIR/plus_minus/train.parquet,$DATA_DIR/plus_minus_mul/train.parquet,$DATA_DIR/plus_minus_mul_div/train.parquet"
VAL_FILES="$DATA_DIR/plus/test.parquet,$DATA_DIR/plus_minus/test.parquet,$DATA_DIR/plus_minus_mul/test.parquet,$DATA_DIR/plus_minus_mul_div/test.parquet"

echo "Training files order (this order guarantees progression through operator groups):"
echo "1. $DATA_DIR/plus/train.parquet"
echo "2. $DATA_DIR/plus_minus/train.parquet"
echo "3. $DATA_DIR/plus_minus_mul/train.parquet"
echo "4. $DATA_DIR/plus_minus_mul_div/train.parquet"

echo "Training files: $TRAIN_FILES"
echo "Validation files: $VAL_FILES"

# Prevent model downloads
export TRANSFORMERS_OFFLINE=1

# Run single training process for all groups
WANDB_RUN_NAME="ContinualCountdown1.5B_SingleRun"
log_file="logs/${WANDB_RUN_NAME}.log"

python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
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
    trainer.total_epochs=400 \
    +trainer.val_before_train=True \
    ++reward_model.enable=False \
    ++reward_model.model.path=$TRAINED_MODEL \
    # Enable curriculum learning to process files in order
    data.curriculum_learning=True \
    2>&1 | tee "$log_file"

# Check if training was successful
if ! grep -q "Final validation metrics" "$log_file"; then
    echo "Training failed"
    exit 1
fi

# Save final model checkpoint
latest_checkpoint="metrics/final_single_run"
mkdir -p "$latest_checkpoint"

# Shutdown Ray and any lingering Python processes
echo "Shutting down Ray and Python processes before copying checkpoint..."
python3 -c 'import ray; ray.shutdown()'
pkill -f "ray::" || true
sleep 5

# Copy final checkpoint
echo "Copying final checkpoint to $latest_checkpoint"
rsync -av --delete "$TRAINED_MODEL/" "$latest_checkpoint/"

echo "Training completed successfully on all groups"
exit 0
