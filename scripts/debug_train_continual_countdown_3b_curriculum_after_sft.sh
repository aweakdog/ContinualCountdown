#!/bin/bash

# Activate conda environment
# Use a more cautious approach to Git configuration
if ! git config --global --get-all safe.directory | grep -q "."; then
    git config --global --add safe.directory .
fi

#conda init
#conda activate zero

# Configuration - Set environment variables from docker-compose.yml if not already set
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/cpfs04/user/liyuanhang.p/tmp/checkpoints/continual_countdown3b}
SFT_CHECKPOINT=global_step_15
export BASE_MODEL=${BASE_MODEL:-"/cpfs04/user/liyuanhang.p/tmp/sft_model/${SFT_CHECKPOINT}"}  # Path to mounted Qwen model
export N_GPUS=${N_GPUS:-8}  # Using 4 A800 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}  # Tensor parallel size optimized for 4 GPUs
export WANDB_MODE=${WANDB_MODE:-offline}  # Run WandB in offline mode
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}


# Set up logging with backup
LOG_FILE="./logs/ContinualCountdown3B_SingleRun.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./logs/run"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_SingleRun_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints
rm -rf ./checkpoints/continual_countdown3b
rm -rf ${CHECKPOINT_BASE_DIR}

# Create all required directories first

# Handle log backup and cleanup
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_SingleRun_${TIMESTAMP}.log"
    chmod 644 "$BACKUP_DIR/ContinualCountdown3B_SingleRun_${TIMESTAMP}.log"
fi

# Clean up current log and wandb
rm -f "$LOG_FILE"
rm -rf ./wandb/*
#chmod -R 755 ./checkpoints/continual_countdown3b
chmod -R 755 ./logs
chmod -R 755 ./logs/run

# Set FSDP gradient metric flag (set to true to enable FSDP gradient metrics)
export FSDP_GRAD_METRIC_ENABLED=true
# Set environment variables
export WANDB_MODE=${WANDB_MODE:-"disabled"}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH=.:$PYTHONPATH

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

# Run single training process
WANDB_RUN_NAME="ContinualCountdown3B_SingleRun"
log_file="./logs/${WANDB_RUN_NAME}.log"

# Print debug info
echo "Starting ContinualCountdown3B training at $(date)" | tee -a "$log_file"
echo "Current directory: $(pwd)" | tee -a "$log_file"
echo "Python path: $(which python3)" | tee -a "$log_file"

echo "Training configuration:" | tee -a "$log_file"
echo "  Model: $TRAINED_MODEL" | tee -a "$log_file"
echo "  GPUs: $N_GPUS" | tee -a "$log_file"

# Create a unique subdirectory for this experiment's logs
EXP_LOG_DIR=./logs/debug_continual_countdown3b_sft_${SFT_CHECKPOINT}
mkdir -p "$EXP_LOG_DIR"
cp tmp/monitor_master.sh "$EXP_LOG_DIR/"
MASTER_LOG_FILE="$EXP_LOG_DIR/experiment_master.log"
# Remove previous master log if it exists
if [ -f "$MASTER_LOG_FILE" ]; then
  rm "$MASTER_LOG_FILE"
fi

# Loop over each group and record logs in the experiment log directory
for group in 1; do
  TRAIN_FILES_STR="[\"./data/continual/${group}/train.parquet\"]"
  VAL_FILES_STR="[\"./data/continual/${group}/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Group${group}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training group $group" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  
  python3 -m verl.trainer.main_ppo \
    fsdp_grad_metric_enabled=$FSDP_GRAD_METRIC_ENABLED \
    data.train_files="$TRAIN_FILES_STR" \
    data.val_files="$VAL_FILES_STR" \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_response_length=1024 \
    ++data.curriculum_learning=true \
    ++data.epochs_per_group=30 \
    ++data.total_rounds=1 \
    ++data.train_sample_size="$TRAIN_SAMPLE_SIZE" \
    actor_rollout_ref.model.path=$BASE_MODEL \
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
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=1e-5 \
    critic.model.path=$BASE_MODEL \
    +critic.model.trust_remote_code=true \
    +critic.model.torch_dtype=bfloat16 \
    +critic.model.device_map=auto \
    +critic.model.attn_implementation=flash_attention_2 \
    +critic.model.use_cache=false \
    critic.ppo_micro_batch_size=4 \
    ++actor_rollout_ref.actor.redo_enabled=true \
    ++actor_rollout_ref.actor.redo_metric_freq=1 \
    ++actor_rollout_ref.actor.redo_reset_freq=1 \
    ++actor_rollout_ref.actor.redo_mode=threshold \
    ++actor_rollout_ref.actor.redo_tau=0.1 \
    ++critic.redo_enabled=true \
    ++critic.redo_metric_freq=1 \
    ++critic.redo_reset_freq=1 \
    ++critic.redo_mode=threshold \
    ++critic.redo_tau=0.1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb','console'] \
    +logger.print_to_console=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${CHECKPOINT_BASE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1200 \
    trainer.test_freq=30 \
    trainer.project_name=ContinualCountdown3B \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$BASE_MODEL \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
done
