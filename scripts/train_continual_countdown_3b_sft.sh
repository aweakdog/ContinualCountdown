#!/bin/bash

# Activate conda environment
#conda activate zero

# Configuration - Set environment variables
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/cpfs04/user/liyuanhang.p/tmp/checkpoints/continual_countdown3b_sft}
export BASE_MODEL=${BASE_MODEL:-"/cpfs04/user/liyuanhang.p/model/qwen3b"}  # Path to mounted Qwen model
export N_GPUS=${N_GPUS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# Set up logging with backup
LOG_FILE="./logs/ContinualCountdown3B_SFT.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./logs/run"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_SFT_${TIMESTAMP}.log"
    chmod 644 "$BACKUP_DIR/ContinualCountdown3B_SFT_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints
rm -rf ./checkpoints/continual_countdown3b_sft
rm -rf ${CHECKPOINT_BASE_DIR}

# Clean up current log and wandb
rm -f "$LOG_FILE"
rm -rf ./wandb/*

chmod -R 755 ./logs
chmod -R 755 ./logs/run

# Set environment variables
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH=.:$PYTHONPATH

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory $BASE_MODEL does not exist"
    exit 1
fi

if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "Error: No config.json found in base model"
    exit 1
fi

# Data files for SFT
TRAIN_FILE="./data/continual/sft/0/train.parquet"
VAL_FILE="./data/continual/sft/0/test.parquet"

# Prevent model downloads
export TRANSFORMERS_OFFLINE=1

# Create logs directory if it doesn't exist
mkdir -p ./logs
chmod -R 777 ./logs
chmod -R 777 ./data/continual/sft/0

# Print debug info
echo "Starting ContinualCountdown3B SFT training at $(date)" | tee -a "$LOG_FILE"
echo "Current directory: $(pwd)" | tee -a "$LOG_FILE"
echo "Python path: $(which python3)" | tee -a "$LOG_FILE"
echo "Training configuration:" | tee -a "$LOG_FILE"
echo "  Model: $BASE_MODEL" | tee -a "$LOG_FILE"
echo "  GPUs: $N_GPUS" | tee -a "$LOG_FILE"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "  WANDB_MODE: $WANDB_MODE" | tee -a "$LOG_FILE"
echo "  NCCL_DEBUG: $NCCL_DEBUG" | tee -a "$LOG_FILE"

set -x

# ====== SFT Distributed Training Launcher (Single Node Example) ======
CONFIG_PATH="/cpfs04/user/liyuanhang.p/src/ContinualCountdown/verl/trainer/config/sft_trainer.yaml"  # Absolute path to your config
nproc_per_node=${N_GPUS:-8}  # Number of GPUs on this machine

# Print debug info
echo "Launching SFT with config: $CONFIG_PATH on $nproc_per_node GPUs" | tee -a "$LOG_FILE"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node --master_port=65536 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path /cpfs04/user/liyuanhang.p/src/ContinualCountdown/verl/trainer/config \
    --config-name sft_trainer.yaml \
    2>&1 | tee -a "$LOG_FILE"

# To run: bash scripts/train_continual_countdown_3b_sft.sh 2>&1 | tee -a scripts/log/sft_log_file.txt
