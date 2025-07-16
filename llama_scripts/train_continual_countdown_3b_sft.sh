#!/bin/bash

# Usage: bash llama_scripts/train_continual_countdown_3b_sft.sh <train_size>
# Example: bash llama_scripts/train_continual_countdown_3b_sft.sh 640

if [ -z "$1" ]; then
    echo "Error: Please provide the training size as an argument."
    echo "Usage: bash llama_scripts/train_continual_countdown_3b_sft.sh <train_size>"
    exit 1
fi

TRAIN_SIZE=$1

# Configuration - Set environment variables
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/nas/shared/sys2/yuanhangli/tmp/llama_sft_model}/${TRAIN_SIZE}
export BASE_MODEL=${BASE_MODEL:-"/cpfs04/user/liyuanhang.p/model/llama3b"}  # Path to mounted Llama model
export N_GPUS=${N_GPUS:-8}
export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# Set up logging with backup
LOG_FILE="./logs/ContinualCountdown3B_Llama_SFT_${TRAIN_SIZE}.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./logs/run"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_Llama_SFT_${TRAIN_SIZE}_${TIMESTAMP}.log"
    chmod 644 "$BACKUP_DIR/ContinualCountdown3B_Llama_SFT_${TRAIN_SIZE}_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints for this train size
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

# Data files for SFT - path constructed based on train_size
DATA_DIR="./data/continual/sft/${TRAIN_SIZE}"
TRAIN_FILE="${DATA_DIR}/train.parquet"
VAL_FILE="${DATA_DIR}/test.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training data file $TRAIN_FILE does not exist."
    echo "Please generate the data first using experiments/continual/data_gen_sft_efficient.py"
    exit 1
fi

# Prevent model downloads
export TRANSFORMERS_OFFLINE=1

# Create logs directory if it doesn't exist
mkdir -p ./logs
chmod -R 777 ./logs

# Print debug info
echo "Starting ContinualCountdown3B Llama SFT training at $(date)" | tee -a "$LOG_FILE"
echo "Current directory: $(pwd)" | tee -a "$LOG_FILE"
echo "Python path: $(which python3)" | tee -a "$LOG_FILE"
echo "Training configuration:" | tee -a "$LOG_FILE"
echo "  Model: $BASE_MODEL" | tee -a "$LOG_FILE"
echo "  Train Size: $TRAIN_SIZE" | tee -a "$LOG_FILE"
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
    model.name_or_path=$BASE_MODEL \
    model.checkpoint_path=$CHECKPOINT_BASE_DIR \
    data.train_files=[$TRAIN_FILE] \
    data.val_files=[$VAL_FILE] \
    2>&1 | tee -a "$LOG_FILE"


