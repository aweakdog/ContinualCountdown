#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [SFT_SIZE] [OPTIONS]"
    echo "  SFT_SIZE: Number of training samples (default: 2048)"
    echo "  OPTIONS:"
    echo "    --gpus N          Number of GPUs to use (default: 8)"
    echo "    --model PATH      Path to base model (default: /cpfs04/user/liyuanhang.p/model/qwen3b)"
    echo "    --wandb MODE      WANDB mode: online/offline (default: offline)"
    echo "    --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 2048                    # Train with 2048 samples"
    echo "  $0 4096 --gpus 4          # Train with 4096 samples on 4 GPUs"
    echo "  $0 1024 --wandb online    # Train with 1024 samples with online wandb"
    exit 1
}

# Parse command line arguments
SFT_SIZE=${1:-2048}  # Default to 2048 if not specified
shift  # Remove first argument

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            N_GPUS="$2"
            shift 2
            ;;
        --model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --wandb)
            WANDB_MODE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate SFT_SIZE is a number
if ! [[ "$SFT_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: SFT_SIZE must be a positive integer, got: $SFT_SIZE"
    usage
fi

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

# Generate SFT data with specified size
echo "Generating SFT data with $SFT_SIZE training samples..." | tee -a "$LOG_FILE"
python -c "
import sys
sys.path.append('.')
from experiments.continual.data_gen_sft_efficient import SFTDataGenerator

# Calculate test size as 25% of train size, minimum 128
test_size = max(128, int($SFT_SIZE * 0.25))

print(f'Generating SFT data: train_size={$SFT_SIZE}, test_size={test_size}')
generator = SFTDataGenerator()
generator.generate_group_data(train_size=$SFT_SIZE, test_size=test_size)
print('SFT data generation completed!')
" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "Error: SFT data generation failed" | tee -a "$LOG_FILE"
    exit 1
fi

# Data files for SFT
TRAIN_FILE="./data/continual/sft/0/train.parquet"
VAL_FILE="./data/continual/sft/0/test.parquet"

# Verify data files exist
if [ ! -f "$TRAIN_FILE" ]; then
    echo "Error: Training data file $TRAIN_FILE not found" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$VAL_FILE" ]; then
    echo "Error: Validation data file $VAL_FILE not found" | tee -a "$LOG_FILE"
    exit 1
fi

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
echo "  SFT Size: $SFT_SIZE training samples" | tee -a "$LOG_FILE"
echo "  Model: $BASE_MODEL" | tee -a "$LOG_FILE"
echo "  GPUs: $N_GPUS" | tee -a "$LOG_FILE"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "  WANDB_MODE: $WANDB_MODE" | tee -a "$LOG_FILE"
echo "  NCCL_DEBUG: $NCCL_DEBUG" | tee -a "$LOG_FILE"
echo "  Training data: $TRAIN_FILE" | tee -a "$LOG_FILE"
echo "  Validation data: $VAL_FILE" | tee -a "$LOG_FILE"

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

# Usage examples:
# bash scripts/train_continual_countdown_3b_sft.sh 2048                    # Train with 2048 samples
# bash scripts/train_continual_countdown_3b_sft.sh 4096 --gpus 4          # Train with 4096 samples on 4 GPUs
# bash scripts/train_continual_countdown_3b_sft.sh 1024 --wandb online    # Train with 1024 samples with online wandb
# bash scripts/train_continual_countdown_3b_sft.sh --help                 # Show help message
