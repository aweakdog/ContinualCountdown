#!/bin/bash

# Enhanced SFT Training Script for Llama 3.2 3B Model
# Usage: ./train_continual_countdown_3b_sft_llama.sh [SFT_SIZE] [--gpus NUM_GPUS] [--model MODEL_PATH] [--wandb MODE]
# Example: ./train_continual_countdown_3b_sft_llama.sh 2048 --gpus 8 --wandb online

set -e

# Function to display usage
show_usage() {
    echo "Usage: $0 [SFT_SIZE] [--gpus NUM_GPUS] [--model MODEL_PATH] [--wandb MODE]"
    echo ""
    echo "Arguments:"
    echo "  SFT_SIZE       Number of SFT training samples (default: 2048)"
    echo "  --gpus         Number of GPUs to use (default: 8)"
    echo "  --model        Path to base model (default: ./models/llama_base3b)"
    echo "  --wandb        Wandb mode: online, offline, or disabled (default: disabled)"
    echo ""
    echo "Examples:"
    echo "  $0 1024                                    # Train with 1024 samples"
    echo "  $0 4096 --gpus 4                         # Train with 4096 samples on 4 GPUs"
    echo "  $0 2048 --wandb online                   # Train with wandb logging"
    echo "  $0 8192 --model /path/to/model --gpus 8  # Custom model path"
    echo ""
    exit 1
}

# Parse command line arguments
SFT_SIZE=${1:-2048}
NUM_GPUS=8
BASE_MODEL="./models/llama_base3b"
WANDB_MODE="disabled"

# Parse optional arguments
if [ $# -gt 0 ]; then
    shift
fi
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
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
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Validate SFT_SIZE
if ! [[ "$SFT_SIZE" =~ ^[0-9]+$ ]] || [ "$SFT_SIZE" -lt 128 ]; then
    echo "Error: SFT_SIZE must be a positive integer >= 128"
    exit 1
fi

# Validate NUM_GPUS
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "Error: NUM_GPUS must be between 1 and 8"
    exit 1
fi

# Validate WANDB_MODE
if [[ "$WANDB_MODE" != "online" && "$WANDB_MODE" != "offline" && "$WANDB_MODE" != "disabled" ]]; then
    echo "Error: WANDB_MODE must be one of: online, offline, disabled"
    exit 1
fi

echo "=== Llama 3.2 3B SFT Training Configuration ==="
echo "SFT Size: $SFT_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "Base Model: $BASE_MODEL"
echo "Wandb Mode: $WANDB_MODE"
echo "=============================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export WANDB_MODE=$WANDB_MODE

# Activate conda environment (uncomment if needed)
# conda activate countdown

# Set up logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs/sft_llama"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sft_llama_${SFT_SIZE}_${TIMESTAMP}.log"

echo "Starting Llama 3.2 3B SFT training..." | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Validate base model path
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found: $BASE_MODEL" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "Error: No config.json found in base model" | tee -a "$LOG_FILE"
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

print(f'Generating SFT data for Llama: train_size={$SFT_SIZE}, test_size={test_size}')
generator = SFTDataGenerator(model_type='llama')  # Use llama template
generator.generate_group_data(train_size=$SFT_SIZE, test_size=test_size)
print('SFT data generation completed!')
" 2>&1 | tee -a "$LOG_FILE"

if [ $? -ne 0 ]; then
    echo "Error: SFT data generation failed" | tee -a "$LOG_FILE"
    exit 1
fi

# Check if data files exist
TRAIN_DATA="./data/continual/sft/0/train.parquet"
TEST_DATA="./data/continual/sft/0/test.parquet"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data file not found: $TRAIN_DATA" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data file not found: $TEST_DATA" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Data files verified successfully" | tee -a "$LOG_FILE"
echo "Training data: $TRAIN_DATA" | tee -a "$LOG_FILE"
echo "Test data: $TEST_DATA" | tee -a "$LOG_FILE"

# Verify config file exists
CONFIG_PATH="./verl/trainer/config/sft_llama_base_trainer.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Using config file: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "Base model configured in config: $BASE_MODEL" | tee -a "$LOG_FILE"

# Start SFT training
echo "Starting SFT training with $NUM_GPUS GPUs..." | tee -a "$LOG_FILE"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=29500 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path $(pwd)/verl/trainer/config \
    --config-name sft_llama_base_trainer \
    2>&1 | tee -a "$LOG_FILE"

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "SFT training completed successfully!" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    
    # Display training summary
    echo "" | tee -a "$LOG_FILE"
    echo "=== Training Summary ===" | tee -a "$LOG_FILE"
    echo "Model: Llama 3.2 3B" | tee -a "$LOG_FILE"
    echo "Base Model Path: $BASE_MODEL" | tee -a "$LOG_FILE"
    echo "SFT Training Size: $SFT_SIZE samples" | tee -a "$LOG_FILE"
    TEST_SIZE_CALC=$((SFT_SIZE / 4))
    TEST_SIZE_FINAL=$((TEST_SIZE_CALC > 128 ? TEST_SIZE_CALC : 128))
    echo "Test Size: $TEST_SIZE_FINAL samples" | tee -a "$LOG_FILE"
    echo "GPUs Used: $NUM_GPUS" | tee -a "$LOG_FILE"
    echo "Wandb Mode: $WANDB_MODE" | tee -a "$LOG_FILE"
    echo "Training Data: $TRAIN_DATA" | tee -a "$LOG_FILE"
    echo "Test Data: $TEST_DATA" | tee -a "$LOG_FILE"
    echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "======================" | tee -a "$LOG_FILE"
else
    echo "SFT training failed with exit code: $TRAINING_EXIT_CODE" | tee -a "$LOG_FILE"
    exit $TRAINING_EXIT_CODE
fi
