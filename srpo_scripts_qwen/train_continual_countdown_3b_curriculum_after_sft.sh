#!/bin/bash

# Phase repetition control parameters
export PHASE1_REPEAT_COUNT=${PHASE1_REPEAT_COUNT:-1}  # Default: run Phase 1 once
export PHASE2_REPEAT_COUNT=${PHASE2_REPEAT_COUNT:-2}  # Default: run Phase 2 twice
export PHASE3_REPEAT_COUNT=${PHASE3_REPEAT_COUNT:-2}  # Default: run Phase 3 twice

echo "[Phase Config] Phase 1 will run $PHASE1_REPEAT_COUNT time(s)"
echo "[Phase Config] Phase 2 will run $PHASE2_REPEAT_COUNT time(s)"
echo "[Phase Config] Phase 3 will run $PHASE3_REPEAT_COUNT time(s)"

# Activate conda environment
# Use a more cautious approach to Git configuration
if ! git config --global --get-all safe.directory | grep -q "."; then
    git config --global --add safe.directory .
fi

#conda init
#conda activate zero

# Configuration - Set environment variables from docker-compose.yml if not already set
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/nas/shared/sys2/yuanhangli/tmp/checkpoints/continual_countdown3b_srpo}
SFT_CHECKPOINT=global_step_0
export BASE_MODEL=${BASE_MODEL:-"/nas/shared/sys2/yuanhangli/tmp/qwen_sft_model/${SFT_CHECKPOINT}"}  # Path to mounted Qwen model
export N_GPUS=${N_GPUS:-8}  # Using all 8 GPUs for training
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}  # Tensor parallel size optimized for 4 GPUs
export WANDB_MODE=${WANDB_MODE:-offline}  # Run WandB in offline mode
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}  # Use all 8 GPUs for training
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# SRPO-specific parameters
export SRPO_N_SAMPLES=${SRPO_N_SAMPLES:-32}  # Number of samples per prompt for SRPO
export SRPO_KL_COEF=${SRPO_KL_COEF:-0.001}  # KL loss coefficient for SRPO
export SRPO_BETA=${SRPO_BETA:-0.01}  # Beta coefficient for CPR computation
export SRPO_BASELINE_TYPE=${SRPO_BASELINE_TYPE:-mean}  # Baseline type for MSA (mean/max/median)
export SRPO_USE_WHITENING=${SRPO_USE_WHITENING:-true}  # Enable whitening for outcome+MSA (recommended by SRPO authors)

# GPU Resource Allocation Configuration for 8 GPU setup
# Training: All 8 GPUs (0-7), Analyzers: Disabled
export RAY_ANALYZER_GPU_START=${RAY_ANALYZER_GPU_START:-8}  # Disable analyzers by setting start beyond available GPUs
export RAY_ANALYZER_GPU_COUNT=${RAY_ANALYZER_GPU_COUNT:-0}  # Disable analyzers
echo "[GPU Config] Training will use all 8 GPUs (0-7), Analyzers disabled"
echo "[SRPO Config] Using $SRPO_N_SAMPLES samples per prompt, KL coefficient: $SRPO_KL_COEF, Beta: $SRPO_BETA, Baseline: $SRPO_BASELINE_TYPE, Whitening: $SRPO_USE_WHITENING"

# Set up logging with backup
LOG_FILE="./srpo_logs/ContinualCountdown3B_SRPO_SingleRun.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./srpo_logs/run"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_SRPO_SingleRun_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints
rm -rf ./checkpoints/continual_countdown3b_srpo
rm -rf ${CHECKPOINT_BASE_DIR}

# Create all required directories first
mkdir -p ./srpo_logs
mkdir -p ./srpo_logs/run
mkdir -p ./checkpoints/continual_countdown3b_srpo
mkdir -p ${CHECKPOINT_BASE_DIR}

# Handle log backup and cleanup
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_GRPO_SingleRun_${TIMESTAMP}.log"
    chmod 644 "$BACKUP_DIR/ContinualCountdown3B_GRPO_SingleRun_${TIMESTAMP}.log"
fi

# Clean up current log and wandb
rm -f "$LOG_FILE"
rm -rf ./wandb/*
chmod -R 755 ./grpo_logs
chmod -R 755 ./grpo_logs/run

# Set FSDP gradient metric flag (disabled for performance when not using analyzers)
export FSDP_GRAD_METRIC_ENABLED=false
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
WANDB_RUN_NAME="ContinualCountdown3B_SRPO_SingleRun"
log_file="./srpo_logs/${WANDB_RUN_NAME}.log"

# Print debug info
echo "Starting ContinualCountdown3B SRPO training at $(date)" | tee -a "$log_file"
echo "Current directory: $(pwd)" | tee -a "$log_file"
echo "Python path: $(which python3)" | tee -a "$log_file"

echo "SRPO Training configuration:" | tee -a "$log_file"
echo "  Base model: $BASE_MODEL" | tee -a "$log_file"
echo "  N_GPUS: $N_GPUS" | tee -a "$log_file"
echo "  ROLLOUT_TP_SIZE: $ROLLOUT_TP_SIZE" | tee -a "$log_file"
echo "  SRPO N_SAMPLES: $SRPO_N_SAMPLES" | tee -a "$log_file"
echo "  SRPO KL_COEF: $SRPO_KL_COEF" | tee -a "$log_file"
echo "  SRPO BETA: $SRPO_BETA" | tee -a "$log_file"
echo "  SRPO BASELINE_TYPE: $SRPO_BASELINE_TYPE" | tee -a "$log_file"
echo "  WANDB_MODE: $WANDB_MODE" | tee -a "$log_file"
echo "  CHECKPOINT_BASE_DIR: $CHECKPOINT_BASE_DIR" | tee -a "$log_file"

# Set up experiment logging
MASTER_LOG_FILE="./srpo_logs/ContinualCountdown3B_SRPO_MasterLog.log"
EXP_LOG_DIR="./srpo_logs/experiments"
mkdir -p "$EXP_LOG_DIR"

echo "=== Starting ContinualCountdown3B SRPO Curriculum Training ===" | tee -a "$MASTER_LOG_FILE"
echo "Training started at: $(date)" | tee -a "$MASTER_LOG_FILE"
echo "Base model: $BASE_MODEL" | tee -a "$MASTER_LOG_FILE"
echo "Algorithm: SRPO (Step-wise Relative Policy Optimization)" | tee -a "$MASTER_LOG_FILE"

# Phase 1: Train group0 (repeat $PHASE1_REPEAT_COUNT times)
echo "=== PHASE 1: Training Group 0 (${PHASE1_REPEAT_COUNT} repetition(s)) ===" | tee -a "$MASTER_LOG_FILE"
for ((phase1_iter=1; phase1_iter<=PHASE1_REPEAT_COUNT; phase1_iter++)); do
  echo "--- Phase 1 Iteration $phase1_iter/$PHASE1_REPEAT_COUNT ---"
  group=0
  TRAIN_FILES_STR="[\"./data/continual/${group}/train.parquet\"]"
  VAL_FILES_STR="[\"./data/continual/${group}/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Phase1_Group${group}_Iter${phase1_iter}_SRPO_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training group $group with SRPO (iteration $phase1_iter)" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=srpo \
  +algorithm.srpo_beta=$SRPO_BETA \
  +algorithm.srpo_baseline_type=$SRPO_BASELINE_TYPE \
  +algorithm.srpo_use_whitening=$SRPO_USE_WHITENING \
  ++fsdp_grad_metric_enabled=false \
  data.train_files="$TRAIN_FILES_STR" \
  data.val_files="$VAL_FILES_STR" \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_response_length=1024 \
  ++data.curriculum_learning=true \
  ++data.epochs_per_group=15 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$SRPO_KL_COEF \
  +actor_rollout_ref.actor.kl_coeff=$SRPO_KL_COEF \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  critic.ppo_mini_batch_size=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.n=$SRPO_N_SAMPLES \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.model.enable_gradient_checkpointing=True \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  +critic.model.trust_remote_code=true \
  +critic.model.torch_dtype=bfloat16 \
  +critic.model.device_map=auto \
  +critic.model.attn_implementation=flash_attention_2 \
  +critic.model.use_cache=false \
  critic.ppo_micro_batch_size=8 \
  ++fsdp_grad_metric_enabled=false \
  +actor_rollout_ref.actor.fsdp_grad_metric_enabled=false \
  +actor_rollout_ref.actor.fisher_analysis_enabled=false \
  +actor_rollout_ref.actor.fsdp_component_analysis.run_fisher_info_analysis=false \
  actor_rollout_ref.actor.redo_enabled=false \
  actor_rollout_ref.actor.redo_metric_freq=0 \
  actor_rollout_ref.actor.redo_reset_freq=0 \
  algorithm.kl_ctrl.kl_coef=$SRPO_KL_COEF \
  trainer.logger=['wandb','console'] \
  +logger.print_to_console=true \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir=${CHECKPOINT_BASE_DIR} \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=1200 \
  trainer.test_freq=30 \
  trainer.project_name=ContinualCountdown3B_GRPO \
  trainer.experiment_name=$RUN_NAME \
  hydra.verbose=true \
  trainer.total_epochs=1 \
  +trainer.val_before_train=true \
  ++reward_model.enable=False \
  ++reward_model.model.path=$BASE_MODEL \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Phase 1 Iteration $phase1_iter completed"
done
echo "Phase 1 completed after $PHASE1_REPEAT_COUNT iteration(s)"