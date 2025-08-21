#!/bin/bash

SFT_CHECKPOINT=global_step_0

# Phase repetition control parameters
export PHASE1_REPEAT_COUNT=${PHASE1_REPEAT_COUNT:-1}  # Default: run Phase 1 once
export PHASE2_REPEAT_COUNT=${PHASE2_REPEAT_COUNT:-0}  # Default: run Phase 2 twice
export PHASE3_REPEAT_COUNT=${PHASE3_REPEAT_COUNT:-0}  # Default: run Phase 3 twice

echo "[Phase Config] Phase 1 will run $PHASE1_REPEAT_COUNT time(s)"
echo "[Phase Config] Phase 2 will run $PHASE2_REPEAT_COUNT time(s)"
echo "[Phase Config] Phase 3 will run $PHASE3_REPEAT_COUNT time(s)"

# Activate conda environment
# Use a more cautious approach to Git configuration
if ! git config --global --get-all safe.directory | grep -q "."; then
    git config --global --add safe.directory .
fi

# Configuration - Set environment variables
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/nas/shared/sys2/yuanhangli/tmp/checkpoints/continual_countdown3b_qwen_curriculum}
export BASE_MODEL=${BASE_MODEL:-"/nas/shared/sys2/yuanhangli/tmp/qwen_sft_model/${SFT_CHECKPOINT}"}  # Path to mounted Qwen SFT model
export N_GPUS=${N_GPUS:-4}  # Using 8 A100 GPUs
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}  # Tensor parallel size optimized for 8 GPUs
export WANDB_MODE=${WANDB_MODE:-offline}  # Run WandB in offline mode
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}  # Limit training to GPUs 0-3, reserve 4-7 for analyzers
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# GPU Resource Allocation Configuration for 8 A100 setup
# Training: GPUs 0-3, Analyzers: GPUs 4-7
export RAY_ANALYZER_GPU_START=${RAY_ANALYZER_GPU_START:-4}  # Start analyzer GPUs from GPU 4
export RAY_ANALYZER_GPU_COUNT=${RAY_ANALYZER_GPU_COUNT:-4}  # Use 4 GPUs for analyzers (4-7)
echo "[GPU Config] Training will use GPUs 0-3, Analyzers will use GPUs 4-7"

# Layer Reset Configuration (OPTIONAL - defaults to disabled)
# Uncomment and modify the following lines to enable layer reset functionality:
export LAYER_RESET_ENABLE=${LAYER_RESET_ENABLE:-true}
export LAYER_RESET_K_FIRST=${LAYER_RESET_K_FIRST:-0}    # Reset first 4 transformer layers
export LAYER_RESET_K_LAST=${LAYER_RESET_K_LAST:-20}      # Reset last 2 transformer layers
export LAYER_RESET_STEPS=${LAYER_RESET_STEPS:-"[40,80,120,150]"}  # Reset at global steps 120 and 200

# Set up logging with backup
LOG_FILE="./qwen_logs/ContinualCountdown3B_Qwen_Curriculum.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./qwen_logs/run"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_Qwen_Curriculum_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints
rm -rf ${CHECKPOINT_BASE_DIR}

# Clean up current log and wandb
rm -f "$LOG_FILE"
rm -rf ./wandb/*
chmod -R 755 ./qwen_logs
chmod -R 755 ./qwen_logs/run

# Set FSDP gradient metric flag (set to true to enable FSDP gradient metrics)
export FSDP_GRAD_METRIC_ENABLED=true
# Set environment variables
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

# Create a unique subdirectory for this experiment's qwen_logs
EXP_LOG_DIR=./qwen_logs/debug_countdown3b_qwen_sft_${SFT_CHECKPOINT}_reset_k${LAYER_RESET_K_FIRST}f_k${LAYER_RESET_K_LAST}l
mkdir -p "$EXP_LOG_DIR"
cp tmp/monitor_master.sh "$EXP_LOG_DIR/"
MASTER_LOG_FILE="$EXP_LOG_DIR/experiment_master.log"
# Remove previous master log if it exists
if [ -f "$MASTER_LOG_FILE" ]; then
  rm "$MASTER_LOG_FILE"
fi

# Prepare layer reset configuration if enabled
LAYER_RESET_CONFIG=""
echo "[LAYER_RESET_DEBUG] Environment variables:"
echo "[LAYER_RESET_DEBUG]   LAYER_RESET_ENABLE=${LAYER_RESET_ENABLE:-false}"
echo "[LAYER_RESET_DEBUG]   LAYER_RESET_K_FIRST=${LAYER_RESET_K_FIRST:-4}"
echo "[LAYER_RESET_DEBUG]   LAYER_RESET_K_LAST=${LAYER_RESET_K_LAST:-2}"
echo "[LAYER_RESET_DEBUG]   LAYER_RESET_STEPS=${LAYER_RESET_STEPS:-[2,30]}"

if [ "${LAYER_RESET_ENABLE:-false}" = "true" ]; then
    echo "[Layer Reset] Layer reset enabled with k_first=${LAYER_RESET_K_FIRST:-4}, k_last=${LAYER_RESET_K_LAST:-2}, steps=${LAYER_RESET_STEPS:-[120,200]}"
    LAYER_RESET_CONFIG="++layer_reset.enable_reset=${LAYER_RESET_ENABLE:-true} ++layer_reset.reset_k_first=${LAYER_RESET_K_FIRST:-4} ++layer_reset.reset_k_last=${LAYER_RESET_K_LAST:-2} ++layer_reset.reset_steps=\"${LAYER_RESET_STEPS:-[120,200]}\""
    echo "[LAYER_RESET_DEBUG] Generated config: $LAYER_RESET_CONFIG"
else
    echo "[Layer Reset] Layer reset disabled (default)"
    echo "[LAYER_RESET_DEBUG] Layer reset is disabled because LAYER_RESET_ENABLE != 'true'"
fi

# Three-phase training: Phase 1 (group0 only), Phase 2 (group1 + group2 in same session), Phase 3 (group2 only)

# Phase 1: Train only group0 (repeat $PHASE1_REPEAT_COUNT times)
echo "=== PHASE 1: Training Group 0 (${PHASE1_REPEAT_COUNT} repetition(s)) ==="
for ((phase1_iter=1; phase1_iter<=PHASE1_REPEAT_COUNT; phase1_iter++)); do
  echo "--- Phase 1 Iteration $phase1_iter/$PHASE1_REPEAT_COUNT ---"
  group=0
  TRAIN_FILES_STR="[\"./data/continual/${group}/train.parquet\"]"
  VAL_FILES_STR="[\"./data/continual/${group}/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Phase1_Group${group}_Iter${phase1_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training group $group (iteration $phase1_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  
  python3 -m verl.trainer.main_ppo \
    fsdp_grad_metric_enabled=$FSDP_GRAD_METRIC_ENABLED \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
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
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size=8 \
    ++actor_rollout_ref.actor.redo_tau=0.1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb','console'] \
    +logger.print_to_console=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${CHECKPOINT_BASE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1200 \
    trainer.test_freq=30 \
    trainer.project_name=ContinualCountdown3B_Qwen \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$BASE_MODEL \
    $LAYER_RESET_CONFIG \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Phase 1 Iteration $phase1_iter completed"
done
echo "Phase 1 completed after $PHASE1_REPEAT_COUNT iteration(s)"

# Phase 2: Train group1 and group2 in the same training session (repeat $PHASE2_REPEAT_COUNT times)
echo "=== PHASE 2: Training Group 1 + Group 2 (${PHASE2_REPEAT_COUNT} repetition(s)) ==="
for ((phase2_iter=1; phase2_iter<=PHASE2_REPEAT_COUNT; phase2_iter++)); do
  echo "--- Phase 2 Iteration $phase2_iter/$PHASE2_REPEAT_COUNT ---"
  TRAIN_FILES_STR="[\"./data/continual/1/train.parquet\", \"./data/continual/2/train.parquet\"]"
  VAL_FILES_STR="[\"./data/continual/1/test.parquet\", \"./data/continual/2/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560, 2560]"  # Sample sizes for group1 and group2
  RUN_NAME="Phase2_Group1and2_Iter${phase2_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training groups 1 and 2 sequentially (iteration $phase2_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

python3 -m verl.trainer.main_ppo \
  fsdp_grad_metric_enabled=$FSDP_GRAD_METRIC_ENABLED \
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
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=false \
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
  critic.ppo_mini_batch_size=32 \
  critic.ppo_micro_batch_size=8 \
  ++actor_rollout_ref.actor.redo_tau=0.1 \
  ++actor_rollout_ref.actor.enable_gradient_analysis=true \
  ++actor_rollout_ref.actor.gradient_analysis_freq=1 \
  ++actor_rollout_ref.actor.enable_fisher_analysis=true \
  ++actor_rollout_ref.actor.fisher_analysis_freq=1 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['wandb','console'] \
  +logger.print_to_console=true \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir=${CHECKPOINT_BASE_DIR} \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=1200 \
  trainer.test_freq=30 \
  trainer.project_name=ContinualCountdown3B_Qwen \
  trainer.experiment_name=$RUN_NAME \
  trainer.total_epochs=1 \
  +trainer.val_before_train=true \
  ++reward_model.enable=False \
  ++reward_model.model.path=$BASE_MODEL \
  $LAYER_RESET_CONFIG \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Phase 2 Iteration $phase2_iter completed"
done
echo "Phase 2 completed after $PHASE2_REPEAT_COUNT iteration(s)"

# Phase 3: Train only group2 (repeat $PHASE3_REPEAT_COUNT times)
echo "=== PHASE 3: Training Group 2 (${PHASE3_REPEAT_COUNT} repetition(s)) ==="
for ((phase3_iter=1; phase3_iter<=PHASE3_REPEAT_COUNT; phase3_iter++)); do
  echo "--- Phase 3 Iteration $phase3_iter/$PHASE3_REPEAT_COUNT ---"
  group=2
  TRAIN_FILES_STR="[\"./data/continual/${group}/train.parquet\"]"
  VAL_FILES_STR="[\"./data/continual/${group}/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Phase3_Group${group}_Iter${phase3_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training group $group (iteration $phase3_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

  python3 -m verl.trainer.main_ppo \
    fsdp_grad_metric_enabled=$FSDP_GRAD_METRIC_ENABLED \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
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
    critic.ppo_mini_batch_size=32 \
    critic.ppo_micro_batch_size=8 \
    ++actor_rollout_ref.actor.redo_tau=0.1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['wandb','console'] \
    +logger.print_to_console=true \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${CHECKPOINT_BASE_DIR} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1200 \
    trainer.test_freq=30 \
    trainer.project_name=ContinualCountdown3B_Qwen \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$BASE_MODEL \
    $LAYER_RESET_CONFIG \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Phase 3 Iteration $phase3_iter completed"
done
echo "Phase 3 completed after $PHASE3_REPEAT_COUNT iteration(s)"

echo "=== ALL PHASES COMPLETED ==="
echo "Phase 1: $PHASE1_REPEAT_COUNT iteration(s)"
echo "Phase 2: $PHASE2_REPEAT_COUNT iteration(s)"
echo "Phase 3: $PHASE3_REPEAT_COUNT iteration(s)"
echo "Total iterations: $((PHASE1_REPEAT_COUNT + PHASE2_REPEAT_COUNT + PHASE3_REPEAT_COUNT))"
echo "Training completed successfully!"
