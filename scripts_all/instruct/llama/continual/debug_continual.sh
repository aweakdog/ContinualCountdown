#!/bin/bash

# Parse command line arguments
EXPERIMENT_TYPE=${1:-debug}  # Default to 'train' if no argument provided

# Model configuration parameters
SFT_MODEL_BASE_DIR=${SFT_MODEL_BASE_DIR:-"./models"}
SFT_MODEL_NAME=${SFT_MODEL_NAME:-"llama_instruct_sft_models"}
SFT_CHECKPOINT=${SFT_CHECKPOINT:-"global_step_0"}

echo "[Experiment Type] Using type: $EXPERIMENT_TYPE"
echo "[Model Config] SFT model base directory: $SFT_MODEL_BASE_DIR"
echo "[Model Config] SFT model name: $SFT_MODEL_NAME"
echo "[Model Config] SFT checkpoint: $SFT_CHECKPOINT"

# Experiment repetition control parameters
export EXP1_REPEAT_COUNT=${EXP1_REPEAT_COUNT:-0}  # Default: run Experiment 1 once
export EXP2_REPEAT_COUNT=${EXP2_REPEAT_COUNT:-2}  # Default: run Experiment 2 twice
export EXP3_REPEAT_COUNT=${EXP3_REPEAT_COUNT:-0}  # Default: run Experiment 3 twice

echo "[Experiment Config] Experiment 1 will run $EXP1_REPEAT_COUNT time(s)"
echo "[Experiment Config] Experiment 2 will run $EXP2_REPEAT_COUNT time(s)"
echo "[Experiment Config] Experiment 3 will run $EXP3_REPEAT_COUNT time(s)"

# Activate conda environment
# Use a more cautious approach to Git configuration
if ! git config --global --get-all safe.directory | grep -q "."; then
    git config --global --add safe.directory .
fi

# Configuration - Set environment variables
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-./models/checkpoints/llama_instruct/continual_countdown3b_llama_curriculum}
# Construct BASE_MODEL path - handle empty SFT_MODEL_NAME
if [ -z "$SFT_MODEL_NAME" ]; then
    export BASE_MODEL=${BASE_MODEL:-"${SFT_MODEL_BASE_DIR}/${SFT_CHECKPOINT}"}
else
    export BASE_MODEL=${BASE_MODEL:-"${SFT_MODEL_BASE_DIR}/${SFT_MODEL_NAME}/${SFT_CHECKPOINT}"}
fi
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
export LAYER_RESET_ENABLE=${LAYER_RESET_ENABLE:-false}

# If layer reset is disabled, force reset parameters to 0
if [ "$LAYER_RESET_ENABLE" = "false" ]; then
    export LAYER_RESET_K_FIRST=0
    export LAYER_RESET_K_LAST=0
    echo "[Layer Reset] Disabled - K_FIRST and K_LAST set to 0"
else
    export LAYER_RESET_K_FIRST=${LAYER_RESET_K_FIRST:-0}    # Reset first 4 transformer layers
    export LAYER_RESET_K_LAST=${LAYER_RESET_K_LAST:-0}      # Reset last 2 transformer layers
    echo "[Layer Reset] Enabled - K_FIRST=$LAYER_RESET_K_FIRST, K_LAST=$LAYER_RESET_K_LAST"
fi

export LAYER_RESET_STEPS=${LAYER_RESET_STEPS:-"[40,80,120]"}  # Reset at global steps 120 and 200

# Set up logging with backup - organized by script location
LOG_BASE_DIR="./logs/instruct/llama/continual"
LOG_FILE="$LOG_BASE_DIR/ContinualCountdown3B_llama_Curriculum.log"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$LOG_BASE_DIR/run"

# Create log directories
mkdir -p "$LOG_BASE_DIR"
mkdir -p "$BACKUP_DIR"

# Create backup of existing log if it exists
if [ -f "$LOG_FILE" ]; then
    mkdir -p "$BACKUP_DIR"
    cp "$LOG_FILE" "$BACKUP_DIR/ContinualCountdown3B_llama_Curriculum_${TIMESTAMP}.log"
fi

# Clean up previous checkpoints
rm -rf ${CHECKPOINT_BASE_DIR}

# Clean up current log and wandb
rm -f "$LOG_FILE"
rm -rf ./wandb/*
chmod -R 755 "$LOG_BASE_DIR"
chmod -R 755 "$BACKUP_DIR"

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

# Create a unique subdirectory for this experiment's logs with timestamp
EXPERIMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_LOG_DIR=$LOG_BASE_DIR/${EXPERIMENT_TYPE}_continual_llama_sft_${SFT_CHECKPOINT}_reset_k${LAYER_RESET_K_FIRST}f_k${LAYER_RESET_K_LAST}l_${EXPERIMENT_TIMESTAMP}
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

# Three-experiment training: Experiment 1 (group0 only), Experiment 2 (group1 + group2 in same session), Experiment 3 (group2 only)

# Experiment 1: Train only group0 (repeat $EXP1_REPEAT_COUNT times)
echo "=== EXPERIMENT 1: Training Group 0 (${EXP1_REPEAT_COUNT} repetition(s)) ==="
for ((exp1_iter=1; exp1_iter<=EXP1_REPEAT_COUNT; exp1_iter++)); do
  echo "--- Experiment 1 Iteration $exp1_iter/$EXP1_REPEAT_COUNT ---"
  TRAIN_FILES_STR="[\"./data/llama_instruct/group_0/train.parquet\"]"
  VAL_FILES_STR="[\"./data/llama_instruct/group_0/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Exp1_Group0_Iter${exp1_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training group0 (iteration $exp1_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  
  python3 -m verl.trainer.main_ppo \
  --config-path ./verl/trainer/config \
  --config-name ppo_trainer \
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
    trainer.project_name=ContinualCountdown3B_llama \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$BASE_MODEL \
    $LAYER_RESET_CONFIG \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Experiment 1 Iteration $exp1_iter completed"
done
echo "Experiment 1 completed after $EXP1_REPEAT_COUNT iteration(s)"

# Experiment 2: Train group1 and group2 in the same training session (repeat $EXP2_REPEAT_COUNT times)
echo "=== EXPERIMENT 2: Training Group 1 + Group 2 (${EXP2_REPEAT_COUNT} repetition(s)) ==="
for ((exp2_iter=1; exp2_iter<=EXP2_REPEAT_COUNT; exp2_iter++)); do
  echo "--- Experiment 2 Iteration $exp2_iter/$EXP2_REPEAT_COUNT ---"
  TRAIN_FILES_STR="[\"./data/llama_instruct/group_1/train.parquet\", \"./data/llama_instruct/deepmath/train.parquet\"]"
  VAL_FILES_STR="[\"./data/llama_instruct/group_1/test.parquet\", \"./data/llama_instruct/deepmath/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560, 2560]"
  #TRAIN_SAMPLE_SIZE="[512, 512]"
  RUN_NAME="Exp2_Group1and2_Iter${exp2_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training groups 1 and 2 sequentially (iteration $exp2_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

python3 -m verl.trainer.main_ppo \
  --config-path ./verl/trainer/config \
  --config-name ppo_trainer \
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
  trainer.project_name=ContinualCountdown3B_llama \
  trainer.experiment_name=$RUN_NAME \
  trainer.total_epochs=1 \
  +trainer.val_before_train=true \
  ++reward_model.enable=False \
  ++reward_model.model.path=$BASE_MODEL \
  $LAYER_RESET_CONFIG \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Experiment 2 Iteration $exp2_iter completed"
done
echo "Experiment 2 completed after $EXP2_REPEAT_COUNT iteration(s)"

# Experiment 3: Train only group2 (repeat $EXP3_REPEAT_COUNT times)
echo "=== EXPERIMENT 3: Training Group 2 (${EXP3_REPEAT_COUNT} repetition(s)) ==="
for ((exp3_iter=1; exp3_iter<=EXP3_REPEAT_COUNT; exp3_iter++)); do
  echo "--- Experiment 3 Iteration $exp3_iter/$EXP3_REPEAT_COUNT ---"
  TRAIN_FILES_STR="[\"./data/llama_instruct/deepmath/train.parquet\"]"
  VAL_FILES_STR="[\"./data/llama_instruct/deepmath/test.parquet\"]"
  TRAIN_SAMPLE_SIZE="[2560]"
  RUN_NAME="Exp3_deepmath_Iter${exp3_iter}_SFT_${SFT_CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
  export RUN_NAME
  LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
  echo "Training deepmath (iteration $exp3_iter) with SFT model from checkpoint $SFT_CHECKPOINT" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  echo "Val files: $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

  python3 -m verl.trainer.main_ppo \
  --config-path ./verl/trainer/config \
  --config-name ppo_trainer \
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
    trainer.project_name=ContinualCountdown3B_llama \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=1 \
    +trainer.val_before_train=true \
    ++reward_model.enable=False \
    ++reward_model.model.path=$BASE_MODEL \
    $LAYER_RESET_CONFIG \
    2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
  ray stop
  sleep 10
  echo "Experiment 3 Iteration $exp3_iter completed"
done
echo "Experiment 3 completed after $EXP3_REPEAT_COUNT iteration(s)"

echo "=== ALL EXPERIMENTS COMPLETED ==="
echo "Experiment 1: $EXP1_REPEAT_COUNT iteration(s)"
echo "Experiment 2: $EXP2_REPEAT_COUNT iteration(s)"
echo "Experiment 3: $EXP3_REPEAT_COUNT iteration(s)"
echo "Total iterations: $((EXP1_REPEAT_COUNT + EXP2_REPEAT_COUNT + EXP3_REPEAT_COUNT))"
echo "Training completed successfully!"
