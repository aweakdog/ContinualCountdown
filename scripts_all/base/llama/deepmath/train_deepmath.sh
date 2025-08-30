#!/bin/bash

# DeepMath-103K PPO RLHF training script for llama2.5-3B (FSDP + vLLM rollout)
# Single-phase training on data/deepmath/{train,test}.parquet

SFT_CHECKPOINT=${SFT_CHECKPOINT:-global_step_0_base}

# Safety for git in shared mounts
if ! git config --global --get-all safe.directory | grep -q "."; then
    git config --global --add safe.directory .
fi

# ===== Environment config =====
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/nas/shared/sys2/yuanhangli/tmp/checkpoints/deepmath_llama3b_ppo}
export BASE_MODEL=${BASE_MODEL:-"/nas/shared/sys2/yuanhangli/tmp/llama_base_sft_model/${SFT_CHECKPOINT}"}
export N_GPUS=${N_GPUS:-4}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
# Known upstream bug: disable attention logging on llama2
export ATTENTION_LOGGING_ENABLED=${ATTENTION_LOGGING_ENABLED:-false}

# Analyzer GPU reservation (kept for consistency)
export RAY_ANALYZER_GPU_START=${RAY_ANALYZER_GPU_START:-4}
export RAY_ANALYZER_GPU_COUNT=${RAY_ANALYZER_GPU_COUNT:-4}

echo "[GPU Config] Training uses GPUs 0-3 by default; analyzers 4-7 if enabled"

# ===== Layer Reset (disabled by default for DeepMath) =====
export LAYER_RESET_ENABLE=${LAYER_RESET_ENABLE:-false}
export LAYER_RESET_K_FIRST=${LAYER_RESET_K_FIRST:-0}
export LAYER_RESET_K_LAST=${LAYER_RESET_K_LAST:-0}
export LAYER_RESET_STEPS=${LAYER_RESET_STEPS:-"[]"}

# ===== Logging setup =====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_LOG_DIR=./llama_logs/train_deepmath_llama3b_sft_${SFT_CHECKPOINT}
mkdir -p "$EXP_LOG_DIR"
cp tmp/monitor_master.sh "$EXP_LOG_DIR/" 2>/dev/null || echo "Warning: monitor_master.sh not found"
MASTER_LOG_FILE="$EXP_LOG_DIR/experiment_master.log"

# Cleanup old run artifacts
rm -rf ${CHECKPOINT_BASE_DIR}
rm -rf ./wandb/*

# General runtime env
export FSDP_GRAD_METRIC_ENABLED=${FSDP_GRAD_METRIC_ENABLED:-true}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH=.:$PYTHONPATH

# Sanity checks
if [ ! -d "$BASE_MODEL" ]; then
  echo "Error: Base model directory $BASE_MODEL does not exist" | tee -a "$MASTER_LOG_FILE"
  exit 1
fi
if [ ! -f "$BASE_MODEL/config.json" ]; then
  echo "Error: No config.json found in base model ($BASE_MODEL)" | tee -a "$MASTER_LOG_FILE"
  exit 1
fi

# Data paths with curriculum group support (mirror GSM8K style)
export DATA_ROOT=${DATA_ROOT:-./data/deepmath}
export GROUPS=${GROUPS:-"0"}  # comma-separated list, e.g., "0" or "0,1,2"

IFS=',' read -ra GROUP_LIST <<< "$GROUPS"

TRAIN_FILES=()
VAL_FILES=()
for g in "${GROUP_LIST[@]}"; do
  TRAIN_FILES+=("\"${DATA_ROOT}/${g}/train.parquet\"")
  VAL_FILES+=("\"${DATA_ROOT}/${g}/test.parquet\"")
  if [ ! -f "${DATA_ROOT}/${g}/train.parquet" ] || [ ! -f "${DATA_ROOT}/${g}/test.parquet" ]; then
    echo "[Warn] Missing parquet for group ${g} under ${DATA_ROOT}/${g}/. You can generate with:\n  python examples/data_preprocess/deepmath.py --from_local --local_dir ${DATA_ROOT}/${g} --prepend_cot_examples --max_samples 5000" | tee -a "$MASTER_LOG_FILE"
  fi
done

TRAIN_FILES_STR="[${TRAIN_FILES[*]}]"
VAL_FILES_STR="[${VAL_FILES[*]}]"

# Curriculum controls
export CURRICULUM_LEARNING=${CURRICULUM_LEARNING:-true}
export EPOCHS_PER_GROUP=${EPOCHS_PER_GROUP:-15}
export TOTAL_ROUNDS=${TOTAL_ROUNDS:-1}

# Per-group sample sizes (default 2560 each if not provided)
if [ -z "${TRAIN_SAMPLE_SIZE}" ]; then
  DEFAULT_SIZES=()
  for _ in "${GROUP_LIST[@]}"; do DEFAULT_SIZES+=(2560); done
  TRAIN_SAMPLE_SIZE_STR="[${DEFAULT_SIZES[*]}]"
else
  TRAIN_SAMPLE_SIZE_STR="${TRAIN_SAMPLE_SIZE}"
fi

# Build layer reset hydra args
LAYER_RESET_CONFIG=""
if [ "${LAYER_RESET_ENABLE}" = "true" ]; then
  echo "[Layer Reset] Enabled k_first=${LAYER_RESET_K_FIRST}, k_last=${LAYER_RESET_K_LAST}, steps=${LAYER_RESET_STEPS}" | tee -a "$MASTER_LOG_FILE"
  LAYER_RESET_CONFIG="++layer_reset.enable_reset=${LAYER_RESET_ENABLE} ++layer_reset.reset_k_first=${LAYER_RESET_K_FIRST} ++layer_reset.reset_k_last=${LAYER_RESET_K_LAST} ++layer_reset.reset_steps=\"${LAYER_RESET_STEPS}\""
else
  echo "[Layer Reset] Disabled" | tee -a "$MASTER_LOG_FILE"
fi

# ===== Single-phase PPO training on DeepMath-103K =====
RUN_NAME="DeepMath_SFT_${SFT_CHECKPOINT}_${TIMESTAMP}"
LOG_FILE="$EXP_LOG_DIR/${RUN_NAME}.log"
echo "Starting PPO RLHF on DeepMath-103K with base model: $BASE_MODEL" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
echo "Train files: $TRAIN_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"
echo "Val files:   $VAL_FILES_STR" | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

python3 -m verl.trainer.main_ppo \
  fsdp_grad_metric_enabled=$FSDP_GRAD_METRIC_ENABLED \
  data.train_files="$TRAIN_FILES_STR" \
  data.val_files="$VAL_FILES_STR" \
  data.train_batch_size=256 \
  data.val_batch_size=256 \
  data.max_response_length=1024 \
  ++data.curriculum_learning=$CURRICULUM_LEARNING \
  ++data.epochs_per_group=$EPOCHS_PER_GROUP \
  ++data.total_rounds=$TOTAL_ROUNDS \
  ++data.train_sample_size="$TRAIN_SAMPLE_SIZE_STR" \
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
  trainer.project_name=DeepMath_llama3B \
  trainer.experiment_name=$RUN_NAME \
  trainer.total_epochs=1 \
  +trainer.val_before_train=true \
  ++reward_model.enable=False \
  ++reward_model.model.path=$BASE_MODEL \
  $LAYER_RESET_CONFIG \
  2>&1 | tee -a "$LOG_FILE" | tee -a "$MASTER_LOG_FILE"

# Cleanup ray between runs
ray stop
sleep 5

echo "[DONE] PPO RLHF on DeepMath-103K completed: $RUN_NAME" | tee -a "$MASTER_LOG_FILE"
