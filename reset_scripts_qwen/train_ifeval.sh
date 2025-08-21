#!/bin/bash

# IFeval PPO RLHF training script for Qwen2.5-3B (FSDP + vLLM rollout)
# Single-phase training on data/ifeval/0/{train,test}.parquet

SFT_CHECKPOINT=${SFT_CHECKPOINT:-global_step_0_instruct}

# Model configuration
BASE_MODEL=${BASE_MODEL:-/nas/shared/sys2/yuanhangli/tmp/qwen_sft_model}
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/nas/shared/sys2/yuanhangli/tmp/qwen_ifeval_rlhf}

# Training configuration
N_GPUS=${N_GPUS:-8}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Logging configuration
export WANDB_PROJECT=${WANDB_PROJECT:-qwen_ifeval_rlhf}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen_3b_ifeval_$(date +%Y%m%d_%H%M%S)}

# Create directories
mkdir -p "$CHECKPOINT_BASE_DIR"
mkdir -p "./ifeval_logs"

# Set log file
MASTER_LOG_FILE="./ifeval_logs/train_ifeval_$(date +%Y%m%d_%H%M%S).log"

echo "Starting IFeval RLHF training..." | tee "$MASTER_LOG_FILE"
echo "Base model: $BASE_MODEL" | tee -a "$MASTER_LOG_FILE"
echo "Checkpoint dir: $CHECKPOINT_BASE_DIR" | tee -a "$MASTER_LOG_FILE"
echo "Log file: $MASTER_LOG_FILE" | tee -a "$MASTER_LOG_FILE"

# Check if preprocessed data exists
if [ ! -f "./data/ifeval/0/train.parquet" ] || [ ! -f "./data/ifeval/0/test.parquet" ]; then
    echo "[Error] Missing IFeval parquet files. Please run preprocessing first:" | tee -a "$MASTER_LOG_FILE"
    echo "  python examples/data_preprocess/ifeval.py --from_local --local_dir ./data/RLVR-IFeval/data --max_prompt_length 800 --output_dir ./data/ifeval/0" | tee -a "$MASTER_LOG_FILE"
    exit 1
fi

# Data paths
export DATA_ROOT=${DATA_ROOT:-./data/ifeval}
export GROUPS=${GROUPS:-"0"}  # comma-separated list

IFS=',' read -ra GROUP_LIST <<< "$GROUPS"

TRAIN_FILES=()
VAL_FILES=()
for g in "${GROUP_LIST[@]}"; do
  TRAIN_FILES+=("\"${DATA_ROOT}/${g}/train.parquet\"")
  VAL_FILES+=("\"${DATA_ROOT}/${g}/test.parquet\"")
  if [ ! -f "${DATA_ROOT}/${g}/train.parquet" ] || [ ! -f "${DATA_ROOT}/${g}/test.parquet" ]; then
    echo "[Warn] Missing parquet for group ${g} under ${DATA_ROOT}/${g}/." | tee -a "$MASTER_LOG_FILE"
  fi
done

# Convert arrays to comma-separated strings
TRAIN_FILES_STR=$(IFS=,; echo "${TRAIN_FILES[*]}")
VAL_FILES_STR=$(IFS=,; echo "${VAL_FILES[*]}")

echo "Training files: $TRAIN_FILES_STR" | tee -a "$MASTER_LOG_FILE"
echo "Validation files: $VAL_FILES_STR" | tee -a "$MASTER_LOG_FILE"

# Run training
python -m verl.trainer.main_ppo \
  data.train_files=[$TRAIN_FILES_STR] \
  data.val_files=[$VAL_FILES_STR] \
  data.prompt_key=prompt \
  data.response_key=response \
  data.micro_batch_size=16 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=$BASE_MODEL/$SFT_CHECKPOINT \
  +actor_rollout_ref.model.trust_remote_code=true \
  +actor_rollout_ref.model.torch_dtype=bfloat16 \
  +actor_rollout_ref.model.device_map=auto \
  +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
  +actor_rollout_ref.model.use_cache=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.tensor_parallel_size=1 \
  actor_rollout_ref.rollout.max_model_len=1536 \
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
  ++actor_rollout_ref.actor.enable_gradient_analysis=false \
  ++actor_rollout_ref.actor.gradient_analysis_freq=1 \
  ++actor_rollout_ref.actor.enable_fisher_analysis=false \
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
  trainer.total_epochs=3 \
  trainer.max_steps=2000 \
  reward_model.enable=true \
  reward_model.style=rule 2>&1 | tee -a "$MASTER_LOG_FILE"

echo "IFeval RLHF training completed!" | tee -a "$MASTER_LOG_FILE"
echo "Results saved to: $CHECKPOINT_BASE_DIR" | tee -a "$MASTER_LOG_FILE"
echo "Log file: $MASTER_LOG_FILE" | tee -a "$MASTER_LOG_FILE"
