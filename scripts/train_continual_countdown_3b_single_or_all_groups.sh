#!/bin/bash

# Usage:
#   ./train_continual_countdown_3b_single_or_all_groups.sh <group_id|all>
#
# If <group_id> is specified (0, 1, 2, or 3), only that group will be trained.
# If "all" is specified, each group will be trained in order, with separate logs.

set -e

# Activate conda environment if needed
# source activate zero

export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR:-/cpfs04/user/liyuanhang.p/tmp/checkpoints/continual_countdown3b}
export BASE_MODEL=${BASE_MODEL:-"/cpfs04/user/liyuanhang.p/model/qwen3b"}  # Path to mounted Qwen model


export N_GPUS=${N_GPUS:-8}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
export WANDB_MODE=${WANDB_MODE:-offline}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export WANDB_MODE=${WANDB_MODE:-"disabled"}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH=.:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1

groups=(0 1 2 3)
train_files=("./data/continual/0/train.parquet" "./data/continual/1/train.parquet" "./data/continual/2/train.parquet" "./data/continual/3/train.parquet")
val_files=("./data/continual/0/test.parquet" "./data/continual/1/test.parquet" "./data/continual/2/test.parquet" "./data/continual/3/test.parquet")
train_sample_sizes=(2560 2560 2560 2560)

# Directory setup
mkdir -p ./logs
mkdir -p ./logs/run
chmod -R 777 ./logs
chmod -R 777 ./data/continual

# Helper function to train a single group
target_group_train() {
    group_id=$1
    echo "\n==== Training group $group_id ===="
    TRAIN_FILES_STR="[\"${train_files[$group_id]}\"]"
    VAL_FILES_STR="[\"${val_files[$group_id]}\"]"
    TRAIN_SAMPLE_SIZE="[${train_sample_sizes[$group_id]}]"
    WANDB_RUN_NAME="ContinualCountdown3B_Group${group_id}"
    LOG_FILE="./logs/${WANDB_RUN_NAME}.log"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="./logs/run"

    # Backup previous log
    if [ -f "$LOG_FILE" ]; then
        cp "$LOG_FILE" "$BACKUP_DIR/${WANDB_RUN_NAME}_${TIMESTAMP}.log"
        chmod 644 "$BACKUP_DIR/${WANDB_RUN_NAME}_${TIMESTAMP}.log"
    fi
    rm -f "$LOG_FILE"
    rm -rf ./wandb/*
    rm -rf ./checkpoints/continual_countdown3b_group${group_id}
    rm -rf ${CHECKPOINT_BASE_DIR}_group${group_id}

    # Check base model
    if [ ! -d "$BASE_MODEL" ]; then
        echo "Error: Base model directory $BASE_MODEL does not exist"
        exit 1
    fi
    if [ ! -f "$BASE_MODEL/config.json" ]; then
        echo "Error: No config.json found in base model"
        exit 1
    fi

    echo "Training configuration:"
    echo "  Group: $group_id"
    echo "  Model: $BASE_MODEL"
    echo "  GPUs: $N_GPUS"
    echo "  Rollout TP size: $ROLLOUT_TP_SIZE"
    echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "  VLLM_ATTENTION_BACKEND: $VLLM_ATTENTION_BACKEND"
    echo "  WANDB_MODE: $WANDB_MODE"
    echo "  NCCL_DEBUG: $NCCL_DEBUG"

    python3 -m verl.trainer.main_ppo \
        data.train_files="$TRAIN_FILES_STR" \
        data.val_files="$VAL_FILES_STR" \
        data.train_batch_size=256 \
        data.val_batch_size=256 \
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
        critic.ppo_micro_batch_size=8 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['wandb','console'] \
        +logger.print_to_console=true \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=${CHECKPOINT_BASE_DIR}_group${group_id} \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=150 \
        trainer.test_freq=30 \
        trainer.project_name=ContinualCountdown3B \
        trainer.experiment_name=$WANDB_RUN_NAME \
        trainer.total_epochs=1 \
        +trainer.val_before_train=true \
        ++reward_model.enable=False \
        ++reward_model.model.path=$BASE_MODEL \
        2>&1 | tee -a "$LOG_FILE"
}

# Main logic

for gid in "${groups[@]}"; do
    target_group_train $gid
    echo "==== Finished group $gid ===="
done
