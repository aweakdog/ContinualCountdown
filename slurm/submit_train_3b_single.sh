#!/bin/bash
#SBATCH --job-name=train_3b_curriculum_single
#SBATCH --account=pangroup
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Setup environment
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate /home/yliog/.conda/envs/zero

export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
# Fallback to 4 GPUs if nvidia-smi fails
NUM_GPUS=${NUM_GPUS:-4}

# Debug outputs
echo "NUM_GPUS: $NUM_GPUS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Start Ray on single node
ray start --head --port=6379 --dashboard-host=0.0.0.0 \
    --num-cpus=$SLURM_CPUS_PER_TASK \
    --num-gpus=$NUM_GPUS \
    --temp-dir="$RAY_TMPDIR"


# Run training script
cd /home/yliog/src/ContinualCountdown
./scripts/train_continual_countdown_3b_single_or_all_groups.sh

# Clean up
ray stop
