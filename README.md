# Continual Learning Experiments: LLM Plasticity on Countdown Problems

## Overview

This project explores the **plasticity** of Large Language Models (LLMs) through continual learning experiments on countdown mathematical problems. We investigate how well pre-trained models (Qwen2.5-3B and Llama3.2-3B) can adapt to unknown environments after being fine-tuned on known problem domains.

### What is Plasticity?
**Plasticity** is defined as an agent's ability to adapt to unknown environments. In our context, we measure how well an LLM can learn new mathematical reasoning patterns after being trained on a specific subset of problems.

## Problem Definition: Countdown

The countdown problem involves using a set of source numbers and basic arithmetic operators (`+`, `-`, `*`, `/`) to reach a target number.

**Example**: Given source numbers `[22, 23, 2, 4]` and target `30`, one solution is: `22/2 + 23 - 4 = 30`

## Dataset Structure

We generate three distinct groups of countdown problems, each with 512,000 problems:

### Group 0 (Multiplication-based)
- **50%**: 4 source numbers, operators `[+, -, *]`
- **25%**: 3 source numbers, operators `[+, *]`
- **25%**: 3 source numbers, operators `[-, *]`

### Group 1 (Division-based)
- **50%**: 4 source numbers, operators `[+, -, /]`
- **25%**: 3 source numbers, operators `[+, /]`
- **25%**: 3 source numbers, operators `[-, /]`

### Group 2 (Mixed operations)
- **25%**: 4 source numbers, operators `[+, *, /]`
- **25%**: 4 source numbers, operators `[-, *, /]`
- **50%**: 3 source numbers, operators `[*, /]`

## Experimental Setup

### Models
- **Qwen2.5-3B**: `~/model/qwen/models--Qwen--Qwen2.5-3B`
- **Llama3.2-3B**: Various checkpoints

### Hardware Configuration
- **GPUs**: 8x NVIDIA RTX A800
- **Training GPUs**: 0-3 (FSDP training)
- **Analyzer GPUs**: 4-7 (Ray analyzers for Fisher Info and Gradient Analysis)

### Training Parameters
- **Batch Size**: 256
- **PPO Mini/Micro Batch**: 32/8
- **Actor Learning Rate**: 1e-6
- **Critic Learning Rate**: 1e-5
- **Tensor Parallel Size**: 2
- **GPU Memory Utilization**: 0.6

## Experimental Design

### Phase 1: Supervised Fine-Tuning (SFT)
- Train both models on **Group 0** (Known Group) with varying sample sizes
- Generate checkpoints: `SFT_x` where `x * 256 = total_samples`
- Example: `SFT_15` = 15 × 256 = 3,840 samples

#### SFT Training Format
Each SFT sample contains a countdown equation example with:
- An incorrect answer demonstration
- The correct reasoning process
- Proper answer submission format

**Example SFT Trajectory:**
```
User: Using the numbers [3, 19, 75], create an equation that equals 44. You can use basic arithmetic operations (+, -, *, /) and each number should be used exactly once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.

Assistant: Let me solve this step by step.
<think>
Our initial number is [3, 19, 75], and our target is 44.
One possible solution is:
3 + 19 + 75 = 22 + 75 = 97, Incorrect! So Let's try next one.
75 / 3 + 19 = 25 + 19= 44, Correct!
</think>
<answer>(75 / 3) + 19</answer><|end_of_text|>
```

### Phase 2: PPO Reinforcement Learning Experiments

#### Experiment 1: Testing on Group 0
- **Objective**: Baseline performance on known domain
- **Method**: PPO RLHF on Group 0 for 150 training steps
- **Metrics**: Reward performance, C_K, GRAMA plasticity metrics
- **Repetitions**: 3 independent runs per checkpoint

#### Experiment 2: Sequential Transfer (Group 1 → Group 2)
- **Objective**: Measure adaptation and transfer capabilities
- **Method**: 
  1. PPO RLHF on Group 1 for 150 training steps
  2. Switch to Group 2 and evaluate performance
- **Metrics**: Reward performance, C_K, GRAMA plasticity metrics
- **Repetitions**: 3 independent runs per checkpoint

#### Experiment 3: Testing on Group 2
- **Objective**: Baseline performance on unknown domain
- **Method**: PPO RLHF on Group 2 for 150 training steps
- **Metrics**: Reward performance, C_K, GRAMA plasticity metrics
- **Repetitions**: 3 independent runs per checkpoint

## Key Insights

- **Experiment 1**: Measures performance retention on the known domain
- **Experiment 2**: Reveals adaptation behavior and transfer learning capabilities
- **Experiment 3**: Provides baseline performance for the most challenging unknown domain

## Technical Implementation

### PPO Configuration
```bash
# Key parameters for PPO training
data.train_batch_size=256
data.val_batch_size=256
data.max_response_length=1024
actor_rollout_ref.actor.optim.lr=1e-6
critic.optim.lr=1e-5
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size=8
algorithm.kl_ctrl.kl_coef=0.001
```

### Curriculum Learning
- **Enabled**: `++data.curriculum_learning=true`
- **Epochs per Group**: 15
- **Total Rounds**: 1

### REDO (Plasticity Analysis)
- **Actor REDO**: Enabled with threshold mode (τ=0.1)
- **Critic REDO**: Enabled with threshold mode (τ=0.1)
- **Metric Frequency**: Every step
- **Reset Frequency**: Every step

## Project Structure

```
ContinualCountdown/
├── scripts/                    # Qwen2.5 training scripts
│   ├── train_continual_countdown_3b_curriculum_after_sft.sh
│   ├── debug_train_continual_countdown_3b_curriculum_after_sft.sh
│   └── develop_train_continual_countdown_3b_curriculum_after_sft.sh
├── llama_scripts/             # Llama3.2 training scripts
│   ├── train_continual_countdown_3b_curriculum_after_sft_llama.sh
│   ├── debug_train_continual_countdown_3b_curriculum_after_sft_llama.sh
│   └── develop_train_continual_countdown_3b_curriculum_after_sft_llama.sh
├── examples/data_preprocess/  # Data preprocessing scripts
│   ├── gsm8k.py              # GSM8K dataset preprocessing
│   ├── ifeval.py             # IFEval dataset preprocessing
│   ├── deepmath.py           # DeepMath-103K dataset preprocessing
│   └── countdown.py          # Countdown dataset generation
├── verl/trainer/config/       # Training configuration files
│   ├── sft_trainer.yaml      # Qwen instruct SFT config
│   ├── sft_trainer_llama.yaml # Llama instruct SFT config
│   ├── sft_trainer_gsm8k*.yaml # GSM8K SFT configs
│   ├── sft_trainer_ifeval*.yaml # IFEval SFT configs
│   └── sft_trainer_deepmath*.yaml # DeepMath SFT configs
└── README.md                  # This file
```

## Data Generation

### Multi-Template Dataset Generation

All data generation scripts support `--model_type all` to automatically generate datasets for all three model templates (base, qwen-instruct, llama-instruct) in separate directories. This ensures each model type gets properly formatted data without manual configuration.

#### Countdown Dataset (Primary)
```bash
# Generate all model types (recommended)
python experiments/continual/data_gen_efficient_simpler.py \
    --model_type all \
    --train_size 512000 \
    --test_size 512

# Generate specific group for all model types
python experiments/continual/data_gen_efficient_simpler.py \
    --model_type all \
    --group 0 \
    --train_size 512000 \
    --test_size 512
```

#### GSM8K Dataset
```bash
# Generate all model types
python examples/data_preprocess/gsm8k.py \
    --model_type all \
    --from_local \
    --local_json_dir ./data/gsm8k/main \
    --prepend_cot_examples \
    --max_prompt_length 800
```

#### DeepMath Dataset
```bash
# Generate all model types with CoT examples
python examples/data_preprocess/deepmath.py \
    --model_type all \
    --from_local \
    --local_parquet_dir ./data/DeepMath-103K/data \
    --prepend_cot_examples \
    --max_prompt_length 800
```

#### IFEval Dataset
```bash
# Generate all model types
python examples/data_preprocess/ifeval.py \
    --model_type all \
    --from_local \
    --local_parquet_dir ./data/RLVR-IFeval/data \
    --max_prompt_length 700
```

### Generated Directory Structure
```
./data/
├── base/
│   ├── continual/     # Countdown data (base template)
│   ├── gsm8k/         # GSM8K data (base template)
│   ├── deepmath/      # DeepMath data (base template)
│   └── ifeval/        # IFEval data (base template)
├── qwen_instruct/
│   ├── continual/     # Countdown data (qwen instruct template)
│   ├── gsm8k/         # GSM8K data (qwen instruct template)
│   ├── deepmath/      # DeepMath data (qwen instruct template)
│   └── ifeval/        # IFEval data (qwen instruct template)
└── llama_instruct/
    ├── continual/     # Countdown data (llama instruct template)
    ├── gsm8k/         # GSM8K data (llama instruct template)
    ├── deepmath/      # DeepMath data (llama instruct template)
    └── ifeval/        # IFEval data (llama instruct template)
```

### Key Features
- **Unified Field Names**: All datasets use `prompt` and `response` fields
- **Automatic Template Selection**: Each directory contains data formatted for the corresponding model type
- **One-Command Generation**: Use `--model_type all` to generate all templates at once
- **Consistent Structure**: Same directory layout across all datasets

## Supervised Fine-Tuning (SFT)

### Training SFT Models

Use the provided training scripts to train SFT models on the Countdown dataset:

#### Qwen SFT Training
```bash
# Train Qwen instruct model on Countdown data
bash scripts/train_continual_countdown_3b_sft.sh 2048 --gpus 8 --wandb offline
```

#### Llama SFT Training
```bash
# Train Llama instruct model on Countdown data
bash llama_scripts/train_continual_countdown_3b_sft_llama.sh 2048 --gpus 8 --wandb offline
```

### SFT Model Output Locations
- **Qwen SFT Model**: `/nas/shared/sys2/yuanhangli/tmp/qwen_instruct_sft_model`
- **Llama SFT Model**: `/nas/shared/sys2/yuanhangli/tmp/llama_instruct_sft_model`

### SFT Training Format
Each SFT sample contains a countdown equation example with:
- Step-by-step reasoning process
- Proper answer submission format with `<answer>` tags

## Continual Learning Training

### Prerequisites
- Ray cluster setup
- CUDA environment with 8 GPUs
- Trained SFT models (see above)

### Running Continual Learning Experiments

#### Qwen Continual Training
```bash
# Run continual learning experiments on Qwen model
bash scripts/train_continual_countdown_3b_curriculum_after_sft.sh
```

#### Llama Continual Training
```bash
# Run continual learning experiments on Llama model
bash llama_scripts/train_continual_countdown_3b_curriculum_after_sft_llama.sh
```

### Experimental Design

The continual learning experiments follow this sequence:
1. **Phase 1**: Train on Group 0 (multiplication-based operations)
2. **Phase 2**: Train on Group 1 (division-based operations) 
3. **Phase 3**: Train on Group 2 (mixed operations)

Each phase uses PPO reinforcement learning with curriculum learning enabled.

### Key Configuration
- **Batch Size**: 256
- **PPO Mini/Micro Batch**: 32/8
- **Actor Learning Rate**: 1e-6
- **Critic Learning Rate**: 1e-5
- **Epochs per Group**: 15
- **Total Rounds**: 1

## Metrics and Analysis

### Plasticity Metrics
- **C_K**: Fisher Information Matrix eigenvalues-based plasticity metric
- **GRAMA**: Gradient-based plasticity analysis
- **REDO**: Plasticity analysis with threshold mode (τ=0.1)
