# ContinualCountdown - Quick Start Guide

This guide provides step-by-step instructions for running continual learning experiments with different model architectures.

## Prerequisites

- Python environment with required dependencies
- CUDA-compatible GPUs (8x recommended)
- Sufficient disk space for models and datasets

## 1. Data Setup

### Download Dataset from HuggingFace

```bash
# Download the countdown dataset
git clone https://huggingface.co/datasets/aweakdog/countdown_3_groups

# Rename and move to ./data directory
mv countdown_3_groups ./data
```

The dataset contains three groups for continual learning experiments:
- Group 0: Initial training data
- Group 1: First continual learning phase
- Group 2: Second continual learning phase

## 2. Model Setup

Download and organize the required base and instruct models in the `./models/` directory:

### Qwen Models

```bash
# Download Qwen 3B Base Model
git clone https://huggingface.co/Qwen/Qwen2.5-3B ./models/qwen_base3b

# Download Qwen 3B Instruct Model  
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ./models/qwen_instruct3b
```

### Llama Models

```bash
# Download Llama 3B Base Model
git clone https://huggingface.co/meta-llama/Llama-3.2-3B ./models/llama_base3b

# Download Llama 3B Instruct Model
git clone https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct ./models/llama_instruct3b
```

### Expected Directory Structure

```
./models/
├── qwen_base3b/
├── qwen_instruct3b/
├── llama_base3b/
└── llama_instruct3b/

./data/
└── countdown_3_groups/
    ├── group_0/
    ├── group_1/
    └── group_2/
```

## 3. Supervised Fine-Tuning (SFT)

Before running continual learning, you need to create SFT models from the base models.

### Example: Llama Base Model SFT

```bash
# Run SFT training for Llama base model
# The number (2560) represents the training steps
scripts_all/base/llama/sft/train_continual_countdown_3b_sft_llama.sh 2560
```

This will:
- Train the model for 2560 steps
- Generate checkpoints (e.g., `global_step_10`)
- Save the SFT model for continual learning

### Other Model SFT Examples

```bash
# Qwen base model SFT
scripts_all/base/qwen/sft/train_continual_countdown_3b_sft_qwen.sh 2560

# Instruct models (if available)
scripts_all/instruct/llama/sft/train_continual_countdown_3b_sft_llama.sh 2560
scripts_all/instruct/qwen/sft/train_continual_countdown_3b_sft_qwen.sh 2560
```

## 4. Continual Learning Training

After obtaining the SFT model, run the continual learning experiment:

### Example: Llama Base Model Continual Training

```bash
# Run continual learning on the SFT model
scripts_all/base/llama/continual/train_continual.sh
```

This script will:
- Load the SFT checkpoint (e.g., `global_step_10`)
- Execute three-phase continual learning:
  - **Phase 1**: Group 0 training
  - **Phase 2**: Group 1+2 combined training  
  - **Phase 3**: Group 2 only training
- Generate analysis and metrics for each phase

### Other Model Continual Training

```bash
# Qwen base model continual training
scripts_all/base/qwen/continual/train_continual.sh

# Instruct models (if available)
scripts_all/instruct/llama/continual/train_continual.sh
scripts_all/instruct/qwen/continual/train_continual.sh
```

## 5. Complete Workflow Example

Here's a complete example for running experiments with the Llama base model:

```bash
# 1. Setup data and models (one-time setup)
git clone https://huggingface.co/datasets/aweakdog/countdown_3_groups ./data
git clone https://huggingface.co/meta-llama/Llama-3.2-3B ./models/llama_base3b

# 2. Run SFT training
scripts_all/base/llama/sft/train_continual_countdown_3b_sft_llama.sh 2560

# 3. Run continual learning
scripts_all/base/llama/continual/train_continual.sh
```

## 6. Monitoring and Results

- **Logs**: Check training logs for loss curves and metrics
- **Checkpoints**: Models are saved at specified intervals
- **Analysis**: Gradient and Fisher information analysis (if enabled)
- **Metrics**: Performance metrics for each continual learning phase

## 7. Configuration Notes

- **GPU Resources**: Scripts are configured for 8-GPU setups
- **Batch Sizes**: Adjust based on your GPU memory
- **Learning Rates**: Pre-configured for optimal performance
- **Phase Repetition**: Configurable via `PHASE*_REPEAT_COUNT` variables

## Troubleshooting

- Ensure sufficient GPU memory for the model size
- Check CUDA compatibility and driver versions
- Verify all required dependencies are installed
- Monitor disk space during training and checkpointing