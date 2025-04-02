# Dummy Tinyzero Dataset

This directory contains scripts for creating and training on a dummy dataset sampled from the original Tinyzero dataset. This serves as a control group for our continual learning experiments.

## Dataset Generation

The `gen_tinyzero_dummy.py` script randomly samples from the original Tinyzero dataset (`/data/tinyzero/train.parquet`) to create 4 equal-sized groups. Each group contains:
- Training set: 100,000 samples
- Test set: 1,000 samples

### Group Structure
Instead of using operator-based names, we use numerical groups to avoid any confusion about their contents:
- Group '0': Random sample from Tinyzero dataset
- Group '1': Random sample from Tinyzero dataset
- Group '2': Random sample from Tinyzero dataset
- Group '3': Random sample from Tinyzero dataset

All samples are randomly selected without any specific operator constraints, making this a proper control group.

## Training Setup

The training follows the same curriculum learning approach as the original continual learning experiments:
- 3 complete rounds of training
- 15 epochs per group per round
- Total epochs: 180 (3 rounds × 4 groups × 15 epochs)

### Build Instructions

First, build the Docker image:
```bash
docker compose build
```
This will build the image with all required dependencies, including:
- Python 3.9
- PyTorch with CUDA support
- Transformers library
- vLLM for efficient inference
- Ray for distributed training

### Docker Setup

Two Docker services are provided:

1. Data Generator:
```bash
docker compose up tinyzero-dummy-generator
```
This creates the dummy dataset in `/data/dummy_tinyzero/{group}/`.

2. Trainer:
```bash
docker compose run --rm tinyzero-dummy-trainer
```
This runs the training on the dummy dataset. The script will:
- Initialize the model from Qwen 1.5B base model
- Set up proper logging and environment variables
- Train for 1 epoch on the dummy dataset
- Save logs to `/app/logs/TinyZeroDummy1.5B_SingleRun.log`
- Save model checkpoints to `/app/models/tinyzero_dummy`

### Configuration
- Uses the same hardware setup (8 GPUs, tensor parallel size 2)
- Metrics are logged to WandB under "TinyZero-Dummy" project
- All logs and checkpoints follow the same structure as the main experiment

## Purpose

This setup serves as a control experiment to compare against the operator-based continual learning approach. By using random grouping instead of operator-based grouping, we can better understand the impact of the operator-based curriculum in the main experiment.

## Evaluation

After training, you can evaluate the results using the provided evaluation script:

```bash
python experiments/dummy_tinyzero/eval.py
```

The evaluation script will:
1. Find the most recent training log in `logs/tinyzero_dummy/`
2. Extract training metrics:
   - Success rate
   - Policy gradient loss
   - Gradient norm
   - Response length
3. Generate plots showing metrics over time
4. Save results:
   - Plots: `./plots/tinyzero_dummy_1.5b_metrics.png`
   - Raw metrics: `./metrics/tinyzero_dummy_1.5b_metrics.json`

You can compare these results with the original continual learning experiment (`experiments/continual/eval.py`) to analyze how random grouping (control) performs versus operator-based curriculum learning.
