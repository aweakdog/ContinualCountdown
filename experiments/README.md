# Continual Learning Experiments

This directory contains the setup for continual learning experiments on the Countdown task.

## Project Structure

```
experiments/
├── continual/         # Continual learning experiment code
│   └── data_gen.py   # Data generation for continual learning
├── README.md         # This file
└── docker/           # Docker configurations
```

## Training Configuration

### Data Structure
- Four operator groups in sequential order:
  1. `0` (addition only)
  2. `1` (addition and subtraction)
  3. `2` (addition, subtraction, multiplication)
  4. `3` (all operations)
- Each group: 100k training samples, 1k test samples
- Data location: `/data/continual/{0,1,2,3}/`
- Format: Parquet files

### Training Flow
- Sequential training through operator groups
- 3 complete rounds of training
- Each group: 15 epochs per round
- Total epochs: 180 (3 rounds × 4 groups × 15 epochs)

### Available Models
1. **Qwen-1.5B Configuration**
   - 8 GPUs (3090)
   - Tensor parallel size: 2
   - Batch sizes: 128 (train/val)
   - Learning rates: 1e-6 (actor), 1e-5 (critic)

2. **Qwen-3B Configuration**
   - 4 GPUs (A800)
   - Tensor parallel size: 2
   - Batch sizes: 64 (train/val)
   - Learning rates: 8e-7 (actor), 8e-6 (critic)

## Running the Experiments

### Prerequisites
1. Install Docker and NVIDIA Container Toolkit
2. Download Qwen models to respective directories:
   - 1.5B: `~/model/qwen1.5b/`
   - 3B: `~/model/qwen3b/`

### Training Steps

1. Generate training data:
```bash
docker compose up data-generator
```

2. Train models:

For 1.5B model (8 GPUs):
```bash
docker compose up curriculum-trainer-1.5b
```

For 3B model (4 GPUs):
```bash
docker compose up curriculum-trainer-3b
```

### Output Locations
- Checkpoints: `./checkpoints/`
- Metrics: `./metrics/`
- Logs: `./logs/`
- WandB data: `./wandb/`
- Plots: `./plots/`

## Monitoring
- Training progress is logged to WandB (offline mode)
- Real-time logs available in `./logs/`
- Metrics tracked:
  - Success rate
  - Normalized weight change
  - Loss function
  - Normalized gradient
  - Response length
