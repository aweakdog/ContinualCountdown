# Continual Learning Experiments

This directory contains scripts and utilities for running continual learning experiments on the Countdown task.

## Data Generation

The `data_gen.py` script generates training and test data for the Countdown task with different operator groups. The data is organized to support continual learning experiments where the model progressively learns to handle more operators.

### Operator Groups

The data is divided into four groups with increasing complexity:
1. `plus`: Only addition (`+`)
2. `plus_minus`: Addition and subtraction (`+`, `-`)
3. `plus_minus_mul`: Addition, subtraction, and multiplication (`+`, `-`, `*`)
4. `plus_minus_mul_div`: All basic arithmetic operations (`+`, `-`, `*`, `/`)

### Dataset Sizes
- Training set: 100,000 samples per group
- Test set: 1,000 samples per group

### Directory Structure

The generated data will be organized as follows:
```
data/
└── continual/
    ├── plus/
    │   ├── train.parquet
    │   └── test.parquet
    ├── plus_minus/
    │   ├── train.parquet
    │   └── test.parquet
    ├── plus_minus_mul/
    │   ├── train.parquet
    │   └── test.parquet
    └── plus_minus_mul_div/
        ├── train.parquet
        └── test.parquet
```

## Docker Setup

All experiments are containerized using Docker for reproducibility. The setup includes:

1. Base Image: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
2. Python Environment: Conda environment with required packages
3. Services:
   - `data-generator`: Generates training and test data
   - `continual-trainer`: Runs continual learning experiments
   - `countdown-viewer`: Interactive data viewer

### Directory Mapping

The following directories are mapped between host and container:
```
./data → /data/countdown    # Dataset storage
./wandb → /app/wandb        # WandB logs
./logs → /app/logs          # Training logs
./metrics → /app/metrics    # Metrics storage
```

## Running Experiments

### 1. Data Generation

```bash
# Build the data generator image
docker compose build data-generator

# Run data generation
docker compose run --rm data-generator
```

The script will generate data with progress indicators and color-coded output.

### 2. Continual Training

The training process uses the veRL framework to train the model through multiple rounds of increasingly complex operator groups.

#### Training Flow
- Sequential training: `plus → plus_minus → plus_minus_mul → plus_minus_mul_div`
- 3 complete rounds of training
- Each group trained for 15 epochs

#### Training Output

During training, the following artifacts are generated:

1. **Model Checkpoints**
   - Initial state: `metrics/initial_{round}_{group}/`
   - Final state: `metrics/final_{round}_{group}/`
   - Used to track model evolution across training

2. **Training Logs**
   - Location: `logs/ContinualCountdown_R{round}_{group}.log`
   - Contains:
     - Success rates
     - Loss values
     - Gradient norms
     - Response lengths

3. **WandB Dashboard**
   - Real-time training metrics
   - Interactive visualizations
   - Experiment tracking

#### Running Training

1. Prepare directories:
```bash
mkdir -p wandb logs metrics
```

2. Configure training (optional):
```bash
# Edit docker-compose.yml to set:
- BASE_MODEL path
- Number of GPUs
- Tensor parallel size
```

3. Launch training:
```bash
# Build and start training
docker compose build continual-trainer
docker compose run --rm continual-trainer
```

### 3. Evaluation

The evaluation process is decoupled from training and runs in a separate container.

#### Evaluation Process

1. **Metrics Extraction**
   - Processes training logs
   - Computes weight changes between checkpoints
   - Generates consolidated metrics

2. **Tracked Metrics**
   - Success Rate: Percentage of correctly solved equations
   - Weight Change: Normalized difference between model states
   - Loss Values: Training loss progression
   - Gradient Norms: Training stability indicators
   - Response Lengths: Solution complexity analysis

3. **Output Files**
   - Individual metrics: `metrics/metrics_{round}_{group}.json`
   - Consolidated metrics: `metrics/consolidated_metrics.json`
   - Visualizations: `plots/training_metrics.png`

#### Running Evaluation

```bash
# After training completes
docker compose run --rm evaluator
```

### GPU Requirements

The containers are configured to use all available GPUs. You can modify the GPU configuration in `docker-compose.yml`:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # Specify GPU indices (e.g., "0,1,2,3")
  - N_GPUS=4                    # Number of GPUs to use
  - ROLLOUT_TP_SIZE=2           # Tensor parallel size
```

