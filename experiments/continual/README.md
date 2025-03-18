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

#### Metrics System

The experiment tracks the following key metrics across all operator groups:

1. **Success Rate**
   - Definition: Percentage of correctly solved equations
   - Tracked per group and round via veRL's reward system
   - Available in:
     - WandB dashboard (real-time)
     - Training logs

2. **Normalized Weight Change**
   - Definition: `||W_new - W_old|| / ||W_old||`
   - Computed between checkpoints for each group
   - Available in:
     - `metrics/metrics_{round}_{group}.json`

3. **Loss Function**
   - Definition: Training loss values
   - Tracked per batch and epoch via veRL's trainer
   - Available in:
     - WandB dashboard (real-time)
     - Training logs

4. **Response Length**
   - Definition: Statistics of generated solution lengths
   - Computed from training logs
   - Metrics include:
     - Average response length
     - Maximum response length
     - Minimum response length
   - Available in:
     - `metrics/metrics_{round}_{group}.json`

#### Metrics Analysis

Metrics are analyzed at three levels:

1. **Per Training Segment**
   - File: `metrics/metrics_{round}_{group}.json`
   - Contains: Raw metrics for each training segment

2. **Summary Statistics**
   - File: `metrics/summary.json`
   - Contains:
     ```json
     {
       "per_round": {
         // Average metrics for each round
       },
       "per_group": {
         // Average metrics for each operator group
       },
       "overall": {
         // Overall averages across all rounds and groups
       }
     }
     ```

3. **Real-time Monitoring**
   - WandB Dashboard: Real-time metrics visualization
   - Log Files: `logs/ContinualCountdown_R{round}_{group}.log`

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
# Build the training image
docker compose build continual-trainer

# Start training
docker compose run --rm continual-trainer
```

### GPU Requirements

The training container is configured to use all available GPUs. You can modify the GPU configuration in `docker-compose.yml`:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # Specify GPU indices (e.g., "0,1,2,3")
  - N_GPUS=4                    # Number of GPUs to use
  - ROLLOUT_TP_SIZE=2           # Tensor parallel size
```

