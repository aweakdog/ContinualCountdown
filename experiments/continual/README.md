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
   - `continual-trainer-0.5b`: Runs continual learning experiments with Qwen 0.5B
   - `continual-trainer-1.5b`: Runs continual learning experiments with Qwen 1.5B
   - `curriculum-trainer-1.5b`: Runs curriculum learning experiments with Qwen 1.5B
   - `evaluator`: Evaluates training results

## Curriculum Learning

The curriculum learning approach trains the model on operator groups in a progressive manner, from simple to complex operations. This is implemented using a single training run that processes files in a specific order.

### Training Process

1. **Data Organization**
   - Training data is organized in order of increasing complexity
   - Files are processed in sequence based on input order:
     1. plus_minus_mul (operators: +, -, *)
     2. plus_minus_div (operators: +, -, /)
     3. plus_mul_div (operators: +, *, /)
     4. minus_mul_div (operators: -, *, /)

2. **Configuration**
   - Hardware Setup:
     - 8 GPUs total
     - Tensor parallel size: 4
     - Data parallel size: 2
   - Training Parameters:
     - 3 complete rounds of training
     - 15 epochs per group per round
     - Total epochs: 180 (3 rounds × 4 groups × 15 epochs)
   - Data Structure:
     - Each group: 100k training samples, 1k test samples
     - Fixed sequence length: 4 tokens
     - Data format: Parquet files
   - Metrics Tracking:
     - Success rate per group
     - Weight changes
     - Loss function
     - Normalized gradient
     - Response length
   - Validation:
     - Tests on current group and all previous groups
     - Runs every 100 steps and at group boundaries

3. **Running Training**

   a. First, ensure data is generated:
   ```bash
   # Generate training and test data for all operator groups
   docker compose up data-generator
   ```

   b. Verify data exists in the expected locations:
   ```bash
   # Check data directory structure
   ls -R data/continual/
   ```

   c. Start curriculum training:
   ```bash
   # Start curriculum training with Qwen 1.5B
   docker compose up curriculum-trainer-1.5b
   ```

   d. Monitor training progress:
   ```bash
   # View live logs
   tail -f logs/ContinualCountdown1.5B_SingleRun.log
   ```

   The training will automatically process operator groups in sequence:
   1. plus_minus_mul: Training with +, -, * operators
   2. plus_minus_div: Training with +, -, / operators
   3. plus_mul_div: Training with +, *, / operators
   4. minus_mul_div: Training with -, *, / operators
   
   For each group:
   - Training runs for 15 epochs
   - Validation tests current and all previous groups
   - Metrics are logged with group-specific prefixes

   Training progress can be monitored through WandB UI or the console logs.

4. **Monitoring**
   - Training progress is logged to WandB under project 'ContinualCountdown1.5B'
   - Checkpoints saved every 100 steps in 'checkpoints/ContinualCountdown1.5B'
   - Console logs available in 'logs/ContinualCountdown1.5B_SingleRun.log'

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

#### For Qwen 0.5B Model
```bash
# Build and start training
docker compose build continual-trainer-0.5b
docker compose run --rm continual-trainer-0.5b ./scripts/train_continual_countdown.sh
```

#### For Qwen 1.5B Model
```bash
# Build and start training
docker compose build continual-trainer-1.5b
docker compose run --rm continual-trainer-1.5b ./scripts/train_continual_countdown_1.5b.sh
```

### Model Configurations

#### Qwen 0.5B
- Base model path: `/app/models/qwen`
- Trained model path: `/app/models/countdown_continual`
- Tensor parallel size: 2
- Training batch size: 256
- PPO mini batch size: 16
- PPO micro batch size: 8

#### Qwen 1.5B
- Base model path: `/app/models/qwen1.5b`
- Trained model path: `/app/models/countdown_continual_1.5b`
- Tensor parallel size: 4
- Training batch size: 128
- PPO mini batch size: 8
- PPO micro batch size: 4

Both models follow the same training flow:
- Sequential training through operator groups
- 3 complete rounds of training
- 15 epochs per group
- Metrics tracking and logging via WandB

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
   For each model size (0.5B and 1.5B):
   - Individual metrics: `metrics/metrics_{round}_{group}_{size}.json`
   - Consolidated metrics: `metrics/consolidated_metrics_{size}.json`
   - Visualizations: `plots/training_metrics_{size}.png`

#### Running Evaluation

```bash
# Evaluate both models
docker compose run --rm evaluator

# Evaluate specific model size
docker compose run --rm evaluator conda run -n zero bash scripts/eval_continual_countdown.sh --model-size 0.5b
docker compose run --rm evaluator conda run -n zero bash scripts/eval_continual_countdown.sh --model-size 1.5bdocker compose run --rm evaluator conda run -n zero bash scripts/eval_continual_countdown.sh --model-size 1.5b
```

The evaluator will process each model's training logs and checkpoints separately, generating model-specific plots and metrics files.

### GPU Requirements

The containers are configured to use all available GPUs. You can modify the GPU configuration in `docker-compose.yml`:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # Specify GPU indices (e.g., "0,1,2,3")
  - N_GPUS=4                    # Number of GPUs to use
  - ROLLOUT_TP_SIZE=2           # Tensor parallel size
```

