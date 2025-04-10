# ContinualCountdown

A continual learning project for training models on arithmetic operations in a curriculum learning fashion.

## Project Structure

- `experiments/continual/`: Main experiment code
  - `data_gen.py`: Data generation script
  - Training scripts and utilities

## Data Structure

Four operator groups with increasing complexity:
1. `plus`: Addition only
2. `plus_minus`: Addition and subtraction
3. `plus_minus_mul`: Addition, subtraction, and multiplication
4. `plus_minus_mul_div`: All four basic operations

Each group contains:
- 100,000 training samples
- 1,000 test samples
- Stored in parquet format at `/data/continual/<group_name>/`

## Training Configuration

### Hardware Requirements
- 8 GPUs
- Tensor Parallel size: 4
- Data Parallel size: 2

### Training Flow
- Sequential training through operator groups
- 3 complete rounds of training
- 15 epochs per group per round
- Total: 180 epochs (3 rounds × 4 groups × 15 epochs)

### Metrics
- Success rate
- Normalized weight change
- Loss function
- Normalized gradient
- Response length
- Metrics saved every 100 steps
- WandB logging enabled
- Checkpoints saved in `checkpoints/ContinualCountdown/`

## Docker Setup

Built on `nvidia/cuda:12.1.0-devel-ubuntu22.04` with CUDA 12.1 support.

### Services
1. Data Generator
   - Generates training and test data
   - Volume mapping: `./data -> /data/countdown`

2. Curriculum Trainer
   - Handles model training
   - Multi-GPU support
   - WandB integration

3. Evaluator
   - Model evaluation service
   - Metrics and plot generation

### Building and Running

```bash
# Build all services
docker compose build

# Generate data
docker compose up data-generator

# Train model
docker compose up curriculum-trainer-1.5b

# Run evaluation
docker compose up evaluator
```
