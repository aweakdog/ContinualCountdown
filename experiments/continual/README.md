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

### Running Data Generation

The data generation is containerized using Docker for reproducibility. To generate the data:

1. Build the Docker image:
```bash
docker compose build data-generator
```

2. Run the data generation script:
```bash
docker compose run --rm data-generator
```

The script uses the CUDA-enabled base image and will save the generated data in the mounted volume at `./data/continual/`. Progress indicators and color-coded output will help track the generation process.
