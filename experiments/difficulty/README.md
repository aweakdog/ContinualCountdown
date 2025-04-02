# Difficulty Testing

This directory contains tools for testing the difficulty of countdown problems using the DeepSeek API. It analyzes and compares difficulty distributions across different datasets and operator groups.

## Setup

1. Create a file named `deepseek_api.txt` in the `experiments/difficulty` directory and add your DeepSeek API key:
```bash
echo "your_api_key_here" > deepseek_api.txt
```

Note: The API key file is already added to `.gitignore` to prevent accidental commits.

2. Build the Docker image (uses the root project's Dockerfile):
```bash
# Build from project root directory
docker compose -f experiments/difficulty/docker-compose.yml build
```

3. Run the difficulty tester:
```bash
# Analyze and compare both continual and tinyzero datasets
docker compose -f experiments/difficulty/docker-compose.yml run --rm difficulty-tester

# Analyze specific datasets or patterns
docker compose -f experiments/difficulty/docker-compose.yml run --rm difficulty-tester \
  --data-dir /app/data/continual \
  --pattern test.parquet

# Analyze specific files
docker compose -f experiments/difficulty/docker-compose.yml run --rm difficulty-tester \
  --input \
  /app/data/continual/plus/train.parquet \
  /app/data/continual/plus_minus/train.parquet
```

## Configuration

### Input Options
- `--data-dir`: One or more base directories to recursively search for parquet files
  - Example: `/app/data/continual /app/data/tinyzero`
- `--pattern`: File pattern to match (default: train.parquet)
- `--input`: Specific files to analyze (optional)

### Output Options
- `--output-dir`: Directory to save results (default: difficulty_results)

### Environment
- Uses the project's root Dockerfile and conda environment
- Additional dependencies are specified in `requirements.txt`
- Runs in the `zero` conda environment

Results will be organized as follows:
```
difficulty_results/
├── combined_difficulty.parquet     # Combined results from all datasets
├── difficulty_distribution.png     # Overall distribution plot
├── difficulty_heatmap.png         # Heatmap comparing datasets
├── dataset_comparison.png         # Continual vs TinyZero comparison
├── continual/                     # Continual learning dataset results
│   ├── continual_overall.png      # Overall distribution for continual
│   ├── plus/                      # Results for plus-only problems
│   │   ├── train_difficulty.parquet
│   │   └── difficulty_distribution.png
│   ├── plus_minus/                # Results for plus-minus problems
│   │   ├── train_difficulty.parquet
│   │   └── difficulty_distribution.png
│   └── plus_minus_mul/            # Results for plus-minus-mul problems
│       ├── train_difficulty.parquet
│       └── difficulty_distribution.png
└── tinyzero/                      # TinyZero dataset results
    ├── tinyzero_overall.png       # Overall distribution for tinyzero
    └── [similar structure as continual]
```

Key visualizations:
1. Individual dataset distributions in their respective folders
2. Overall distribution combining all problems
3. Side-by-side comparison between Continual and TinyZero datasets
4. Heatmap showing difficulty patterns across all groups
```
- API settings: Adjust rate limiting and other parameters in `difficulty_test.py`

## Output Format

The script generates both data files and visualizations:

### Data Files (Parquet)
- Original problem data (numbers, target, operators)
- Classified difficulty (easy, medium, hard)
- Reasoning behind the classification
- Full API response

### Visualizations (PNG)
1. Overall Distribution:
   - `difficulty_distribution.png`: Bar chart showing the distribution of all problems
   - `difficulty_heatmap.png`: Heatmap showing difficulty percentages across datasets

2. Per-Dataset Distributions:
   - `difficulty_distribution_plus.png`: Distribution for plus-only problems
   - `difficulty_distribution_plus_minus.png`: Distribution for plus-minus problems
   - etc.

Each visualization includes:
- Absolute counts of problems in each difficulty level
- Percentage distribution
- Color-coded bars (green=easy, yellow=medium, red=hard)
- Clear labels and titles

## Example Usage

Test a specific group's dataset:
```bash
docker compose run --rm difficulty-tester \
  --input /app/data/continual/plus_minus/test.parquet \
  --output /app/data/difficulty/plus_minus_results.parquet
```
