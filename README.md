# ContinualCountdown
This is a repo for experiments on the Countdown task with continual learning setting.

## Training Approaches

### 1. Regular Continual Learning
Trains on operator groups sequentially with separate training runs:
```bash
docker-compose up continual-trainer-1.5b
```

### 2. Curriculum Learning
Trains on all operator groups in a single run while maintaining optimizer state:
```bash
docker-compose up curriculum-trainer-1.5b
```

## Data Organization

### Operator Groups (100k samples each)
- {+}: Basic addition
- {+ -}: Addition and subtraction
- {+ - *}: Addition, subtraction, and multiplication
- {+ - * /}: All operators

### Data Location
- Training data: `/data/countdown/continual/{group}/`
- Each group has:
  - train.parquet (100,000 samples)
  - test.parquet (1,000 samples)

## Metrics Tracked
- Success rate
- Normalized weight change
- Loss function
- Normalized gradient
- Activations
- Response length

## Technical Details
- Uses verl.trainer.main_ppo framework
- Multi-GPU support (8 GPUs with tensor parallel size 4)
- WandB logging enabled
- Checkpoints saved in `checkpoints/ContinualCountdown/`
- Metrics saved in `/app/metrics/`

## Curriculum Learning Implementation
- Custom CurriculumSampler ensures ordered data processing
- Maintains optimizer momentum across all groups
- Processes 400 epochs total (approximately 100 per group)
- Files processed in order: plus → plus_minus → plus_minus_mul → plus_minus_mul_div