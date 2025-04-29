# Gradient Analyzer & ReDo Utilities Usage

This module provides utilities for gradient-based analysis and neuron reset (ReDo) in distributed training with the verl framework. You can control the frequency and behavior of these utilities directly from your training shell script.

## Parameters

You can set the following parameters for both the actor and critic in your training shell script (e.g., `scripts/train_continual_countdown_3b_curriculum.sh`) by passing them as arguments to your Python training command:

### Actor Parameters
- `actor_rollout_ref.actor.redo_metric_freq`: Frequency (in steps) to calculate and print global nullspace and zero gradient ratios for the actor model.  
  **Example:** `actor_rollout_ref.actor.redo_metric_freq=1000`
- `actor_rollout_ref.actor.redo_reset_freq`: Frequency (in steps) to perform gradient-based neuron reset for the actor model.  
  **Example:** `actor_rollout_ref.actor.redo_reset_freq=5000`
- `actor_rollout_ref.actor.redo_tau`: The tau threshold for neuron reset (see ReDo documentation for details).  
  **Example:** `actor_rollout_ref.actor.redo_tau=0.1`

### Critic Parameters
- `critic.redo_metric_freq`: Frequency (in steps) to calculate and print global nullspace and zero gradient ratios for the critic model.  
  **Example:** `critic.redo_metric_freq=1000`
- `critic.redo_reset_freq`: Frequency (in steps) to perform gradient-based neuron reset for the critic model.  
  **Example:** `critic.redo_reset_freq=5000`
- `critic.redo_tau`: The tau threshold for neuron reset for the critic model.  
  **Example:** `critic.redo_tau=0.1`

## How to Use

1. **Edit your training shell script** (e.g., `scripts/train_continual_countdown_3b_curriculum.sh`).
2. **Add the desired parameters** to the `python3 -m verl.trainer.main_ppo ...` command, for example:

```bash
python3 -m verl.trainer.main_ppo \
    ... \
    actor_rollout_ref.actor.redo_metric_freq=1000 \
    actor_rollout_ref.actor.redo_reset_freq=5000 \
    actor_rollout_ref.actor.redo_tau=0.1 \
    critic.redo_metric_freq=1000 \
    critic.redo_reset_freq=5000 \
    critic.redo_tau=0.1 \
    ...
```
3. **Run your script as usual.** The specified frequencies and tau values will be used for gradient analysis and neuron reset during training.

## Notes
- The metrics are computed globally across all processes (distributed-safe).
- Only the process with rank 0 will print the metrics to avoid log spam.
- You can change these values for each run to experiment with different settings.

## Example

To print metrics every 500 steps and reset neurons every 2000 steps for both actor and critic, use:
```bash
    actor_rollout_ref.actor.redo_metric_freq=500 \
    actor_rollout_ref.actor.redo_reset_freq=2000 \
    critic.redo_metric_freq=500 \
    critic.redo_reset_freq=2000 \
```

---

For more details, see the implementation in `gradanalyzer.py` and `redo.py`.
