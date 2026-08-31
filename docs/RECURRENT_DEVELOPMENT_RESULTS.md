# Recurrent Development Results

This record is a configuration check, not a result for the paper.  It uses
Pusher-v5, endogenous actuator wear, 8 workers, RecurrentPPO (`MlpLstmPolicy`),
one million training transitions, learning rate `3e-4`, and two development
seeds.  Every trained checkpoint was re-evaluated deterministically for 200
task episodes (10 lifetimes) on CPU; evaluation does not perform learning.

| Recurrent-state reset | Seed 1000 | Seed 1001 | Mean across development seeds |
|---|---:|---:|---:|
| Every task episode (`episode`) | -31.72 | -30.88 | -31.30 |
| Every 20-task lifetime (`lifetime`) | -36.19 | -34.06 | -35.13 |

The initial metadata written by the training jobs had different return units:
one task for `episode` and an entire 20-task lifetime for `lifetime`.  Those
numbers must not be compared.  The values above replace that comparison with a
common per-task-episode unit.

At this one configuration, lifetime memory is lower by 3.83 reward points on
the two-seed development mean.  This is insufficient to draw a scientific
conclusion: it may reflect recurrent optimization, the training schedule, or
the benchmark parameterization.  The next required baseline is a finite
history (frame-stack) PPO policy, which tests whether recent observations alone
are sufficient before assigning value to persistent recurrent memory.
