# GPU Smoke and Small-Sweep Results

Date: 2026-08-04

These are engineering smoke results used to choose a stable initial PPO
configuration. They are not benchmark results, are not a method comparison, and
must not be cited as paper evidence.

## PPO/SAC smoke comparison

Both runs used Pusher-v5, endogenous actuator wear, eight workers, one million
environment steps, seed 1000, and 20 deterministic evaluation episodes.

| Algorithm | Mean evaluation reward | Evaluation std. |
|---|---:|---:|
| PPO | -26.650 | 3.606 |
| SAC | -31.571 | 6.649 |

PPO is the provisional development baseline because its single-seed smoke run
had higher reward and lower evaluation variation. This does not establish a
statistically supported PPO-versus-SAC conclusion.

## PPO short sweep

Each trial used 250,000 steps, eight workers, endogenous wear in Pusher-v5, and
20 evaluation episodes.

| Learning rate | Seed | Mean evaluation reward | Evaluation std. |
|---:|---:|---:|---:|
| 3e-4 | 1000 | -33.819 | 5.186 |
| 3e-4 | 1001 | -36.988 | 3.313 |
| 1e-4 | 1000 | -39.089 | 4.141 |
| 1e-4 | 1001 | -40.349 | 4.184 |

Across the two development seeds, `3e-4` averaged `-35.403` and `1e-4`
averaged `-39.719`. The next PPO development run therefore retains
`learning_rate=3e-4`. The 250,000-step sweep is deliberately too short to select
the final protocol or estimate uncertainty for the paper.

## PPO duration check

Development seed 1001 was trained with the selected `3e-4` learning rate and
eight workers. The two independent runs confirm that the 250,000-step sweep was
undertrained and that one million steps is the correct initial PPO development
budget.

| Training steps | Mean evaluation reward | Evaluation std. |
|---:|---:|---:|
| 500,000 | -31.586 | 4.212 |
| 1,000,000 | -24.063 | 3.095 |

The one-million-step run improved mean reward by `+7.523` and reduced evaluation
variation. For this small MLP policy and 8-worker MuJoCo setup, completed runs
take minutes rather than the earlier conservative hours-scale estimate. This
does not predict recurrent or Transformer training time, where learner updates
and sequence storage are materially more expensive.

## Next action

Freeze the PPO development configuration at one million steps, then implement
and test episode-reset and lifetime-persistent recurrent baselines. Do not yet
run the full multi-seed benchmark.
