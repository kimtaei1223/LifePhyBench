# Recurrent Baseline Protocol

The recurrent baselines differ only in the boundary at which recurrent state is
reset. They are frozen-adaptation baselines: network weights do not update during
evaluation.

| Baseline | Task episode boundary | Physical health | Recurrent state reset |
|---|---|---|---|
| `episode` | Returned as Gym terminal transition | persists through ordinary reset | every task episode |
| `lifetime` | Internal nonterminal transition | persists through ordinary reset | only at lifetime boundary |

`LifetimeEpisodeScheduler` implements the first condition. `LifetimeStreamWrapper`
implements the second: it resets the task internally after each task terminal
state, returns the next task observation without a terminal signal, and emits a
Gym truncation only after the fixed number of task episodes in a lifetime.

This distinction is essential. A standard RecurrentPPO policy clears its LSTM
state on a Gym done signal, so merely preserving MuJoCo health while returning
ordinary task terminals would not create a lifetime-memory baseline.

## Development campaign

The first campaign is Pusher-v5 with endogenous actuator wear, eight workers,
one million steps, learning rate `3e-4`, and development seeds 1000 and 1001.
It executes four runs sequentially: episode/lifetime memory by two seeds.

```bash
./.venv-mujoco/bin/python scripts/run_recurrent_campaign.py
```

This is a development comparison, not the final multi-task, multi-law benchmark.
For a direct comparison between the two reset conventions, the primary reported
quantity is mean reward per task episode, evaluated on the same number of task
episodes and seeded deterministically.  Independent lifetimes remain the unit
of uncertainty for the final benchmark; task-transition scores are retained
only as diagnostic logs.

## Evaluation safeguard

`evaluate_policy` treats a Gym terminal transition as the end of one evaluation
episode.  That is not comparable here: an `episode` policy emits a terminal
after every task, whereas a `lifetime` policy emits one only after 20 tasks.
`lifephybench.recurrent_evaluation.evaluate_task_episodes` therefore splits
the evaluation return at the explicit `lifephy/inner_task_boundary` marker
without resetting the recurrent state there.  It resets recurrent state only
at an actual Gym lifetime boundary.
