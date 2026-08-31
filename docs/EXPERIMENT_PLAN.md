# Experiment Plan

## Stage gates

### Gate A — semantic validity

- Episode reset preserves physical state.
- Lifetime reset clears physical state.
- Actions causally change future dynamics.
- Identical visible states can yield different transitions after different
  histories.
- Runs are exactly reproducible from seed and configuration.

Status: implemented for the Tier-0 wear diagnostic.

### Gate B — phenomenon existence

Before training a novel neural method, demonstrate with transparent controllers
that persistent-state evaluation differs materially from per-episode-reset
evaluation.  If no meaningful gap exists over plausible parameter ranges, revise
or stop the corresponding task.

Status: complete for the v10 hierarchical Pusher thermal diagnostic. Its
planning oracle beats the exhaustive deterministic task-reactive rule by
`+0.9739` reward per task. This does not close Gate B for other tasks or
mechanisms.

### Gate C — benchmark validity

- At least three degradation mechanisms.
- At least two simulator/task families.
- Train, interpolation, extrapolation, and stochastic test conditions.
- A planning-oracle gap proves that useful performance remains recoverable;
  state-estimation error is reported separately.
- Sensor-noise and simulator-parameter sensitivity studies.

CPU checkpoint: three health channels, two MuJoCo task families, and
power/threshold/stochastic law families now have semantic tests. The remaining
Gate C evidence is learned-policy performance, sensor-noise sensitivity, and
final parameter-range justification.

Held-out checkpoint: one dynamic-versus-static Pusher diagnostic has a
20-training-seed result. Stochastic conditions, held-out physical parameters,
and cross-task learned-policy agreement remain open.

### Gate D — learning contribution

Run two separate tracks and never describe frozen recurrent inference as
continual parameter learning.

**Track A — frozen adaptation:** compare parameter-matched implementations where
possible:

1. reactive PPO/SAC;
2. frame-stack policy;
3. episode-reset GRU/LSTM;
4. lifetime-persistent GRU/LSTM;
5. Transformer-XL or another long-context baseline;
6. latent-context/system-identification baseline;
7. continual-RL regularization or replay baseline;
8. privileged myopic state oracle and, where tractable, a lifetime-planning
   oracle;
9. proposed dual-timescale method.

**Track B — online learning:** compare replay, regularization, latent-dynamics
updating, and online RL methods under matched interaction, gradient-step, memory,
and wall-clock budgets.  This track reports forgetting and forward transfer;
Track A does not.

The proposed method proceeds only if it improves held-out lifetime performance,
not merely training-law reward.

Status: not complete. V10 compares two reset modes of standard RecurrentPPO;
the learned task-reset arm collapsed to Always Low, while a transparent
task-reactive rule is stronger. No proposed method or strongest-baseline
superiority has been established.


### Gate E — paper-quality evidence

- Fix sample-size and seed lists before final runs.
- Report bootstrap 95% confidence intervals over independent lifetimes.
- Use paired comparisons and effect sizes.
- Correct families of secondary hypothesis tests.
- Repeat the primary conclusion across at least two independent implementations
  or physics engines where feasible.
- Release code, exact configs, raw per-lifetime metrics, and analysis scripts.

Status: partial. V10 froze disjoint seed sets and a local analysis manifest,
retained all 20 paired seed effects, and produced a seed-bootstrap interval and
reproducible figure script. It was not externally preregistered, the hash set
does not bind the complete dependency graph, and the cross-task/release
requirements remain incomplete.

## Primary endpoints

1. **Normalized area under the lifetime curve:** AULC relative to preregistered
   random and planning-oracle anchors.
2. **Mean episode return:** lifetime return divided by its fixed episode count.
3. **Success AUC:** area under success-versus-lifetime-progress curve.
4. **Worst-window success:** minimum rolling success rate.

## Secondary endpoints

- cumulative damage and energy;
- performance at matched damage budgets;
- adaptation/recovery time after a regime change;
- latent-state calibration error where an estimator is present;
- catastrophic-failure probability;
- compute, samples, and wall-clock time.

## Planned ablations

- reset slow memory at every episode;
- reset fast memory never / at every episode;
- remove action history;
- remove proprioception or vision;
- match parameter count and context length;
- train with exogenous drift instead of action-dependent degradation;
- remove damage term from training reward but retain separate evaluation;
- vary lifetime length and degradation timescale;
- vary observability and sensor noise;
- train/test on held-out degradation-law families.

## Compute policy for the available machine

- Use CPU-parallel state simulation and one GPU learner initially.
- Cap concurrent heavy jobs to avoid exceeding 32 GB system RAM.
- Run state observations first; enable RGB-D only for representative experiments.
- Store scalar logs and periodic checkpoints by default, not every rendered frame.
- Conduct short successive-halving sweeps before full-lifetime seed runs.

## Stop/revise criteria

Revise the study before large-scale training if any occurs:

- the reset-bias effect is negligible under physically defensible ranges;
- a lifetime-planning oracle is also poor, indicating an invalid or impossible
  task;
- simple lifetime EMA or recurrent baselines close the entire proposed-method gap;
- the conclusion depends on one seed, one task, or one arbitrary reward weight;
- held-out degradation laws reverse the main conclusion;
- simulator artifacts rather than persistent dynamics explain the result.
