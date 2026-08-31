# Decision Log

## 2026-08-03 — Target venue and framing

- Target venue: Transactions on Machine Learning Research (TMLR).
- Primary framing: general continual decision-making under action-dependent,
  cross-episode physical dynamics.
- Robotics is the high-value application and validation domain, not the sole
  definition of the learning problem.
- Hardware experiments are out of scope for the initial project.
- All required software must have a no-cost path.

Rationale: TMLR explicitly welcomes new learning-task formalizations, evaluation
methods, algorithms with sound empirical validation, and studies that reveal
strengths and weaknesses of learning systems.  A simulator-only project must
therefore deliver general learning insight rather than rely on robot novelty.

## 2026-08-03 — Two reset operations

- `reset_episode`: task reset; physical state persists.
- `reset_lifetime`: task and physical state reset; defines an independent sample.

Rationale: overloading the conventional `reset` operation is a likely source of
silent benchmark errors and train/test leakage.

## 2026-08-03 — Pusher geometric-contact friction is rejected as a mechanism

- Directly mutating Pusher's `geom_friction` did not change object-slider
  velocity decay under controlled MuJoCo rollouts.
- The Pusher object is supported by slider constraints; its baseline damping,
  not the geometric contact-friction parameter, governs that motion.
- The repository will not present this mutation as a contact-wear mechanism.
- Contact-friction validation is deferred to a task whose contact response is
  demonstrably sensitive to the mutated parameter.

Rationale: changing a simulator array without a directionality effect would
create a benchmark artifact rather than a physical mechanism.

## 2026-08-03 — CPU semantic protocol checkpoint

- CPU semantic tasks are Pusher-v5 and Reacher-v5.
- Implemented health mechanisms are persistent actuator wear, recoverable
  thermal derating, and persistent actuated-joint aging.
- Implemented law families are power, threshold, and stochastic shock.
- Every scripted pilot logs one JSONL record per independent lifetime and uses
  paired bootstrap over lifetimes for comparison.
- Learned-policy protocols remain unfrozen until GPU training begins; no pilot
  result is treated as paper evidence or used to select a method.

## Pending decisions

- Final simulator versions and asset licenses.
- Exact degradation laws and physically defensible parameter ranges.
- Whether the method is recurrent RL, latent state-space inference, or a hybrid.
- Final project license and public repository location.
- Final seed count after a pilot variance and power analysis.

## 2026-08-27 — v10 hierarchical thermal held-out result

- The physical design was selected as the first eligible point in a
  predeclared CPU oracle grid, rather than the highest-reward point.
- Calibration seeds 5300--5304 were kept disjoint from confirmatory seeds
  6300--6319.
- The design, training strategy, low-level model hash, source hashes, seed
  list, primary estimand, test, interval, and success threshold were frozen
  before confirmatory training.
- All 80 factorial models completed. The dynamic lifetime-minus-task primary
  contrast was `+1.1269` reward per task with seed-bootstrap 95% CI
  `[0.7950, 1.4328]` and locally pre-specified one-sided `p = 1.15e-6`; the static
  zero-dose memory contrast was exactly zero.
- The result is recorded as a passed learned-comparator study, not as a proof
  of universal memory necessity. Every learned task-reset policy converged to
  Always Low, while a calibration-stage deterministic task-reactive rule
  achieved `+0.9407` per task over Always Low. This comparator limitation must
  accompany the primary result in the paper.

Rationale: preserving the successful frozen endpoint and its strongest
alternative explanation is more informative than either discarding the
confirmatory result or overstating it.

The analysis plan was stored in the local campaign manifest before held-out
training, but it was not externally preregistered. The current hash set also
covers only the low-level model and four central source files. The next frozen
study must bind the full source and analysis tree, environment, commit, and an
external timestamp.

## 2026-08-30 — v12 final attribution and claim boundary

- The v12.2 scoped confirmation passed for hybrid `z=1.5` versus physics `z=0`
  on 100 held-out lifetime seeds: target-OOD mean `+0.9987` reward/task, 95%
  bootstrap CI `[0.7441, 1.2545]`.
- Because that comparison jointly changed the residual and uncertainty margin,
  a fresh-seed 2 x 2 factorial was run on seeds 23000--23099.
- At matched `z=1.5`, the residual effect was `+0.103` with a confidence
  interval crossing zero. The uncertainty effect was `+0.965` without the
  residual and `+0.736` with it. Independent residual attribution did not pass
  the frozen gate.
- The complete `+1.068` composite effect decomposed into `-1.191` base-task
  return, `-0.029` throughput bonus, and `+2.288` avoided-trip contribution.
- Fixed-trajectory reward sensitivity located the mean break-even thermal-trip
  penalty near 40 when the high-power bonus is 2. This is a post-hoc accounting
  analysis, not policy retraining under alternative utilities.
- The paper will claim uncertainty-aware belief supervision for risk-sensitive
  trip avoidance in the scoped simulated setting. It will not claim that the
  learned residual independently caused the full gain or that immediate task
  performance improved.

Rationale: the factorial and decomposition preserve the successful application
result while removing an unsupported mechanism claim and making the utility
preference required for the result explicit.
