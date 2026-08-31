# Physical-state benchmark redesign gates

## Why a redesign is required

The fair recurrent experiment controlled Gym termination, GAE boundaries, the
task-boundary marker, and the exogenous dose. Lifetime memory outperformed
forced task-boundary reset, but the same effect remained in the static-health
control where thermal load stayed 0.0 and actuator efficiency stayed 1.0.

The current Pusher protocol therefore measures a general cost of resetting
recurrent state. It does not isolate an advantage caused by a changing physical
health state. No further GPU campaign should be launched under that protocol
for a physical-state-specific claim.

## CPU gates completed on 2026-08-17

1. **MuJoCo physical response:** `tests/test_mujoco_pusher.py` passed 15/15.
   In particular, the thermal derating counterfactual verifies that a hot
   actuator produces a smaller rollout response under the same action.
2. **Action-history identifiability:** the endogenous toy audit gave time-only
   wear RMSE `0.005256` and action-history RMSE approximately `0`; the
   exogenous clock control reversed this preference as intended.
3. **Planning diagnostic:** both the myopic privileged controller and the
   lifetime DP oracle achieved success AUC `1.0` in the current deterministic
   toy configuration. This is a semantic smoke test, not evidence of a useful
   planning gap.
4. **Canonical thermal probe:** over canonical task seeds 700--709, the
   task-boundary observations of the cold and hot states were exactly equal
   (maximum absolute difference `0.0`). The hot state retained thermal load
   `0.125`, and the same five-step probe produced a minimum response-norm gap
   of `0.2925` in the cold-minus-hot direction.

The audit outputs are retained under `outputs/cpu_semantic_gates/`.

## Revised benchmark: canonical-reset thermal probe

The next environment must use a canonical reset at every task boundary: the
robot state, object state, goal geometry, and reward schedule are matched
across task episodes. The hidden thermal state alone persists through the
lifetime. A short, fixed probe action sequence at the beginning of each task
must therefore have dynamics that depend on thermal state but not on any
previous task geometry.

The controller should then choose a subsequent target-reaching action under
the same visible task observation. The appropriate action depends on the
unobserved actuator efficiency. A lifetime-memory policy can infer that
efficiency from prior action-response history; a task-reset policy cannot
recover it immediately after the canonical boundary.

## Required CPU acceptance criteria before GPU training

1. **Canonical-reset equivalence:** after every task boundary, all non-health
   simulator state and all policy observations must be equal for paired
   low-health and high-health rollouts.
2. **Counterfactual dynamics:** applying the same probe sequence to paired
   low-health and high-health states must produce a preregistered minimum
   separation in actuator-response norm.
3. **History necessity:** before the probe, the visible observation must not
   predict health above the static baseline; after prior probes, action-response
   history must predict health substantially better than elapsed task index.
4. **Decision relevance:** a CPU privileged-health oracle must outperform a
   health-blind myopic controller on the lifetime objective by a preregistered
   practical margin across at least ten seeds.
5. **Static negative control:** with thermal dose fixed to zero, the
   health-specific interaction (degrading minus static memory effect) must be
   statistically compatible with zero under the same protocol.

## GPU entry criterion

Only after all five CPU gates pass should the project run a one-seed GPU wiring
pilot. The full held-out GPU campaign must freeze the seed list, thermal range,
probe sequence, primary endpoint, and static-control analysis before training.

## CPU calibration status

The deterministic toy planning smoke configuration is too easy: the planning
oracle has no meaningful advantage over the myopic privileged controller.
A 144-setting CPU calibration sweep found a candidate with a planner return
advantage of `0.2068` and success-AUC advantage of `0.0833`:

- horizon `2`, target `0.8`, wear rate `0.1`;
- minimum gain `0.15`, damage cost `0.5`;
- 12 episodes per lifetime.

This selection is calibration only, not held-out evidence. Before it can serve
as a planning gate, freeze the candidate and evaluate it under a separately
specified variation or stochastic task distribution. No inferential claim is
made from the search itself.

The first frozen variation check used a coarse DP approximation and failed at
target `0.75`. A solver-convergence audit then showed that this sign reversal
was numerical: at that target, the planner gap moved from `-0.0436` with the
coarse 21-action / `0.05`-position / `0.02`-wear grid to `+0.2804` and
`+0.3843` at progressively finer grids. The coarse output is retained at
`outputs/cpu_semantic_gates/held_out_toy_planning_gap_validation.json` for
auditability, but it is not a valid planning gate.

The frozen variation check was rerun with the benchmark's default converged
41-action / `0.02`-position / `0.005`-wear solver. All four unseen variations
passed the `0.05` practical return margin, with return gaps from `+0.2496` to
`+0.3843`. The successful result is stored in
`outputs/cpu_semantic_gates/held_out_toy_planning_gap_validation_high_resolution.json`.
The validation remains deterministic: repeated seed labels verify
reproducibility, not stochastic statistical inference.
