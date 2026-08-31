# Integrated Pusher–Reacher evidence and error audit

Status: post-confirmatory audit, 2026-08-31.

## Audit verdict

No critical wiring, seed-leakage, checkpoint-drift, reward-reconstruction, or
analysis-unit error was identified in the frozen Pusher or Reacher studies.
The Reacher confirmatory result is nevertheless a **partial replication**, not
a full gate pass: its uncertainty-aware physics supervisor reached a 2.3%
maximum trip rate against the prespecified 2.0% limit.

The original result and failure flag must remain unchanged. Any subsequently
calibrated Reacher margin is a separately frozen post-confirmatory extension,
not a correction or replacement of the inherited-margin experiment.

## Evidence that can be combined

Both experiments implement the same scientific intervention:

- task state resets while hidden thermal state persists;
- thermal change depends on selected actions;
- the deployed policy observes a noisy sensor rather than exact health;
- a fixed low-level task controller is supervised by a high/low-power decision;
- paired independent lifetimes are evaluated under sensor, cooling, shock, and
  combined shifts;
- the uncertainty margin is compared at fixed physics-belief structure.

The within-task effects agree in direction:

| Task | Uncertainty contrast | Mean reward/task | 95% bootstrap CI | Maximum treatment trip rate |
|---|---|---:|---:|---:|
| Pusher | physics `z=1.5` minus physics `z=0` | +0.965 | [0.733, 1.196] | below the frozen 2% gate |
| Reacher | physics `z=1.5` minus physics `z=0` | +0.721 | [0.474, 0.978] | 2.3% |

Raw rewards must not be pooled across tasks because their reward scales differ.
The defensible cross-task statement is agreement in the direction and
mechanism of the paired within-task effect.

## Central causal interpretation

The learned residual is not the supported cause of the robust gain.

| Task | Residual effect at matched `z=1.5` | Interpretation |
|---|---:|---|
| Pusher | +0.103, CI includes zero | not independently established |
| Reacher | +0.077, CI includes zero | not independently established |

By contrast, the uncertainty margin produced a positive target-OOD mean in
both tasks. Pusher reward decomposition further showed that avoided trip cost
outweighed lower base task return. The supported mechanism is therefore
selective conservatism under hidden persistent health, conditional on thermal
trips carrying meaningful utility cost.

## Strong-baseline interpretation

The selected monolithic Reacher recurrent policy performed well in-domain but
failed under cooling and combined shifts. Its target-OOD effect relative to
physics `z=0` was -9.068 reward/task, and every one of the 100 paired aggregate
effects was negative. This supports an OOD robustness advantage for explicit
model structure in this benchmark.

It does not prove that every end-to-end recurrent architecture must fail. Only
one matched RecurrentPPO family, five development training seeds, and one
selection rule were tested.

The so-called privileged oracle is a privileged **current-state threshold
baseline**, not a planning upper bound. It must not be described as the best
achievable privileged policy.

## Statistical cautions

- The independent unit is a 20-task lifetime; tasks are never treated as
  independent samples.
- Reacher's primary mean effect was strongly positive under the prespecified
  magnitude-sensitive sign-flip test, but only 54 of 100 seed effects were
  positive. The exact sign test was not significant. The claim is an expected
  utility improvement, not a claim that most individual lifetimes improve.
- The Reacher 2.3% versus 2.0% gate miss is 0.3 percentage points but remains a
  failure under the frozen rule.
- The hybrid Reacher supervisor passed 1.4% trip rate and had a positive
  secondary effect, but it cannot be substituted post hoc for the primary
  physics-margin contrast.

## Uncertainty limitation

The Reacher hybrid's nominal interval coverage was 97.1% in-domain but only
26.1% in the combined shift. The margin improved decisions despite severe OOD
miscalibration; it is a heuristic uncertainty-aware supervisor, not a calibrated
probabilistic safety certificate.

## External-validity boundary

Reacher supplies a second task and actuator morphology inside MuJoCo, but both
tasks share the same phenomenological thermal law and simulator. The study does
not establish:

- a motor model calibrated in physical units;
- transfer to real robot telemetry or hardware;
- formal safety guarantees;
- universal advantage across reward preferences;
- algorithmic novelty of residual learning, belief filtering, or shielding.

## Recommended final extension

The inherited Pusher margin narrowly missed the Reacher safety gate. A useful
and scientifically distinct follow-up is to ask whether a margin selected only
on Reacher development seeds restores the safety–utility trade-off.

The extension must:

1. retain the inherited-margin failure unchanged;
2. select one cutoff/margin pair using only seeds 25100–25119;
3. require a safety buffer during selection rather than optimize on the
   confirmatory boundary;
4. freeze the selected pair and all source/checkpoint hashes;
5. evaluate once on new seeds 25300–25399;
6. report the result as morphology-specific calibration, not universal
   zero-shot transfer.

## Extension outcome

The recommended extension was subsequently frozen and completed on a disjoint
set of 100 fresh lifetime seeds (25300--25399). Development-only selection chose
cutoff `0.06` and uncertainty multiplier `z=2.0`. Relative to physics `z=0`,
the selected policy improved target-OOD reward by `+0.692`, bootstrap 95% CI
`[+0.434, +0.962]`, while its maximum condition-level trip rate was 1.6%.
All frozen extension criteria passed.

The selected margin did not improve mean reward over the inherited `z=1.5`
setting: `-0.008`, 95% CI `[-0.121, +0.108]`. Its contribution is safety-margin
recovery without a detectable expected-utility loss. The original inherited-
margin failure (2.3% versus the 2.0% gate) remains the authoritative zero-shot
replication result.

Only 52 of 100 selected-versus-`z=0` lifetime effects were positive (exact sign
test `p = 0.764`). The extension strengthens the mean expected-utility and
safety-calibration claim, not a majority-lifetime improvement claim.
