# TMLR manuscript blueprint

Status: evidence-grounded writing plan, corrected 2026-09-01.

This document fixes the communication structure of the paper. It does not
change any experimental endpoint or promote post-confirmatory evidence into a
primary result.

## Recommended title

**Uncertainty-Aware Belief Supervision under Persistent Hidden Thermal
Dynamics: Evidence from Two Robot-Control Tasks**

The title leads with the actionable finding rather than an unsupported claim of
algorithmic novelty or physical realism.

Alternatives:

1. **Uncertainty-Aware Belief Supervision under Persistent Hidden Thermal
   Dynamics**
2. **Calibrating Conservatism for Persistent Hidden Thermal Dynamics in Robot
   Control**

## One-sentence contribution

In two controlled MuJoCo tasks, an uncertainty margin around an explicit
persistent-state belief improves mean risk-sensitive lifetime utility under
deployment shifts, while empirical trip-rate gates near 2% vary with policy
composition and finite evaluation sample.

## Abstract draft

Robot-control benchmarks commonly reset the environment between tasks, hiding
failure modes caused by physical state that accumulates across those resets.
We study fixed low-level controllers in two MuJoCo manipulation tasks where
task state resets but an unobserved, action-driven thermal state persists and
changes later actuator dynamics. An explicit belief supervisor uses a noisy
temperature signal and an uncertainty margin to choose between high- and
low-power control. Under prespecified sensor and cooling shifts, a Pusher
physics-only margin improved mean utility by `+0.965` reward per task (95%
bootstrap CI `[0.733, 1.196]`) but reached a 2.15% maximum trip rate; the
corresponding hybrid policy's margin effect was `+0.736` (`[0.524, 0.950]`) and
its maximum was 1.3%. On Reacher, the inherited physics-only margin retained a
positive mean effect (`+0.721`, `[0.474, 0.978]`) but reached 2.3%, while the
secondary hybrid policy reached 1.4%. A separately frozen development
procedure selected a more conservative physics-only margin; on 100 new
lifetimes it reached 1.6% and retained a positive mean effect (`+0.692`,
`[0.434, 0.962]`). The inherited margin also passed on that second sample at
1.9%, and the calibrated setting had no detectable mean-utility advantage.
Fresh factorial evidence does not establish the residual as an independent
source of improvement, and only 52 of 100 calibrated Reacher lifetime effects
were positive. These results support uncertainty-aware expected-utility
claims, not calibration superiority or a transferable safety guarantee.

## Reader takeaway in plain language

The controller tracks a hidden physical condition that survives ordinary task
resets. A conservative uncertainty buffer helps on average, but whether a
near-boundary point tolerance passes depends on the complete policy and fresh
sample. Target-task calibration can select a passing policy, but these data do
not show that calibration is uniquely responsible for passing.

## Claim hierarchy

| Level | Claim | Evidence | Boundary |
|---|---|---|---|
| Primary | Uncertainty margins improve mean risk-sensitive lifetime utility under the tested persistent hidden dynamics and shifts. | Pusher fresh factorial and inherited Reacher confirmation. | Expected mean, not most lifetimes; utility depends on trip cost. |
| Deployment | Mean-utility improvement does not imply that a complete policy passes a near-boundary empirical trip-rate tolerance. | Physics-only `z=1.5` reaches 2.15% on Pusher and 2.3% on original Reacher; hybrid policies reach 1.3% and 1.4%. | A point rule on finite samples, not a safety probability bound. |
| Calibration | Target-task development-only calibration can select a policy that passes a fresh tolerance gate. | Reacher cutoff `0.06`, `z=2.0`: 1.6% maximum; inherited `z=1.5` is 1.9% on the same fresh seeds and their mean-difference CI crosses zero. | Feasibility only; no calibration necessity, superiority, or unique causal recovery. |
| Mechanism | Selective conservatism and avoided trip cost, rather than an independently verified residual gain, explain the robust benefit. | Pusher 2 x 2 factorial and reward decomposition; Reacher residual contrast. | Not a formal causal model of hardware failures. |
| Baseline | Explicit structure outperformed the tested recurrent baseline under target OOD. | Selected RecurrentPPO result on Reacher. | One architecture, training budget, and selection procedure only. |

## Non-claims that must appear explicitly

- No real-motor thermal calibration or hardware validation.
- No formal safety certificate or probability-calibrated OOD guarantee.
- No universal superiority over recurrent, adaptive, or model-free methods.
- No majority-lifetime benefit on calibrated Reacher.
- No independent residual advantage at the deployed uncertainty margin.
- No claim that calibration is necessary, superior, or uniquely causes a fresh
  tolerance pass.
- No claim that the privileged threshold comparator is an optimal oracle.
- No claim that the inherited Reacher confirmation passed its complete frozen
  gate.

## Paper structure and evidence placement

### 1. Introduction

Use four short moves:

1. Ordinary task resets can erase a physically persistent state from the
   benchmark while the controller still experiences its future consequences.
2. Explicit beliefs offer a transparent response, but their uncertainty margin
   becomes a deployment parameter.
3. Ask two questions: does the margin help across tasks, and how do policy
   composition and development-only margin selection affect a point tolerance?
4. State the three findings: positive mean effects on two tasks, physics-only
   versus hybrid gate differences, and a feasible but non-superior calibrated
   Reacher policy.

Do not lead with the learned residual. It was part of the experimental path but
is not the established cause of the final result.

### 2. Related work

Organize by the scientific gap rather than by an undifferentiated paper list:

1. persistent degradation and selective resets in RL benchmarks;
2. thermal-aware and wear-aware robot control;
3. belief-state control and latent-dynamics inference;
4. uncertainty-aware shielding and supervisory switching;
5. continuing, recurrent, and adaptive RL baselines.

For every closest work, compare the reset semantics, source of degradation,
observability, adaptation mechanism, OOD protocol, and evidential unit. State
that individual components have prior art; the contribution is the controlled
two-task evidence and the transfer/calibration lesson. Source the comparison
from [`LITERATURE_AUDIT_2026-08-31.md`](LITERATURE_AUDIT_2026-08-31.md) and
[`NOVELTY_LEDGER.md`](NOVELTY_LEDGER.md), then refresh it on the submission
date.

### 3. Problem setting

Define one lifetime as 20 tasks. Explain exactly which task variables reset and
which thermal state persists. Give the action-driven transition, noisy
observation, actuator-efficiency effect, trip event, and risk-sensitive utility.
Make clear that the model is phenomenological and shared by both tasks.

### 4. Supervisor and comparators

Describe the fixed low-level controller, explicit physics belief, uncertainty
margin, optional residual, privileged current-state threshold, and tested
RecurrentPPO baseline. Separate a policy's available observations from values
used only for analysis. Include parameter and training budgets, while bounding
all comparator claims to their tested configurations.

### 5. Evaluation protocol

Present the chronology visibly:

1. Pusher development and v12.2 confirmation;
2. fresh Pusher residual-by-uncertainty factorial;
3. inherited-margin Reacher confirmation;
4. post-confirmatory Reacher development calibration;
5. one evaluation on disjoint fresh Reacher lifetimes.

List seed roles, OOD conditions, primary estimands, 2% gate, bootstrap method,
paired sign-flip test, exact sign test, and why lifetime—not task—is the
independent unit. The chronology prevents the calibrated extension from being
mistaken for a repaired primary endpoint.

### 6. Results

Recommended order:

1. **Pusher mechanism:** show the 2 x 2 factorial and establish that uncertainty
   is the robust component.
2. **Policy composition across tasks:** show positive mean effects together
   with physics-only failures and hybrid passes in the designated tests.
3. **Target calibration:** show the development frontier and fresh 1.6% result,
   while also reporting inherited 1.9% on the same seeds and the inconclusive
   calibrated-versus-inherited mean contrast.
4. **Cautions:** report positive-lifetime counts, recurrent comparison,
   condition-level heterogeneity, and reward sensitivity.

Never pool raw rewards across tasks. Compare paired within-task effects and
their direction.

### 7. Discussion and limitations

The practical workflow is: model the persistent state explicitly, select a
conservatism margin using target development lifetimes and a buffered
tolerance, freeze it, then test once. Present this as a feasible procedure, not
a demonstrated necessity. Discuss sample sensitivity, why positive expected
utility can coexist with only 52/100 positive lifetime effects, and why an
empirical trip rate is not a safety proof. End with the simulator, shared-law,
hardware, reward-preference, and baseline limitations.

### 8. Conclusion

Use one restrained paragraph: uncertainty-aware supervision helped on average
in both tasks; physics-only and full-hybrid policies differed at the point
gate; target-specific calibration selected a passing policy but did not prove
necessity or superiority; physical and algorithmic generality remain open.

## Main-paper figure and table map

| Placement | Artifact | Purpose |
|---|---|---|
| Figure 1 | New schematic derived from the environment specification | Show selective reset, persistent hidden thermal state, noisy sensing, and supervisory switch. |
| Figure 2 | `paper_artifacts/physics_residual_v12_3/figure_v12_3_factorial.svg` | Attribute the Pusher effect to uncertainty versus residual. |
| Figure 3 | `paper_artifacts/reacher_replication/figure_reacher_replication_summary.svg` | Show the original inherited test and separately frozen calibrated evaluation without implying unique recovery. |
| Table 1 | New compact protocol table | Align tasks, seed roles, OOD conditions, estimands, and gates. |
| Table 2 | Pusher and Reacher within-task contrasts | Report means, intervals, tests, positive counts, and trip rates without pooling reward scales. |
| Table 3 | `table_reacher_calibration_frontier.csv`, condensed | Make the development-only selection rule auditable. |

Place the full reward-sensitivity curve, condition tables, recurrent training
details, extra ablations, protocol hashes, and full calibration frontier in the
appendix. Keep the inherited Reacher failure and calibrated result in the main
paper because appendix review is optional.

## Claim-to-artifact map

| Claim | Authoritative source |
|---|---|
| Pusher uncertainty effect and residual attribution | [`PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md`](PHYSICS_RESIDUAL_V12_FINAL_RESULTS.md) and `paper_artifacts/physics_residual_v12_3/` |
| Reacher inherited transfer result | [`REACHER_REPLICATION_FINAL_RESULTS.md`](REACHER_REPLICATION_FINAL_RESULTS.md) and `paper_artifacts/reacher_replication/table_reacher_contrasts.csv` |
| Reacher calibrated fresh-test result | [`REACHER_REPLICATION_FINAL_RESULTS.md`](REACHER_REPLICATION_FINAL_RESULTS.md) and `paper_artifacts/reacher_replication/table_reacher_calibration_frontier.csv` |
| Cross-task interpretation and limitations | [`INTEGRATED_PUSHER_REACHER_AUDIT.md`](INTEGRATED_PUSHER_REACHER_AUDIT.md) |
| Clean-environment reproducibility | [`CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md`](CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md) |
| Closest-work boundary | [`LITERATURE_AUDIT_2026-08-31.md`](LITERATURE_AUDIT_2026-08-31.md) and [`NOVELTY_LEDGER.md`](NOVELTY_LEDGER.md) |

## Final writing checks

- The Abstract answers problem, intervention, test, result, implication, and
  limitation without requiring robotics background.
- Every number in the Abstract appears in an immutable result artifact.
- “Safety” is qualified as a frozen empirical trip-rate tolerance.
- Pusher and Reacher reward scales are never pooled.
- The zero-shot Reacher failure precedes the calibrated extension.
- Mean effects are not translated into claims about most lifetimes.
- All negative evidence remains in the main text or an explicitly referenced
  table.
- The manuscript contains no identifying paths, hostnames, usernames, emails,
  repository owner names, acknowledgments, or self-identifying links during
  double-blind review.

## Immediate next implementation step

Complete and compile the anonymous TMLR manuscript, then run the full
claim-to-evidence, privacy, archive-size, and clean-reproduction audits. Refresh
the literature ledger immediately before submission.
