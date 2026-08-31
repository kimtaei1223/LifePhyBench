# Thermal protection commitment protocol

## Motivation

The canonical thermal-probe campaign found a lifetime-memory benefit in the
dynamic cell, but the dynamic-minus-static interaction was unresolved. Frozen
policy cross-evaluation showed that the same benefit persisted after thermal
physics was removed. The previous task therefore did not identify memory use
specific to persistent physical health.

## Redesigned decision

At the first action of every canonical Pusher task, the policy commits to a
high-power or low-power actuator mode. The policy observation at that boundary
is identical for cold and hot physical states.

- High power applies the nominal robot command and receives a one-time `2.0`
  throughput bonus when operation is safe.
- Low power applies `0.40` of the commanded action and receives no throughput
  bonus.
- High power at thermal load `0.10` or above triggers a protection shutdown and
  task reward `-75.0`.
- The mode remains fixed until the next task boundary.

The throughput bonus is explicit benchmark reward shaping, representing the
productivity advantage of unrestricted actuator operation. It must not be
described as an emergent MuJoCo quantity. Both modes pay the Pusher control
cost of the policy's requested action. Low mode therefore reduces physical
actuation but does not receive an unintended control-cost discount.

In the dynamic cell, thermal load is hidden, action-coupled, and persistent
across 20 canonical tasks. A lifetime-memory policy can integrate previous
actions and responses to choose high power while cold and low power near the
trip threshold. A task-reset policy must make the same decision from the same
boundary observation without cross-task state. In the static cell thermal load
is fixed at zero, so high power is always optimal and persistent memory has no
health-specific role.

## CPU entry gates

The current retained report is
`outputs/cpu_semantic_gates/thermal_commitment_gate_v4.json`. All criteria
passed:

- maximum cold/hot boundary-observation difference: `0.0`;
- high-minus-low five-step response-norm gap: `3.2998`;
- history thermal estimator RMSE: `8.73e-9`, versus clock-only RMSE `0.1273`;
- privileged mode oracle minus best health-blind fixed mode: reported by the
  v2 gate;
- corresponding static-control oracle gap: `0.0`.

These are semantic design checks, not learned-policy evidence.

## GPU entry sequence

1. Run one calibration seed in the four dynamic/static × task/lifetime cells
   and evaluate conditional mode selection.
2. Verify model serialization, mode-selection logging, trip logging, dynamic
   thermal change, and exact static health.
3. Inspect the pilot before freezing held-out seeds and launching a full
   campaign. Pilot reward differences are not used as confirmatory evidence.

## Calibration history

The first GPU wiring pilot used bonus `2.0` and trip penalty `25.0`. All four
policies selected low power on every deterministic evaluation task. Forced-mode
evaluation then exposed two reward-design problems:

- static high-power break-even bonuses were `4.65` and `7.00` for the task and
  lifetime policies, respectively;
- a `-25` early trip was better than completing a typical `-43` to `-47` task,
  creating an incentive to trip deliberately.

The v2 calibration candidate therefore uses bonus `10.0` and trip penalty
`75.0`. These values are calibration choices, not confirmatory findings. They
must pass a new learned-policy gate before held-out seeds are frozen.

The v2 300k-step calibration failed that gate. Static lifetime learned high
power, but static task and dynamic lifetime remained fixed at low power. The
remaining structural confound was that low mode paid Pusher control cost on
the scaled applied action, making it artificially cheaper. The v3 candidate
charges both modes using the requested action, retains trip penalty `75.0`, and
returns the throughput bonus to `2.0`. The v1/v2 outputs retain their original
applied-action cost semantics and are not overwritten.

The v3 300k-step calibration fixed the static behavior: both static policies
selected high power on every evaluation task. Dynamic lifetime selected high
power on 25% of tasks, but its boundary thermal load had median `0.0790`, 75th
percentile `0.1152`, and maximum `0.1312`; the old `0.70` trip threshold was
unreachable. The v4 calibration threshold is therefore `0.10`, selected from
that retained distribution before v4 training. At load `0.10`, the five-step
same-action MuJoCo response gap is checked to exceed `0.20`. Heat rate remains
`0.1`; no reward parameter is changed from v3.

Before observing the v2 learned-policy results, the following behavioral entry
criteria are fixed:

- static task and lifetime policies: high-power selection rate at least `0.80`;
- dynamic lifetime policy: at least 40 cold and 40 hot boundary decisions;
- dynamic lifetime cold high-power rate at least `0.60`;
- dynamic lifetime hot high-power rate at most `0.40`;
- dynamic lifetime thermal-trip rate at most `0.20`.

The future primary estimand remains the paired dynamic-minus-static difference
of lifetime-minus-task memory effects. A full campaign is not authorized by
this document until the wiring pilot passes.

## v4 verdict and v5 curriculum calibration

The v4 seed-4995 calibration completed all four 300k-step cells. Both static
policies selected high power on every evaluation task, confirming that the
requested-action control-cost correction removed the low-mode discount. Both
dynamic policies nevertheless selected low power on every task. Consequently
the dynamic lifetime policy produced no hot boundary decisions and failed the
frozen learnability gate. This is a behavioral failure, not a wiring failure.

The first v4 wiring-validation attempt also exposed a stale validator constant:
the validator compared metadata against the old `0.70` threshold even though
the manifest and all four runs correctly used `0.10`. The validator now reads
the expected threshold from the immutable campaign manifest. Revalidation
passes the wiring checks and retains the failed behavioral report separately.

The v5 calibration introduces a training-only threshold curriculum to avoid
premature collapse to the always-low policy. For each training worker, the trip
threshold decreases linearly from `0.70` at lifetime 0 to `0.10` at lifetime
10 and remains `0.10` thereafter. The evaluation environment never uses the
curriculum: it is instantiated directly at the target threshold `0.10` for all
400 evaluation tasks. Thermal health remains hidden, reward parameters and
physical dynamics are unchanged, and v5 remains calibration rather than
confirmatory evidence. The frozen v2 behavioral gates are not changed.

## v5 long-budget stress test

The 300k-step v5 calibration retained the always-low dynamic policy. Before
changing the task again, a frozen long-budget stress test distinguishes
undertraining and seed sensitivity from structural failure. It is exploratory
calibration and cannot be reported as confirmatory evidence.

- calibration seeds: `4990`, `4991`, `4992`, `4993`;
- cells: dynamic task-memory and dynamic lifetime-memory only;
- budget: 2M training steps and 1,000 evaluation tasks per model;
- training curriculum: `0.70` to `0.10` over the first 10 lifetimes per worker;
- evaluation threshold: fixed `0.10` with no curriculum;
- all other v5 physical and reward parameters remain unchanged.

A lifetime seed passes only if it has at least 40 cold and 40 hot decisions,
selects high power on at least 60% of cold decisions, selects high power on at
most 40% of hot decisions, and trips on at most 20% of decisions. The stress
test passes only if at least three of the four lifetime seeds pass every
criterion. Failure blocks the confirmatory campaign and triggers structural
redesign. Static cells are omitted because independent v4 and v5 calibrations
already produced 100% high-power selection in both memory arms; they must be
restored in any later confirmatory factorial campaign.

## v6 decision-only mode loss

The four-seed 2M v5 stress test completed all eight dynamic models but produced
zero passing lifetime seeds. Every deterministic policy selected low power on
every task, so increased budget and seed replication ruled out the 300k budget
as a sufficient explanation.

An optimization-semantics audit then identified a causal credit mismatch. The
mode action coordinate affects the environment only on the first transition of
each 100-step task, while the default diagonal-Gaussian PPO log probability
sums all action coordinates on every transition. Thus 99 of every 100 mode-loss
contributions came from transitions on which that coordinate could not affect
the environment. The retained CPU report is
`outputs/cpu_semantic_gates/thermal_commitment_credit_mask_v6.json`.

The v6 policy leaves the environment, reward, observations, recurrent state,
curriculum, and physical dynamics unchanged. It excludes only the latched mode
coordinate from PPO log probability and entropy after `mode_selected` becomes
true. The seven physical-action coordinates remain in the objective on every
transition, and the mode coordinate remains in the objective on every actual
commitment decision. Both stored rollout log probabilities and training-time
log probabilities use the same mask.

V6 uses calibration seed `4989`, 300k steps and 400 evaluation tasks in the
full dynamic/static by task/lifetime wiring factorial. It is calibration, not
confirmatory evidence, and uses the unchanged frozen behavioral gates. A full
campaign remains blocked unless all wiring checks and the learned-policy gate
pass.

## v6 local mode counterfactual

V6 passed every wiring check but again learned the always-low dynamic policy.
A paired one-task counterfactual then forced High and Low from identical
canonical task states at controlled initial thermal loads, using each learned
physical controller under endogenous thermal physics. The retained report is
`outputs/thermal_commitment_calibration_v6/local_mode_counterfactual.json`.

At zero thermal load, forced High completed all 100 steps without a trip for
all four controllers and exceeded forced Low by `5.06` to `12.42` reward
points (mean `8.25`). Always-Low is therefore not the correct cold-state
policy. The safe High range was controller dependent: the first harmful or
tripping grid point was `0.05` for the dynamic-lifetime and static-lifetime
controllers, `0.075` for static-task, and `0.09` for dynamic-task. This
confirms a nontrivial health-contingent switching problem, but also shows that
the previous one-step oracle gate was insufficient because it did not capture
within-task heating and trips.

The v6 result is classified as joint exploration/representation failure rather
than reward dominance by Low. Before any v7 training, the CPU decision gate
must be upgraded from one-step returns to full-task paired returns, and the
agent input/training design must make prior action-response history usable
without exposing privileged thermal health at evaluation.

## v7 non-privileged action history

The standard recurrent policy consumes observations but does not explicitly
receive its preceding action. Thermal load is action-coupled, so estimating it
requires the action-response history. V7 appends the previous applied physical
action to the next within-task observation. This quantity is available to a
real controller and is not privileged health information. It is set to an
all-zero vector on every task and lifetime reset, preserving exact cold/hot
boundary-observation equivalence. Because the wrapper is inside the commitment
wrapper, the appended action is the physical command after the selected power
scale, which is also the action used by the thermal-dose dynamics.

V7 retains the v6 decision-only mode loss and every v5 environment, reward and
curriculum parameter. It changes only the non-privileged recurrent input,
uses calibration seed `4987`, and runs the unchanged four-cell 300k/400-task
learnability gate. The representation wrapper, complete train/save/evaluation
path, and boundary-zero semantics passed CPU tests before GPU training.

## v8 hierarchical decision redesign

V8 separated the one consequential Low/High commitment from the 100-step
physical controller. A frozen low-level controller executes each physical
task, while the high-level recurrent policy makes exactly one discrete mode
decision per task. Task and lifetime arms differ only in whether the high-level
LSTM state is reset at the task boundary. This removes the 99-in-100 inactive
action-coordinate credit problem identified in v6.

The one-seed v8 wiring pilot passed serialization and static-control checks,
but both dynamic policies still selected Low on every task. It was retained as
a failed calibration, not confirmatory evidence.

## v9 bounded optimization calibration

V9 fixed the physical task and searched four training-only curriculum and
entropy candidates in a declared order. Every candidate failed its first-seed
behavioral screen, so the automatic procedure stopped without touching held-out
seeds. This ruled out the declared optimizer-only fixes and triggered a
physical-decision design audit.

## v10 oracle design, freeze, and held-out study

V10 searched a finite ordered physical-design grid using the frozen low-level
controller. The first design satisfying all oracle and static-control gates was
selected: trip load `0.10`, heat rate `0.05`, Low scale `0.40`, High bonus
`2.0`, and trip penalty `75.0`. Its lifetime prefix oracle uses High for the
first two tasks and then Low. It beats Always Low by `38.2918` reward per
lifetime and the best deterministic task-reactive rule by `19.4773`.

Calibration seeds 5300--5304 selected the first passing training strategy:
50,000 high-level decisions, trip-load curriculum `0.30` to `0.10` over 120
lifetimes, entropy coefficient `0.005`, learning rate `3e-4`, and no teacher
shaping. The design, strategy, seeds, selected source hashes, and primary
analysis were written to local frozen files before held-out training. The plan
was not externally preregistered.

All four cells completed for held-out seeds 6300--6319. The locally specified
dynamic Lifetime-minus-Task effect was `+1.1269` reward per task (seed-bootstrap
95% CI `[0.7950, 1.4328]`; one-sided `p = 1.15e-6`), while the static zero-dose
effect was exactly zero. However, every learned Task policy selected Always
Low; the calibration-stage task-reactive rule already recovers `+0.9407` per
task over Always Low. The study therefore confirms the matched learned-policy
comparison but not the stronger claim that lifetime memory is necessary or
that the policy inferred hidden health. The full report is
[`HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md`](HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md).
