# Reacher cross-task replication protocol

## Purpose

This study tests the largest remaining empirical weakness of the v12 result:
all final evidence currently comes from one Pusher task.  The replication moves
the same hidden, action-driven, persistent thermal-state problem to Reacher-v5.
It is an external task replication, not a new method-selection round.

## Bias controls

- The Pusher-selected belief cutoff (`0.060`) and uncertainty multiplier
  (`z=1.5`) transfer unchanged.
- Development, controller-calibration, and final confirmatory seeds are
  disjoint.
- Confirmatory seeds `25200` through `25299` are not used by the low-level
  controller stage.
- All completed, failed, and interrupted runs are retained.
- Failure of the replication is reported; it does not trigger a silent change
  of the primary contrast or thresholds.

The machine-readable frozen design is
[`configs/reacher_cross_task_replication_v1.json`](../configs/reacher_cross_task_replication_v1.json).

## Stage 1: frozen low-level controller

Five development-only RecurrentPPO controllers are trained on canonical-reset
Reacher-v5 with zero thermal dose.  The controller with the highest mean task
reward over 1,000 evaluation tasks is frozen.  This choice cannot consume the
replication's confirmatory seeds and is not itself evidence for the paper.

## Stage 2: belief and baseline development

The frozen low-level controller is placed below the same binary high/low-power
thermal supervisor used for Pusher.  Development lifetimes identify the compact
thermal transition and train any residual component.  A separate calibration
set is used for the monolithic temperature-aware recurrent baseline.  The
measured-temperature oracle and model-only limiter are mandatory comparators.

The stage-2 addendum was frozen after low-level seed 5100 was selected and
before any stage-2 lifetime was generated. It fixes 3,000 development
lifetimes, the residual train/validation/test split, 20 controller-calibration
seeds, and five 100,000-decision monolithic-baseline training runs. See
[`configs/reacher_cross_task_stage2_v1.json`](../configs/reacher_cross_task_stage2_v1.json).

## Stage 3: untouched confirmatory evaluation

One hundred fresh paired lifetimes are evaluated in-domain and under the same
prespecified sensor-noise, cooling, shock, and combined shifts.  The primary
contrast is the inherited uncertainty-aware physics supervisor against the
same physics supervisor without an uncertainty margin, averaged over the
target OOD cells.

Success requires a positive paired mean effect, a paired-bootstrap 95% lower
bound above zero, a two-sided paired sign-flip p-value below 0.05, and no cell
with a trip rate above 0.02.  These gates are intentionally frozen before any
Reacher training begins.
