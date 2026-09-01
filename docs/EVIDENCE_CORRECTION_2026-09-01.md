# Evidence-interpretation correction

Date: 2026-09-01  
Scope: manuscript narrative and evidence map only; no protocol, raw result,
checkpoint, or statistical output was changed.

## Issue found

An earlier writing draft combined two valid Pusher v12.3 facts that belong to
different policies:

- `+0.965` reward/task is the uncertainty effect for the **physics-only**
  `z=1.5` treatment relative to physics `z=0`;
- `1.3%` is the maximum trip rate of the **full hybrid** `z=1.5` treatment.

The physics-only treatment associated with `+0.965` has a 2.15% maximum trip
rate and therefore fails the frozen 2.0% point rule. The hybrid margin effect is
`+0.736`, and that hybrid treatment passes at 1.3%. These quantities must not be
presented as one policy result.

## Corrected cross-task evidence map

| Task and sample | Treatment contrast | Mean effect (95% bootstrap CI) | Treatment maximum trip rate |
|---|---|---:|---:|
| Pusher fresh factorial | physics `z=1.5` minus physics `z=0` | +0.965 [0.733, 1.196] | 2.15% |
| Pusher fresh factorial | hybrid `z=1.5` minus hybrid `z=0` | +0.736 [0.524, 0.950] | 1.30% |
| Reacher original test | physics `z=1.5` minus physics `z=0` | +0.721 [0.474, 0.978] | 2.30% |
| Reacher original test | hybrid `z=1.5` minus physics `z=0` | +0.797 [0.546, 1.059] | 1.40% |
| Reacher extension test | calibrated physics `z=2` minus physics `z=0` | +0.692 [0.434, 0.962] | 1.60% |

On the same Reacher extension seeds, inherited physics `z=1.5` also has a 1.9%
maximum trip rate. The selected `z=2` policy differs from inherited `z=1.5` by
`-0.008` reward/task, with 95% CI `[-0.121, 0.108]`. Consequently, the extension
shows that a frozen development-only procedure can select a policy that passes
a fresh point gate, but it does not establish that calibration is necessary,
superior, or uniquely responsible for passing.

## Allowed conclusions after correction

1. Uncertainty margins improve paired mean risk-sensitive utility in both
   simulated tasks under the tested shifts.
2. In the designated original tests, physics-only `z=1.5` misses the 2% point
   rule and the corresponding full hybrid policy passes it.
3. The learned residual is not independently established as the cause of the
   improvement.
4. The calibrated Reacher policy passes its fresh gate, but calibration
   superiority and necessity remain unproven.
5. The 2% value is an empirical point decision rule, not a confidence bound,
   calibrated risk probability, hardware claim, or formal safety guarantee.

## Documents corrected

- `manuscript/tmlr/main.tex` and its results, discussion, and conclusion files;
- `docs/TMLR_MANUSCRIPT_BLUEPRINT.md`;
- `docs/TMLR_CHECKLIST.md`;
- `docs/INTEGRATED_PUSHER_REACHER_AUDIT.md`;
- `docs/REACHER_REPLICATION_FINAL_RESULTS.md`.

The correction preserves the chronological status of the original Reacher
failure and the separately frozen post-confirmatory extension.
