# Phase-0 Diagnostic Results

Date: 2026-08-03

These numbers validate that the software exposes the intended phenomenon.  They
are not paper results, are not based on a robotics simulator, and must not be
used as evidence for physical realism or method superiority.

## Command

```bash
PYTHONPATH=src python3 scripts/run_toy_benchmark.py --episodes 80 --seeds 10
```

The diagnostic used a 3-step episode horizon, quadratic action-dependent wear,
small process noise, and rare wear shocks.  Values below are mean ± standard
deviation across 10 independent lifetimes.

| Physics protocol | Controller | Success AUC | Mean episode return | Final wear |
|---|---|---:|---:|---:|
| persistent | nominal reactive | 0.660 ± 0.016 | 0.300 ± 0.023 | 0.716 ± 0.010 |
| persistent | episode EMA | 0.889 ± 0.015 | 0.574 ± 0.021 | 0.782 ± 0.014 |
| persistent | lifetime EMA | 0.736 ± 0.012 | 0.274 ± 0.022 | 1.000 ± 0.000 |
| persistent | myopic state oracle | 0.804 ± 0.008 | 0.381 ± 0.020 | 1.000 ± 0.000 |
| episode physical reset | nominal reactive | 1.000 ± 0.000 | 0.988 ± 0.000 | 0.008 ± 0.000 |

## Interpretation

1. Resetting physical state at every episode changes nominal-reactive success
   from 0.660 to 1.000, so the implementation can express the planned reset-bias
   phenomenon.
2. Episode-local adaptation improves immediate task performance under persistent
   wear in this setting.
3. Naively carrying an actuator-gain estimate across episodes is harmful: the
   controller compensates with larger actions, accelerates damage, and ends with
   worse lifetime return.  Persistent memory alone is therefore not the proposed
   solution; lifetime-aware planning must account for endogenous damage.
4. A myopic state oracle is not a lifetime upper bound.  True privileged
   comparison requires a controller optimized for the full lifetime objective.

## Next diagnostic

Implement a small dynamic-programming planning oracle and multiple held-out wear
law families.  This separates estimation difficulty from planning difficulty
before any neural RL experiment begins.

## Clock-shortcut audit

Command:

```bash
python scripts/run_clock_shortcut_audit.py
```

With 20 training and 20 held-out lifetimes, the action-history predictor had
held-out wear RMSE `~0` in the endogenous-action condition, whereas the
time-only predictor had RMSE `0.00526`. In the exogenous-clock control,
time-only RMSE was `~0` and action-history RMSE was `0.00371`.

This is a semantic unit test rather than a learning result: it confirms that
the diagnostic can reject a clock-only shortcut precisely when degradation is
caused by actions.

## Lifetime-planning oracle diagnostic

`scripts/run_planning_oracle_diagnostic.py` compares the privileged myopic
controller with a discretized finite-lifetime dynamic-programming controller.
The latter is a planning anchor, not an exact continuous-control optimum: its
action, position, and wear grids are committed in code and must be reported
with any use of its score. The selected 41-action, 0.02-position, and
0.005-wear resolution matched the myopic controller's success in the current
deterministic smoke configuration while optimizing the remaining lifetime.
