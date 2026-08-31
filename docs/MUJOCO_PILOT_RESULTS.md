# MuJoCo Factorial Pilot Results

Date: 2026-08-03

Purpose: verify that the simulator implementation independently expresses
cross-episode persistence and action-dependent degradation.  This is a scripted
software/phenomenon pilot, not a learned-policy result and not paper evidence.

## Command

```bash
.venv-mujoco/bin/python scripts/run_mujoco_factorial_pilot.py \
  --episodes 10 --episode-steps 100 --seeds 5 \
  --wear-rate 0.001 --exogenous-dose-per-step 0.25
```

## Key result: terminal wear

| Degradation cause | Scripted action | Episode physical reset | Persistent lifetime |
|---|---|---:|---:|
| endogenous | zero | 0.000 | 0.000 |
| endogenous | low constant | 0.006 | 0.063 |
| endogenous | random uniform | 0.033 | 0.337 |
| endogenous | high constant | 0.100 | 1.000 |
| exogenous clock | zero | 0.025 | 0.250 |
| exogenous clock | low constant | 0.025 | 0.250 |
| exogenous clock | random uniform | 0.025 | 0.250 |
| exogenous clock | high constant | 0.025 | 0.250 |

## What this establishes

1. Under endogenous degradation, more action dose produces more wear.
2. Under the exogenous control, wear is invariant to action dose.
3. Persistent lifetimes accumulate approximately ten episodes of wear, whereas
   episode-reset controls retain only one episode's wear.
4. The same wrapper can therefore realize the main causal controls without
   changing the underlying Pusher task.

## What this does not establish

- that these wear rates are physically calibrated;
- that Pusher is an adequate final benchmark;
- that any learning algorithm adapts successfully;
- that constant-clock exogenous drift is dose-matched to a learned policy;
- that reward changes are caused only by wear rather than scripted behavior.

The final endogenous/exogenous comparison will replay a preregistered or
policy-matched exogenous health trajectory so that marginal degradation exposure
is controlled.

## Recoverable thermal pilot

Command:

```bash
.venv-mujoco/bin/python scripts/run_thermal_pilot.py \
  --steps 400 --episode-steps 100
```

With action-dependent heating (`0.005`), transition cooling (`0.01`), and
episode-boundary cooling (`0.10`), random actions produced terminal thermal
loads of `0.101`, `0.140`, `0.156`, and `0.152` over four 100-step episodes.
The final multiplicative thermal actuator efficiency was `0.939` while
persistent wear was disabled. Regression tests separately confirm that a
nonzero thermal state changes the actual Pusher joint-velocity response and
that a lifetime reset, unlike an episode reset, clears thermal state.

This is a recoverable actuator-derating mechanism, distinct in temporal
semantics from persistent wear but not yet a distinct contact or joint-physics
mechanism. Contact friction remains required before the final multi-mechanism
benchmark claim.

## Persistent joint-aging pilot

Command:

```bash
.venv-mujoco/bin/python scripts/run_joint_aging_pilot.py \
  --steps 400 --episode-steps 100
```

With wear and thermal health disabled, action-dependent aging (`0.002`) reached
`0.268` after 400 random-action steps across four episodes. This increased the
seven actuated joints' damping by a factor of `2.073` while leaving actuator
gain at `1.0`. Tests confirm that episode reset preserves aging, lifetime reset
restores base damping, exogenous-clock aging is action-independent, and higher
damping changes the actual Pusher joint-velocity response.

Wear, thermal derating, and joint aging now differ in persistence and recovery,
but their rates are diagnostic parameters rather than physical calibration.
