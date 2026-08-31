# Canonical thermal-probe protocol

## Status

This is a frozen, GPU-unexecuted experiment design. It replaces neither the
completed fair campaign nor its static-health negative result.

## Physical construction

Every task reset uses MuJoCo seed `811`, so robot state, object state, goal
geometry, and policy observation are canonical at the task boundary. Thermal
health remains hidden and persists until the lifetime reset.

The dynamic cell uses endogenous action-coupled thermal accumulation. The
static control keeps the same thermal wrapper but fixes the exogenous thermal
dose to zero. Both cells use heat rate `0.1`, cooling rate `0.0`, and episode
cooling `0.0`.

The CPU counterfactual audit verifies that paired cold and hot boundary
observations are exactly equal while the same five-step action probe produces
a minimum cold-minus-hot velocity-response gap of `0.2925` across ten canonical
task seeds.

## Frozen learned-policy comparison

- seeds: 4000--4009;
- 2M timesteps per policy;
- 1,000 task episodes of evaluation per policy;
- four cells: dynamic/static × task-reset/lifetime-memory;
- all cells receive the same boundary marker and lifetime-only Gym/GAE
  boundary;
- the only memory intervention is forced LSTM reset at task boundaries.

## Entry gate

Before any GPU pilot, retain the CPU audit outputs and the high-resolution toy
planning validation. The pilot must confirm canonical reset and health logging
in every cell. Only then may the full 40-run campaign be launched.
