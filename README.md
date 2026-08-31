# LifePhyBench

LifePhyBench is a research codebase for **decision-making under endogenous,
persistent physical degradation and selective resets**.  Unlike ordinary
episodic benchmarks, task state resets between episodes while a hidden physical
state (wear, aging, fatigue, recovery, or damage) persists and keeps evolving as
a consequence of the agent's actions.

The intended publication target is *Transactions on Machine Learning Research
(TMLR)*.  The project is therefore framed as a general learning problem, not as
a single robot application.

## Current status

The repository now contains a completed held-out hierarchical thermal study in
addition to the semantic diagnostics and earlier development pilots. In the
frozen Pusher thermal-commitment protocol, lifetime-state RecurrentPPO
outperformed a matched task-reset arm by `+1.1269` reward per task across 20
independent held-out training seeds (seed-bootstrap 95% CI
`[0.7950, 1.4328]`; locally pre-specified one-sided `p = 1.15e-6`). The static
zero-dose control difference was exactly zero.

This is a valid single-diagnostic learned-policy result, not the final TMLR
evidence package. The task-reset policies collapsed to an Always-Low strategy,
and a transparent task-reactive rule closes most of the learned comparison
gap. Cross-task, cross-mechanism, and stronger-baseline evidence remains open.
See [`docs/HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md`](docs/HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md)
for the result and its claim limits.

## Core semantics

- `reset_episode()` resets the task state but preserves physical state.
- `reset_lifetime()` resets both task and physical state.
- Actions affect immediate task progress and future physical dynamics.
- Train/validation/test units are complete lifetimes, never shuffled steps.
- Hidden physical parameters are exposed only to explicitly labelled oracle
  baselines and diagnostic logging.

## Quick start

The Phase-0 smoke test runs with Python 3.10+ and the standard library only.

```bash
cd LifePhyBench
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_toy_benchmark.py --episodes 40 --seeds 5
PYTHONPATH=src python3 scripts/run_clock_shortcut_audit.py
```

The first simulator-backed environment uses the pinned free MuJoCo stack:

```bash
conda env create -f environment.yml
conda activate lifephybench
pytest -q
python scripts/smoke_mujoco.py --steps 600 --episode-steps 100
python scripts/run_thermal_pilot.py --steps 400 --episode-steps 100
python scripts/run_joint_aging_pilot.py --steps 400 --episode-steps 100
python scripts/run_multitask_health_factorial.py \
  --output outputs/cpu_semantic_pilot_v1.jsonl
python scripts/calibrate_training_stack.py

# After CUDA preflight passes
./.venv-mujoco/bin/python -m pip install "stable-baselines3[extra]"
./.venv-mujoco/bin/python scripts/train_sb3_smoke.py \
  --algorithm ppo --workers 8 --total-timesteps 1000000 \
  --run-name ppo-smoke-seed1000

# After the PPO/SAC smoke comparison, check the development training budget
./.venv-mujoco/bin/python scripts/check_ppo_duration.py

# Recurrent development campaign: episode-RNN and lifetime-RNN, sequential runs
./.venv-mujoco/bin/python scripts/run_recurrent_campaign.py
```

On the current workstation an equivalent environment is already available at
`.venv-mujoco/` (ignored by Git).

Results are printed as JSON.  To save them, redirect stdout to a file under
`outputs/` (ignored by Git).

The sealed Pusher--Reacher publication package can be verified without
retraining. The command checks repository privacy, snapshot and protocol
hashes, loads both retained RecurrentPPO archives on CPU, regenerates all final
publication artifacts, compares their hashes, and optionally runs the complete
test suite:

```bash
python scripts/reproduce_clean_checkout.py --run-tests \
  --report clean_checkout_reproduction.json
```

The verified clean-checkout result is documented in
[`docs/CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md`](docs/CLEAN_CHECKOUT_REPRODUCTION_2026-08-31.md).

For a fully empty Python 3.11 environment, the validated CPU-only dependency
lock and bootstrap command are:

```bash
LIFEPHYBENCH_PYTHON=python3.11 \
  ./scripts/bootstrap_reproduction_env.sh .venv-reproduction
```

This installs [`requirements-reproduction.txt`](requirements-reproduction.txt),
checks dependency consistency, installs LifePhyBench without re-resolving
dependencies, and executes the full clean-checkout reproduction audit.

The completed v10 result can be reanalyzed and rendered without retraining:

```bash
./.venv-mujoco/bin/python \
  scripts/analyze_frozen_hierarchical_confirmatory.py \
  --input-root outputs/hierarchical_autonomous_v10/confirmatory
./.venv-mujoco/bin/python \
  scripts/render_hierarchical_confirmatory_artifacts.py
```

## Repository map

```text
configs/                 Versioned experimental protocols
docs/                    Research specification and decision records
scripts/                 Reproducible entry points
src/lifephybench/        Environment, policies, and evaluation code
tests/                   Semantic and determinism tests
```

Start with [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md),
[`docs/NOVELTY_LEDGER.md`](docs/NOVELTY_LEDGER.md), and
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). The completed CPU scope
and the GPU boundary are recorded in
[`docs/CPU_STAGE_RESULTS.md`](docs/CPU_STAGE_RESULTS.md); the current GPU
preflight result is in [`docs/GPU_PREFLIGHT.md`](docs/GPU_PREFLIGHT.md), and
engineering-only GPU smoke results are in
[`docs/GPU_SMOKE_RESULTS.md`](docs/GPU_SMOKE_RESULTS.md). Development results
for recurrent and finite-history baselines are in
[`docs/RECURRENT_DEVELOPMENT_RESULTS.md`](docs/RECURRENT_DEVELOPMENT_RESULTS.md)
and [`docs/FRAMESTACK_DEVELOPMENT_RESULTS.md`](docs/FRAMESTACK_DEVELOPMENT_RESULTS.md).
The current held-out result is documented in
[`docs/HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md`](docs/HIERARCHICAL_THERMAL_CONFIRMATORY_V10.md).

## Reproducibility rule

No headline result should be generated from an untracked interactive notebook.
Every reported number must be reproducible from a committed configuration,
seed list, environment fingerprint, and command-line entry point.
