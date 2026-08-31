# Hierarchical thermal confirmatory study (v10)

Status: completed held-out study on 2026-08-27.

This document is the authoritative human-readable summary of the v10 study.
It supersedes the project-status claims in the earlier continuous-action
thermal pilots, but it does not erase those negative development results.

## Question and scope

The study asks a deliberately narrow learned-policy question:

> In the frozen hierarchical Pusher thermal-commitment diagnostic, does a
> RecurrentPPO policy whose recurrent state persists across a 20-task lifetime
> outperform the same policy and training configuration when recurrent state
> is forcibly reset at every task boundary?

The answer is yes for the learned-versus-learned comparison that was locally
specified before held-out training in the dynamic endogenous-thermal cell.
This is not yet evidence that cross-task memory is universally necessary, that
the result generalizes beyond this Pusher diagnostic, or that LifePhyBench
provides a new learning algorithm.

## Frozen protocol

The physical design and training strategy were selected using calibration
seeds 5300--5304. They were frozen before any confirmatory training. Held-out
training seeds 6300--6319 were then used in a 2 x 2 factorial:

- dynamic endogenous thermal state versus static zero-dose control;
- recurrent state reset at every task versus retained for the lifetime.

The static condition is named `exogenous_clock` in legacy metadata, but its
dose is zero. It is a static zero-dose negative control, not a dose-matched
exogenous-drift condition.

| Item | Frozen value |
|---|---:|
| High-level action | Discrete Low/High; one decision per physical task |
| Physical task / lifetime | 100 simulator steps / 20 tasks |
| Low-power action scale | 0.40 |
| Safe High reward bonus | 2.0 |
| Thermal heat rate | 0.05 |
| Protection-trip load / penalty | 0.10 / 75.0 |
| Policy observation summary | previous mode and observed trip only |
| Privileged thermal health | not exposed |
| Training budget | 50,000 high-level decisions per model, 8 workers |
| Optimizer settings | learning rate 3e-4, entropy coefficient 0.005 |
| Training-only curriculum | trip load 0.30 to 0.10 over 120 lifetimes |
| Teacher shaping | 0.0 |
| Evaluation | 1,000 tasks = 50 lifetimes per trained model |
| Inferential unit | independent training seed, n = 20 |

The evaluation reward is unscaled. The 0.02 reward scale applies only during
training. The frozen low-level controller has SHA-256
`4e10e871d48b823133784a9c8410ddae7cadae7ade0fd24f6ad51b478f434fd3`.

The per-run metadata retain the legacy labels
`hierarchical_thermal_mode_calibration` and
`calibration_not_confirmatory_evidence` because changing that source after the
freeze would have broken the recorded source hashes. The campaign
`manifest.json` is the authoritative phase record.

## Held-out learned-policy results

Values are mean +/- sample SD across 20 independent training seeds. The 1,000
evaluation tasks per model reduce evaluation noise; they are not 1,000
independent inferential samples.

| Physical condition | Memory reset | Reward per task | High rate | Trip rate |
|---|---|---:|---:|---:|
| Dynamic endogenous thermal | Task | -43.7003 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Dynamic endogenous thermal | Lifetime | -42.5734 +/- 0.7574 | 0.0750 +/- 0.0414 | 0.0050 +/- 0.0154 |
| Static zero dose | Task | -22.51295 +/- 0.00000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |
| Static zero dose | Lifetime | -22.51295 +/- 0.00000 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 |

| Paired seed-level contrast | Mean | SD | Seed-bootstrap 95% CI | Pre-specified / secondary test |
|---|---:|---:|---:|---:|
| Dynamic Lifetime - Task (primary) | +1.1269 | 0.7574 | [0.7950, 1.4328] | one-sided t p = 1.15e-6 |
| Static Lifetime - Task | 0.0000 | 0.0000 | [0.0000, 0.0000] | control |
| Dynamic-minus-static difference in differences | +1.1269 | 0.7574 | [0.7963, 1.4328] | secondary |

The primary effect was positive in 17 of 20 seeds, zero in two, and negative
in one. The locally pre-specified one-sided Wilcoxon sensitivity test gave
`p = 1.12e-4`. The lifetime policy's mean cold-minus-hot High-selection gap was
`0.2653`. The primary success rule required both one-sided `p < 0.05` and a
positive bootstrap lower bound, so the frozen primary comparison passed.

Both static policies selected High on every task and achieved the same
deterministic ceiling. The exact-zero static contrast shows that the learned
difference disappears when thermal accumulation is disabled in this setup,
but it is not an independent replication of the dynamic effect. Consequently,
the difference-in-differences is numerically identical to the primary contrast.

The absolute change is `+1.1269` reward per task, or `+22.5379` over a 20-task
lifetime. A percentage improvement is intentionally not reported because the
reward is negative and its zero point is arbitrary.

## Strong-comparator audit

The pre-specified result is a learned-versus-learned comparison, not a proof
that all of its difference is caused by a representational need for lifetime
memory. All 20 learned task-reset policies selected Low on every held-out
evaluation task. The calibration-stage exhaustive search over the 16
deterministic rules using only start/previous-mode/trip found a stronger
task-reactive rule: choose High on the first lifetime task and Low thereafter.

These transparent anchors must be kept separate from the held-out inferential
table because they were constructed during physical-design calibration.

| Calibration-stage anchor | Mean reward per task | Schedule |
|---|---:|---|
| Always Low | -43.7003 | 0 High tasks |
| Best deterministic task-reactive rule | -42.7596 | first task High, then Low |
| Lifetime prefix oracle | -41.7857 | first two tasks High, then Low |

The reactive rule recovers `+0.9407` per task over Always Low, approximately
83% of the held-out learned-versus-learned mean gap. The held-out lifetime-RNN
mean is only `+0.1862` above that calibration-stage reactive anchor; this
post-hoc comparison has eight seeds above, eight tied within numerical
tolerance, and four below. No held-out test was specified for this anchor, so
the comparison is descriptive and does not establish superiority.
The oracle's structural advantage over the best reactive rule is `+0.9739`
per task (`+19.4773` per lifetime), showing that a genuine lifetime-information
opportunity exists even though the current learned task baseline did not
realize its representable reactive strategy.

The lifetime policies also reduced to a small set of deterministic schedules:
eight used one High decision, eight used two High decisions (including prefix
and `High-Low-High` variants), two selected Always Low, and two used three High
decisions and incurred a trip. This behavior is consistent with counting
lifetime position. Because the canonical task and thermal law are
deterministic, the current study does not separate clock/counting from online
inference of hidden thermal health.

Accordingly, the supported conclusion is:

> Under the frozen v10 Pusher thermal-commitment protocol, retaining recurrent
> state across the lifetime improved RecurrentPPO performance relative to the
> matched task-reset training arm, and the effect disappeared in the static
> zero-dose control.

The study does not by itself support claims of universal memory necessity,
cross-task or cross-mechanism generalization, calibrated real-world thermal
prediction, or algorithmic novelty.

## Reproducibility and retained evidence

Local evidence is retained under
`outputs/hierarchical_autonomous_v10/confirmatory/`:

- `manifest.json`: frozen seed list, design, source hashes, and statistical plan;
- `PROGRESS.json`: all 20 confirmatory seeds completed;
- 80 `metadata.json` files: 20 seeds x 4 factorial cells;
- `CONFIRMATORY_RESULTS.json`: final seed-level analysis;
- `paper_artifacts/`: regenerated tables and figures.

The key retained files currently have these SHA-256 digests:

```text
38085161392b24ac6f910941508df7cfb1a84be1006bb7d62ebbc96a2d7b6807  FROZEN_PROTOCOL.json
cd66d89e66cf23254673747817e17055ac707fa0c5ccdee5a754a0a7bbac35c7  confirmatory/manifest.json
808d36ee89569ae86a28d1d7a098fcd35769a11f29167c64798c07f93efd3c6f  confirmatory/CONFIRMATORY_RESULTS.json
```

The analysis plan was written to the local campaign manifest before the first
held-out model was trained. It was not externally preregistered. The frozen
hash list covers the low-level model and four central source files, not the
entire dependency graph, runner, analyzer, or environment lockfile; the current
worktree is also not a versioned public release. Future confirmatory campaigns
must bind the complete source tree and software environment to a commit and an
external timestamp before training.

Re-run the frozen wiring preflight and analysis without retraining:

```bash
./.venv-mujoco/bin/python scripts/run_frozen_hierarchical_confirmatory.py \
  --preflight-only

./.venv-mujoco/bin/python \
  scripts/analyze_frozen_hierarchical_confirmatory.py \
  --input-root outputs/hierarchical_autonomous_v10/confirmatory

./.venv-mujoco/bin/python \
  scripts/render_hierarchical_confirmatory_artifacts.py
```

## Remaining evidence before a TMLR-level central claim

1. Add and tune a strong learned task-reactive baseline that reaches the
   transparent reactive anchor; do not compare only against the collapsed arm.
2. Randomize task state and thermal evolution, add an explicit task-index
   baseline, and test held-out heat rates so clock counting can be separated
   from health inference.
3. Replicate the central interaction across additional tasks or morphologies
   and at least one additional persistent physical-state family.
4. Add stronger adaptation/system-identification baselines with matched
   interaction and optimization budgets.
5. Freeze new protocols before their final test seeds and retain all failures.
6. Package versioned raw rows, environment fingerprints, and generated figures
   for an anonymous reproducibility release.
