# LifePhyBench v12.3 evidence snapshot

This directory preserves the evidence chain used for the final v12.2
confirmation and the fresh-seed v12.3 factorial attribution study. The source
tree was clean when the snapshot began.

## Identity

- Created: 2026-08-30T21:43:14+09:00
- Source branch: `research/lifephybench`
- Source commit: `a8c0307c8655e6dfd82a21249ee34a20e82b01e0`
- Checkpoint SHA-256:
  `c38734683b1aeebebd709a543354b14bc7bc12c440a3ef6ee0622ab882ea07f2`
- v12.2 frozen-protocol SHA-256:
  `24c72ce43ccfca7414795a4d4480d866eaedf13ce0d1441d65e647bb6646918b`
- v12.3 frozen-protocol SHA-256:
  `9c67f858870d19ca6487ff78f96b867b99ba9e4ae3a8913d7f01690ac3f98bad`

## Preserved evidence

- `physics_residual_v12_1_recovery`: development search and fresh audit
- `physics_residual_v12_refinement`: selected residual checkpoint and pilot
- `physics_residual_v12_2_scoped_confirmatory`: frozen held-out confirmation,
  per-cell data, statistical results, tables, and figures
- `physics_residual_v12_3_factorial_ablation`: fresh-seed 2 x 2 attribution
  protocol, per-cell data, and statistical results

The snapshot contains 29 artifact files. Every artifact is covered by
`manifests/ARTIFACTS.sha256`; the manifest and runtime metadata are covered by
`SNAPSHOT_ROOT.sha256`.

## Result boundary

The preserved v12.2 result supports the scoped comparison between the hybrid
supervisor at uncertainty margin `z=1.5` and the physics-only supervisor at
`z=0` on the predefined target OOD aggregate: mean improvement `0.998673`
reward per task, 95% bootstrap CI `[0.744080, 1.254547]`, with 79 of 100 paired
seeds positive.

The v12.3 factorial study does not support attributing the complete gain to the
learned residual. At matched uncertainty, the residual effect at `z=1.5` was
`0.103` with a confidence interval crossing zero, whereas the uncertainty
margin produced the larger and more stable benefit. These files therefore
support an uncertainty-aware belief-supervision claim, not an unrestricted
claim that residual learning independently caused the full improvement.

## Verification performed before capture

```text
./.venv-mujoco/bin/python -m pytest -q
150 passed in 16.35s
```

The focused v12 test subset also passed: 20 tests in 10.16 seconds.

To verify the preserved files:

```bash
cd evidence/snapshots/2026-08-30_v12.3_final
sha256sum -c manifests/ARTIFACTS.sha256
sha256sum -c SNAPSHOT_ROOT.sha256
```

The original `outputs/` tree remains ignored by Git. This snapshot is the
version-controlled copy of the evidence required to audit the final claims.
