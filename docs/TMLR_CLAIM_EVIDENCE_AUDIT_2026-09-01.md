# TMLR claim-to-evidence audit — 2026-09-01

Status: **PASS**. Principal manuscript numbers are traced to immutable study
artifacts and the corrected policy/result boundaries are present in source.

| Manuscript claim | Verified value | Authoritative artifact | Status |
|---|---:|---|---|
| Pusher physics-only uncertainty effect | +0.965, CI [0.733, 1.196], 77/100 positive | `paper_artifacts/physics_residual_v12_3/factorial_contrasts.csv`; `paper_artifacts/physics_residual_v12_3/paired_seed_contrasts.csv` | PASS |
| Pusher physics-only maximum OOD trip rate | 2.15% (fails 2.0% point rule) | `paper_artifacts/physics_residual_v12_3/policy_summaries.csv` | PASS |
| Pusher hybrid uncertainty effect | +0.736, CI [0.524, 0.950], 72/100 positive | `paper_artifacts/physics_residual_v12_3/factorial_contrasts.csv`; `paper_artifacts/physics_residual_v12_3/paired_seed_contrasts.csv` | PASS |
| Pusher hybrid maximum OOD trip rate | 1.30% (passes point rule) | `paper_artifacts/physics_residual_v12_3/policy_summaries.csv` | PASS |
| Pusher residual effect at matched margin | +0.103, CI [-0.004, 0.215] | `paper_artifacts/physics_residual_v12_3/factorial_contrasts.csv` | PASS; independent benefit not established |
| Reacher inherited physics margin | +0.721, CI [0.474, 0.978], 54/100 positive | `paper_artifacts/reacher_replication/table_reacher_contrasts.csv` | PASS |
| Reacher inherited physics maximum trip rate | 2.30% (original gate failure) | `evidence/snapshots/2026-08-31_pusher_reacher_final/artifacts/outputs/reacher_replication/confirmatory/CONFIRMATORY_CELLS.json` | PASS |
| Reacher original hybrid comparator | +0.797; maximum trip rate 1.40% | `paper_artifacts/reacher_replication/table_reacher_contrasts.csv`; `evidence/snapshots/2026-08-31_pusher_reacher_final/artifacts/outputs/reacher_replication/confirmatory/CONFIRMATORY_CELLS.json` | PASS |
| Reacher selected margin | +0.692, CI [0.434, 0.962], 52/100 positive, exact sign p=0.764 | `paper_artifacts/reacher_replication/table_reacher_contrasts.csv` | PASS |
| Reacher selected maximum trip rate | 1.60% on extension lifetimes | `evidence/snapshots/2026-08-31_pusher_reacher_final/artifacts/outputs/reacher_replication/margin_extension/FRESH_CELLS.json` | PASS |
| Inherited margin on the same extension sample | 1.90% | `evidence/snapshots/2026-08-31_pusher_reacher_final/artifacts/outputs/reacher_replication/margin_extension/FRESH_CELLS.json` | PASS |
| Selected versus inherited margin | -0.008, CI [-0.121, 0.108] | `paper_artifacts/reacher_replication/table_reacher_contrasts.csv` | PASS; no detectable superiority |

## Boundary audit

- The +0.965 Pusher effect is paired with the physics-only 2.15% rate, not the
  hybrid 1.30% rate.
- The original Reacher inherited-margin failure at 2.30% remains authoritative;
  the later 1.90% observation is explicitly a different fresh sample.
- The selected Reacher margin demonstrates a feasible development-only
  procedure, not calibration necessity, superiority, or unique recovery.
- The residual confidence interval crosses zero at the deployed margin, so the
  residual is an evaluated component rather than the established mechanism.
- The 2.0% threshold is an empirical point gate, not a confidence-bound or
  formal safety certificate.

## Claims intentionally outside the evidence

The artifacts do not establish physical-unit thermal validity, hardware safety,
formal coverage, universal reward preferences, cross-simulator generalization,
majority-lifetime benefit, or general algorithmic superiority. These limits are
required in the paper wherever the main result is summarized.
