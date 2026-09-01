#!/usr/bin/env python3
"""Trace the principal TMLR manuscript claims to immutable study artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PUSHER = REPOSITORY / "paper_artifacts" / "physics_residual_v12_3"
REACHER = REPOSITORY / "paper_artifacts" / "reacher_replication"
SNAPSHOT = (
    REPOSITORY
    / "evidence"
    / "snapshots"
    / "2026-08-31_pusher_reacher_final"
    / "artifacts"
    / "outputs"
)
MANUSCRIPT = REPOSITORY / "manuscript" / "tmlr"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    matches = [item for item in rows if all(item.get(k) == v for k, v in keys.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {keys}, found {len(matches)}")
    return matches[0]


def close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPOSITORY).as_posix()
    except ValueError:
        # Reports may be written outside the extracted anonymous supplement.
        # Emit only the filename so machine-local paths never enter logs.
        return path.name


def max_trip(cells: dict, policy: str) -> float:
    return max(
        float(condition[policy]["summary"]["trip_rate"])
        for condition in cells.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs" / "TMLR_CLAIM_EVIDENCE_AUDIT_2026-09-01.md",
    )
    args = parser.parse_args()

    pusher_contrast_path = PUSHER / "factorial_contrasts.csv"
    pusher_policy_path = PUSHER / "policy_summaries.csv"
    pusher_seed_path = PUSHER / "paired_seed_contrasts.csv"
    reacher_contrast_path = REACHER / "table_reacher_contrasts.csv"
    original_cells_path = (
        SNAPSHOT / "reacher_replication" / "confirmatory" / "CONFIRMATORY_CELLS.json"
    )
    extension_cells_path = (
        SNAPSHOT
        / "reacher_replication"
        / "margin_extension"
        / "FRESH_CELLS.json"
    )

    pusher_contrasts = read_csv(pusher_contrast_path)
    pusher_policies = read_csv(pusher_policy_path)
    pusher_seeds = read_csv(pusher_seed_path)
    reacher_contrasts = read_csv(reacher_contrast_path)
    original_cells = json.loads(original_cells_path.read_text(encoding="utf-8"))
    extension_cells = json.loads(extension_cells_path.read_text(encoding="utf-8"))

    manuscript_files = [
        MANUSCRIPT / "main.tex",
        *sorted((MANUSCRIPT / "sections").glob("*.tex")),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_files)

    pusher_physics = row(
        pusher_contrasts,
        scope="target_ood_aggregate",
        contrast="uncertainty_without_residual",
    )
    pusher_hybrid = row(
        pusher_contrasts,
        scope="target_ood_aggregate",
        contrast="uncertainty_with_residual",
    )
    pusher_residual = row(
        pusher_contrasts,
        scope="target_ood_aggregate",
        contrast="residual_at_z1_5",
    )
    original_primary = row(
        reacher_contrasts,
        phase="inherited_confirmatory",
        contrast="primary_uncertainty",
    )
    original_hybrid = row(
        reacher_contrasts,
        phase="inherited_confirmatory",
        contrast="hybrid_vs_physics_z0",
    )
    original_monolithic = row(
        reacher_contrasts,
        phase="inherited_confirmatory",
        contrast="monolithic_vs_physics_z0",
    )
    extension_selected = row(
        reacher_contrasts,
        phase="post_confirmatory_extension",
        contrast="selected_vs_physics_z0",
    )
    extension_comparison = row(
        reacher_contrasts,
        phase="post_confirmatory_extension",
        contrast="selected_vs_inherited",
    )

    checks: list[tuple[str, float, float]] = [
        ("Pusher physics-only mean", float(pusher_physics["mean"]), 0.964710947659376),
        ("Pusher physics-only CI low", float(pusher_physics["bootstrap_95_ci_low"]), 0.7328840680602233),
        ("Pusher physics-only CI high", float(pusher_physics["bootstrap_95_ci_high"]), 1.1963021810734489),
        ("Pusher hybrid mean", float(pusher_hybrid["mean"]), 0.7355278803213932),
        ("Pusher hybrid CI low", float(pusher_hybrid["bootstrap_95_ci_low"]), 0.5238573079434723),
        ("Pusher hybrid CI high", float(pusher_hybrid["bootstrap_95_ci_high"]), 0.950208877071415),
        ("Pusher residual mean", float(pusher_residual["mean"]), 0.10345894894131487),
        ("Pusher residual CI low", float(pusher_residual["bootstrap_95_ci_low"]), -0.004452076459085999),
        ("Pusher residual CI high", float(pusher_residual["bootstrap_95_ci_high"]), 0.21465768847720065),
        ("Reacher inherited mean", float(original_primary["mean_reward_per_task"]), 0.720523381094315),
        ("Reacher inherited CI low", float(original_primary["bootstrap_95_ci_lower"]), 0.4744752393432859),
        ("Reacher inherited CI high", float(original_primary["bootstrap_95_ci_upper"]), 0.9784987106636537),
        ("Reacher original hybrid mean", float(original_hybrid["mean_reward_per_task"]), 0.7970240266978742),
        ("Reacher monolithic mean", float(original_monolithic["mean_reward_per_task"]), -9.067785000846065),
        ("Reacher monolithic CI low", float(original_monolithic["bootstrap_95_ci_lower"]), -9.507488386291085),
        ("Reacher monolithic CI high", float(original_monolithic["bootstrap_95_ci_upper"]), -8.629553291699162),
        ("Reacher selected mean", float(extension_selected["mean_reward_per_task"]), 0.6919715990702785),
        ("Reacher selected CI low", float(extension_selected["bootstrap_95_ci_lower"]), 0.43446918971675674),
        ("Reacher selected CI high", float(extension_selected["bootstrap_95_ci_upper"]), 0.961789445773937),
        ("Reacher selected-vs-inherited mean", float(extension_comparison["mean_reward_per_task"]), -0.008144875535866507),
        ("Reacher selected-vs-inherited CI low", float(extension_comparison["bootstrap_95_ci_lower"]), -0.1210223136323574),
        ("Reacher selected-vs-inherited CI high", float(extension_comparison["bootstrap_95_ci_upper"]), 0.10775717207616331),
    ]
    for label, actual, expected in checks:
        close(actual, expected, label)

    pusher_trip = {}
    for policy in ("physics_z1_5", "hybrid_z1_5"):
        pusher_trip[policy] = max(
            float(item["trip_rate"])
            for item in pusher_policies
            if item["policy"] == policy and item["condition"] != "in_domain"
        )
    close(pusher_trip["physics_z1_5"], 0.0215, "Pusher physics trip")
    close(pusher_trip["hybrid_z1_5"], 0.013, "Pusher hybrid trip")

    pusher_positive = {
        key: sum(float(item[key]) > 0.0 for item in pusher_seeds)
        for key in ("uncertainty_without_residual", "uncertainty_with_residual")
    }
    if pusher_positive != {
        "uncertainty_without_residual": 77,
        "uncertainty_with_residual": 72,
    }:
        raise AssertionError(f"unexpected Pusher positive counts: {pusher_positive}")

    trips = {
        "original_physics": max_trip(original_cells, "physics_z1_5"),
        "original_hybrid": max_trip(original_cells, "hybrid_z1_5"),
        "original_monolithic": max_trip(original_cells, "monolithic_recurrent"),
        "extension_selected": max_trip(extension_cells, "selected_calibrated_margin"),
        "extension_inherited": max_trip(extension_cells, "inherited_physics_z1_5"),
    }
    expected_trips = {
        "original_physics": 0.023,
        "original_hybrid": 0.014,
        "original_monolithic": 0.236,
        "extension_selected": 0.016,
        "extension_inherited": 0.019,
    }
    for key, expected in expected_trips.items():
        close(trips[key], expected, f"Reacher {key} trip")

    integer_checks = {
        "inherited positive lifetimes": (int(original_primary["positive_lifetimes"]), 54),
        "inherited exact sign p": (float(original_primary["exact_sign_two_sided_p"]), 0.4841184136072915),
        "original hybrid positive lifetimes": (int(original_hybrid["positive_lifetimes"]), 62),
        "selected positive lifetimes": (int(extension_selected["positive_lifetimes"]), 52),
        "selected exact sign p": (float(extension_selected["exact_sign_two_sided_p"]), 0.7643534344026668),
    }
    for label, (actual, expected) in integer_checks.items():
        close(float(actual), float(expected), label)

    required_fragments = [
        "$+0.965$", "$[0.733,1.196]$", "2.15\\%",
        "$+0.736$", "$[0.524,0.950]$", "1.3\\%",
        "$+0.721$", "$[0.474,0.978]$", "2.3\\%", "1.4\\%",
        "$-9.068$", "$[-9.507,-8.630]$", "23.6\\%",
        "$+0.692$", "$[0.434,0.962]$", "1.6\\%", "1.9\\%",
        "$-0.008$", "$[-0.121,0.108]$", "not that calibration was necessary",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise AssertionError(f"manuscript fragments missing: {missing}")

    forbidden = [
        "calibration recovered the tolerance",
        "calibration restored the tolerance",
        "calibration was superior",
    ]
    lowered = source.lower()
    present_forbidden = [phrase for phrase in forbidden if phrase in lowered]
    if present_forbidden:
        raise AssertionError(f"unsupported manuscript claims: {present_forbidden}")

    report = f"""# TMLR claim-to-evidence audit — 2026-09-01

Status: **PASS**. Principal manuscript numbers are traced to immutable study
artifacts and the corrected policy/result boundaries are present in source.

| Manuscript claim | Verified value | Authoritative artifact | Status |
|---|---:|---|---|
| Pusher physics-only uncertainty effect | +0.965, CI [0.733, 1.196], 77/100 positive | `{rel(pusher_contrast_path)}`; `{rel(pusher_seed_path)}` | PASS |
| Pusher physics-only maximum OOD trip rate | 2.15% (fails 2.0% point rule) | `{rel(pusher_policy_path)}` | PASS |
| Pusher hybrid uncertainty effect | +0.736, CI [0.524, 0.950], 72/100 positive | `{rel(pusher_contrast_path)}`; `{rel(pusher_seed_path)}` | PASS |
| Pusher hybrid maximum OOD trip rate | 1.30% (passes point rule) | `{rel(pusher_policy_path)}` | PASS |
| Pusher residual effect at matched margin | +0.103, CI [-0.004, 0.215] | `{rel(pusher_contrast_path)}` | PASS; independent benefit not established |
| Reacher inherited physics margin | +0.721, CI [0.474, 0.978], 54/100 positive | `{rel(reacher_contrast_path)}` | PASS |
| Reacher inherited physics maximum trip rate | 2.30% (original gate failure) | `{rel(original_cells_path)}` | PASS |
| Reacher original hybrid comparator | +0.797; maximum trip rate 1.40% | `{rel(reacher_contrast_path)}`; `{rel(original_cells_path)}` | PASS |
| Reacher monolithic recurrent comparator | -9.068, CI [-9.507, -8.630]; maximum trip rate 23.60% | `{rel(reacher_contrast_path)}`; `{rel(original_cells_path)}` | PASS; bounded to tested model and budget |
| Reacher selected margin | +0.692, CI [0.434, 0.962], 52/100 positive, exact sign p=0.764 | `{rel(reacher_contrast_path)}` | PASS |
| Reacher selected maximum trip rate | 1.60% on extension lifetimes | `{rel(extension_cells_path)}` | PASS |
| Inherited margin on the same extension sample | 1.90% | `{rel(extension_cells_path)}` | PASS |
| Selected versus inherited margin | -0.008, CI [-0.121, 0.108] | `{rel(reacher_contrast_path)}` | PASS; no detectable superiority |

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
"""
    output = args.output if args.output.is_absolute() else REPOSITORY / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "pass", "checks": len(checks) + 16, "report": rel(output)}))


if __name__ == "__main__":
    main()
