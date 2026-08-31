#!/usr/bin/env python3
"""Recover v12 after its development gate miss without touching held-out seeds.

The pipeline performs a focused controller search on already-consumed
development seeds, audits the selected controller on fresh development seeds,
and only then freezes a protocol and evaluates the untouched confirmatory
seeds.  Every stage is restartable and every stop is an intended scientific
decision point rather than a process failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scripts.run_physics_residual_v12_confirmatory_pipeline import (
        OOD_CONDITIONS,
        analyze_confirmatory,
        paired_values,
        sha256,
        summarize_values,
        trip_rate,
    )
    from scripts.run_physics_residual_v12_pilot import _designs, evaluate_jobs
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        OOD_CONDITIONS,
        analyze_confirmatory,
        paired_values,
        sha256,
        summarize_values,
        trip_rate,
    )
    from run_physics_residual_v12_pilot import (  # type: ignore[no-redef]
        _designs,
        evaluate_jobs,
    )


ALL_CONDITIONS = ("in_domain",) + OOD_CONDITIONS


def atomic_json(path: Path, document: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def candidate_specs() -> list[dict[str, Any]]:
    """Focused grid spanning the safe/reward frontier seen in v12 development."""
    return [
        {
            "name": f"hybrid-c{cutoff:.4f}-s{scale:.2f}-z{z:.2f}",
            "type": "hybrid_belief",
            "cutoff": cutoff,
            "residual_scale": scale,
            "uncertainty_multiplier": z,
        }
        for cutoff in (0.0600, 0.0625, 0.0650)
        for scale in (0.50, 1.00, 1.50)
        for z in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]


def physics_spec() -> dict[str, Any]:
    return {
        "name": "physics_belief",
        "type": "physics_belief",
        "cutoff": 0.060,
        "uncertainty_multiplier": 0.0,
    }


def build_jobs(
    specs: list[dict[str, Any]], seeds: tuple[int, ...]
) -> list[tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]]:
    return [
        (condition, spec, asdict(design), seeds)
        for condition, design in _designs().items()
        for spec in specs
    ]


def fast_candidate_row(
    cells: dict[str, dict[str, dict[str, Any]]], policy: str
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for condition in ALL_CONDITIONS:
        seeds, values = paired_values(cells, condition, policy, "physics_belief")
        effects[condition] = {
            "seeds": seeds,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "positive": int(np.count_nonzero(values > 0.0)),
            "negative": int(np.count_nonzero(values < 0.0)),
            "hybrid_trip_rate": trip_rate(cells, condition, policy),
            "physics_trip_rate": trip_rate(cells, condition, "physics_belief"),
        }
    ood_means = [effects[name]["mean"] for name in OOD_CONDITIONS]
    max_trip = max(effects[name]["hybrid_trip_rate"] for name in ALL_CONDITIONS)
    ood_wins = sum(value >= 0.25 for value in ood_means)
    criteria = {
        "in_domain_margin_at_least_minus_0_10": effects["in_domain"]["mean"] >= -0.10,
        "shock_mean_nonnegative": effects["ood_shocks"]["mean"] >= 0.0,
        "at_least_three_ood_mean_wins_of_0_25": ood_wins >= 3,
        "all_trip_rates_at_most_0_018": max_trip <= 0.018,
    }
    return {
        "policy": policy,
        "spec": cells["in_domain"][policy]["spec"],
        "effects": effects,
        "ood_mean": float(np.mean(ood_means)),
        "worst_ood_mean": float(min(ood_means)),
        "ood_wins_at_least_0_25": int(ood_wins),
        "maximum_trip_rate": float(max_trip),
        "criteria": criteria,
        "admissible": all(criteria.values()),
    }


def select_candidate(
    cells: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    names = sorted(
        name for name in cells["in_domain"] if name != "physics_belief"
    )
    rows = [fast_candidate_row(cells, name) for name in names]
    admissible = [row for row in rows if row["admissible"]]
    selected = None
    if admissible:
        selected = max(
            admissible,
            key=lambda row: (
                row["worst_ood_mean"],
                row["ood_mean"],
                row["effects"]["in_domain"]["mean"],
                -row["maximum_trip_rate"],
            ),
        )
    return {
        "phase": "physics_residual_v12_1_focused_development_search",
        "confirmatory_evidence": False,
        "candidate_count": len(rows),
        "admissible_count": len(admissible),
        "selection_rule": {
            "hard_constraints": {
                "in_domain_gain_at_least": -0.10,
                "shock_gain_at_least": 0.0,
                "ood_wins_of_0_25_at_least": 3,
                "maximum_trip_rate": 0.018,
            },
            "ranking": [
                "maximum worst-condition OOD gain",
                "maximum mean OOD gain",
                "maximum in-domain gain",
                "minimum maximum trip rate",
            ],
        },
        "selected": selected,
        "rows": rows,
        "passed": selected is not None,
    }


def detailed_audit(
    cells: dict[str, dict[str, dict[str, Any]]],
    *,
    policy: str,
    random_seed: int,
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    ood_by_seed: dict[int, list[float]] = defaultdict(list)
    for ordinal, condition in enumerate(ALL_CONDITIONS):
        seeds, values = paired_values(cells, condition, policy, "physics_belief")
        summary = summarize_values(values, seed=random_seed + ordinal)
        summary["seeds"] = seeds
        summary["hybrid_trip_rate"] = trip_rate(cells, condition, policy)
        summary["physics_trip_rate"] = trip_rate(
            cells, condition, "physics_belief"
        )
        effects[condition] = summary
        if condition in OOD_CONDITIONS:
            for seed, value in zip(seeds, values, strict=True):
                ood_by_seed[seed].append(float(value))
    aggregate_seeds = sorted(ood_by_seed)
    if any(len(ood_by_seed[seed]) != len(OOD_CONDITIONS) for seed in aggregate_seeds):
        raise ValueError("development OOD aggregate is incomplete")
    aggregate_values = np.asarray(
        [np.mean(ood_by_seed[seed]) for seed in aggregate_seeds], dtype=np.float64
    )
    aggregate = summarize_values(aggregate_values, seed=random_seed + 100)
    aggregate["seeds"] = aggregate_seeds
    ood_wins = sum(effects[name]["mean"] >= 0.25 for name in OOD_CONDITIONS)
    criteria = {
        "in_domain_mean_at_least_minus_0_25": effects["in_domain"]["mean"] >= -0.25,
        "shock_mean_nonnegative": effects["ood_shocks"]["mean"] >= 0.0,
        "ood_aggregate_mean_at_least_0_25": aggregate["mean"] >= 0.25,
        "at_least_three_ood_mean_wins_of_0_25": ood_wins >= 3,
        "all_hybrid_trip_rates_at_most_0_02": all(
            effects[name]["hybrid_trip_rate"] <= 0.02 for name in ALL_CONDITIONS
        ),
    }
    return {
        "phase": "physics_residual_v12_1_fresh_development_audit",
        "confirmatory_evidence": False,
        "effects": effects,
        "ood_aggregate_hybrid_minus_physics": aggregate,
        "ood_wins_at_least_0_25": int(ood_wins),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def final_policy_specs(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.055},
        {
            "name": "ema_history",
            "type": "ema",
            "alpha": 0.60,
            "cutoff": 0.060,
        },
        physics_spec(),
        {**selected, "name": "hybrid_belief"},
        {"name": "privileged_oracle", "type": "privileged_oracle", "cutoff": 0.060},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refinement-root",
        type=Path,
        default=Path("outputs/physics_residual_v12_refinement"),
    )
    parser.add_argument(
        "--low-level-model",
        type=Path,
        default=Path(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/physics_residual_v12_1_recovery"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise SystemExit(f"output root exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    checkpoint = args.refinement_root / "residual_model.pt"
    refinement_result = args.refinement_root / "PILOT_RESULTS.json"
    refinement = read_json(refinement_result)
    if refinement.get("gate", {}).get("passed") is not True:
        raise SystemExit("v12 residual estimator refinement did not pass")

    search_cells_path = root / "SEARCH_CELLS.json"
    search_analysis_path = root / "SEARCH_ANALYSIS.json"
    if search_analysis_path.exists() and args.resume:
        search_cells = read_json(search_cells_path)
        search_analysis = read_json(search_analysis_path)
    else:
        atomic_json(
            status_path,
            {"phase": "v12_1_recovery", "status": "running_focused_search"},
            overwrite=True,
        )
        search_specs = [physics_spec(), *candidate_specs()]
        search_cells = evaluate_jobs(
            build_jobs(search_specs, tuple(range(12000, 12030))),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=checkpoint,
        )
        search_analysis = select_candidate(search_cells)
        atomic_json(search_cells_path, search_cells, overwrite=False)
        atomic_json(search_analysis_path, search_analysis, overwrite=False)
    if search_analysis.get("passed") is not True:
        atomic_json(
            status_path,
            {"phase": "v12_1_recovery", "status": "stopped_no_admissible_candidate"},
            overwrite=True,
        )
        print(json.dumps(search_analysis, indent=2, sort_keys=True))
        return

    selected = search_analysis["selected"]["spec"]
    audit_cells_path = root / "FRESH_DEVELOPMENT_AUDIT_CELLS.json"
    audit_analysis_path = root / "FRESH_DEVELOPMENT_AUDIT.json"
    if audit_analysis_path.exists() and args.resume:
        audit_cells = read_json(audit_cells_path)
        audit = read_json(audit_analysis_path)
    else:
        atomic_json(
            status_path,
            {"phase": "v12_1_recovery", "status": "running_fresh_development_audit"},
            overwrite=True,
        )
        audit_specs = [physics_spec(), {**selected, "name": "hybrid_belief"}]
        audit_cells = evaluate_jobs(
            build_jobs(audit_specs, tuple(range(13000, 13050))),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=checkpoint,
        )
        audit = detailed_audit(
            audit_cells, policy="hybrid_belief", random_seed=51000
        )
        atomic_json(audit_cells_path, audit_cells, overwrite=False)
        atomic_json(audit_analysis_path, audit, overwrite=False)
    if audit.get("passed") is not True:
        atomic_json(
            status_path,
            {"phase": "v12_1_recovery", "status": "stopped_fresh_development_audit_failed"},
            overwrite=True,
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
        return

    final_path = root / "CONFIRMATORY_RESULTS.json"
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return
    protocol_path = root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = root / "FROZEN_PROTOCOL.sha256"
    project_root = Path(__file__).resolve().parent.parent
    sources = [
        Path(__file__).resolve(),
        project_root / "scripts/run_physics_residual_v12_confirmatory_pipeline.py",
        project_root / "scripts/run_physics_residual_v12_pilot.py",
        project_root / "scripts/qualify_physics_belief_v12.py",
        project_root / "scripts/qualify_hierarchical_v11.py",
        project_root / "src/lifephybench/envs/hierarchical_thermal_v11.py",
    ]
    if protocol_path.exists() and args.resume:
        protocol = read_json(protocol_path)
        protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
        if sha256(protocol_path) != protocol_hash:
            raise SystemExit("frozen v12.1 protocol hash mismatch")
    else:
        protocol = {
            "phase": "physics_residual_v12_1_frozen_confirmatory_protocol",
            "status": "frozen_before_heldout_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "confirmatory_evidence": True,
            "focused_search_seeds": list(range(12000, 12030)),
            "fresh_development_audit_seeds": list(range(13000, 13050)),
            "heldout_evaluation_seeds": list(range(22000, 22100)),
            "selected_spec": selected,
            "designs": {name: asdict(value) for name, value in _designs().items()},
            "policy_specs": final_policy_specs(selected),
            "primary_comparison": "hybrid_belief_minus_physics_belief",
            "primary_unit": "paired independent lifetime seed",
            "primary_criteria": {
                "in_domain_mean_at_least": -0.25,
                "in_domain_bootstrap_lower_above": -0.50,
                "ood_aggregate_mean_at_least": 0.25,
                "ood_aggregate_bootstrap_lower_above": 0.0,
                "ood_aggregate_sign_flip_p_below": 0.05,
                "shock_mean_at_least": 0.0,
                "maximum_hybrid_trip_rate_each_condition": 0.02,
            },
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
            "refinement_result": {
                "path": str(refinement_result.resolve()),
                "sha256": sha256(refinement_result),
            },
            "low_level_model": {
                "path": str(args.low_level_model.resolve()),
                "sha256": sha256(args.low_level_model),
            },
            "source_hashes": {
                str(path.relative_to(project_root)): sha256(path) for path in sources
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        atomic_json(protocol_path, protocol, overwrite=False)
        protocol_hash = sha256(protocol_path)
        protocol_hash_path.write_text(protocol_hash + "\n", encoding="utf-8")

    if sha256(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise SystemExit("residual checkpoint drift after v12.1 freeze")
    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after v12.1 freeze: {relative}")

    confirmatory_cells_path = root / "CONFIRMATORY_CELLS.json"
    if confirmatory_cells_path.exists() and args.resume:
        confirmatory_cells = read_json(confirmatory_cells_path)
    else:
        atomic_json(
            status_path,
            {
                "phase": "v12_1_recovery",
                "status": "running_heldout_confirmatory_evaluation",
                "protocol_sha256": protocol_hash,
            },
            overwrite=True,
        )
        confirmatory_cells = evaluate_jobs(
            build_jobs(
                final_policy_specs(protocol["selected_spec"]),
                tuple(protocol["heldout_evaluation_seeds"]),
            ),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=checkpoint,
        )
        atomic_json(confirmatory_cells_path, confirmatory_cells, overwrite=False)
    report = analyze_confirmatory(confirmatory_cells)
    report["criteria"]["shock_mean_nonnegative"] = (
        report["condition_effects"]["ood_shocks"]["mean"] >= 0.0
    )
    report["confirmatory_passed"] = all(report["criteria"].values())
    report.update(
        {
            "phase": "physics_residual_v12_1_heldout_confirmatory_analysis",
            "protocol_sha256": protocol_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "development_search_passed": True,
            "fresh_development_audit_passed": True,
            "scientific_null_is_normal_completion": not report["confirmatory_passed"],
        }
    )
    atomic_json(final_path, report, overwrite=False)
    atomic_json(
        status_path,
        {
            "phase": "v12_1_recovery",
            "status": "complete",
            "confirmatory_passed": report["confirmatory_passed"],
            "protocol_sha256": protocol_hash,
        },
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "confirmatory_passed": report["confirmatory_passed"],
                "criteria": report["criteria"],
                "output": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
