#!/usr/bin/env python3
"""Freeze and run the scoped v12.2 held-out confirmation.

Development established a defensible scope: sensor-noise shift, cooling-model
shift, and their combined shift.  Physical shocks remain a reported secondary
boundary condition and cannot change the primary decision.  The protocol is
written and hashed before any held-out evaluation starts.
"""

from __future__ import annotations

import argparse
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
        paired_values,
        sha256,
        summarize_values,
        trip_rate,
    )
    from scripts.run_physics_residual_v12_pilot import _designs, evaluate_jobs
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        paired_values,
        sha256,
        summarize_values,
        trip_rate,
    )
    from run_physics_residual_v12_pilot import (  # type: ignore[no-redef]
        _designs,
        evaluate_jobs,
    )


TARGET_OOD_CONDITIONS = (
    "ood_sensor_noise",
    "ood_cooling",
    "ood_combined",
)
SECONDARY_CONDITIONS = ("ood_shocks",)
ALL_CONDITIONS = ("in_domain",) + TARGET_OOD_CONDITIONS + SECONDARY_CONDITIONS


def atomic_json(path: Path, document: dict[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
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
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def policy_specs(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.055},
        {
            "name": "ema_history",
            "type": "ema",
            "alpha": 0.60,
            "cutoff": 0.060,
        },
        {
            "name": "physics_belief",
            "type": "physics_belief",
            "cutoff": 0.060,
            "uncertainty_multiplier": 0.0,
        },
        {**selected, "name": "hybrid_belief"},
        {"name": "privileged_oracle", "type": "privileged_oracle", "cutoff": 0.060},
    ]


def build_jobs(
    specs: list[dict[str, Any]], seeds: tuple[int, ...]
) -> list[tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]]:
    return [
        (condition, spec, asdict(design), seeds)
        for condition, design in _designs().items()
        for spec in specs
    ]


def analyze_scoped_confirmatory(
    cells: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    targeted_by_seed: dict[int, list[float]] = defaultdict(list)
    for ordinal, condition in enumerate(ALL_CONDITIONS):
        seeds, values = paired_values(
            cells, condition, "hybrid_belief", "physics_belief"
        )
        summary = summarize_values(values, seed=61000 + ordinal)
        summary["seeds"] = seeds
        summary["hybrid_trip_rate"] = trip_rate(
            cells, condition, "hybrid_belief"
        )
        summary["physics_trip_rate"] = trip_rate(
            cells, condition, "physics_belief"
        )
        effects[condition] = summary
        if condition in TARGET_OOD_CONDITIONS:
            for seed, value in zip(seeds, values, strict=True):
                targeted_by_seed[seed].append(float(value))

    aggregate_seeds = sorted(targeted_by_seed)
    if any(
        len(targeted_by_seed[seed]) != len(TARGET_OOD_CONDITIONS)
        for seed in aggregate_seeds
    ):
        raise ValueError("targeted OOD aggregate is incomplete")
    aggregate_values = np.asarray(
        [np.mean(targeted_by_seed[seed]) for seed in aggregate_seeds],
        dtype=np.float64,
    )
    aggregate = summarize_values(aggregate_values, seed=62000)
    aggregate["seeds"] = aggregate_seeds

    in_domain = effects["in_domain"]
    criteria = {
        "in_domain_mean_noninferior_at_minus_0_25": in_domain["mean"] >= -0.25,
        "in_domain_bootstrap_lower_above_minus_0_50": (
            in_domain["bootstrap_95_ci"][0] > -0.50
        ),
        "each_target_ood_mean_at_least_0_25": all(
            effects[name]["mean"] >= 0.25 for name in TARGET_OOD_CONDITIONS
        ),
        "each_target_ood_bootstrap_lower_above_zero": all(
            effects[name]["bootstrap_95_ci"][0] > 0.0
            for name in TARGET_OOD_CONDITIONS
        ),
        "target_ood_aggregate_mean_at_least_0_25": aggregate["mean"] >= 0.25,
        "target_ood_aggregate_bootstrap_lower_above_zero": (
            aggregate["bootstrap_95_ci"][0] > 0.0
        ),
        "target_ood_aggregate_sign_flip_p_below_0_05": (
            aggregate["sign_flip_two_sided_p"] < 0.05
        ),
        "all_condition_hybrid_trip_rates_at_most_0_02": all(
            effects[name]["hybrid_trip_rate"] <= 0.02 for name in ALL_CONDITIONS
        ),
    }
    return {
        "phase": "physics_residual_v12_2_scoped_heldout_confirmation",
        "status": "final_heldout_result",
        "primary_scope": list(TARGET_OOD_CONDITIONS),
        "secondary_boundary_conditions": list(SECONDARY_CONDITIONS),
        "condition_effects": effects,
        "target_ood_aggregate_hybrid_minus_physics": aggregate,
        "criteria": criteria,
        "confirmatory_passed": all(criteria.values()),
    }


def development_scope_is_supported(audit: dict[str, Any]) -> bool:
    effects = audit["effects"]
    return bool(
        effects["in_domain"]["mean"] >= -0.25
        and all(effects[name]["mean"] >= 0.25 for name in TARGET_OOD_CONDITIONS)
        and all(
            effects[name]["bootstrap_95_ci"][0] > 0.0
            for name in TARGET_OOD_CONDITIONS
        )
        and all(effects[name]["hybrid_trip_rate"] <= 0.02 for name in ALL_CONDITIONS)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recovery-root",
        type=Path,
        default=Path("outputs/physics_residual_v12_1_recovery"),
    )
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
        default=Path("outputs/physics_residual_v12_2_scoped_confirmatory"),
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
    final_path = root / "CONFIRMATORY_RESULTS.json"
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return

    search_path = args.recovery_root / "SEARCH_ANALYSIS.json"
    audit_path = args.recovery_root / "FRESH_DEVELOPMENT_AUDIT.json"
    checkpoint = args.refinement_root / "residual_model.pt"
    refinement_result = args.refinement_root / "PILOT_RESULTS.json"
    search = read_json(search_path)
    audit = read_json(audit_path)
    if search.get("passed") is not True or search.get("selected") is None:
        raise SystemExit("v12.1 focused search did not select a controller")
    if not development_scope_is_supported(audit):
        raise SystemExit("development evidence does not support the scoped claim")
    selected = search["selected"]["spec"]

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
            raise SystemExit("frozen v12.2 protocol hash mismatch")
    else:
        protocol = {
            "phase": "physics_residual_v12_2_scoped_frozen_protocol",
            "status": "frozen_before_any_v12_2_heldout_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "confirmatory_evidence": True,
            "scope_rationale": (
                "Fresh development evidence supported sensor-noise, cooling-model, "
                "and combined shifts; physical shocks were null and are therefore "
                "reported only as a secondary boundary condition."
            ),
            "primary_scope": list(TARGET_OOD_CONDITIONS),
            "secondary_boundary_conditions": list(SECONDARY_CONDITIONS),
            "heldout_evaluation_seeds": list(range(22000, 22100)),
            "heldout_untouched_before_freeze": True,
            "selected_spec": selected,
            "designs": {name: asdict(value) for name, value in _designs().items()},
            "policy_specs": policy_specs(selected),
            "primary_comparison": "hybrid_belief_minus_physics_belief",
            "primary_unit": "paired independent lifetime seed",
            "primary_criteria": {
                "in_domain_mean_at_least": -0.25,
                "in_domain_bootstrap_lower_above": -0.50,
                "each_target_ood_mean_at_least": 0.25,
                "each_target_ood_bootstrap_lower_above": 0.0,
                "target_ood_aggregate_mean_at_least": 0.25,
                "target_ood_aggregate_bootstrap_lower_above": 0.0,
                "target_ood_aggregate_sign_flip_p_below": 0.05,
                "maximum_hybrid_trip_rate_each_condition": 0.02,
            },
            "development_evidence": {
                "focused_search": {"path": str(search_path.resolve()), "sha256": sha256(search_path)},
                "fresh_audit": {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
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
        raise SystemExit("v12.2 residual checkpoint drift after freeze")
    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after v12.2 freeze: {relative}")

    cells_path = root / "CONFIRMATORY_CELLS.json"
    if cells_path.exists() and args.resume:
        cells = read_json(cells_path)
    else:
        atomic_json(
            status_path,
            {
                "phase": "v12_2_scoped_confirmatory",
                "status": "running_heldout_confirmatory_evaluation",
                "protocol_sha256": protocol_hash,
            },
            overwrite=True,
        )
        cells = evaluate_jobs(
            build_jobs(
                policy_specs(protocol["selected_spec"]),
                tuple(protocol["heldout_evaluation_seeds"]),
            ),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=checkpoint,
        )
        atomic_json(cells_path, cells, overwrite=False)

    report = analyze_scoped_confirmatory(cells)
    report.update(
        {
            "protocol_sha256": protocol_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "wiring_passed": True,
            "scientific_null_is_normal_completion": not report["confirmatory_passed"],
        }
    )
    atomic_json(final_path, report, overwrite=False)
    atomic_json(
        status_path,
        {
            "phase": "v12_2_scoped_confirmatory",
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
                "target_ood_aggregate": report[
                    "target_ood_aggregate_hybrid_minus_physics"
                ],
                "output": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
