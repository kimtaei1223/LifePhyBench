#!/usr/bin/env python3
"""Restartable v12 development audit, protocol freeze, and confirmation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    from scripts.run_physics_residual_v12_pilot import _designs, evaluate_jobs
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_physics_residual_v12_pilot import _designs, evaluate_jobs  # type: ignore[no-redef]


OOD_CONDITIONS = (
    "ood_sensor_noise",
    "ood_cooling",
    "ood_shocks",
    "ood_combined",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def bootstrap_ci(values: np.ndarray, *, seed: int, resamples: int = 100_000) -> list[float]:
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = resamples
    while remaining:
        count = min(10_000, remaining)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        chunks.append(values[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(chunks)
    return [float(value) for value in np.quantile(distribution, [0.025, 0.975])]


def sign_flip_p(values: np.ndarray, *, seed: int, draws: int = 1_000_000) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = 0
    remaining = draws
    while remaining:
        count = min(50_000, remaining)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        randomized = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def exact_sign_p(values: np.ndarray) -> float:
    positive = int(np.count_nonzero(values > 0.0))
    negative = int(np.count_nonzero(values < 0.0))
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    tail = min(positive, negative)
    return float(min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(tail + 1)) / 2**nonzero))


def paired_values(
    result: dict[str, dict[str, dict[str, Any]]],
    condition: str,
    treatment: str,
    control: str,
) -> tuple[list[int], np.ndarray]:
    def rewards(policy: str) -> dict[int, float]:
        rows = result[condition][policy]["summary"]["lifetime_rows"]
        return {int(row["seed"]): float(row["mean_reward_per_task"]) for row in rows}

    treatment_rows = rewards(treatment)
    control_rows = rewards(control)
    if set(treatment_rows) != set(control_rows):
        raise ValueError(f"paired seed mismatch: {condition} {treatment} {control}")
    seeds = sorted(treatment_rows)
    return seeds, np.asarray(
        [treatment_rows[seed] - control_rows[seed] for seed in seeds],
        dtype=np.float64,
    )


def summarize_values(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "bootstrap_95_ci": bootstrap_ci(values, seed=seed),
        "sign_flip_two_sided_p": sign_flip_p(values, seed=seed + 1000),
        "exact_sign_two_sided_p": exact_sign_p(values),
        "positive": int(np.count_nonzero(values > 0.0)),
        "negative": int(np.count_nonzero(values < 0.0)),
    }


def trip_rate(
    result: dict[str, dict[str, dict[str, Any]]], condition: str, policy: str
) -> float:
    return float(result[condition][policy]["summary"]["trip_rate"])


def analyze_development(
    result: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    ood_wins = 0
    for ordinal, condition in enumerate(("in_domain",) + OOD_CONDITIONS):
        seeds, values = paired_values(
            result, condition, "hybrid_belief", "physics_belief"
        )
        summary = summarize_values(values, seed=31000 + ordinal)
        summary["seeds"] = seeds
        summary["hybrid_trip_rate"] = trip_rate(result, condition, "hybrid_belief")
        summary["physics_trip_rate"] = trip_rate(result, condition, "physics_belief")
        effects[condition] = summary
        if condition in OOD_CONDITIONS and summary["mean"] >= 0.25:
            ood_wins += 1
    criteria = {
        "in_domain_noninferior_mean_at_least_minus_0_25": effects["in_domain"]["mean"] >= -0.25,
        "in_domain_hybrid_trip_rate_at_most_0_02": effects["in_domain"]["hybrid_trip_rate"] <= 0.02,
        "at_least_three_ood_mean_wins_of_0_25": ood_wins >= 3,
        "all_ood_hybrid_trip_rates_at_most_0_02": all(
            effects[condition]["hybrid_trip_rate"] <= 0.02
            for condition in OOD_CONDITIONS
        ),
    }
    return {
        "phase": "v12_large_development_ablation",
        "effects": effects,
        "ood_wins_at_least_0_25": ood_wins,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def analyze_confirmatory(
    result: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    condition_effects: dict[str, Any] = {}
    ood_by_seed: dict[int, list[float]] = defaultdict(list)
    for ordinal, condition in enumerate(("in_domain",) + OOD_CONDITIONS):
        seeds, values = paired_values(
            result, condition, "hybrid_belief", "physics_belief"
        )
        condition_effects[condition] = summarize_values(
            values, seed=41000 + ordinal
        )
        condition_effects[condition]["hybrid_trip_rate"] = trip_rate(
            result, condition, "hybrid_belief"
        )
        condition_effects[condition]["physics_trip_rate"] = trip_rate(
            result, condition, "physics_belief"
        )
        if condition in OOD_CONDITIONS:
            for seed, value in zip(seeds, values, strict=True):
                ood_by_seed[seed].append(float(value))
    aggregate_seeds = sorted(ood_by_seed)
    if any(len(ood_by_seed[seed]) != len(OOD_CONDITIONS) for seed in aggregate_seeds):
        raise ValueError("confirmatory OOD aggregate is incomplete")
    aggregate_values = np.asarray(
        [np.mean(ood_by_seed[seed]) for seed in aggregate_seeds], dtype=np.float64
    )
    aggregate = summarize_values(aggregate_values, seed=42000)
    aggregate["seeds"] = aggregate_seeds
    in_domain = condition_effects["in_domain"]
    criteria = {
        "in_domain_mean_noninferior_at_minus_0_25": in_domain["mean"] >= -0.25,
        "in_domain_bootstrap_lower_above_minus_0_50": in_domain["bootstrap_95_ci"][0] > -0.50,
        "ood_aggregate_mean_at_least_0_25": aggregate["mean"] >= 0.25,
        "ood_aggregate_bootstrap_lower_above_zero": aggregate["bootstrap_95_ci"][0] > 0.0,
        "ood_aggregate_sign_flip_p_below_0_05": aggregate["sign_flip_two_sided_p"] < 0.05,
        "all_hybrid_trip_rates_at_most_0_02": all(
            condition_effects[condition]["hybrid_trip_rate"] <= 0.02
            for condition in condition_effects
        ),
    }
    return {
        "phase": "physics_residual_v12_heldout_confirmatory_analysis",
        "status": "final_heldout_result",
        "condition_effects": condition_effects,
        "ood_aggregate_hybrid_minus_physics": aggregate,
        "criteria": criteria,
        "confirmatory_passed": all(criteria.values()),
    }


def policy_specs(selected: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.055},
        {"name": "ema_history", "type": "ema", "alpha": 0.60, "cutoff": 0.060},
        {
            "name": "physics_belief",
            "type": "physics_belief",
            "cutoff": 0.060,
            "uncertainty_multiplier": 0.0,
        },
        {**selected, "name": "hybrid_belief"},
        {
            **selected,
            "name": "hybrid_no_uncertainty",
            "uncertainty_multiplier": 0.0,
        },
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
        default=Path("outputs/physics_residual_v12_confirmatory"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    root = args.output_root.resolve()
    final_path = root / "CONFIRMATORY_RESULTS.json"
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise SystemExit(f"output root exists; use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return
    status_path = root / "status.json"
    refinement_result_path = args.refinement_root / "PILOT_RESULTS.json"
    residual_checkpoint = args.refinement_root / "residual_model.pt"
    refinement = read_json(refinement_result_path)
    if refinement.get("gate", {}).get("passed") is not True:
        raise SystemExit("v12 refinement gate did not pass")
    selected = refinement["selected_hybrid_spec"]
    specs = policy_specs(selected)

    development_cells_path = root / "DEVELOPMENT_ABLATION_CELLS.json"
    development_analysis_path = root / "DEVELOPMENT_ABLATION_ANALYSIS.json"
    if development_analysis_path.exists() and args.resume:
        development_cells = read_json(development_cells_path)
        development_analysis = read_json(development_analysis_path)
    else:
        atomic_json(
            status_path,
            {"status": "running_large_development_ablation", "phase": "v12_automatic_confirmatory_pipeline"},
            overwrite=True,
        )
        development_cells = evaluate_jobs(
            build_jobs(specs, tuple(range(12000, 12030))),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=residual_checkpoint,
        )
        development_analysis = analyze_development(development_cells)
        atomic_json(development_cells_path, development_cells, overwrite=False)
        atomic_json(development_analysis_path, development_analysis, overwrite=False)
    if development_analysis.get("passed") is not True:
        atomic_json(
            status_path,
            {"status": "stopped_development_gate_failed", "phase": "v12_automatic_confirmatory_pipeline"},
            overwrite=True,
        )
        print(json.dumps(development_analysis, indent=2, sort_keys=True))
        return

    protocol_path = root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = root / "FROZEN_PROTOCOL.sha256"
    project_root = Path(__file__).resolve().parent.parent
    frozen_sources = [
        Path(__file__).resolve(),
        project_root / "scripts/run_physics_residual_v12_pilot.py",
        project_root / "scripts/qualify_physics_belief_v12.py",
        project_root / "scripts/qualify_hierarchical_v11.py",
        project_root / "src/lifephybench/envs/hierarchical_thermal_v11.py",
    ]
    if protocol_path.exists() and args.resume:
        protocol = read_json(protocol_path)
        expected_protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
        if sha256(protocol_path) != expected_protocol_hash:
            raise SystemExit("frozen v12 protocol hash mismatch")
    else:
        protocol = {
            "phase": "physics_residual_v12_frozen_confirmatory_protocol",
            "status": "frozen_before_v12_heldout_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "confirmatory_evidence": True,
            "development_seeds": list(range(12000, 12030)),
            "heldout_evaluation_seeds": list(range(22000, 22100)),
            "heldout_untouched_before_freeze": True,
            "designs": {name: asdict(design) for name, design in _designs().items()},
            "policy_specs": specs,
            "primary_comparison": "hybrid_belief_minus_physics_belief",
            "primary_analysis": {
                "unit": "paired independent lifetime seed",
                "bootstrap_resamples": 100_000,
                "sign_flip_draws": 1_000_000,
                "criteria": {
                    "in_domain_mean_noninferior_at": -0.25,
                    "in_domain_bootstrap_lower_above": -0.50,
                    "ood_aggregate_mean_at_least": 0.25,
                    "ood_aggregate_bootstrap_lower_above": 0.0,
                    "ood_aggregate_sign_flip_p_below": 0.05,
                    "maximum_hybrid_trip_rate_each_condition": 0.02,
                },
            },
            "checkpoint": {
                "path": str(residual_checkpoint.resolve()),
                "sha256": sha256(residual_checkpoint),
            },
            "refinement_result": {
                "path": str(refinement_result_path.resolve()),
                "sha256": sha256(refinement_result_path),
            },
            "low_level_model": {
                "path": str(args.low_level_model.resolve()),
                "sha256": sha256(args.low_level_model),
            },
            "source_hashes": {
                str(path.relative_to(project_root)): sha256(path)
                for path in frozen_sources
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
        atomic_json(protocol_path, protocol, overwrite=False)
        protocol_hash_path.write_text(sha256(protocol_path) + "\n", encoding="utf-8")
        expected_protocol_hash = sha256(protocol_path)

    if sha256(residual_checkpoint) != protocol["checkpoint"]["sha256"]:
        raise SystemExit("residual checkpoint drift after freeze")
    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after v12 freeze: {relative}")

    confirmatory_cells_path = root / "CONFIRMATORY_CELLS.json"
    if confirmatory_cells_path.exists() and args.resume:
        confirmatory_cells = read_json(confirmatory_cells_path)
    else:
        atomic_json(
            status_path,
            {
                "status": "running_heldout_confirmatory_evaluation",
                "phase": "v12_automatic_confirmatory_pipeline",
                "protocol_sha256": expected_protocol_hash,
            },
            overwrite=True,
        )
        confirmatory_cells = evaluate_jobs(
            build_jobs(specs, tuple(protocol["heldout_evaluation_seeds"])),
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=residual_checkpoint,
        )
        atomic_json(confirmatory_cells_path, confirmatory_cells, overwrite=False)
    report = analyze_confirmatory(confirmatory_cells)
    report.update(
        {
            "protocol_sha256": expected_protocol_hash,
            "checkpoint_sha256": sha256(residual_checkpoint),
            "wiring_passed": True,
            "development_gate_passed": True,
            "scientific_null_is_normal_completion": not report["confirmatory_passed"],
        }
    )
    atomic_json(final_path, report, overwrite=False)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "phase": "v12_automatic_confirmatory_pipeline",
            "confirmatory_passed": report["confirmatory_passed"],
            "protocol_sha256": expected_protocol_hash,
        },
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "confirmatory_passed": report["confirmatory_passed"],
                "criteria": report["criteria"],
                "ood_aggregate": report["ood_aggregate_hybrid_minus_physics"],
                "output": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
