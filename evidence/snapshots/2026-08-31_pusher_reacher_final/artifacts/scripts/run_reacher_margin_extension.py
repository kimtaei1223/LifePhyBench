#!/usr/bin/env python3
"""Calibrate one Reacher safety margin, then test it on untouched fresh seeds."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.run_physics_residual_v12_confirmatory_pipeline import (
        sha256,
        summarize_values,
    )
    from scripts.run_physics_residual_v12_pilot import _designs
    from scripts.run_reacher_belief_development import (
        evaluate_calibration_job,
        initialize_calibration,
    )
    from scripts.run_reacher_confirmatory import paired_values
except ModuleNotFoundError:
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        sha256,
        summarize_values,
    )
    from run_physics_residual_v12_pilot import _designs  # type: ignore[no-redef]
    from run_reacher_belief_development import (  # type: ignore[no-redef]
        evaluate_calibration_job,
        initialize_calibration,
    )
    from run_reacher_confirmatory import paired_values  # type: ignore[no-redef]


ALL_CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_shocks",
    "ood_combined",
)


def atomic_json(
    path: Path, document: dict[str, Any], *, overwrite: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": f"physics_c{cutoff:.3f}_z{multiplier:g}",
            "type": "physics_belief",
            "cutoff": float(cutoff),
            "uncertainty_multiplier": float(multiplier),
        }
        for cutoff in config["cutoffs"]
        for multiplier in config["uncertainty_multipliers"]
    ]


def collect_cells(
    jobs: list[tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]],
    *,
    workers: int,
    low_model: Path,
    residual_checkpoint: Path,
) -> dict[str, dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(workers, len(jobs)),
        initializer=initialize_calibration,
        initargs=(str(low_model), str(residual_checkpoint)),
    ) as pool:
        rows = pool.map(evaluate_calibration_job, jobs)
    cells: dict[str, dict[str, Any]] = {}
    for condition, policy, row in rows:
        cells.setdefault(condition, {})[policy] = row
    return cells


def select_candidate(
    cells: dict[str, dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    target_ood: tuple[str, ...],
    safety_buffer: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frontier = []
    for spec in specs:
        name = str(spec["name"])
        max_trip = max(
            float(cells[condition][name]["summary"]["trip_rate"])
            for condition in ALL_CONDITIONS
        )
        target_reward = float(
            np.mean(
                [
                    cells[condition][name]["summary"]["mean_reward_per_task"]
                    for condition in target_ood
                ]
            )
        )
        frontier.append(
            {
                "spec": spec,
                "maximum_trip_rate": max_trip,
                "target_ood_mean_reward_per_task": target_reward,
                "buffered_safe": max_trip <= safety_buffer,
            }
        )
    safe = [row for row in frontier if row["buffered_safe"]]
    if safe:
        selected = max(
            safe,
            key=lambda row: (
                row["target_ood_mean_reward_per_task"],
                -row["maximum_trip_rate"],
            ),
        )
    else:
        selected = min(
            frontier,
            key=lambda row: (
                row["maximum_trip_rate"],
                -row["target_ood_mean_reward_per_task"],
            ),
        )
    return dict(selected["spec"]), frontier


def reward_decomposition(summary: dict[str, Any]) -> dict[str, float]:
    reward = float(summary["mean_reward_per_task"])
    high_rate = float(summary["high_rate"])
    trip_rate = float(summary["trip_rate"])
    bonus = 2.0 * max(0.0, high_rate - trip_rate)
    penalty = -75.0 * trip_rate
    return {
        "mean_reward_per_task": reward,
        "mean_base_task_return": reward - bonus - penalty,
        "mean_throughput_bonus": bonus,
        "mean_trip_penalty": penalty,
    }


def analyze(
    cells: dict[str, dict[str, Any]], selected_name: str, target_ood: tuple[str, ...]
) -> dict[str, Any]:
    contrasts = {
        "selected_vs_physics_z0": (selected_name, "physics_z0"),
        "selected_vs_inherited": (selected_name, "inherited_physics_z1_5"),
        "selected_vs_current_sensor": (selected_name, "current_sensor"),
        "hybrid_vs_physics_z0": ("hybrid_z1_5", "physics_z0"),
    }
    aggregate: dict[str, Any] = {}
    condition_results: dict[str, Any] = {condition: {} for condition in ALL_CONDITIONS}
    for contrast_index, (name, (treatment, control)) in enumerate(contrasts.items()):
        condition_values: list[np.ndarray] = []
        seed_reference: list[int] | None = None
        for condition_index, condition in enumerate(ALL_CONDITIONS):
            seeds, values = paired_values(cells, condition, treatment, control)
            summary = summarize_values(
                values, seed=85300 + 100 * contrast_index + condition_index
            )
            summary["seeds"] = seeds
            condition_results[condition][name] = summary
            if condition in target_ood:
                if seed_reference is None:
                    seed_reference = seeds
                elif seeds != seed_reference:
                    raise ValueError("fresh extension seed mismatch")
                condition_values.append(values)
        values = np.mean(np.stack(condition_values), axis=0)
        summary = summarize_values(values, seed=86300 + contrast_index)
        summary["seeds"] = seed_reference
        aggregate[name] = summary
    max_trip = max(
        float(cells[condition][selected_name]["summary"]["trip_rate"])
        for condition in ALL_CONDITIONS
    )
    primary = aggregate["selected_vs_physics_z0"]
    criteria = {
        "selected_minus_physics_z0_mean_above_zero": primary["mean"] > 0.0,
        "paired_bootstrap_95_ci_lower_above_zero": primary["bootstrap_95_ci"][0] > 0.0,
        "paired_sign_flip_two_sided_p_below_0_05": primary["sign_flip_two_sided_p"] < 0.05,
        "maximum_selected_trip_rate_each_condition_at_most_0_02": max_trip <= 0.02,
    }
    decomposition = {
        condition: {
            policy: reward_decomposition(cells[condition][policy]["summary"])
            for policy in cells[condition]
        }
        for condition in ALL_CONDITIONS
    }
    return {
        "phase": "reacher_margin_post_confirmatory_extension_analysis",
        "condition_contrasts": condition_results,
        "target_ood_aggregate_contrasts": aggregate,
        "maximum_selected_trip_rate": max_trip,
        "reward_decomposition": decomposition,
        "criteria": criteria,
        "extension_success": all(criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/reacher_margin_extension_v1.json")
    )
    parser.add_argument(
        "--low-level-selection", type=Path, default=Path("outputs/reacher_replication/low_level/SELECTION.json")
    )
    parser.add_argument(
        "--belief-root", type=Path, default=Path("outputs/reacher_replication/belief_development")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication/margin_extension")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    project_root = Path(__file__).resolve().parent.parent

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    config_path = resolved(args.config)
    low_selection_path = resolved(args.low_level_selection)
    belief_root = resolved(args.belief_root)
    output_root = resolved(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    calibration_path = output_root / "CALIBRATION_FRONTIER_CELLS.json"
    selection_path = output_root / "CALIBRATED_MARGIN_SELECTION.json"
    protocol_path = output_root / "FROZEN_FRESH_PROTOCOL.json"
    protocol_hash_path = output_root / "FROZEN_FRESH_PROTOCOL.sha256"
    cells_path = output_root / "FRESH_CELLS.json"
    results_path = output_root / "FRESH_RESULTS.json"
    status_path = output_root / "status.json"
    if results_path.exists():
        print(results_path.read_text(encoding="utf-8"))
        return
    config = read_json(config_path)
    expected_config_hash = "19d8d4248d3ee3622390bddd43124d0bc638834bad1ab38328c703cc5d269b1b"
    if sha256(config_path) != expected_config_hash:
        raise SystemExit("margin-extension config hash mismatch")
    low_selection = read_json(low_selection_path)
    low_model = Path(low_selection["selected"]["model"])
    residual_checkpoint = belief_root / "residual_model.pt"
    calibration = config["calibration"]
    cal_seeds = tuple(range(int(calibration["seeds"][0]), int(calibration["seeds"][1]) + 1))
    target_ood = tuple(config["fresh_evaluation"]["target_ood"])
    designs = _designs()
    specs = candidate_specs(calibration)
    if calibration_path.exists() and args.resume:
        calibration_cells = read_json(calibration_path)
    else:
        if calibration_path.exists():
            raise SystemExit("calibration frontier exists; use --resume")
        atomic_json(status_path, {"status": "calibrating_margin_frontier"}, overwrite=True)
        jobs = [
            (condition, spec, asdict(design), cal_seeds)
            for condition, design in designs.items()
            for spec in specs
        ]
        calibration_cells = collect_cells(
            jobs,
            workers=args.workers,
            low_model=low_model,
            residual_checkpoint=residual_checkpoint,
        )
        atomic_json(calibration_path, calibration_cells)
    selected, frontier = select_candidate(
        calibration_cells,
        specs,
        target_ood=target_ood,
        safety_buffer=float(calibration["selection_safety_buffer_max_trip_rate"]),
    )
    selection = {
        "phase": "post_confirmatory_reacher_margin_calibration",
        "confirmatory_evidence": False,
        "selection_rule": calibration["selection_rule"],
        "selected": selected,
        "frontier": frontier,
    }
    if selection_path.exists() and args.resume:
        if read_json(selection_path) != selection:
            raise SystemExit("calibrated margin selection drift")
    else:
        atomic_json(selection_path, selection)

    fresh_seeds = tuple(
        range(
            int(config["fresh_evaluation"]["seeds"][0]),
            int(config["fresh_evaluation"]["seeds"][1]) + 1,
        )
    )
    fresh_specs = [
        {**selected, "name": "selected_calibrated_margin"},
        {"name": "physics_z0", "type": "physics_belief", "cutoff": 0.060, "uncertainty_multiplier": 0.0},
        {"name": "inherited_physics_z1_5", "type": "physics_belief", "cutoff": 0.060, "uncertainty_multiplier": 1.5},
        {"name": "hybrid_z1_5", "type": "hybrid_belief", "cutoff": 0.060, "residual_scale": 1.0, "uncertainty_multiplier": 1.5},
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.060},
        {"name": "privileged_state_threshold", "type": "privileged_oracle", "cutoff": 0.060},
    ]
    sources = [
        Path(__file__).resolve(),
        project_root / "scripts/run_reacher_belief_development.py",
        project_root / "scripts/run_reacher_confirmatory.py",
        project_root / "scripts/qualify_physics_belief_v12.py",
        project_root / "scripts/qualify_hierarchical_v11.py",
        project_root / "src/lifephybench/envs/hierarchical_thermal_v11.py",
    ]
    if protocol_path.exists() and args.resume:
        protocol = read_json(protocol_path)
        protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
        if sha256(protocol_path) != protocol_hash:
            raise SystemExit("fresh extension protocol hash mismatch")
    else:
        if protocol_path.exists():
            raise SystemExit("fresh extension protocol exists; use --resume")
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        protocol = {
            "phase": "reacher_margin_extension_frozen_fresh_protocol",
            "status": "frozen_after_calibration_before_any_25300_25399_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "config_sha256": expected_config_hash,
            "calibration_cells_sha256": sha256(calibration_path),
            "selection_sha256": sha256(selection_path),
            "selected_spec": selected,
            "fresh_policy_specs": fresh_specs,
            "conditions": {name: asdict(design) for name, design in designs.items()},
            "target_ood": list(target_ood),
            "fresh_seeds": list(fresh_seeds),
            "fresh_seeds_untouched_before_freeze": True,
            "checkpoints": {
                "low_level": {"path": str(low_model), "sha256": sha256(low_model)},
                "residual": {"path": str(residual_checkpoint), "sha256": sha256(residual_checkpoint)},
            },
            "source_hashes": {
                str(path.relative_to(project_root)): sha256(path) for path in sources
            },
            "success_criteria": config["success_criteria"],
            "interpretation": config["interpretation"],
        }
        atomic_json(protocol_path, protocol)
        protocol_hash = sha256(protocol_path)
        protocol_hash_path.write_text(protocol_hash + "\n", encoding="utf-8")
    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after extension freeze: {relative}")
    for label, row in protocol["checkpoints"].items():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"{label} checkpoint drift after extension freeze")

    if cells_path.exists() and args.resume:
        fresh_cells = read_json(cells_path)
    else:
        if cells_path.exists():
            raise SystemExit("fresh extension cells exist; use --resume")
        atomic_json(
            status_path,
            {"status": "evaluating_fresh_margin_extension", "protocol_sha256": protocol_hash},
            overwrite=True,
        )
        jobs = [
            (condition, spec, asdict(design), fresh_seeds)
            for condition, design in designs.items()
            for spec in protocol["fresh_policy_specs"]
        ]
        fresh_cells = collect_cells(
            jobs,
            workers=args.workers,
            low_model=low_model,
            residual_checkpoint=residual_checkpoint,
        )
        atomic_json(cells_path, fresh_cells)
    report = analyze(fresh_cells, "selected_calibrated_margin", target_ood)
    report.update(
        {
            "status": "complete",
            "protocol_sha256": protocol_hash,
            "selected_spec": selected,
            "original_inherited_margin_result_unchanged": True,
        }
    )
    atomic_json(results_path, report)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "extension_success": report["extension_success"],
            "protocol_sha256": protocol_hash,
        },
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "selected_spec": selected,
                "extension_success": report["extension_success"],
                "criteria": report["criteria"],
                "primary": report["target_ood_aggregate_contrasts"]["selected_vs_physics_z0"],
                "maximum_selected_trip_rate": report["maximum_selected_trip_rate"],
                "output": str(results_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
