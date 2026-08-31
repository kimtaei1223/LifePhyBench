#!/usr/bin/env python3
"""Freeze and execute the untouched-seed Reacher confirmatory evaluation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from scripts.qualify_hierarchical_v11 import (
        QualificationDesign,
        _physical_steps_from_info,
        _trip_from_info,
        make_default_environment_factory,
    )
    from scripts.run_physics_residual_v12_confirmatory_pipeline import (
        sha256,
        summarize_values,
    )
    from scripts.run_physics_residual_v12_pilot import _designs
    from scripts.run_reacher_belief_development import (
        evaluate_calibration_job,
        initialize_calibration,
        policy_specs,
    )
except ModuleNotFoundError:
    from qualify_hierarchical_v11 import (  # type: ignore[no-redef]
        QualificationDesign,
        _physical_steps_from_info,
        _trip_from_info,
        make_default_environment_factory,
    )
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        sha256,
        summarize_values,
    )
    from run_physics_residual_v12_pilot import _designs  # type: ignore[no-redef]
    from run_reacher_belief_development import (  # type: ignore[no-redef]
        evaluate_calibration_job,
        initialize_calibration,
        policy_specs,
    )


TARGET_OOD = ("ood_sensor_noise", "ood_cooling", "ood_combined")
ALL_CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_shocks",
    "ood_combined",
)

_MONOLITHIC_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_MONOLITHIC_MODEL: Any | None = None


def atomic_json(
    path: Path, document: dict[str, Any], *, overwrite: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
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
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def initialize_monolithic(low_level_model: str, monolithic_model: str) -> None:
    global _MONOLITHIC_FACTORY, _MONOLITHIC_MODEL
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    from sb3_contrib import RecurrentPPO

    _MONOLITHIC_FACTORY = make_default_environment_factory(
        Path(low_level_model), environment_id="Reacher-v5"
    )
    _MONOLITHIC_MODEL = RecurrentPPO.load(monolithic_model, device="cpu")


def evaluate_monolithic_job(
    job: tuple[str, dict[str, float], tuple[int, ...]]
) -> tuple[str, str, dict[str, Any]]:
    if _MONOLITHIC_FACTORY is None or _MONOLITHIC_MODEL is None:
        raise RuntimeError("monolithic worker is not initialized")
    condition, design_document, seeds = job
    design = QualificationDesign(**design_document)
    lifetime_rows = []
    physical_audits = 0
    for seed in seeds:
        environment = _MONOLITHIC_FACTORY(design, seed)
        rewards: list[float] = []
        actions: list[int] = []
        trips: list[bool] = []
        state = None
        episode_start = np.asarray([True])
        try:
            observation, _reset_info = environment.reset(seed=seed)
            for task_index in range(20):
                action, state = _MONOLITHIC_MODEL.predict(
                    observation,
                    state=state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                action_value = int(np.asarray(action).item())
                observation, reward, terminated, truncated, info = environment.step(
                    action_value
                )
                boundary = bool(terminated or truncated)
                if boundary != (task_index == 19):
                    raise RuntimeError("monolithic confirmatory lifetime boundary mismatch")
                if (_physical_steps_from_info(info) or 0) > 1:
                    physical_audits += 1
                rewards.append(float(reward))
                actions.append(action_value)
                trips.append(_trip_from_info(info))
                episode_start = np.asarray([boundary])
        finally:
            environment.close()
        lifetime_rows.append(
            {
                "seed": int(seed),
                "mean_reward_per_task": float(np.mean(rewards)),
                "trip_rate": float(np.mean(trips)),
                "high_rate": float(np.mean(actions)),
            }
        )
    summary = {
        "lifetimes": len(lifetime_rows),
        "tasks": 20 * len(lifetime_rows),
        "mean_reward_per_task": float(
            np.mean([row["mean_reward_per_task"] for row in lifetime_rows])
        ),
        "trip_rate": float(np.mean([row["trip_rate"] for row in lifetime_rows])),
        "high_rate": float(np.mean([row["high_rate"] for row in lifetime_rows])),
        "low_level_rollout_audit_observations": physical_audits,
        "uncertainty_calibration": None,
        "lifetime_rows": lifetime_rows,
    }
    spec = {"name": "monolithic_recurrent", "type": "learned_recurrent_baseline"}
    return condition, "monolithic_recurrent", {"spec": spec, "summary": summary}


def paired_values(
    cells: dict[str, dict[str, Any]],
    condition: str,
    treatment: str,
    control: str,
    field: str = "mean_reward_per_task",
) -> tuple[list[int], np.ndarray]:
    def mapping(policy: str) -> dict[int, float]:
        return {
            int(row["seed"]): float(row[field])
            for row in cells[condition][policy]["summary"]["lifetime_rows"]
        }

    treated = mapping(treatment)
    controlled = mapping(control)
    if set(treated) != set(controlled):
        raise ValueError(f"paired seed mismatch: {condition} {treatment} {control}")
    seeds = sorted(treated)
    return seeds, np.asarray(
        [treated[seed] - controlled[seed] for seed in seeds], dtype=np.float64
    )


def analyze(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contrasts = {
        "primary_uncertainty": ("physics_z1_5", "physics_z0"),
        "hybrid_vs_physics_z0": ("hybrid_z1_5", "physics_z0"),
        "residual_at_z1_5": ("hybrid_z1_5", "physics_z1_5"),
        "monolithic_vs_physics_z0": ("monolithic_recurrent", "physics_z0"),
        "uncertainty_vs_current_sensor": ("physics_z1_5", "current_sensor"),
    }
    condition_results: dict[str, Any] = {}
    values_by_contrast: dict[str, dict[str, tuple[list[int], np.ndarray]]] = {
        name: {} for name in contrasts
    }
    for condition_index, condition in enumerate(ALL_CONDITIONS):
        condition_results[condition] = {}
        for contrast_index, (name, (treatment, control)) in enumerate(contrasts.items()):
            seeds, values = paired_values(cells, condition, treatment, control)
            values_by_contrast[name][condition] = seeds, values
            summary = summarize_values(
                values, seed=82500 + 100 * condition_index + contrast_index
            )
            summary["seeds"] = seeds
            condition_results[condition][name] = summary

    aggregate_results: dict[str, Any] = {}
    for contrast_index, name in enumerate(contrasts):
        seed_lists = [values_by_contrast[name][condition][0] for condition in TARGET_OOD]
        if not all(seeds == seed_lists[0] for seeds in seed_lists[1:]):
            raise ValueError(f"target OOD seed mismatch: {name}")
        values = np.mean(
            np.stack([values_by_contrast[name][condition][1] for condition in TARGET_OOD]),
            axis=0,
        )
        summary = summarize_values(values, seed=83500 + contrast_index)
        summary["seeds"] = seed_lists[0]
        aggregate_results[name] = summary

    primary = aggregate_results["primary_uncertainty"]
    max_trip_rate = max(
        float(cells[condition]["physics_z1_5"]["summary"]["trip_rate"])
        for condition in ALL_CONDITIONS
    )
    criteria = {
        "paired_mean_reward_difference_above_zero": primary["mean"] > 0.0,
        "paired_bootstrap_95_ci_lower_above_zero": primary["bootstrap_95_ci"][0] > 0.0,
        "paired_sign_flip_two_sided_p_below_0_05": primary["sign_flip_two_sided_p"] < 0.05,
        "maximum_trip_rate_each_condition_at_most_0_02": max_trip_rate <= 0.02,
    }
    coverage = {
        condition: {
            policy: cells[condition][policy]["summary"]["uncertainty_calibration"]
            for policy in ("physics_z0", "physics_z1_5", "hybrid_z1_5")
        }
        for condition in ALL_CONDITIONS
    }
    return {
        "phase": "reacher_cross_task_confirmatory_analysis",
        "condition_contrasts": condition_results,
        "target_ood_aggregate_contrasts": aggregate_results,
        "uncertainty_calibration": coverage,
        "maximum_physics_z1_5_trip_rate": max_trip_rate,
        "primary_criteria": criteria,
        "primary_success": all(criteria.values()),
    }


def git_head(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1-protocol", type=Path, default=Path("configs/reacher_cross_task_replication_v1.json")
    )
    parser.add_argument(
        "--stage2-protocol", type=Path, default=Path("configs/reacher_cross_task_stage2_v1.json")
    )
    parser.add_argument(
        "--low-level-selection", type=Path, default=Path("outputs/reacher_replication/low_level/SELECTION.json")
    )
    parser.add_argument(
        "--belief-root", type=Path, default=Path("outputs/reacher_replication/belief_development")
    )
    parser.add_argument(
        "--monolithic-selection", type=Path, default=Path("outputs/reacher_replication/monolithic_baseline/SELECTION.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication/confirmatory")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    project_root = Path(__file__).resolve().parent.parent

    def resolved(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    stage1_protocol = resolved(args.stage1_protocol)
    stage2_protocol = resolved(args.stage2_protocol)
    low_selection_path = resolved(args.low_level_selection)
    belief_root = resolved(args.belief_root)
    mono_selection_path = resolved(args.monolithic_selection)
    output_root = resolved(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = output_root / "FROZEN_PROTOCOL.sha256"
    cells_path = output_root / "CONFIRMATORY_CELLS.json"
    result_path = output_root / "CONFIRMATORY_RESULTS.json"
    status_path = output_root / "status.json"
    if result_path.exists():
        print(result_path.read_text(encoding="utf-8"))
        return

    expected_stage1 = "4c1bbbeae5b5a28a0691e6b3f12c4d20a0443ef6a3b14c0e934fa90a1e7f509d"
    expected_stage2 = "520058ce41af8e925dccc986129e18a01e39945fc3d30427e0ec616d0c732ff9"
    if sha256(stage1_protocol) != expected_stage1 or sha256(stage2_protocol) != expected_stage2:
        raise SystemExit("parent Reacher protocol hash mismatch")
    low_selection = read_json(low_selection_path)
    mono_selection = read_json(mono_selection_path)
    low_model = Path(low_selection["selected"]["model"])
    residual_checkpoint = belief_root / "residual_model.pt"
    monolithic_model = Path(mono_selection["selected"]["model"])
    for path in (low_model, residual_checkpoint, monolithic_model):
        if not path.is_file():
            raise SystemExit(f"required checkpoint missing: {path}")
    sources = [
        Path(__file__).resolve(),
        project_root / "scripts/run_reacher_belief_development.py",
        project_root / "scripts/run_physics_residual_v12_pilot.py",
        project_root / "scripts/qualify_physics_belief_v12.py",
        project_root / "scripts/qualify_hierarchical_v11.py",
        project_root / "src/lifephybench/envs/hierarchical_thermal_v11.py",
        project_root / "src/lifephybench/envs/thermal_commitment.py",
        project_root / "src/lifephybench/envs/mujoco_pusher.py",
    ]
    designs = _designs()
    seeds = tuple(range(25200, 25300))
    if protocol_path.exists() and args.resume:
        protocol = read_json(protocol_path)
        protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
        if sha256(protocol_path) != protocol_hash:
            raise SystemExit("Reacher confirmatory protocol hash mismatch")
    else:
        if protocol_path.exists():
            raise SystemExit("frozen protocol exists; use --resume")
        protocol = {
            "phase": "reacher_cross_task_frozen_confirmatory_protocol",
            "status": "frozen_before_any_25200_25299_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_head(project_root),
            "parent_protocols": {
                str(stage1_protocol.relative_to(project_root)): expected_stage1,
                str(stage2_protocol.relative_to(project_root)): expected_stage2,
            },
            "seeds": list(seeds),
            "seeds_untouched_before_freeze": True,
            "conditions": {name: asdict(design) for name, design in designs.items()},
            "target_ood": list(TARGET_OOD),
            "analytic_policy_specs": policy_specs(),
            "monolithic_policy": mono_selection["selected"],
            "primary_estimand": "target-OOD paired physics_z1_5 minus physics_z0 reward per task",
            "primary_criteria": {
                "paired_mean_reward_difference_above": 0.0,
                "paired_bootstrap_95_ci_lower_above": 0.0,
                "paired_sign_flip_two_sided_p_below": 0.05,
                "maximum_physics_z1_5_trip_rate_each_condition": 0.02,
            },
            "checkpoints": {
                "low_level": {"path": str(low_model), "sha256": sha256(low_model)},
                "residual": {"path": str(residual_checkpoint), "sha256": sha256(residual_checkpoint)},
                "monolithic": {"path": str(monolithic_model), "sha256": sha256(monolithic_model)},
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
        atomic_json(protocol_path, protocol)
        protocol_hash = sha256(protocol_path)
        protocol_hash_path.write_text(protocol_hash + "\n", encoding="utf-8")

    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after Reacher freeze: {relative}")
    for label, row in protocol["checkpoints"].items():
        if sha256(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"{label} checkpoint drift after Reacher freeze")

    if cells_path.exists() and args.resume:
        cells = read_json(cells_path)
    else:
        if cells_path.exists():
            raise SystemExit("confirmatory cells exist; use --resume")
        atomic_json(
            status_path,
            {
                "status": "evaluating_analytic_baselines",
                "protocol_sha256": protocol_hash,
            },
            overwrite=True,
        )
        analytic_jobs = [
            (condition, spec, asdict(design), seeds)
            for condition, design in designs.items()
            for spec in protocol["analytic_policy_specs"]
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=min(args.workers, len(analytic_jobs)),
            initializer=initialize_calibration,
            initargs=(str(low_model), str(residual_checkpoint)),
        ) as pool:
            analytic_rows = pool.map(evaluate_calibration_job, analytic_jobs)
        atomic_json(
            status_path,
            {
                "status": "evaluating_monolithic_baseline",
                "protocol_sha256": protocol_hash,
            },
            overwrite=True,
        )
        monolithic_jobs = [
            (condition, asdict(design), seeds)
            for condition, design in designs.items()
        ]
        with context.Pool(
            processes=min(5, args.workers),
            initializer=initialize_monolithic,
            initargs=(str(low_model), str(monolithic_model)),
        ) as pool:
            monolithic_rows = pool.map(evaluate_monolithic_job, monolithic_jobs)
        cells = {}
        for condition, policy, row in analytic_rows + monolithic_rows:
            cells.setdefault(condition, {})[policy] = row
        atomic_json(cells_path, cells)

    report = analyze(cells)
    report.update(
        {
            "status": "complete",
            "protocol_sha256": protocol_hash,
            "confirmatory_lifetimes_per_policy_condition": len(seeds),
            "confirmatory_seed_count": len(seeds),
            "wiring_passed": True,
        }
    )
    atomic_json(result_path, report)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "primary_success": report["primary_success"],
            "protocol_sha256": protocol_hash,
        },
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "primary_success": report["primary_success"],
                "primary_criteria": report["primary_criteria"],
                "primary": report["target_ood_aggregate_contrasts"]["primary_uncertainty"],
                "maximum_physics_z1_5_trip_rate": report["maximum_physics_z1_5_trip_rate"],
                "output": str(result_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
