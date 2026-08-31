#!/usr/bin/env python3
"""Frozen 2x2 residual-by-uncertainty mechanism ablation on fresh seeds."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

try:
    from scripts.qualify_hierarchical_v11 import (
        HIGH_POWER_BONUS,
        TRIP_PENALTY,
        QualificationDesign,
        _exact_load,
        _physical_steps_from_info,
        _sensor_from_observation,
        _trip_from_info,
        make_default_environment_factory,
    )
    from scripts.qualify_physics_belief_v12 import PhysicsBeliefPolicy, TransitionModel
    from scripts.run_physics_residual_v12_confirmatory_pipeline import (
        bootstrap_ci,
        sha256,
        summarize_values,
    )
    from scripts.run_physics_residual_v12_pilot import (
        HybridBeliefPolicy,
        _designs,
        load_residual_checkpoint,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qualify_hierarchical_v11 import (  # type: ignore[no-redef]
        HIGH_POWER_BONUS,
        TRIP_PENALTY,
        QualificationDesign,
        _exact_load,
        _physical_steps_from_info,
        _sensor_from_observation,
        _trip_from_info,
        make_default_environment_factory,
    )
    from qualify_physics_belief_v12 import (  # type: ignore[no-redef]
        PhysicsBeliefPolicy,
        TransitionModel,
    )
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        bootstrap_ci,
        sha256,
        summarize_values,
    )
    from run_physics_residual_v12_pilot import (  # type: ignore[no-redef]
        HybridBeliefPolicy,
        _designs,
        load_residual_checkpoint,
    )


TARGET_OOD = ("ood_sensor_noise", "ood_cooling", "ood_combined")
ALL_CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_combined",
    "ood_shocks",
)
COMPONENT_FIELDS = (
    "mean_base_task_return",
    "mean_throughput_bonus",
    "mean_trip_penalty",
)


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


def policy_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "physics_z0",
            "type": "physics_belief",
            "cutoff": 0.060,
            "uncertainty_multiplier": 0.0,
        },
        {
            "name": "physics_z1_5",
            "type": "physics_belief",
            "cutoff": 0.060,
            "uncertainty_multiplier": 1.5,
        },
        {
            "name": "hybrid_z0",
            "type": "hybrid_belief",
            "cutoff": 0.060,
            "residual_scale": 1.0,
            "uncertainty_multiplier": 0.0,
        },
        {
            "name": "hybrid_z1_5",
            "type": "hybrid_belief",
            "cutoff": 0.060,
            "residual_scale": 1.0,
            "uncertainty_multiplier": 1.5,
        },
    ]


def decompose_task_return(total_reward: float, *, action: int, tripped: bool) -> dict[str, float]:
    throughput_bonus = HIGH_POWER_BONUS if action == 1 and not tripped else 0.0
    trip_penalty = -TRIP_PENALTY if tripped else 0.0
    base_return = float(total_reward) - throughput_bonus - trip_penalty
    return {
        "base_task_return": float(base_return),
        "throughput_bonus": float(throughput_bonus),
        "trip_penalty": float(trip_penalty),
    }


_WORKER_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_WORKER_MODEL: Any | None = None
_WORKER_CHECKPOINT: dict[str, Any] | None = None


def _initialize_worker(low_level_model_path: str, checkpoint_path: str) -> None:
    global _WORKER_FACTORY, _WORKER_MODEL, _WORKER_CHECKPOINT
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _WORKER_FACTORY = make_default_environment_factory(Path(low_level_model_path))
    _WORKER_MODEL, _WORKER_CHECKPOINT = load_residual_checkpoint(
        Path(checkpoint_path), torch.device("cpu")
    )


def _policy_factory(spec: dict[str, Any]) -> Callable[[], Any]:
    if _WORKER_MODEL is None or _WORKER_CHECKPOINT is None:
        raise RuntimeError("factorial worker was not initialized")
    if spec["type"] == "physics_belief":
        transition = TransitionModel(**_WORKER_CHECKPOINT["transition_model"])
        return lambda: PhysicsBeliefPolicy(
            transition,
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    if spec["type"] == "hybrid_belief":
        return lambda: HybridBeliefPolicy(
            _WORKER_MODEL,
            _WORKER_CHECKPOINT,
            residual_scale=float(spec["residual_scale"]),
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    raise ValueError(f"unknown factorial policy type: {spec['type']}")


def evaluate_policy_decomposed(
    design: QualificationDesign,
    environment_factory: Callable[[QualificationDesign, int], Any],
    policy_factory: Callable[[], Any],
    *,
    seeds: tuple[int, ...],
    tasks_per_lifetime: int = 20,
) -> dict[str, Any]:
    lifetime_rows = []
    total_physical_audits = 0
    for seed in seeds:
        environment = environment_factory(design, seed)
        policy = policy_factory()
        rewards: list[float] = []
        base_returns: list[float] = []
        bonuses: list[float] = []
        penalties: list[float] = []
        actions: list[int] = []
        trips: list[bool] = []
        try:
            observation, reset_info = environment.reset(seed=seed)
            info = dict(reset_info)
            for task_index in range(tasks_per_lifetime):
                action = int(
                    policy.act(
                        task_index=task_index,
                        sensor=_sensor_from_observation(observation),
                        exact_load=_exact_load(environment, info),
                    )
                )
                observation, reward, terminated, truncated, step_info = environment.step(action)
                info = dict(step_info)
                tripped = _trip_from_info(info)
                components = decompose_task_return(reward, action=action, tripped=tripped)
                physical_steps = _physical_steps_from_info(info)
                if physical_steps is not None and physical_steps > 1:
                    total_physical_audits += 1
                rewards.append(float(reward))
                base_returns.append(components["base_task_return"])
                bonuses.append(components["throughput_bonus"])
                penalties.append(components["trip_penalty"])
                actions.append(action)
                trips.append(tripped)
                boundary = bool(terminated or truncated)
                if boundary != (task_index == tasks_per_lifetime - 1):
                    raise RuntimeError("factorial lifetime boundary mismatch")
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        mean_reward = float(np.mean(rewards))
        mean_base = float(np.mean(base_returns))
        mean_bonus = float(np.mean(bonuses))
        mean_penalty = float(np.mean(penalties))
        if not np.isclose(mean_reward, mean_base + mean_bonus + mean_penalty, atol=1e-10):
            raise RuntimeError("reward decomposition failed to reconstruct total return")
        lifetime_rows.append(
            {
                "seed": int(seed),
                "mean_reward_per_task": mean_reward,
                "mean_base_task_return": mean_base,
                "mean_throughput_bonus": mean_bonus,
                "mean_trip_penalty": mean_penalty,
                "trip_rate": float(np.mean(trips)),
                "high_rate": float(np.mean(actions)),
            }
        )
    if total_physical_audits == 0:
        raise RuntimeError("factorial evaluation observed no physical rollout")
    return {
        "lifetimes": len(lifetime_rows),
        "tasks": len(lifetime_rows) * tasks_per_lifetime,
        "mean_reward_per_task": float(np.mean([row["mean_reward_per_task"] for row in lifetime_rows])),
        "mean_base_task_return": float(np.mean([row["mean_base_task_return"] for row in lifetime_rows])),
        "mean_throughput_bonus": float(np.mean([row["mean_throughput_bonus"] for row in lifetime_rows])),
        "mean_trip_penalty": float(np.mean([row["mean_trip_penalty"] for row in lifetime_rows])),
        "trip_rate": float(np.mean([row["trip_rate"] for row in lifetime_rows])),
        "high_rate": float(np.mean([row["high_rate"] for row in lifetime_rows])),
        "low_level_rollout_audit_observations": total_physical_audits,
        "lifetime_rows": lifetime_rows,
    }


def _evaluate_job(
    job: tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]
) -> tuple[str, str, dict[str, Any]]:
    if _WORKER_FACTORY is None:
        raise RuntimeError("factorial worker has no environment factory")
    condition, spec, design_document, seeds = job
    summary = evaluate_policy_decomposed(
        QualificationDesign(**design_document),
        _WORKER_FACTORY,
        _policy_factory(spec),
        seeds=seeds,
    )
    return condition, str(spec["name"]), {"spec": spec, "summary": summary}


def evaluate_jobs(
    jobs: list[tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]],
    *,
    workers: int,
    low_level_model: Path,
    checkpoint: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=(str(low_level_model.resolve()), str(checkpoint.resolve())),
    ) as pool:
        rows = pool.map(_evaluate_job, jobs)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for condition, policy, row in rows:
        result.setdefault(condition, {})[policy] = row
    return result


def paired_metric_values(
    cells: dict[str, dict[str, dict[str, Any]]],
    condition: str,
    treatment: str,
    control: str,
    field: str,
) -> tuple[list[int], np.ndarray]:
    def values(policy: str) -> dict[int, float]:
        return {
            int(row["seed"]): float(row[field])
            for row in cells[condition][policy]["summary"]["lifetime_rows"]
        }

    treated = values(treatment)
    controlled = values(control)
    if set(treated) != set(controlled):
        raise ValueError(f"paired seed mismatch: {condition} {treatment} {control}")
    seeds = sorted(treated)
    return seeds, np.asarray(
        [treated[seed] - controlled[seed] for seed in seeds], dtype=np.float64
    )


CONTRASTS = {
    "residual_at_z0": ("hybrid_z0", "physics_z0"),
    "residual_at_z1_5": ("hybrid_z1_5", "physics_z1_5"),
    "uncertainty_without_residual": ("physics_z1_5", "physics_z0"),
    "uncertainty_with_residual": ("hybrid_z1_5", "hybrid_z0"),
}


def _fast_summary(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "bootstrap_95_ci": bootstrap_ci(values, seed=seed),
        "positive": int(np.count_nonzero(values > 0.0)),
        "negative": int(np.count_nonzero(values < 0.0)),
    }


def analyze(cells: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    condition_contrasts: dict[str, Any] = {}
    values_by_contrast_condition: dict[str, dict[str, tuple[list[int], np.ndarray]]] = {
        name: {} for name in (*CONTRASTS, "interaction")
    }
    for condition_ordinal, condition in enumerate(ALL_CONDITIONS):
        rows: dict[str, Any] = {}
        for contrast_ordinal, (name, (treatment, control)) in enumerate(CONTRASTS.items()):
            seeds, values = paired_metric_values(
                cells, condition, treatment, control, "mean_reward_per_task"
            )
            values_by_contrast_condition[name][condition] = (seeds, values)
            rows[name] = _fast_summary(
                values, seed=71000 + 100 * condition_ordinal + contrast_ordinal
            )
        uncertainty_hybrid = values_by_contrast_condition["uncertainty_with_residual"][condition][1]
        uncertainty_physics = values_by_contrast_condition["uncertainty_without_residual"][condition][1]
        interaction_values = uncertainty_hybrid - uncertainty_physics
        values_by_contrast_condition["interaction"][condition] = (seeds, interaction_values)
        rows["interaction"] = _fast_summary(
            interaction_values, seed=71500 + condition_ordinal
        )
        condition_contrasts[condition] = rows

    aggregate_contrasts: dict[str, Any] = {}
    for ordinal, name in enumerate(values_by_contrast_condition):
        seed_sets = [
            values_by_contrast_condition[name][condition][0]
            for condition in TARGET_OOD
        ]
        if not all(seeds == seed_sets[0] for seeds in seed_sets[1:]):
            raise ValueError(f"target OOD seed mismatch for {name}")
        aggregate_values = np.mean(
            np.stack(
                [
                    values_by_contrast_condition[name][condition][1]
                    for condition in TARGET_OOD
                ]
            ),
            axis=0,
        )
        summary = summarize_values(aggregate_values, seed=72000 + ordinal)
        summary["seeds"] = seed_sets[0]
        aggregate_contrasts[name] = summary

    decomposition: dict[str, Any] = {}
    component_arrays = []
    for ordinal, field in enumerate(COMPONENT_FIELDS):
        condition_arrays = []
        seeds_reference: list[int] | None = None
        for condition in TARGET_OOD:
            seeds, values = paired_metric_values(
                cells, condition, "hybrid_z1_5", "physics_z0", field
            )
            if seeds_reference is None:
                seeds_reference = seeds
            elif seeds != seeds_reference:
                raise ValueError("decomposition target OOD seed mismatch")
            condition_arrays.append(values)
        aggregate = np.mean(np.stack(condition_arrays), axis=0)
        component_arrays.append(aggregate)
        decomposition[field] = summarize_values(aggregate, seed=73000 + ordinal)
    total_values = aggregate_contrasts["residual_at_z0"]["values"]
    uncertainty_values = aggregate_contrasts["uncertainty_with_residual"]["values"]
    original_composite_values = np.asarray(total_values) + np.asarray(uncertainty_values)
    reconstructed = np.sum(np.stack(component_arrays), axis=0)
    decomposition_identity_error = float(
        np.max(np.abs(original_composite_values - reconstructed))
    )
    decomposition["total_hybrid_z1_5_minus_physics_z0"] = summarize_values(
        original_composite_values, seed=73010
    )

    primary = aggregate_contrasts["residual_at_z1_5"]
    max_hybrid_trip = max(
        float(cells[condition]["hybrid_z1_5"]["summary"]["trip_rate"])
        for condition in ALL_CONDITIONS
    )
    criteria = {
        "matched_uncertainty_residual_mean_at_least_0_10": primary["mean"] >= 0.10,
        "matched_uncertainty_residual_bootstrap_lower_above_zero": (
            primary["bootstrap_95_ci"][0] > 0.0
        ),
        "matched_uncertainty_residual_sign_flip_p_below_0_05": (
            primary["sign_flip_two_sided_p"] < 0.05
        ),
        "hybrid_z1_5_all_condition_trip_rate_at_most_0_02": max_hybrid_trip <= 0.02,
        "reward_decomposition_identity_error_at_most_1e_10": (
            decomposition_identity_error <= 1e-10
        ),
    }
    return {
        "phase": "physics_residual_v12_3_factorial_mechanism_ablation",
        "status": "final_fresh_seed_ablation_result",
        "condition_contrasts": condition_contrasts,
        "target_ood_aggregate_contrasts": aggregate_contrasts,
        "target_ood_reward_decomposition": decomposition,
        "reward_decomposition_identity_max_abs_error": decomposition_identity_error,
        "maximum_hybrid_z1_5_trip_rate": max_hybrid_trip,
        "criteria": criteria,
        "residual_attribution_supported": all(criteria.values()),
    }


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
        default=Path("outputs/physics_residual_v12_3_factorial_ablation"),
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
    protocol_path = root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = root / "FROZEN_PROTOCOL.sha256"
    cells_path = root / "FACTORIAL_CELLS.json"
    final_path = root / "FACTORIAL_RESULTS.json"
    checkpoint = args.refinement_root / "residual_model.pt"
    refinement_result = args.refinement_root / "PILOT_RESULTS.json"
    project_root = Path(__file__).resolve().parent.parent
    sources = [
        Path(__file__).resolve(),
        project_root / "scripts/run_physics_residual_v12_pilot.py",
        project_root / "scripts/qualify_physics_belief_v12.py",
        project_root / "scripts/qualify_hierarchical_v11.py",
        project_root / "src/lifephybench/envs/hierarchical_thermal_v11.py",
        project_root / "src/lifephybench/envs/thermal_commitment.py",
        project_root / "src/lifephybench/envs/mujoco_pusher.py",
    ]
    if protocol_path.exists() and args.resume:
        protocol = read_json(protocol_path)
        protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
        if sha256(protocol_path) != protocol_hash:
            raise SystemExit("v12.3 frozen protocol hash mismatch")
    else:
        protocol = {
            "phase": "physics_residual_v12_3_frozen_factorial_protocol",
            "status": "frozen_before_any_v12_3_factorial_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "fresh-seed post-confirmatory mechanism ablation",
            "seeds": list(range(23000, 23100)),
            "seeds_untouched_before_freeze": True,
            "designs": {name: asdict(value) for name, value in _designs().items()},
            "policy_specs": policy_specs(),
            "factorial_design": {
                "residual": ["off", "on"],
                "uncertainty_multiplier": [0.0, 1.5],
                "cutoff": 0.060,
                "residual_scale_when_on": 1.0,
            },
            "primary_estimand": (
                "target-OOD aggregate hybrid_z1_5 minus physics_z1_5 reward/task"
            ),
            "primary_criteria": {
                "mean_at_least": 0.10,
                "bootstrap_95_ci_lower_above": 0.0,
                "paired_sign_flip_p_below": 0.05,
                "maximum_hybrid_z1_5_trip_rate_each_condition": 0.02,
            },
            "reward_decomposition": (
                "base task return excluding discrete mode bonus and trip penalty, "
                "throughput bonus, and trip penalty"
            ),
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

    for relative, digest in protocol["source_hashes"].items():
        if sha256(project_root / relative) != digest:
            raise SystemExit(f"source drift after v12.3 freeze: {relative}")
    if sha256(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise SystemExit("checkpoint drift after v12.3 freeze")
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return

    if cells_path.exists() and args.resume:
        cells = read_json(cells_path)
    else:
        atomic_json(
            status_path,
            {
                "phase": "v12_3_factorial_ablation",
                "status": "running_fresh_seed_factorial_evaluation",
                "protocol_sha256": protocol_hash,
            },
            overwrite=True,
        )
        jobs = [
            (condition, spec, asdict(design), tuple(protocol["seeds"]))
            for condition, design in _designs().items()
            for spec in protocol["policy_specs"]
        ]
        cells = evaluate_jobs(
            jobs,
            workers=args.workers,
            low_level_model=args.low_level_model,
            checkpoint=checkpoint,
        )
        atomic_json(cells_path, cells, overwrite=False)
    report = analyze(cells)
    report.update(
        {
            "protocol_sha256": protocol_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "wiring_passed": True,
        }
    )
    atomic_json(final_path, report, overwrite=False)
    atomic_json(
        status_path,
        {
            "phase": "v12_3_factorial_ablation",
            "status": "complete",
            "residual_attribution_supported": report["residual_attribution_supported"],
            "protocol_sha256": protocol_hash,
        },
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "residual_attribution_supported": report["residual_attribution_supported"],
                "criteria": report["criteria"],
                "matched_uncertainty_residual": report[
                    "target_ood_aggregate_contrasts"
                ]["residual_at_z1_5"],
                "reward_decomposition": report["target_ood_reward_decomposition"],
                "output": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
