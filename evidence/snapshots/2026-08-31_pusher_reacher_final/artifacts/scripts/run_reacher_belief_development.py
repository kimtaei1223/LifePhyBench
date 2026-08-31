#!/usr/bin/env python3
"""Build the Reacher thermal belief and calibrate mandatory analytic baselines."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
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
        _exact_load,
        _physical_steps_from_info,
        _sensor_from_observation,
        _trip_from_info,
        make_default_environment_factory,
    )
    from scripts.qualify_physics_belief_v12 import (
        CurrentSensorPolicy,
        EmaPolicy,
        PhysicsBeliefPolicy,
        PrivilegedOraclePolicy,
        TransitionModel,
        fit_transition_model,
    )
    from scripts.run_physics_residual_v12_pilot import (
        HybridBeliefPolicy,
        _designs,
        atomic_torch_save,
        load_residual_checkpoint,
        train_residual,
    )
except ModuleNotFoundError:
    from qualify_hierarchical_v11 import (  # type: ignore[no-redef]
        QualificationDesign,
        _exact_load,
        _physical_steps_from_info,
        _sensor_from_observation,
        _trip_from_info,
        make_default_environment_factory,
    )
    from qualify_physics_belief_v12 import (  # type: ignore[no-redef]
        CurrentSensorPolicy,
        EmaPolicy,
        PhysicsBeliefPolicy,
        PrivilegedOraclePolicy,
        TransitionModel,
        fit_transition_model,
    )
    from run_physics_residual_v12_pilot import (  # type: ignore[no-redef]
        HybridBeliefPolicy,
        _designs,
        atomic_torch_save,
        load_residual_checkpoint,
        train_residual,
    )


_DATA_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_DATA_LIFETIMES = 0
_CAL_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_CAL_MODEL: Any | None = None
_CAL_CHECKPOINT: dict[str, Any] | None = None


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def configure_worker(low_level_model: str) -> None:
    global _DATA_FACTORY
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _DATA_FACTORY = make_default_environment_factory(
        Path(low_level_model), environment_id="Reacher-v5"
    )


def behavior_action(
    *,
    rng: np.random.Generator,
    lifetime: int,
    task_index: int,
    sensor: float,
    exact_load: float,
    memory: dict[str, Any],
) -> int:
    """Indexed diverse development behavior; never used as a reported policy."""

    family = lifetime % 4
    if family == 0:
        cutoff = float(memory.setdefault("cutoff", rng.uniform(0.035, 0.085)))
        return int(exact_load < cutoff)
    if family == 1:
        prefix = int(memory.setdefault("prefix", rng.integers(0, 8)))
        return int(task_index < prefix and exact_load < 0.095)
    if family == 2:
        phase = int(memory.setdefault("phase", rng.integers(0, 2)))
        return int((task_index + phase) % 2 == 0 and exact_load < 0.095)
    previous = int(memory.setdefault("state", rng.integers(0, 2)))
    if rng.random() < 0.25:
        previous = 1 - previous
    if exact_load >= 0.095 or sensor >= 0.10:
        previous = 0
    memory["state"] = previous
    return previous


def generate_seed_rows(job: tuple[int, int]) -> tuple[int, list[dict[str, Any]]]:
    if _DATA_FACTORY is None:
        raise RuntimeError("development-data worker is not initialized")
    seed, lifetimes = job
    design = _designs()["in_domain"]
    environment = _DATA_FACTORY(design, seed)
    rng = np.random.default_rng(seed + 92_003)
    rows: list[dict[str, Any]] = []
    try:
        for lifetime in range(lifetimes):
            observation, reset_info = environment.reset(seed=seed)
            info = dict(reset_info)
            memory: dict[str, Any] = {}
            for task_index in range(20):
                sensor = _sensor_from_observation(observation)
                exact = _exact_load(environment, info)
                action = behavior_action(
                    rng=rng,
                    lifetime=lifetime,
                    task_index=task_index,
                    sensor=sensor,
                    exact_load=exact,
                    memory=memory,
                )
                observation, reward, terminated, truncated, step_info = environment.step(
                    action
                )
                info = dict(step_info)
                rows.append(
                    {
                        "seed": seed,
                        "lifetime_ordinal": lifetime,
                        "task_index": task_index,
                        "action": action,
                        "reward": float(reward),
                        "tripped": _trip_from_info(info),
                        "sensor_load": sensor,
                        "true_load_at_selection": exact,
                        "true_load_after_task": float(info["lifephy/thermal_load"]),
                        "initial_load": float(
                            info["lifephy/v11_lifetime_initial_thermal_load"]
                        ),
                    }
                )
                if bool(terminated or truncated) != (task_index == 19):
                    raise RuntimeError("development lifetime boundary mismatch")
    finally:
        environment.close()
    return seed, rows


def atomic_jsonl(path: Path, runs: list[tuple[int, list[dict[str, Any]]]]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for seed, rows in runs:
                for row in rows:
                    handle.write(json.dumps({"seed": seed, **row}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_runs(path: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped.setdefault(int(row["seed"]), []).append(row)
    return sorted(grouped.items())


def initialize_calibration(
    low_level_model: str, residual_checkpoint: str
) -> None:
    global _CAL_FACTORY, _CAL_MODEL, _CAL_CHECKPOINT
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _CAL_FACTORY = make_default_environment_factory(
        Path(low_level_model), environment_id="Reacher-v5"
    )
    _CAL_MODEL, _CAL_CHECKPOINT = load_residual_checkpoint(
        Path(residual_checkpoint), torch.device("cpu")
    )


def calibration_policy(spec: dict[str, Any]) -> Any:
    if _CAL_CHECKPOINT is None or _CAL_MODEL is None:
        raise RuntimeError("calibration worker is not initialized")
    policy_type = spec["type"]
    if policy_type == "current_sensor":
        return CurrentSensorPolicy(float(spec["cutoff"]))
    if policy_type == "ema":
        return EmaPolicy(float(spec["alpha"]), float(spec["cutoff"]))
    if policy_type == "physics_belief":
        return PhysicsBeliefPolicy(
            TransitionModel(**_CAL_CHECKPOINT["transition_model"]),
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    if policy_type == "hybrid_belief":
        return HybridBeliefPolicy(
            _CAL_MODEL,
            _CAL_CHECKPOINT,
            residual_scale=float(spec["residual_scale"]),
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    if policy_type == "privileged_oracle":
        return PrivilegedOraclePolicy(float(spec["cutoff"]))
    raise ValueError(f"unknown policy type: {policy_type}")


def predictive_distribution(policy: Any) -> tuple[float, float] | None:
    if isinstance(policy, PhysicsBeliefPolicy):
        if policy.mean is None or policy.variance is None:
            return None
        return float(policy.mean), math.sqrt(max(0.0, float(policy.variance)))
    if isinstance(policy, HybridBeliefPolicy):
        if policy.last_corrected_mean is None or policy.last_total_std is None:
            return None
        return float(policy.last_corrected_mean), float(policy.last_total_std)
    return None


def evaluate_calibration_job(
    job: tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]
) -> tuple[str, str, dict[str, Any]]:
    if _CAL_FACTORY is None:
        raise RuntimeError("calibration environment factory is not initialized")
    condition, spec, design_document, seeds = job
    design = QualificationDesign(**design_document)
    lifetime_rows = []
    prediction_errors: list[float] = []
    prediction_stds: list[float] = []
    physical_audits = 0
    for seed in seeds:
        environment = _CAL_FACTORY(design, seed)
        policy = calibration_policy(spec)
        rewards: list[float] = []
        actions: list[int] = []
        trips: list[bool] = []
        try:
            observation, reset_info = environment.reset(seed=seed)
            info = dict(reset_info)
            for task_index in range(20):
                sensor = _sensor_from_observation(observation)
                exact = _exact_load(environment, info)
                action = int(
                    policy.act(
                        task_index=task_index, sensor=sensor, exact_load=exact
                    )
                )
                distribution = predictive_distribution(policy)
                if distribution is not None:
                    mean, std = distribution
                    prediction_errors.append(exact - mean)
                    prediction_stds.append(std)
                observation, reward, terminated, truncated, step_info = environment.step(
                    action
                )
                info = dict(step_info)
                if (_physical_steps_from_info(info) or 0) > 1:
                    physical_audits += 1
                rewards.append(float(reward))
                actions.append(action)
                trips.append(_trip_from_info(info))
                if bool(terminated or truncated) != (task_index == 19):
                    raise RuntimeError("calibration lifetime boundary mismatch")
        finally:
            environment.close()
        lifetime_rows.append(
            {
                "seed": seed,
                "mean_reward_per_task": float(np.mean(rewards)),
                "trip_rate": float(np.mean(trips)),
                "high_rate": float(np.mean(actions)),
            }
        )
    coverage = None
    if prediction_errors:
        errors = np.asarray(prediction_errors)
        stds = np.asarray(prediction_stds)
        coverage = {
            f"z{z:g}": float(np.mean(np.abs(errors) <= z * stds))
            for z in (0.5, 1.0, 1.5, 1.96)
        }
        coverage.update(
            {
                "observations": len(errors),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_predicted_std": float(np.mean(stds)),
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
        "uncertainty_calibration": coverage,
        "lifetime_rows": lifetime_rows,
    }
    return condition, str(spec["name"]), {"spec": spec, "summary": summary}


def policy_specs() -> list[dict[str, Any]]:
    return [
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.060},
        {"name": "ema_history", "type": "ema", "alpha": 0.60, "cutoff": 0.060},
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
            "name": "hybrid_z1_5",
            "type": "hybrid_belief",
            "cutoff": 0.060,
            "residual_scale": 1.0,
            "uncertainty_multiplier": 1.5,
        },
        {"name": "privileged_oracle", "type": "privileged_oracle", "cutoff": 0.060},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/reacher_cross_task_stage2_v1.json")
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("outputs/reacher_replication/low_level/SELECTION.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication/belief_development")
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    project_root = Path(__file__).resolve().parent.parent
    protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    selection_path = args.selection if args.selection.is_absolute() else project_root / args.selection
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    data_path = output_root / "DEVELOPMENT_LIFETIMES.jsonl"
    data_report_path = output_root / "DEVELOPMENT_DATA_REPORT.json"
    checkpoint_path = output_root / "residual_model.pt"
    training_path = output_root / "TRAINING_RESULTS.json"
    calibration_path = output_root / "CALIBRATION_RESULTS.json"
    final_path = output_root / "STAGE2_BELIEF_RESULTS.json"
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return
    protocol = read_json(protocol_path)
    selection = read_json(selection_path)
    low_level_model = Path(selection["selected"]["model"])
    if int(selection["selected"]["seed"]) != int(protocol["selected_low_level_seed"]):
        raise SystemExit("selected low-level seed does not match frozen stage2 protocol")
    if not low_level_model.is_file():
        raise SystemExit(f"selected low-level model is missing: {low_level_model}")

    development = protocol["development_data"]
    seed_first, seed_last = map(int, development["seeds"])
    development_seeds = tuple(range(seed_first, seed_last + 1))
    lifetimes = int(development["lifetimes_per_seed"])
    if data_path.exists() and args.resume:
        runs = load_runs(data_path)
    else:
        if data_path.exists():
            raise SystemExit("development data exists; use --resume")
        atomic_json(
            status_path,
            {"status": "generating_development_lifetimes", "updated_utc": datetime.now(timezone.utc).isoformat()},
        )
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=min(args.workers, len(development_seeds)),
            initializer=configure_worker,
            initargs=(str(low_level_model),),
        ) as pool:
            runs = pool.map(
                generate_seed_rows,
                [(seed, lifetimes) for seed in development_seeds],
            )
        atomic_jsonl(data_path, runs)
        all_rows = [row for _seed, rows in runs for row in rows]
        atomic_json(
            data_report_path,
            {
                "phase": "reacher_belief_development_data",
                "seeds": list(development_seeds),
                "lifetimes": len(development_seeds) * lifetimes,
                "tasks": len(all_rows),
                "high_rate": float(np.mean([row["action"] for row in all_rows])),
                "trip_rate": float(np.mean([row["tripped"] for row in all_rows])),
            },
        )

    residual = protocol["residual_model"]
    transition_seeds = set(range(*[int(value) for value in residual["transition_fit_seeds"]]))
    transition_seeds.add(int(residual["transition_fit_seeds"][1]))
    train_seeds = set(range(int(residual["residual_train_seeds"][0]), int(residual["residual_train_seeds"][1]) + 1))
    validation_seeds = set(range(int(residual["residual_validation_seeds"][0]), int(residual["residual_validation_seeds"][1]) + 1))
    test_seeds = set(range(int(residual["residual_test_seeds"][0]), int(residual["residual_test_seeds"][1]) + 1))
    transition = fit_transition_model(runs, transition_seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if checkpoint_path.exists() and training_path.exists() and args.resume:
        training_report = read_json(training_path)
    else:
        atomic_json(
            status_path,
            {"status": "training_residual", "updated_utc": datetime.now(timezone.utc).isoformat()},
        )
        checkpoint, training_report = train_residual(
            runs=runs,
            transition=transition,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            device=device,
            hidden_dim=int(residual["hidden_dim"]),
            epochs=int(residual["epochs"]),
            learning_rate=float(residual["learning_rate"]),
            rng_seed=int(residual["rng_seed"]),
            mean_loss_weight=float(residual["mean_loss_weight"]),
            nll_loss_weight=float(residual["nll_loss_weight"]),
        )
        atomic_torch_save(checkpoint_path, checkpoint)
        atomic_json(training_path, training_report)

    if calibration_path.exists() and args.resume:
        calibration = read_json(calibration_path)
    else:
        atomic_json(
            status_path,
            {"status": "calibrating_analytic_baselines", "updated_utc": datetime.now(timezone.utc).isoformat()},
        )
        calibration_config = protocol["controller_calibration"]
        cal_first, cal_last = map(int, calibration_config["seeds"])
        calibration_seeds = tuple(range(cal_first, cal_last + 1))
        designs = _designs()
        jobs = [
            (condition, spec, asdict(design), calibration_seeds)
            for condition, design in designs.items()
            for spec in policy_specs()
        ]
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=min(args.workers, len(jobs)),
            initializer=initialize_calibration,
            initargs=(str(low_level_model), str(checkpoint_path)),
        ) as pool:
            rows = pool.map(evaluate_calibration_job, jobs)
        cells: dict[str, dict[str, Any]] = {}
        for condition, policy, row in rows:
            cells.setdefault(condition, {})[policy] = row
        calibration = {
            "phase": "reacher_stage2_controller_calibration",
            "confirmatory_evidence": False,
            "seeds": list(calibration_seeds),
            "cells": cells,
        }
        atomic_json(calibration_path, calibration)

    report = {
        "phase": "reacher_cross_task_stage2_belief_development",
        "status": "complete",
        "confirmatory_evidence": False,
        "confirmatory_seeds_consumed": False,
        "selected_low_level": selection["selected"],
        "transition_model": asdict(transition),
        "residual_training": training_report,
        "calibration_result": str(calibration_path),
    }
    atomic_json(final_path, report)
    atomic_json(
        status_path,
        {"status": "complete", "updated_utc": datetime.now(timezone.utc).isoformat()},
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
