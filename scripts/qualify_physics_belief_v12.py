#!/usr/bin/env python3
"""CPU feasibility gate for a physics-informed thermal belief supervisor.

V11 held-out results are treated as development data after the frozen v11
analysis has completed.  Half of the old seeds identify a transparent linear
thermal transition, while the other half audit its state-estimation error.
Fresh v12 development seeds then compare a current-sensor rule, an EMA rule,
the fitted belief supervisor, and a privileged thermal-state oracle.

This script is a development gate, not confirmatory evidence.  It never uses
the privileged thermal state as an input to the deployable belief supervisor.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from scripts.qualify_hierarchical_v11 import (
        QualificationDesign,
        evaluate_policy,
        make_default_environment_factory,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qualify_hierarchical_v11 import (  # type: ignore[no-redef]
        QualificationDesign,
        evaluate_policy,
        make_default_environment_factory,
    )


@dataclass(frozen=True)
class TransitionModel:
    intercept: float
    load_coefficient: float
    high_power_increment: float
    process_variance: float
    measurement_variance: float
    observations: int
    r_squared: float


@dataclass
class CurrentSensorPolicy:
    cutoff: float

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del task_index, exact_load
        return int(sensor < self.cutoff)


@dataclass
class EmaPolicy:
    alpha: float
    cutoff: float
    estimate: float | None = None

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del task_index, exact_load
        self.estimate = (
            sensor
            if self.estimate is None
            else self.alpha * sensor + (1.0 - self.alpha) * self.estimate
        )
        return int(self.estimate < self.cutoff)


@dataclass
class PrivilegedOraclePolicy:
    cutoff: float

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del task_index, sensor
        return int(exact_load < self.cutoff)


@dataclass
class PhysicsBeliefPolicy:
    model: TransitionModel
    cutoff: float
    uncertainty_multiplier: float
    mean: float | None = None
    variance: float | None = None
    previous_action: int | None = None

    def update(self, *, task_index: int, sensor: float) -> tuple[float, float]:
        if task_index == 0 or self.mean is None or self.previous_action is None:
            self.mean = float(sensor)
            self.variance = self.model.measurement_variance
            return self.mean, self.variance

        variance = float(self.variance or 0.0)
        prior_mean = (
            self.model.intercept
            + self.model.load_coefficient * self.mean
            + self.model.high_power_increment * self.previous_action
        )
        prior_variance = (
            self.model.load_coefficient**2 * variance
            + self.model.process_variance
        )
        denominator = prior_variance + self.model.measurement_variance
        gain = 0.0 if denominator <= 0.0 else prior_variance / denominator
        self.mean = float(prior_mean + gain * (sensor - prior_mean))
        self.variance = float(max(0.0, (1.0 - gain) * prior_variance))
        return self.mean, self.variance

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del exact_load
        mean, variance = self.update(task_index=task_index, sensor=sensor)
        upper_bound = mean + self.uncertainty_multiplier * math.sqrt(variance)
        action = int(upper_bound < self.cutoff)
        self.previous_action = action
        return action


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def _iter_stochastic_runs(input_root: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    runs: list[tuple[int, list[dict[str, Any]]]] = []
    for directory in sorted(input_root.glob("v11-heldout-stochastic-*-seed*-decisions100k")):
        metadata = _read_json(directory / "metadata.json")
        arguments = metadata.get("arguments", {})
        seed = int(arguments.get("seed"))
        rows = [
            json.loads(line)
            for line in (directory / "evaluation_tasks.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if not rows:
            raise ValueError(f"empty evaluation rows: {directory}")
        runs.append((seed, rows))
    if not runs:
        raise ValueError(f"no stochastic v11 development runs found: {input_root}")
    return runs


def fit_transition_model(
    runs: list[tuple[int, list[dict[str, Any]]]], training_seeds: set[int]
) -> TransitionModel:
    predictors: list[list[float]] = []
    targets: list[float] = []
    measurement_errors: list[float] = []
    for seed, rows in runs:
        if seed not in training_seeds:
            continue
        for row in rows:
            measurement_errors.append(
                float(row["sensor_load"]) - float(row["true_load_at_selection"])
            )
        for current, following in zip(rows, rows[1:]):
            same_lifetime = (
                int(current["lifetime_ordinal"])
                == int(following["lifetime_ordinal"])
            )
            consecutive = int(following["task_index"]) == int(current["task_index"]) + 1
            if same_lifetime and consecutive:
                predictors.append(
                    [
                        1.0,
                        float(current["true_load_at_selection"]),
                        float(current["action"]),
                    ]
                )
                targets.append(float(following["true_load_at_selection"]))
    if len(targets) < 4 or len(measurement_errors) < 2:
        raise ValueError("insufficient v11 rows for transition identification")
    design = np.asarray(predictors, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    target_ss = float(np.sum((target - target.mean()) ** 2))
    residual_ss = float(np.sum(residual**2))
    return TransitionModel(
        intercept=float(coefficients[0]),
        load_coefficient=float(coefficients[1]),
        high_power_increment=float(coefficients[2]),
        process_variance=float(np.var(residual, ddof=3)),
        measurement_variance=float(np.var(measurement_errors, ddof=1)),
        observations=len(targets),
        r_squared=float(1.0 - residual_ss / target_ss),
    )


def audit_estimator(
    runs: list[tuple[int, list[dict[str, Any]]]],
    validation_seeds: set[int],
    model: TransitionModel,
    *,
    ema_alpha: float,
) -> dict[str, float | int]:
    errors: dict[str, list[float]] = {
        "raw_sensor": [],
        "ema": [],
        "physics_belief": [],
    }
    rows_used = 0
    for seed, rows in runs:
        if seed not in validation_seeds:
            continue
        lifetimes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            lifetimes[int(row["lifetime_ordinal"])].append(row)
        for lifetime_rows in lifetimes.values():
            policy = PhysicsBeliefPolicy(model, cutoff=1.0, uncertainty_multiplier=0.0)
            ema: float | None = None
            for row in sorted(lifetime_rows, key=lambda value: int(value["task_index"])):
                sensor = float(row["sensor_load"])
                exact = float(row["true_load_at_selection"])
                ema = sensor if ema is None else ema_alpha * sensor + (1.0 - ema_alpha) * ema
                mean, _variance = policy.update(
                    task_index=int(row["task_index"]), sensor=sensor
                )
                errors["raw_sensor"].append(sensor - exact)
                errors["ema"].append(ema - exact)
                errors["physics_belief"].append(mean - exact)
                policy.previous_action = int(row["action"])
                rows_used += 1
    if rows_used == 0:
        raise ValueError("no estimator-validation rows matched the requested seeds")

    def rmse(values: list[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        return float(np.sqrt(np.mean(array**2)))

    raw_rmse = rmse(errors["raw_sensor"])
    ema_rmse = rmse(errors["ema"])
    belief_rmse = rmse(errors["physics_belief"])
    return {
        "rows": rows_used,
        "raw_sensor_rmse": raw_rmse,
        "ema_rmse": ema_rmse,
        "physics_belief_rmse": belief_rmse,
        "belief_improvement_over_raw_fraction": 1.0 - belief_rmse / raw_rmse,
        "belief_improvement_over_ema_fraction": 1.0 - belief_rmse / ema_rmse,
    }


def _policy_factory(spec: dict[str, Any], model: TransitionModel) -> Callable[[], Any]:
    policy_type = spec["type"]
    if policy_type == "current_sensor":
        return lambda: CurrentSensorPolicy(float(spec["cutoff"]))
    if policy_type == "ema":
        return lambda: EmaPolicy(float(spec["alpha"]), float(spec["cutoff"]))
    if policy_type == "privileged_oracle":
        return lambda: PrivilegedOraclePolicy(float(spec["cutoff"]))
    if policy_type == "physics_belief":
        return lambda: PhysicsBeliefPolicy(
            model,
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    raise ValueError(f"unknown policy type: {policy_type}")


_WORKER_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_WORKER_DESIGN: QualificationDesign | None = None
_WORKER_MODEL: TransitionModel | None = None


def _initialize_worker(
    low_level_model_path: str,
    design_document: dict[str, float],
    model_document: dict[str, Any],
) -> None:
    global _WORKER_DESIGN, _WORKER_FACTORY, _WORKER_MODEL
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    _WORKER_FACTORY = make_default_environment_factory(Path(low_level_model_path))
    _WORKER_DESIGN = QualificationDesign(**design_document)
    _WORKER_MODEL = TransitionModel(**model_document)


def _evaluate_job(job: tuple[dict[str, Any], tuple[int, ...]]) -> tuple[str, dict[str, Any]]:
    if _WORKER_FACTORY is None or _WORKER_DESIGN is None or _WORKER_MODEL is None:
        raise RuntimeError("v12 feasibility worker was not initialized")
    spec, seeds = job
    summary = evaluate_policy(
        _WORKER_DESIGN,
        _WORKER_FACTORY,
        _policy_factory(spec, _WORKER_MODEL),
        seeds=seeds,
        tasks_per_lifetime=20,
        require_physical_rollouts=True,
    )
    return str(spec["name"]), {"spec": spec, "summary": summary}


def evaluate_jobs(
    jobs: list[tuple[dict[str, Any], tuple[int, ...]]],
    *,
    workers: int,
    low_level_model_path: Path,
    design: QualificationDesign,
    model: TransitionModel,
) -> dict[str, dict[str, Any]]:
    initializer_args = (
        str(low_level_model_path.resolve()),
        asdict(design),
        asdict(model),
    )
    if workers == 1:
        _initialize_worker(*initializer_args)
        results = [_evaluate_job(job) for job in jobs]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=initializer_args,
        ) as pool:
            results = pool.map(_evaluate_job, jobs)
    return dict(results)


def assess_gate(
    estimator: dict[str, float | int],
    validation: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current = validation["current_sensor"]["summary"]
    history = validation["ema_history"]["summary"]
    belief = validation["physics_belief"]["summary"]
    oracle = validation["privileged_oracle"]["summary"]
    contrasts = {
        "oracle_minus_current_reward": (
            oracle["mean_reward_per_task"] - current["mean_reward_per_task"]
        ),
        "history_minus_current_reward": (
            history["mean_reward_per_task"] - current["mean_reward_per_task"]
        ),
        "belief_minus_current_reward": (
            belief["mean_reward_per_task"] - current["mean_reward_per_task"]
        ),
        "belief_minus_ema_reward": (
            belief["mean_reward_per_task"] - history["mean_reward_per_task"]
        ),
        "belief_minus_best_nonprivileged_baseline_reward": (
            belief["mean_reward_per_task"]
            - max(
                current["mean_reward_per_task"],
                history["mean_reward_per_task"],
            )
        ),
        "belief_oracle_gap_closed_fraction": (
            (belief["mean_reward_per_task"] - current["mean_reward_per_task"])
            / max(
                1e-12,
                oracle["mean_reward_per_task"] - current["mean_reward_per_task"],
            )
        ),
        "belief_minus_current_trip_rate": (
            belief["trip_rate"] - current["trip_rate"]
        ),
    }
    criteria = {
        "transition_r_squared_at_least_0_90": True,
        "belief_rmse_25_percent_below_raw": (
            float(estimator["belief_improvement_over_raw_fraction"]) >= 0.25
        ),
        "belief_rmse_20_percent_below_ema": (
            float(estimator["belief_improvement_over_ema_fraction"]) >= 0.20
        ),
        "oracle_reward_advantage_at_least_0_50": (
            contrasts["oracle_minus_current_reward"] >= 0.50
        ),
        "history_reward_advantage_at_least_0_25": (
            contrasts["history_minus_current_reward"] >= 0.25
        ),
        "belief_reward_advantage_at_least_0_25": (
            contrasts["belief_minus_current_reward"] >= 0.25
        ),
        "belief_beats_best_nonprivileged_baseline_by_0_25": (
            contrasts["belief_minus_best_nonprivileged_baseline_reward"] >= 0.25
        ),
        "belief_closes_at_least_25_percent_oracle_gap": (
            contrasts["belief_oracle_gap_closed_fraction"] >= 0.25
        ),
        "belief_trip_rate_at_most_0_02": belief["trip_rate"] <= 0.02,
        "belief_trip_increase_at_most_0_005": (
            contrasts["belief_minus_current_trip_rate"] <= 0.005
        ),
    }
    return {
        "contrasts": contrasts,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def atomic_write_new(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v12 feasibility result: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v11-input-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/confirmatory"),
    )
    parser.add_argument(
        "--low-level-model",
        type=Path,
        default=Path(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=list(range(9300, 9305)))
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(9310, 9320)))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/physics_belief_v12/FEASIBILITY.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    selection_seeds = tuple(args.selection_seeds)
    validation_seeds = tuple(args.validation_seeds)
    if (
        not selection_seeds
        or not validation_seeds
        or set(selection_seeds).intersection(validation_seeds)
        or len(set(selection_seeds)) != len(selection_seeds)
        or len(set(validation_seeds)) != len(validation_seeds)
    ):
        raise SystemExit("selection and validation seeds must be nonempty, unique, and disjoint")

    runs = _iter_stochastic_runs(args.v11_input_root)
    available_seeds = sorted({seed for seed, _rows in runs})
    midpoint = len(available_seeds) // 2
    identification_seeds = set(available_seeds[:midpoint])
    estimator_audit_seeds = set(available_seeds[midpoint:])
    model = fit_transition_model(runs, identification_seeds)
    estimator = audit_estimator(
        runs, estimator_audit_seeds, model, ema_alpha=0.60
    )
    design = QualificationDesign(
        thermal_episode_cooling=0.15,
        sensor_noise_sd=0.01,
        shock_probability=5.0e-4,
        shock_size=0.01,
    )

    cutoff_grid = (0.045, 0.050, 0.055, 0.060, 0.065)
    current_specs = [
        {
            "name": f"current-cutoff{cutoff:.3f}",
            "type": "current_sensor",
            "cutoff": cutoff,
        }
        for cutoff in cutoff_grid
    ]
    history_specs = [
        {
            "name": f"ema-alpha{alpha:.1f}-cutoff{cutoff:.3f}",
            "type": "ema",
            "alpha": alpha,
            "cutoff": cutoff,
        }
        for alpha in (0.20, 0.40, 0.60, 0.80)
        for cutoff in cutoff_grid
    ]
    oracle_spec = {
        "name": "privileged_oracle",
        "type": "privileged_oracle",
        "cutoff": 0.06,
    }
    belief_specs = [
        {
            "name": f"belief-cutoff{cutoff:.3f}-z{multiplier:.1f}",
            "type": "physics_belief",
            "cutoff": cutoff,
            "uncertainty_multiplier": multiplier,
        }
        for cutoff in cutoff_grid
        for multiplier in (0.0, 0.5, 1.0, 1.5)
    ]
    selection_specs = current_specs + history_specs + [oracle_spec] + belief_specs
    selection = evaluate_jobs(
        [(spec, selection_seeds) for spec in selection_specs],
        workers=args.workers,
        low_level_model_path=args.low_level_model,
        design=design,
        model=model,
    )
    def select_safe(specs: list[dict[str, Any]]) -> dict[str, Any]:
        admissible = [
            spec
            for spec in specs
            if selection[spec["name"]]["summary"]["trip_rate"] <= 0.02
        ]
        candidate_pool = admissible or specs
        return max(
            candidate_pool,
            key=lambda spec: (
                selection[spec["name"]]["summary"]["mean_reward_per_task"],
                -selection[spec["name"]]["summary"]["trip_rate"],
                -float(spec.get("uncertainty_multiplier", 0.0)),
                -float(spec["cutoff"]),
            ),
        )

    selected_current = select_safe(current_specs)
    selected_history = select_safe(history_specs)
    selected_belief = select_safe(belief_specs)
    validation_specs = [
        {**selected_current, "name": "current_sensor"},
        {**selected_history, "name": "ema_history"},
        oracle_spec,
        {**selected_belief, "name": "physics_belief"},
    ]
    validation = evaluate_jobs(
        [(spec, validation_seeds) for spec in validation_specs],
        workers=min(args.workers, len(validation_specs)),
        low_level_model_path=args.low_level_model,
        design=design,
        model=model,
    )
    gate = assess_gate(estimator, validation)
    gate["criteria"]["transition_r_squared_at_least_0_90"] = model.r_squared >= 0.90
    gate["passed"] = all(gate["criteria"].values())
    report = {
        "phase": "physics_informed_belief_v12_cpu_feasibility",
        "status": "development_gate_passed" if gate["passed"] else "development_gate_failed",
        "confirmatory_evidence": False,
        "gpu_used": False,
        "v11_results_reclassified_as_development_only": True,
        "v11_identification_seeds": sorted(identification_seeds),
        "v11_estimator_audit_seeds": sorted(estimator_audit_seeds),
        "v12_selection_seeds": list(selection_seeds),
        "v12_validation_seeds": list(validation_seeds),
        "design": asdict(design),
        "transition_model": asdict(model),
        "estimator_audit": estimator,
        "selection_rule": {
            "candidate_counts": {
                "current_sensor": len(current_specs),
                "ema_history": len(history_specs),
                "physics_belief": len(belief_specs),
            },
            "maximum_selection_trip_rate": 0.02,
            "primary": "maximum reward within each policy family among trip-admissible candidates",
            "fallback_if_none_admissible": "maximum reward; feasibility gate remains authoritative",
        },
        "selection_results": selection,
        "selected_current_spec": selected_current,
        "selected_history_spec": selected_history,
        "selected_belief_spec": selected_belief,
        "validation_results": validation,
        "gate": gate,
    }
    atomic_write_new(args.output, report)
    print(
        json.dumps(
            {
                "passed": gate["passed"],
                "selected_belief_spec": selected_belief,
                "contrasts": gate["contrasts"],
                "estimator_audit": estimator,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
