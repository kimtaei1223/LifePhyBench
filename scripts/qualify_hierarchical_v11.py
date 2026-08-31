"""CPU-only analytic qualification for the v11 hierarchical thermal design.

This is a development-seed gate, not confirmatory evidence.  Every analytic
decision rule is evaluated by stepping the full hierarchical environment, so
the default path executes the frozen low-level Pusher controller for each
high-level decision.  The first design that passes the predeclared ordered
grid is selected; passing designs are never ranked by reward.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

DEV_SEEDS = (7300, 7301, 7302, 7303, 7304)
RESERVED_CONFIRMATORY_SEEDS = frozenset(range(8300, 8330))
COOLING_GRID = (0.10, 0.05, 0.15)
SENSOR_NOISE_GRID = (0.02, 0.01, 0.03)
SHOCK_GRID = ((5.0e-4, 0.01), (1.0e-3, 0.01), (5.0e-4, 0.02))

TRIP_LOAD = 0.10
LOW_POWER_SCALE = 0.40
TRIP_PENALTY = 75.0
HIGH_POWER_BONUS = 2.0
THERMAL_HEAT_RATE = 0.05
FIXED_INITIAL_LOAD = 0.04
STOCHASTIC_INITIAL_LOAD_LOW = 0.0
STOCHASTIC_INITIAL_LOAD_HIGH = 0.08
CANONICAL_TASK_SEED = 811
TASKS_PER_LIFETIME = 20

MIN_PRIVILEGED_OVER_INDEX = 0.50
MIN_HISTORY_OVER_CURRENT = 0.25
MAX_ORACLE_TRIP_RATE = 0.02
MIN_ORACLE_MIXED_LIFETIME_RATE = 0.80


@dataclass(frozen=True)
class QualificationDesign:
    """The only physical-observation fields varied by this qualification."""

    thermal_episode_cooling: float
    sensor_noise_sd: float
    shock_probability: float
    shock_size: float


ORDERED_GRID = tuple(
    QualificationDesign(cooling, sensor_sd, shock_probability, shock_size)
    for cooling, sensor_sd, (shock_probability, shock_size) in itertools.product(
        COOLING_GRID, SENSOR_NOISE_GRID, SHOCK_GRID
    )
)


class AnalyticPolicy(Protocol):
    """A non-learning high-level decision rule used by the CPU gate."""

    def act(
        self,
        *,
        task_index: int,
        sensor: float,
        exact_load: float,
    ) -> int: ...


EnvironmentFactory = Callable[[QualificationDesign, int], Any]
EnvironmentFactoryBuilder = Callable[[Any], EnvironmentFactory]
PolicyFactory = Callable[[], AnalyticPolicy]
ProgressCallback = Callable[[dict[str, Any]], None]

_WORKER_ENVIRONMENT_FACTORY: EnvironmentFactory | None = None
_WORKER_DESIGN: QualificationDesign | None = None
_WORKER_SEEDS: tuple[int, ...] = ()
_WORKER_TASKS_PER_LIFETIME = 0
_WORKER_REQUIRE_PHYSICAL_ROLLOUTS = False


@dataclass
class _AlwaysPolicy:
    action: int

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del task_index, sensor, exact_load
        return self.action


@dataclass
class _SchedulePolicy:
    schedule: tuple[int, ...]

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del sensor, exact_load
        return self.schedule[task_index]


@dataclass
class _ThresholdPolicy:
    source: str
    cutoff: float
    alpha: float | None = None
    estimate: float | None = None

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del task_index
        if self.source == "privileged_health":
            value = exact_load
        elif self.source == "current_sensor":
            value = sensor
        elif self.source == "history_filter":
            self.estimate = (
                sensor
                if self.estimate is None
                else float(self.alpha) * sensor
                + (1.0 - float(self.alpha)) * self.estimate
            )
            value = self.estimate
        else:  # pragma: no cover - construction is internal and exhaustive.
            raise ValueError(f"unknown threshold source: {self.source}")
        return int(value < self.cutoff)


def _sensor_from_observation(observation: Any) -> float:
    """Read the v11 summary ``[..., mode, sensor, index, trip, marker]``."""

    try:
        value = float(observation[-4])
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError("v11 observation is missing the noisy sensor") from error
    if not math.isfinite(value):
        raise RuntimeError("v11 noisy sensor is not finite")
    return value


def _exact_load(environment: Any, info: dict[str, Any]) -> float:
    """Read privileged health for an oracle without exposing it to policies."""

    for key in (
        "lifephy/v11_exact_thermal_load",
        "lifephy/v11_thermal_load",
        "lifephy/thermal_load_at_mode_selection",
        "lifephy/thermal_load",
    ):
        if key in info:
            value = float(info[key])
            if math.isfinite(value):
                return value
    for owner, attribute in (
        (environment, "current_thermal_load"),
        (environment, "thermal_load"),
        (getattr(environment, "_health", None), "thermal_load"),
    ):
        if owner is not None and hasattr(owner, attribute):
            value = float(getattr(owner, attribute))
            if math.isfinite(value):
                return value
    raise RuntimeError(
        "v11 environment must expose exact thermal load through audit info or "
        "an oracle-only environment attribute"
    )


def _trip_from_info(info: dict[str, Any]) -> bool:
    return bool(
        info.get("lifephy/thermal_trip", info.get("lifephy/v11_thermal_trip", False))
    )


def _physical_steps_from_info(info: dict[str, Any]) -> int | None:
    value = info.get("lifephy/hierarchical_physical_steps")
    return None if value is None else int(value)


def evaluate_policy(
    design: QualificationDesign,
    environment_factory: EnvironmentFactory,
    policy_factory: PolicyFactory,
    *,
    seeds: tuple[int, ...],
    tasks_per_lifetime: int,
    require_physical_rollouts: bool,
) -> dict[str, Any]:
    """Evaluate one analytic rule on paired development lifetimes."""

    lifetime_rows = []
    total_reward = 0.0
    total_tasks = 0
    total_trips = 0
    physical_step_audits = 0
    for seed in seeds:
        environment = environment_factory(design, seed)
        policy = policy_factory()
        actions: list[int] = []
        rewards: list[float] = []
        trips = 0
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
                if action not in (0, 1):
                    raise RuntimeError(f"analytic policy returned invalid action {action}")
                observation, reward, terminated, truncated, step_info = environment.step(
                    action
                )
                info = dict(step_info)
                physical_steps = _physical_steps_from_info(info)
                # A protection trip can end on the commitment check without
                # stepping MuJoCo.  More than one recorded step proves that
                # the analytic layer actually exercised the low-level policy.
                if physical_steps is not None and physical_steps > 1:
                    physical_step_audits += 1
                actions.append(action)
                rewards.append(float(reward))
                trips += int(_trip_from_info(info))
                boundary = bool(terminated or truncated)
                if boundary != (task_index == tasks_per_lifetime - 1):
                    raise RuntimeError(
                        "v11 lifetime boundary does not match the frozen task horizon"
                    )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        total_reward += sum(rewards)
        total_tasks += len(rewards)
        total_trips += trips
        lifetime_rows.append(
            {
                "seed": seed,
                "mean_reward_per_task": sum(rewards) / len(rewards),
                "trip_rate": trips / len(actions),
                "used_both_modes": len(set(actions)) == 2,
                "high_rate": sum(actions) / len(actions),
            }
        )
    if require_physical_rollouts and physical_step_audits == 0:
        raise RuntimeError(
            "qualification did not observe low-level physical rollout audit fields"
        )
    return {
        "mean_reward_per_task": total_reward / total_tasks,
        "trip_rate": total_trips / total_tasks,
        "mixed_mode_lifetime_rate": sum(
            row["used_both_modes"] for row in lifetime_rows
        )
        / len(lifetime_rows),
        "lifetimes": len(lifetime_rows),
        "tasks": total_tasks,
        "low_level_rollout_audit_observations": physical_step_audits,
        "lifetime_rows": lifetime_rows,
    }


def _threshold_values() -> tuple[float, ...]:
    return tuple(TRIP_LOAD * index / 20 for index in range(21))


def _schedule_candidates(horizon: int) -> tuple[tuple[int, ...], ...]:
    """Predeclared finite class for a strong non-learning clock oracle."""

    schedules: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(schedule: tuple[int, ...]) -> None:
        if schedule not in seen:
            seen.add(schedule)
            schedules.append(schedule)

    for high_tasks in range(horizon + 1):
        add((1,) * high_tasks + (0,) * (horizon - high_tasks))
        add((0,) * (horizon - high_tasks) + (1,) * high_tasks)
    for start in range(horizon):
        for stop in range(start + 1, horizon + 1):
            add(tuple(int(start <= index < stop) for index in range(horizon)))
    for period in range(2, min(6, horizon) + 1):
        for pattern in itertools.product((0, 1), repeat=period):
            add(tuple(pattern[index % period] for index in range(horizon)))
    return tuple(schedules)


def _candidate_specs(tasks_per_lifetime: int) -> tuple[dict[str, Any], ...]:
    """Return all categories in one stable, predeclared candidate order."""

    specs: list[dict[str, Any]] = [
        {"category": "always_low", "class": "always", "action": 0},
        {"category": "always_high", "class": "always", "action": 1},
    ]
    specs.extend(
        {
            "category": "privileged_health_oracle",
            "class": "privileged_health_threshold",
            "cutoff": cutoff,
        }
        for cutoff in _threshold_values()
    )
    specs.extend(
        {
            "category": "task_index_only_oracle",
            "class": "finite_task_index_schedule",
            "schedule": list(schedule),
        }
        for schedule in _schedule_candidates(tasks_per_lifetime)
    )
    specs.extend(
        {
            "category": "current_sensor_rule",
            "class": "current_sensor_threshold",
            "cutoff": cutoff,
        }
        for cutoff in _threshold_values()
    )
    specs.extend(
        {
            "category": "history_filter_heuristic",
            "class": "exponentially_weighted_sensor_history",
            "alpha": alpha,
            "cutoff": cutoff,
        }
        for alpha in (0.20, 0.40, 0.60, 0.80)
        for cutoff in _threshold_values()
    )
    return tuple(specs)


def _sensor_independent_cache_key(
    design: QualificationDesign,
) -> tuple[float, float, float]:
    """Identify physics shared across sensor-noise-only grid variants."""

    return (
        design.thermal_episode_cooling,
        design.shock_probability,
        design.shock_size,
    )


def _cached_sensor_independent_results(
    row: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Recover selected optima whose policy and physics never read the sensor."""

    summaries = row["summaries"]
    rules = row["selected_analytic_rules"]
    return [
        (
            {"category": "always_low", "class": "always", "action": 0},
            summaries["always_low"],
        ),
        (
            {"category": "always_high", "class": "always", "action": 1},
            summaries["always_high"],
        ),
        (
            {
                "category": "privileged_health_oracle",
                **rules["privileged_health_oracle"],
            },
            summaries["privileged_health_oracle"],
        ),
        (
            {
                "category": "task_index_only_oracle",
                **rules["task_index_only_oracle"],
            },
            summaries["task_index_only_oracle"],
        ),
    ]


def _policy_from_spec(spec: dict[str, Any]) -> AnalyticPolicy:
    policy_class = spec["class"]
    if policy_class == "always":
        return _AlwaysPolicy(int(spec["action"]))
    if policy_class == "finite_task_index_schedule":
        return _SchedulePolicy(tuple(int(action) for action in spec["schedule"]))
    if policy_class == "privileged_health_threshold":
        return _ThresholdPolicy("privileged_health", float(spec["cutoff"]))
    if policy_class == "current_sensor_threshold":
        return _ThresholdPolicy("current_sensor", float(spec["cutoff"]))
    if policy_class == "exponentially_weighted_sensor_history":
        return _ThresholdPolicy(
            "history_filter",
            float(spec["cutoff"]),
            alpha=float(spec["alpha"]),
        )
    raise ValueError(f"unknown analytic candidate class: {policy_class}")


def _evaluate_candidate(
    design: QualificationDesign,
    environment_factory: EnvironmentFactory,
    spec: dict[str, Any],
    *,
    seeds: tuple[int, ...],
    tasks_per_lifetime: int,
    require_physical_rollouts: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = evaluate_policy(
        design,
        environment_factory,
        lambda: _policy_from_spec(spec),
        seeds=seeds,
        tasks_per_lifetime=tasks_per_lifetime,
        require_physical_rollouts=require_physical_rollouts,
    )
    return spec, summary


def _initialize_candidate_worker(
    factory_builder: EnvironmentFactoryBuilder,
    factory_argument: Any,
    design_document: dict[str, float],
    seeds: tuple[int, ...],
    tasks_per_lifetime: int,
    require_physical_rollouts: bool,
) -> None:
    """Load the frozen controller once and bind immutable design state."""

    global _WORKER_DESIGN
    global _WORKER_ENVIRONMENT_FACTORY
    global _WORKER_REQUIRE_PHYSICAL_ROLLOUTS
    global _WORKER_SEEDS
    global _WORKER_TASKS_PER_LIFETIME

    _WORKER_ENVIRONMENT_FACTORY = factory_builder(factory_argument)
    _WORKER_DESIGN = QualificationDesign(**design_document)
    _WORKER_SEEDS = seeds
    _WORKER_TASKS_PER_LIFETIME = tasks_per_lifetime
    _WORKER_REQUIRE_PHYSICAL_ROLLOUTS = require_physical_rollouts


def _evaluate_candidate_in_worker(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _WORKER_DESIGN is None or _WORKER_ENVIRONMENT_FACTORY is None:
        raise RuntimeError("parallel qualification worker was not initialized")
    return _evaluate_candidate(
        _WORKER_DESIGN,
        _WORKER_ENVIRONMENT_FACTORY,
        spec,
        seeds=_WORKER_SEEDS,
        tasks_per_lifetime=_WORKER_TASKS_PER_LIFETIME,
        require_physical_rollouts=_WORKER_REQUIRE_PHYSICAL_ROLLOUTS,
    )


def _best_in_category(
    results: list[tuple[dict[str, Any], dict[str, Any]]], category: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = [row for row in results if row[0]["category"] == category]
    if not candidates:
        raise RuntimeError(f"analytic category has no candidates: {category}")
    # max() returns the first candidate on exact ties, preserving sequential
    # predeclared order even when pool tasks finish out of order.
    spec, summary = max(
        candidates, key=lambda row: row[1]["mean_reward_per_task"]
    )
    public_spec = {key: value for key, value in spec.items() if key != "category"}
    return public_spec, summary


def _assemble_design_result(
    design: QualificationDesign,
    candidate_results: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    _always_low_spec, always_low = _best_in_category(
        candidate_results, "always_low"
    )
    _always_high_spec, always_high = _best_in_category(
        candidate_results, "always_high"
    )
    privileged_spec, privileged = _best_in_category(
        candidate_results, "privileged_health_oracle"
    )
    index_spec, task_index = _best_in_category(
        candidate_results, "task_index_only_oracle"
    )
    current_spec, current_sensor = _best_in_category(
        candidate_results, "current_sensor_rule"
    )
    history_spec, history_filter = _best_in_category(
        candidate_results, "history_filter_heuristic"
    )

    privileged_over_index = (
        privileged["mean_reward_per_task"] - task_index["mean_reward_per_task"]
    )
    history_over_current = (
        history_filter["mean_reward_per_task"]
        - current_sensor["mean_reward_per_task"]
    )
    criteria = {
        "privileged_health_over_task_index_at_least_half": (
            privileged_over_index >= MIN_PRIVILEGED_OVER_INDEX
        ),
        "history_filter_over_current_sensor_at_least_quarter": (
            history_over_current >= MIN_HISTORY_OVER_CURRENT
        ),
        "privileged_oracle_trip_rate_at_most_two_percent": (
            privileged["trip_rate"] <= MAX_ORACLE_TRIP_RATE
        ),
        "privileged_oracle_uses_both_modes_in_eighty_percent": (
            privileged["mixed_mode_lifetime_rate"]
            >= MIN_ORACLE_MIXED_LIFETIME_RATE
        ),
        "always_low_not_optimal": (
            privileged["mean_reward_per_task"]
            > always_low["mean_reward_per_task"]
        ),
        "always_high_not_optimal": (
            privileged["mean_reward_per_task"]
            > always_high["mean_reward_per_task"]
        ),
    }
    return {
        "design": asdict(design),
        "analytic_candidates_evaluated": len(candidate_results),
        "selected_analytic_rules": {
            "privileged_health_oracle": privileged_spec,
            "task_index_only_oracle": index_spec,
            "current_sensor_rule": current_spec,
            "history_filter_heuristic": history_spec,
        },
        "summaries": {
            "privileged_health_oracle": privileged,
            "task_index_only_oracle": task_index,
            "current_sensor_rule": current_sensor,
            "history_filter_heuristic": history_filter,
            "always_low": always_low,
            "always_high": always_high,
        },
        "contrasts": {
            "privileged_health_minus_task_index": privileged_over_index,
            "history_filter_minus_current_sensor": history_over_current,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def evaluate_design(
    design: QualificationDesign,
    environment_factory: EnvironmentFactory,
    *,
    seeds: tuple[int, ...] = DEV_SEEDS,
    tasks_per_lifetime: int = TASKS_PER_LIFETIME,
    require_physical_rollouts: bool = False,
) -> dict[str, Any]:
    """Evaluate all predeclared analytic decision layers for one design."""
    results = [
        _evaluate_candidate(
            design,
            environment_factory,
            spec,
            seeds=seeds,
            tasks_per_lifetime=tasks_per_lifetime,
            require_physical_rollouts=require_physical_rollouts,
        )
        for spec in _candidate_specs(tasks_per_lifetime)
    ]
    return _assemble_design_result(design, results)


def evaluate_design_parallel(
    design: QualificationDesign,
    factory_builder: EnvironmentFactoryBuilder,
    factory_argument: Any,
    *,
    parallel_workers: int,
    seeds: tuple[int, ...] = DEV_SEEDS,
    tasks_per_lifetime: int = TASKS_PER_LIFETIME,
    require_physical_rollouts: bool = False,
    sensor_independent_cache_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every category in one deterministic spawn-based process pool."""

    if parallel_workers <= 0:
        raise ValueError("parallel_workers must be positive")
    all_specs = _candidate_specs(tasks_per_lifetime)
    if sensor_independent_cache_row is None:
        specs = all_specs
        cached_results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    else:
        cached_design = QualificationDesign(**sensor_independent_cache_row["design"])
        if _sensor_independent_cache_key(cached_design) != (
            _sensor_independent_cache_key(design)
        ):
            raise ValueError("sensor-independent cache physics mismatch")
        if cached_design.sensor_noise_sd == design.sensor_noise_sd:
            raise ValueError("sensor cache is intended only for a noise-grid variant")
        specs = tuple(
            spec
            for spec in all_specs
            if spec["category"]
            in {"current_sensor_rule", "history_filter_heuristic"}
        )
        cached_results = _cached_sensor_independent_results(
            sensor_independent_cache_row
        )
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=parallel_workers,
        initializer=_initialize_candidate_worker,
        initargs=(
            factory_builder,
            factory_argument,
            asdict(design),
            seeds,
            tasks_per_lifetime,
            require_physical_rollouts,
        ),
    ) as pool:
        # Pool.map preserves input order. Indexed environment RNG makes each
        # candidate invariant to scheduling and worker assignment.
        results = pool.map(_evaluate_candidate_in_worker, specs, chunksize=1)
    result = _assemble_design_result(design, cached_results + results)
    if sensor_independent_cache_row is not None:
        result["analytic_candidates_declared"] = len(all_specs)
        result["new_rollout_candidates_evaluated"] = len(specs)
        result["sensor_independent_candidates_reused"] = len(all_specs) - len(specs)
        result["sensor_independent_cache_source_design"] = (
            sensor_independent_cache_row["design"]
        )
    return result


def qualify_ordered_grid(
    environment_factory: EnvironmentFactory,
    *,
    designs: tuple[QualificationDesign, ...] = ORDERED_GRID,
    seeds: tuple[int, ...] = DEV_SEEDS,
    tasks_per_lifetime: int = TASKS_PER_LIFETIME,
    require_physical_rollouts: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stop at the first eligible design without reward-ranking passers."""

    if not designs or not seeds or tasks_per_lifetime <= 0:
        raise ValueError("designs, seeds, and task horizon must be non-empty")
    if RESERVED_CONFIRMATORY_SEEDS.intersection(seeds):
        raise ValueError("confirmatory seeds are forbidden in CPU qualification")
    rows = []
    selected = None
    for design in designs:
        row = evaluate_design(
            design,
            environment_factory,
            seeds=seeds,
            tasks_per_lifetime=tasks_per_lifetime,
            require_physical_rollouts=require_physical_rollouts,
        )
        rows.append(row)
        if row["passed"]:
            selected = asdict(design)
        current = _qualification_report(
            designs=designs,
            seeds=seeds,
            tasks_per_lifetime=tasks_per_lifetime,
            require_physical_rollouts=require_physical_rollouts,
            parallel_workers=1,
            rows=rows,
            selected=selected,
        )
        if progress_callback is not None:
            progress_callback(current)
        if selected is not None:
            break
    return current


def _qualification_report(
    *,
    designs: tuple[QualificationDesign, ...],
    seeds: tuple[int, ...],
    tasks_per_lifetime: int,
    require_physical_rollouts: bool,
    parallel_workers: int,
    rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
) -> dict[str, Any]:
    passed = selected is not None
    return {
        "phase": "hierarchical_v11_cpu_qualification",
        "status": (
            "first_ordered_design_passed_dev_gate"
            if passed
            else "ordered_design_grid_exhausted_without_pass"
        ),
        "confirmatory_evidence": False,
        "development_seeds": list(seeds),
        "reserved_confirmatory_seeds_not_accessed": True,
        "tasks_per_lifetime": tasks_per_lifetime,
        "ordered_grid": [asdict(design) for design in designs],
        "gate_thresholds": {
            "min_privileged_health_minus_task_index_reward_per_task": (
                MIN_PRIVILEGED_OVER_INDEX
            ),
            "min_history_filter_minus_current_sensor_reward_per_task": (
                MIN_HISTORY_OVER_CURRENT
            ),
            "max_privileged_oracle_trip_rate": MAX_ORACLE_TRIP_RATE,
            "min_privileged_oracle_mixed_mode_lifetime_rate": (
                MIN_ORACLE_MIXED_LIFETIME_RATE
            ),
            "always_low_and_always_high_must_be_suboptimal": True,
        },
        "analytic_decision_layer_only": True,
        "actual_low_level_rollouts_required": require_physical_rollouts,
        "parallel_workers": parallel_workers,
        "multiprocessing_start_method": (
            "spawn" if parallel_workers > 1 else "not_used"
        ),
        "evaluated_designs": rows,
        "selected_design": selected,
        "passed": passed,
    }


def qualify_ordered_grid_parallel(
    factory_builder: EnvironmentFactoryBuilder,
    factory_argument: Any,
    *,
    parallel_workers: int,
    designs: tuple[QualificationDesign, ...] = ORDERED_GRID,
    seeds: tuple[int, ...] = DEV_SEEDS,
    tasks_per_lifetime: int = TASKS_PER_LIFETIME,
    require_physical_rollouts: bool = False,
    progress_callback: ProgressCallback | None = None,
    sensor_independent_cache_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parallel gate with one full candidate pool per ordered design."""

    if not designs or not seeds or tasks_per_lifetime <= 0:
        raise ValueError("designs, seeds, and task horizon must be non-empty")
    if parallel_workers <= 1:
        raise ValueError("parallel qualification requires at least two workers")
    if RESERVED_CONFIRMATORY_SEEDS.intersection(seeds):
        raise ValueError("confirmatory seeds are forbidden in CPU qualification")
    rows = []
    selected = None
    sensor_cache: dict[tuple[float, float, float], dict[str, Any]] = {}
    for previous in sensor_independent_cache_rows or []:
        previous_design = QualificationDesign(**previous["design"])
        sensor_cache.setdefault(
            _sensor_independent_cache_key(previous_design), previous
        )
    for design in designs:
        cache_key = _sensor_independent_cache_key(design)
        cached_row = sensor_cache.get(cache_key)
        row = evaluate_design_parallel(
            design,
            factory_builder,
            factory_argument,
            parallel_workers=parallel_workers,
            seeds=seeds,
            tasks_per_lifetime=tasks_per_lifetime,
            require_physical_rollouts=require_physical_rollouts,
            sensor_independent_cache_row=cached_row,
        )
        rows.append(row)
        sensor_cache.setdefault(cache_key, row)
        if row["passed"]:
            selected = asdict(design)
        current = _qualification_report(
            designs=designs,
            seeds=seeds,
            tasks_per_lifetime=tasks_per_lifetime,
            require_physical_rollouts=require_physical_rollouts,
            parallel_workers=parallel_workers,
            rows=rows,
            selected=selected,
        )
        if progress_callback is not None:
            progress_callback(current)
        if selected is not None:
            break
    return current


def make_default_environment_factory(
    low_level_model_path: Path,
    *,
    environment_id: str = "Pusher-v5",
) -> EnvironmentFactory:
    """Build a real v11 MuJoCo environment while loading its controller once."""

    try:
        import torch
        from sb3_contrib import RecurrentPPO

        from lifephybench.envs.hierarchical_thermal_v11 import (
            HierarchicalThermalV11Config,
            HierarchicalThermalV11Env,
        )
    except ImportError as error:  # pragma: no cover - exercised in the real stack.
        raise RuntimeError(
            "v11 environment and MuJoCo RL dependencies must be installed"
        ) from error

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_path = low_level_model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"frozen low-level model not found: {model_path}")
    low_level_model = RecurrentPPO.load(str(model_path), device="cpu")

    def factory(design: QualificationDesign, seed: int) -> Any:
        del seed  # The evaluator passes the development seed to env.reset.
        config = HierarchicalThermalV11Config(
            low_level_model_path=str(model_path),
            condition="stochastic",
            environment_id=environment_id,
            episode_steps=100,
            episodes_per_lifetime=TASKS_PER_LIFETIME,
            canonical_task_seed=CANONICAL_TASK_SEED,
            trip_load=TRIP_LOAD,
            low_power_scale=LOW_POWER_SCALE,
            trip_penalty=TRIP_PENALTY,
            high_power_bonus=HIGH_POWER_BONUS,
            thermal_heat_rate=THERMAL_HEAT_RATE,
            thermal_episode_cooling=design.thermal_episode_cooling,
            fixed_initial_load=FIXED_INITIAL_LOAD,
            stochastic_initial_load_low=STOCHASTIC_INITIAL_LOAD_LOW,
            stochastic_initial_load_high=STOCHASTIC_INITIAL_LOAD_HIGH,
            sensor_noise_sd=design.sensor_noise_sd,
            shock_probability=design.shock_probability,
            shock_size=design.shock_size,
            low_level_device="cpu",
        )
        return HierarchicalThermalV11Env(config, low_level_model=low_level_model)

    return factory


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resume_prefix(
    path: Path,
    *,
    seeds: tuple[int, ...],
    tasks_per_lifetime: int,
) -> list[dict[str, Any]]:
    """Validate and return an exact failed ordered-grid prefix."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume invalid qualification report: {path}") from error
    if document.get("phase") != "hierarchical_v11_cpu_qualification":
        raise ValueError("resume report phase mismatch")
    if document.get("passed") is not False or document.get("selected_design") is not None:
        raise ValueError("only a failed, unselected qualification can be resumed")
    if document.get("development_seeds") != list(seeds):
        raise ValueError("resume development seeds mismatch")
    if document.get("tasks_per_lifetime") != tasks_per_lifetime:
        raise ValueError("resume task horizon mismatch")
    rows = document.get("evaluated_designs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("resume report contains no completed designs")
    if len(rows) >= len(ORDERED_GRID):
        raise ValueError("ordered grid is already exhausted")
    expected_prefix = [asdict(design) for design in ORDERED_GRID[: len(rows)]]
    actual_prefix = [row.get("design") for row in rows]
    if actual_prefix != expected_prefix or any(row.get("passed") is not False for row in rows):
        raise ValueError("resume report is not an exact failed ordered-grid prefix")
    return rows


def _merge_resumed_report(
    prefix_rows: list[dict[str, Any]],
    continuation: dict[str, Any],
    *,
    parallel_workers: int,
) -> dict[str, Any]:
    """Combine prior completed designs with a deterministic continuation."""

    merged = dict(continuation)
    merged["evaluated_designs"] = prefix_rows + continuation["evaluated_designs"]
    merged["ordered_grid"] = [asdict(design) for design in ORDERED_GRID]
    merged["parallel_workers"] = parallel_workers
    merged["resumed_from_completed_designs"] = len(prefix_rows)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low-level-model",
        type=Path,
        default=Path(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_v11/CPU_QUALIFICATION.json"),
    )
    parser.add_argument("--dev-seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    parser.add_argument(
        "--max-designs",
        type=int,
        default=0,
        help="development smoke only; zero evaluates until first pass or exhaustion",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="spawned CPU workers; use 8 on the 32 GB reference host",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from an exact failed prefix already stored at --output",
    )
    args = parser.parse_args()
    seeds = tuple(args.dev_seeds)
    if not set(seeds).issubset(DEV_SEEDS):
        raise SystemExit(f"CPU qualification accepts development seeds only: {DEV_SEEDS}")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("development seeds must be unique")
    if args.parallel_workers <= 0:
        raise SystemExit("parallel workers must be positive")
    prefix_rows: list[dict[str, Any]] = []
    start_index = 0
    if args.resume:
        prefix_rows = _resume_prefix(
            args.output,
            seeds=seeds,
            tasks_per_lifetime=TASKS_PER_LIFETIME,
        )
        start_index = len(prefix_rows)
    available_designs = ORDERED_GRID[start_index:]
    designs = (
        available_designs[: args.max_designs]
        if args.max_designs > 0
        else available_designs
    )

    def checkpoint(current: dict[str, Any]) -> None:
        document = (
            _merge_resumed_report(
                prefix_rows, current, parallel_workers=args.parallel_workers
            )
            if prefix_rows
            else current
        )
        _write_json(args.output, document)

    if args.parallel_workers == 1:
        factory = make_default_environment_factory(args.low_level_model)
        report = qualify_ordered_grid(
            factory,
            designs=designs,
            seeds=seeds,
            require_physical_rollouts=True,
            progress_callback=checkpoint,
        )
    else:
        report = qualify_ordered_grid_parallel(
            make_default_environment_factory,
            args.low_level_model,
            parallel_workers=args.parallel_workers,
            designs=designs,
            seeds=seeds,
            require_physical_rollouts=True,
            progress_callback=checkpoint,
            sensor_independent_cache_rows=prefix_rows,
        )
    if prefix_rows:
        report = _merge_resumed_report(
            prefix_rows, report, parallel_workers=args.parallel_workers
        )
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("v11 CPU qualification failed; held-out training remains blocked")
    print("[V11 CPU QUALIFICATION PASS — STOP BEFORE HELD-OUT TRAINING]")


if __name__ == "__main__":
    main()
