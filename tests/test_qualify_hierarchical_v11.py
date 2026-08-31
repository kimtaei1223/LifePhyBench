from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import qualify_hierarchical_v11 as qualification


class FakeLowLevelV11Environment:
    """Cheap deterministic stand-in with a slowly varying hidden load."""

    def __init__(
        self,
        design: qualification.QualificationDesign,
        seed: int,
        *,
        horizon: int = 20,
    ) -> None:
        self.design = design
        self.seed = seed
        self.horizon = horizon
        self.task_index = 0
        self.current_thermal_load = 0.0
        self.closed = False
        self.step_calls = 0

    def _set_load(self) -> None:
        phase = 2 * ((self.seed - 7300) % 5)
        hot = ((self.task_index + phase) // 10) % 2 == 1
        self.current_thermal_load = 0.12 if hot else 0.02

    def _sensor(self) -> float:
        hot = self.current_thermal_load >= 0.10
        if hot:
            return 0.18 if self.task_index % 2 == 0 else 0.06
        return 0.00 if self.task_index % 2 == 0 else 0.14

    def _observation(self) -> np.ndarray:
        normalized_index = self.task_index / max(1, self.horizon - 1)
        return np.asarray(
            [0.0, self._sensor(), normalized_index, 0.0, 1.0],
            dtype=np.float64,
        )

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self.task_index = 0
        self._set_load()
        return self._observation(), {
            "lifephy/v11_exact_thermal_load": self.current_thermal_load
        }

    def step(self, action: int):
        self.step_calls += 1
        safe_high = self.current_thermal_load < 0.10
        correct = bool(action == int(safe_high))
        reward = 3.0 if correct else 0.0
        trip = bool(action == 1 and not safe_high)
        self.task_index += 1
        truncated = self.task_index == self.horizon
        if not truncated:
            self._set_load()
        info = {
            "lifephy/v11_exact_thermal_load": self.current_thermal_load,
            "lifephy/thermal_trip": trip,
            "lifephy/hierarchical_physical_steps": 100,
        }
        return self._observation(), reward, False, truncated, info

    def close(self) -> None:
        self.closed = True


def fake_factory(design: qualification.QualificationDesign, seed: int):
    return FakeLowLevelV11Environment(design, seed)


def small_fake_factory(design: qualification.QualificationDesign, seed: int):
    return FakeLowLevelV11Environment(design, seed, horizon=4)


def small_fake_factory_builder(_argument):
    return small_fake_factory


def test_ordered_grid_matches_predeclared_nested_order():
    assert len(qualification.ORDERED_GRID) == 27
    assert qualification.ORDERED_GRID[:4] == (
        qualification.QualificationDesign(0.10, 0.02, 5.0e-4, 0.01),
        qualification.QualificationDesign(0.10, 0.02, 1.0e-3, 0.01),
        qualification.QualificationDesign(0.10, 0.02, 5.0e-4, 0.02),
        qualification.QualificationDesign(0.10, 0.01, 5.0e-4, 0.01),
    )


def test_real_rollout_interface_is_exercised_for_every_task():
    created: list[FakeLowLevelV11Environment] = []

    def recording_factory(design, seed):
        environment = FakeLowLevelV11Environment(design, seed, horizon=4)
        created.append(environment)
        return environment

    design = qualification.ORDERED_GRID[0]
    result = qualification.evaluate_policy(
        design,
        recording_factory,
        lambda: qualification._AlwaysPolicy(0),
        seeds=(7300, 7301),
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
    )

    assert result["tasks"] == 8
    assert result["low_level_rollout_audit_observations"] == 8
    assert sum(environment.step_calls for environment in created) == 8
    assert all(environment.closed for environment in created)


def test_first_passing_design_is_selected_with_required_gate_margins():
    designs = qualification.ORDERED_GRID[:2]
    report = qualification.qualify_ordered_grid(
        fake_factory,
        designs=designs,
        seeds=qualification.DEV_SEEDS,
        tasks_per_lifetime=20,
        require_physical_rollouts=True,
    )

    assert report["passed"] is True
    assert report["selected_design"] == qualification.asdict(designs[0])
    assert len(report["evaluated_designs"]) == 1
    result = report["evaluated_designs"][0]
    assert result["contrasts"]["privileged_health_minus_task_index"] >= 0.50
    assert result["contrasts"]["history_filter_minus_current_sensor"] >= 0.25
    assert result["summaries"]["privileged_health_oracle"]["trip_rate"] <= 0.02
    assert (
        result["summaries"]["privileged_health_oracle"][
            "mixed_mode_lifetime_rate"
        ]
        >= 0.80
    )
    assert all(result["criteria"].values())


def test_confirmatory_seed_is_rejected_before_any_evaluation():
    calls = 0

    def forbidden_factory(design, seed):
        nonlocal calls
        calls += 1
        return FakeLowLevelV11Environment(design, seed)

    with pytest.raises(ValueError, match="confirmatory seeds"):
        qualification.qualify_ordered_grid(
            forbidden_factory,
            designs=qualification.ORDERED_GRID[:1],
            seeds=(8300,),
        )
    assert calls == 0


def test_spawn_parallel_result_exactly_matches_sequential_result():
    design = qualification.ORDERED_GRID[0]
    seeds = (7300, 7301)
    sequential = qualification.evaluate_design(
        design,
        small_fake_factory,
        seeds=seeds,
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
    )
    parallel = qualification.evaluate_design_parallel(
        design,
        small_fake_factory_builder,
        None,
        parallel_workers=2,
        seeds=seeds,
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
    )

    assert parallel == sequential


def test_resume_accepts_only_exact_failed_ordered_prefix(tmp_path):
    first = {
        "phase": "hierarchical_v11_cpu_qualification",
        "passed": False,
        "selected_design": None,
        "development_seeds": list(qualification.DEV_SEEDS),
        "tasks_per_lifetime": 20,
        "evaluated_designs": [
            {
                "design": qualification.asdict(qualification.ORDERED_GRID[0]),
                "passed": False,
            }
        ],
    }
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    rows = qualification._resume_prefix(
        path, seeds=qualification.DEV_SEEDS, tasks_per_lifetime=20
    )
    assert rows == first["evaluated_designs"]

    first["evaluated_designs"][0]["design"] = qualification.asdict(
        qualification.ORDERED_GRID[1]
    )
    path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="exact failed ordered-grid prefix"):
        qualification._resume_prefix(
            path, seeds=qualification.DEV_SEEDS, tasks_per_lifetime=20
        )


def test_resumed_report_preserves_prefix_and_full_order():
    prefix = [{"design": qualification.asdict(qualification.ORDERED_GRID[0])}]
    continuation = {
        "evaluated_designs": [
            {"design": qualification.asdict(qualification.ORDERED_GRID[1])}
        ],
        "ordered_grid": [],
        "passed": True,
        "selected_design": qualification.asdict(qualification.ORDERED_GRID[1]),
    }
    merged = qualification._merge_resumed_report(
        prefix, continuation, parallel_workers=8
    )
    assert merged["evaluated_designs"] == prefix + continuation["evaluated_designs"]
    assert merged["ordered_grid"] == [
        qualification.asdict(design) for design in qualification.ORDERED_GRID
    ]
    assert merged["resumed_from_completed_designs"] == 1


def test_progress_callback_receives_each_completed_design():
    reports = []
    designs = qualification.ORDERED_GRID[:2]
    result = qualification.qualify_ordered_grid(
        fake_factory,
        designs=designs,
        seeds=qualification.DEV_SEEDS,
        tasks_per_lifetime=20,
        require_physical_rollouts=True,
        progress_callback=reports.append,
    )
    assert result["passed"] is True
    assert len(reports) == 1
    assert reports[0] == result


def test_parallel_sensor_cache_is_exact_for_noise_only_variant():
    first = qualification.ORDERED_GRID[0]
    noise_variant = qualification.ORDERED_GRID[3]
    first_result = qualification.evaluate_design_parallel(
        first,
        small_fake_factory_builder,
        None,
        parallel_workers=2,
        seeds=(7300, 7301),
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
    )
    cached = qualification.evaluate_design_parallel(
        noise_variant,
        small_fake_factory_builder,
        None,
        parallel_workers=2,
        seeds=(7300, 7301),
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
        sensor_independent_cache_row=first_result,
    )
    sequential = qualification.evaluate_design(
        noise_variant,
        small_fake_factory,
        seeds=(7300, 7301),
        tasks_per_lifetime=4,
        require_physical_rollouts=True,
    )

    for key in (
        "selected_analytic_rules",
        "summaries",
        "contrasts",
        "criteria",
        "passed",
    ):
        assert cached[key] == sequential[key]
    assert cached["new_rollout_candidates_evaluated"] == 105
    assert cached["sensor_independent_candidates_reused"] == (
        len(qualification._candidate_specs(4)) - 105
    )
