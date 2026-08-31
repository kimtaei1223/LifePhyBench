from __future__ import annotations

import math

from scripts.render_physics_residual_v12_3_artifacts import (
    aggregate_alternative_contrast,
    alternative_reward,
)


def _row(seed: int, base: float, bonus: float, penalty: float):
    return {
        "seed": seed,
        "mean_base_task_return": base,
        "mean_throughput_bonus": bonus,
        "mean_trip_penalty": penalty,
        "mean_reward_per_task": base + bonus + penalty,
    }


def _cells():
    conditions = ("ood_sensor_noise", "ood_cooling", "ood_combined")
    cells = {}
    for condition in conditions:
        cells[condition] = {
            "treatment": {
                "summary": {
                    "lifetime_rows": [
                        _row(1, -5.0, 1.0, -1.0),
                        _row(2, -4.0, 1.0, -1.0),
                    ]
                }
            },
            "control": {
                "summary": {
                    "lifetime_rows": [
                        _row(1, -4.0, 2.0, -3.0),
                        _row(2, -3.0, 2.0, -3.0),
                    ]
                }
            },
        }
    return cells


def test_alternative_reward_reweights_recorded_components():
    row = _row(1, -5.0, 2.0, -75.0)
    assert alternative_reward(row, throughput_bonus=2.0, trip_penalty=75.0) == -78.0
    assert alternative_reward(row, throughput_bonus=1.0, trip_penalty=50.0) == -54.0


def test_aggregate_alternative_contrast_preserves_seed_pairing():
    seeds, values = aggregate_alternative_contrast(
        _cells(),
        treatment="treatment",
        control="control",
        throughput_bonus=2.0,
        trip_penalty=75.0,
    )
    assert seeds == [1, 2]
    assert all(math.isclose(value, 0.0) for value in values)


def test_alternative_contrast_changes_with_trip_cost():
    _, values = aggregate_alternative_contrast(
        _cells(),
        treatment="treatment",
        control="control",
        throughput_bonus=2.0,
        trip_penalty=150.0,
    )
    assert all(math.isclose(value, 2.0) for value in values)
