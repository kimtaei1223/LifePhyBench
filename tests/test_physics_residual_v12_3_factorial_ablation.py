from __future__ import annotations

import math

import pytest

from scripts.run_physics_residual_v12_3_factorial_ablation import (
    decompose_task_return,
    paired_metric_values,
)


def test_reward_decomposition_reconstructs_high_power_task():
    row = decompose_task_return(-10.0, action=1, tripped=False)
    assert row == {
        "base_task_return": -12.0,
        "throughput_bonus": 2.0,
        "trip_penalty": 0.0,
    }
    assert math.isclose(sum(row.values()), -10.0)


def test_reward_decomposition_reconstructs_trip():
    row = decompose_task_return(-75.0, action=1, tripped=True)
    assert row == {
        "base_task_return": 0.0,
        "throughput_bonus": 0.0,
        "trip_penalty": -75.0,
    }


def test_paired_metric_values_rejects_seed_mismatch():
    cells = {
        "condition": {
            "a": {"summary": {"lifetime_rows": [{"seed": 1, "metric": 2.0}]}},
            "b": {"summary": {"lifetime_rows": [{"seed": 2, "metric": 1.0}]}},
        }
    }
    with pytest.raises(ValueError, match="paired seed mismatch"):
        paired_metric_values(cells, "condition", "a", "b", "metric")
