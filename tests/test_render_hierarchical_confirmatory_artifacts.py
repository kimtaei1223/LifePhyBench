from __future__ import annotations

import copy
import math

import matplotlib.pyplot as plt
import pytest

from scripts.render_hierarchical_confirmatory_artifacts import (
    SVG_HASH_SALT,
    configure_plot_style,
    crosscheck_result_rows,
    summarize_cells,
)


def cell(
    seed: int,
    degradation: str,
    memory: str,
    reward: float,
    high_rate: float,
    trip_rate: float,
    *,
    cold: float | None = None,
    hot: float | None = None,
) -> dict:
    return {
        "seed": seed,
        "degradation_mode": degradation,
        "memory_mode": memory,
        "reward": reward,
        "high_rate": high_rate,
        "trip_rate": trip_rate,
        "cold_high_rate": cold,
        "hot_high_rate": hot,
    }


def test_four_cell_summary_uses_training_seed_sample_sd():
    rows = []
    for seed, offset in [(1, 0.0), (2, 2.0)]:
        rows.extend(
            [
                cell(seed, "endogenous_action", "task", 1 + offset, 0.1, 0.0),
                cell(
                    seed,
                    "endogenous_action",
                    "lifetime",
                    4 + offset,
                    0.2,
                    0.0,
                ),
                cell(seed, "exogenous_clock", "task", 7.0, 1.0, 0.0),
                cell(seed, "exogenous_clock", "lifetime", 7.0, 1.0, 0.0),
            ]
        )

    summaries = summarize_cells(rows)
    dynamic_task = summaries[0]
    assert dynamic_task["condition"] == "Dynamic"
    assert dynamic_task["memory"] == "Task-reset"
    assert dynamic_task["n_seeds"] == 2
    assert dynamic_task["reward_mean"] == 2.0
    assert math.isclose(dynamic_task["reward_sample_sd"], math.sqrt(2.0))
    assert summaries[2]["reward_sample_sd"] == 0.0


def test_result_metadata_crosscheck_fails_on_changed_reward():
    cells = [
        cell(10, "endogenous_action", "task", -10.0, 0.0, 0.0),
        cell(
            10,
            "endogenous_action",
            "lifetime",
            -8.0,
            0.2,
            0.1,
            cold=0.5,
            hot=0.1,
        ),
        cell(10, "exogenous_clock", "task", -2.0, 1.0, 0.0),
        cell(10, "exogenous_clock", "lifetime", -2.0, 1.0, 0.0),
    ]
    result = {
        "seed": 10,
        "dynamic_task_reward": -10.0,
        "dynamic_lifetime_reward": -8.0,
        "dynamic_memory_effect": 2.0,
        "static_memory_effect": 0.0,
        "interaction": 2.0,
        "dynamic_lifetime_high_rate": 0.2,
        "dynamic_lifetime_adaptation_gap": 0.4,
        "dynamic_lifetime_trip_rate": 0.1,
        "static_task_high_rate": 1.0,
        "static_lifetime_high_rate": 1.0,
    }
    crosscheck_result_rows([result], cells)

    changed = copy.deepcopy(result)
    changed["dynamic_lifetime_reward"] = -7.9
    with pytest.raises(ValueError, match="dynamic_lifetime_reward"):
        crosscheck_result_rows([changed], cells)


def test_svg_hash_salt_is_explicitly_fixed():
    configure_plot_style()
    assert plt.rcParams["svg.hashsalt"] == SVG_HASH_SALT
