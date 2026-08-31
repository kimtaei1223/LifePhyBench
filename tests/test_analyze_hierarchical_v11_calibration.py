from __future__ import annotations

import pytest

from scripts.analyze_hierarchical_v11_calibration import (
    assess_baseline_competence,
    select_reactive_arm,
)


def summaries() -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    values = {
        "fixed": {
            "lifetime_lstm": -42.7,
            "task_reset_lstm": -42.8,
            "reactive_mlp_64": -41.6,
            "reactive_mlp_256": -42.7,
        },
        "stochastic": {
            "lifetime_lstm": -42.9,
            "task_reset_lstm": -43.5,
            "reactive_mlp_64": -43.5,
            "reactive_mlp_256": -42.1,
        },
    }
    for condition, arms in values.items():
        result[condition] = {}
        for arm, reward in arms.items():
            result[condition][arm] = {
                "mean_task_episode_reward": reward,
                "thermal_trip_rate": 0.001,
                "both_modes_lifetime_rate": 0.9,
            }
    return result


def test_selects_strongest_stochastic_reactive_arm_without_lifetime() -> None:
    values = summaries()
    values["stochastic"]["lifetime_lstm"]["mean_task_episode_reward"] = 100.0
    assert select_reactive_arm(values) == "reactive_mlp_256"


def test_baseline_competence_requires_gain_mixed_modes_and_safety() -> None:
    report = assess_baseline_competence(
        selected_arm="reactive_mlp_256",
        summaries=summaries(),
        all_low_arms=["reactive_mlp_64"],
    )
    assert report["passed"] is True
    assert report["selected_gain_over_always_low"] == pytest.approx(1.4)


def test_baseline_competence_fails_collapsed_selected_arm() -> None:
    values = summaries()
    values["stochastic"]["reactive_mlp_256"]["both_modes_lifetime_rate"] = 0.0
    report = assess_baseline_competence(
        selected_arm="reactive_mlp_256",
        summaries=values,
        all_low_arms=["reactive_mlp_64"],
    )
    assert report["passed"] is False
