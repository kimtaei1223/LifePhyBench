from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_hierarchical_v11_confirmatory import (
    exact_two_sided_sign_p,
    monte_carlo_sign_flip_p,
    summarize_estimand,
)
from scripts.run_hierarchical_v11_confirmatory import (
    build_command,
    expected_run_name,
)


def minimal_protocol() -> dict:
    return {
        "budgets": {
            "workers": 8,
            "total_task_decisions_per_run": 100_000,
            "evaluation_task_episodes": 4_000,
            "device": "cuda",
            "torch_threads_per_process": 1,
        },
        "physics": {
            "episode_steps": 100,
            "tasks_per_lifetime": 20,
            "canonical_task_seed": 811,
            "trip_load": 0.1,
            "low_power_scale": 0.4,
            "trip_penalty": 75.0,
            "high_power_bonus": 2.0,
            "thermal_heat_rate": 0.05,
            "thermal_episode_cooling": 0.15,
            "sensor_noise_sd": 0.01,
            "shock_probability": 0.0005,
            "shock_size": 0.01,
            "conditions": {
                "fixed": {
                    "initial_thermal_load": {"distribution": "constant", "value": 0.04}
                },
                "stochastic": {
                    "initial_thermal_load": {
                        "distribution": "uniform",
                        "low": 0.0,
                        "high": 0.08,
                    }
                },
            },
        },
        "arms": {
            "common_training_hyperparameters": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "n_steps": 64,
                "batch_size": 256,
                "ent_coef": 0.005,
                "training_reward_scale": 0.02,
            }
        },
        "inputs": {"low_level_checkpoint": {"path": "outputs/low/model.zip"}},
    }


def test_confirmatory_command_binds_phase_hash_and_frozen_budget(tmp_path: Path) -> None:
    digest = "a" * 64
    command = build_command(
        trainer=tmp_path / "trainer.py",
        output_root=tmp_path / "out",
        run_name="heldout",
        condition="stochastic",
        arm="reactive_mlp_256",
        seed=8300,
        evaluation_seed=123456,
        protocol=minimal_protocol(),
        protocol_sha256=digest,
    )
    assert command[command.index("--study-phase") + 1] == "confirmatory"
    assert command[command.index("--protocol-sha256") + 1] == digest
    assert command[command.index("--total-task-decisions") + 1] == "100000"
    assert command[command.index("--eval-task-episodes") + 1] == "4000"
    assert command[command.index("--evaluation-seed") + 1] == "123456"


def test_confirmatory_cell_name_is_unambiguous() -> None:
    assert expected_run_name(
        "stochastic", "reactive_mlp_256", 8300, 100_000
    ) == "v11-heldout-stochastic-reactive_mlp_256-seed8300-decisions100k"


def test_sign_flip_and_exact_sign_tests_are_two_sided_and_deterministic() -> None:
    values = [1.0] * 8
    first = monte_carlo_sign_flip_p(values, draws=100_000, seed=760_000)
    second = monte_carlo_sign_flip_p(values, draws=100_000, seed=760_000)
    assert first == second
    assert first == pytest.approx(0.0078125, abs=0.001)
    exact = exact_two_sided_sign_p(values)
    assert exact["two_sided_p"] == pytest.approx(2 / 256)


def test_estimand_requires_all_three_frozen_criteria() -> None:
    report = summarize_estimand(
        [1.0] * 8,
        bootstrap_resamples=10_000,
        bootstrap_seed=1,
        sign_flip_draws=100_000,
        sign_flip_seed=2,
        criteria={
            "mean_reward_per_task_at_least": 0.25,
            "paired_seed_bootstrap_95_ci_lower_above": 0.0,
            "monte_carlo_two_sided_sign_flip_p_below": 0.05,
        },
    )
    assert report["estimand_passed"] is True

    failed = summarize_estimand(
        [0.1] * 8,
        bootstrap_resamples=10_000,
        bootstrap_seed=1,
        sign_flip_draws=100_000,
        sign_flip_seed=2,
        criteria={
            "mean_reward_per_task_at_least": 0.25,
            "paired_seed_bootstrap_95_ci_lower_above": 0.0,
            "monte_carlo_two_sided_sign_flip_p_below": 0.05,
        },
    )
    assert failed["criteria_passed"]["mean_reward_per_task_at_least"] is False
    assert failed["estimand_passed"] is False
