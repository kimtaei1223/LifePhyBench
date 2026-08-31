import json
import sys

from scripts import run_thermal_commitment_v5_long_stress
from scripts import validate_thermal_commitment_v5_long_stress


def test_long_stress_runner_freezes_eight_dynamic_cells(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(
        run_thermal_commitment_v5_long_stress.subprocess, "run", fake_run
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_v5_long_stress.py",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_thermal_commitment_v5_long_stress.main()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["seeds"] == [4990, 4991, 4992, 4993]
    assert manifest["total_timesteps"] == 2_000_000
    assert manifest["eval_task_episodes"] == 1_000
    assert manifest["frozen_success_rule"]["minimum_passing_lifetime_seeds"] == 3
    assert len(commands) == 8
    observed = set()
    for command, kwargs in commands:
        assert kwargs["check"] is True
        assert command[command.index("--degradation-mode") + 1] == "endogenous_action"
        assert command[command.index("--total-timesteps") + 1] == "2000000"
        assert command[command.index("--eval-task-episodes") + 1] == "1000"
        assert command[command.index("--commitment-trip-load") + 1] == "0.10"
        assert (
            command[
                command.index("--commitment-curriculum-start-trip-load") + 1
            ]
            == "0.70"
        )
        observed.add(
            (
                int(command[command.index("--seed") + 1]),
                command[command.index("--memory-mode") + 1],
            )
        )
    assert observed == {
        (seed, memory)
        for seed in (4990, 4991, 4992, 4993)
        for memory in ("task", "lifetime")
    }


def test_long_stress_validator_applies_three_of_four_frozen_rule(
    monkeypatch, tmp_path
):
    commands = []
    monkeypatch.setattr(
        run_thermal_commitment_v5_long_stress.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_v5_long_stress.py",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_thermal_commitment_v5_long_stress.main()

    for command in commands:
        seed = int(command[command.index("--seed") + 1])
        memory = command[command.index("--memory-mode") + 1]
        run_name = command[command.index("--run-name") + 1]
        run_directory = tmp_path / run_name
        run_directory.mkdir()
        (run_directory / "model.zip").write_bytes(b"model")
        lifetime = memory == "lifetime"
        passing_lifetime = lifetime and seed != 4993
        evaluation = {
            "mean_task_episode_reward": -30.0 if lifetime else -40.0,
            "high_power_selection_rate": 0.5 if passing_lifetime else 0.0,
            "cold_high_power_selection_rate": (
                0.8 if passing_lifetime else 0.0
            ),
            "hot_high_power_selection_rate": (
                0.2 if passing_lifetime else None
            ),
            "cold_mode_selections": 500,
            "hot_mode_selections": 500 if passing_lifetime else 0,
            "thermal_trip_rate": 0.1 if passing_lifetime else 0.0,
        }
        metadata = {
            "arguments": {
                "seed": seed,
                "memory_mode": memory,
                "thermal_commitment": True,
                "degradation_mode": "endogenous_action",
                "total_timesteps": 2_000_000,
                "eval_task_episodes": 1_000,
                "commitment_trip_load": 0.10,
                "commitment_curriculum_start_trip_load": 0.70,
                "commitment_curriculum_lifetimes": 10,
            },
            "controlled_semantics": {
                "training_trip_load_curriculum_only": True,
                "evaluation_trip_load": 0.10,
            },
            "task_episode_evaluation": evaluation,
        }
        (run_directory / "metadata.json").write_text(json.dumps(metadata))

    output = tmp_path / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_thermal_commitment_v5_long_stress.py",
            "--input-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    validate_thermal_commitment_v5_long_stress.main()
    report = json.loads(output.read_text())
    assert report["wiring_passed"] is True
    assert report["behavior_passed"] is True
    assert report["passing_lifetime_seeds"] == 3
    assert report["passed"] is True
