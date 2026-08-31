import json
import sys

from scripts import run_thermal_commitment_masked_pilot


def test_masked_pilot_builds_four_decision_masked_cells(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(
        run_thermal_commitment_masked_pilot.subprocess, "run", fake_run
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_masked_pilot.py",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_thermal_commitment_masked_pilot.main()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["optimization"]["decision_only_mode_loss"] is True
    assert manifest["seed"] == 4989
    assert len(commands) == 4
    observed = set()
    for command, kwargs in commands:
        assert kwargs["check"] is True
        assert "--commitment-mask-mode-loss" in command
        assert command[command.index("--commitment-trip-load") + 1] == "0.10"
        assert (
            command[
                command.index("--commitment-curriculum-start-trip-load") + 1
            ]
            == "0.70"
        )
        label = "dynamic" if "endogenous_action" in command else "static"
        memory = command[command.index("--memory-mode") + 1]
        observed.add((label, memory))
    assert observed == {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }


def test_action_history_variant_is_frozen_in_manifest_and_commands(
    monkeypatch, tmp_path
):
    commands = []
    monkeypatch.setattr(
        run_thermal_commitment_masked_pilot.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_masked_pilot.py",
            "--seed",
            "4987",
            "--version-label",
            "v7",
            "--append-previous-applied-action",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_thermal_commitment_masked_pilot.main()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["representation"] == {
        "previous_applied_action_observed": True,
        "privileged_health_exposed": False,
        "zeroed_at_every_task_boundary": True,
    }
    assert manifest["phase"].startswith("thermal_commitment_v7_")
    assert all("--append-previous-applied-action" in command for command in commands)
    assert all("thermal-commitment-v7-" in command[-1] for command in commands)
