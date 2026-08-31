import sys

from scripts import run_thermal_commitment_pilot


def test_pilot_runner_builds_all_frozen_cells(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(run_thermal_commitment_pilot.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_pilot.py",
            "--output-root",
            str(tmp_path),
            "--seed",
            "4999",
        ],
    )
    run_thermal_commitment_pilot.main()

    assert len(commands) == 4
    observed = set()
    for command, kwargs in commands:
        assert kwargs["check"] is True
        assert "--thermal-commitment" in command
        assert command[command.index("--canonical-task-seed") + 1] == "811"
        assert command[command.index("--commitment-trip-load") + 1] == "0.10"
        assert command[command.index("--commitment-trip-penalty") + 1] == "75.0"
        assert command[command.index("--commitment-high-power-bonus") + 1] == "2.0"
        assert (
            command[command.index("--commitment-control-cost-basis") + 1]
            == "requested_action"
        )
        label = "dynamic" if "endogenous_action" in command else "static"
        memory = command[command.index("--memory-mode") + 1]
        observed.add((label, memory))
    assert observed == {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }
