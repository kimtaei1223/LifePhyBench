import sys

from scripts import run_thermal_commitment_curriculum_pilot


def test_curriculum_pilot_builds_four_training_only_curriculum_cells(
    monkeypatch, tmp_path
):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))

    monkeypatch.setattr(
        run_thermal_commitment_curriculum_pilot.subprocess, "run", fake_run
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_thermal_commitment_curriculum_pilot.py",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_thermal_commitment_curriculum_pilot.main()

    assert len(commands) == 4
    observed = set()
    for command, kwargs in commands:
        assert kwargs["check"] is True
        assert command[command.index("--commitment-trip-load") + 1] == "0.10"
        assert (
            command[
                command.index("--commitment-curriculum-start-trip-load") + 1
            ]
            == "0.7"
        )
        assert (
            command[command.index("--commitment-curriculum-lifetimes") + 1]
            == "10"
        )
        label = "dynamic" if "endogenous_action" in command else "static"
        memory = command[command.index("--memory-mode") + 1]
        observed.add((label, memory))
    assert observed == {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }
