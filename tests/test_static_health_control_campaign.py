import json
import sys

from scripts import run_static_health_control_campaign


def test_static_health_manifest_and_commands_share_zero_dose(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_static_health_control_campaign.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--seeds",
            "3000",
            "--total-timesteps",
            "1",
            "--eval-task-episodes",
            "1",
            "--device",
            "cpu",
            "--output-root",
            str(tmp_path),
        ],
    )
    run_static_health_control_campaign.main()

    assert len(calls) == 2
    assert {command[command.index("--memory-mode") + 1] for command in calls} == {
        "task",
        "lifetime",
    }
    assert {
        command[command.index("--thermal-exogenous-dose-per-step") + 1]
        for command in calls
    } == {"0.0"}
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    semantics = manifest["controlled_semantics"]
    assert semantics["thermal_exogenous_dose_per_step"] == 0.0
    assert semantics["expected_thermal_load"] == 0.0
    assert semantics["expected_actuator_efficiency"] == 1.0
