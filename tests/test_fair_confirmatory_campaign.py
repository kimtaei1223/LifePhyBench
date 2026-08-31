import json
import sys

from scripts import run_fair_confirmatory_campaign


def test_default_fixed_dose_is_shared_by_manifest(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_fair_confirmatory_campaign.subprocess,
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
    run_fair_confirmatory_campaign.main()
    assert len(calls) == 4
    doses = []
    for command in calls:
        position = command.index("--thermal-exogenous-dose-per-step")
        doses.append(command[position + 1])
    assert len(set(doses)) == 1
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert manifest["fixed_exogenous_dose"] == float(doses[0])
