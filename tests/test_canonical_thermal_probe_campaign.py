import json
import sys

from scripts import run_canonical_thermal_probe_campaign


def test_canonical_probe_manifest_freezes_all_four_cells(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_canonical_thermal_probe_campaign.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--seeds",
            "4000",
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
    run_canonical_thermal_probe_campaign.main()

    assert len(calls) == 4
    for command in calls:
        assert command[command.index("--canonical-task-seed") + 1] == "811"
        assert command[command.index("--thermal-heat-rate") + 1] == "0.1"
        assert command[command.index("--thermal-cooling-rate") + 1] == "0.0"
        assert command[command.index("--thermal-episode-cooling") + 1] == "0.0"
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert manifest["canonical_task_seed"] == 811
    assert manifest["thermal_parameters"]["heat_rate"] == 0.1
    assert {cell["label"] for cell in manifest["cells"]} == {"dynamic", "static"}
