import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_hierarchical_v11_calibration.py"


def load_script():
    spec = importlib.util.spec_from_file_location("run_hierarchical_v11_calibration", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cell_names_are_unambiguous():
    module = load_script()
    assert module.expected_cell(
        condition="stochastic",
        policy_arm="lifetime_lstm",
        seed=7300,
        decisions=100_000,
    ) == "v11-stochastic-lifetime_lstm-seed7300-decisions100k"


def test_partial_cell_is_never_silently_skipped(tmp_path):
    module = load_script()
    run = tmp_path / "cell"
    run.mkdir()
    (run / "status.json").write_text('{"status":"training"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial"):
        module.validate_completed_cell(
            run,
            condition="fixed",
            policy_arm="reactive_mlp_64",
            seed=7300,
            decisions=100_000,
            eval_tasks=4_000,
        )


def test_complete_cell_requires_exact_arguments_and_raw_count(tmp_path):
    module = load_script()
    run = tmp_path / "cell"
    run.mkdir()
    arguments = {
        "condition": "fixed",
        "policy_arm": "reactive_mlp_64",
        "seed": 7300,
        "total_task_decisions": 100_000,
        "eval_task_episodes": 2,
    }
    (run / "metadata.json").write_text(
        json.dumps({"arguments": arguments}), encoding="utf-8"
    )
    (run / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (run / "model.zip").write_bytes(b"model")
    (run / "evaluation_tasks.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    assert module.validate_completed_cell(
        run,
        condition="fixed",
        policy_arm="reactive_mlp_64",
        seed=7300,
        decisions=100_000,
        eval_tasks=2,
    )
