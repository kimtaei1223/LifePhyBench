from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_hierarchical_v11_protocol import (
    CALIBRATION_SEEDS,
    HELDOUT_SEEDS,
    FreezeError,
    freeze_protocol,
)
from scripts.validate_hierarchical_v11_freeze import (
    FreezeValidationError,
    validate_frozen_protocol,
)


def fake_environment() -> dict:
    return {
        "pip_freeze": {
            "lines": ["gymnasium==1.3.0", "torch==2.11.0+cu128"],
            "line_count": 2,
            "sha256": "pip-freeze-test-hash",
        },
        "python": {"version": "3.11.test"},
        "system": {"platform": "test-linux"},
        "cuda": {
            "torch_version": "2.11.0+cu128",
            "torch_cuda_available": True,
            "device_count": 1,
            "devices": [{"index": 0, "name": "test-gpu"}],
        },
        "nvidia_smi": {"returncode": 0, "stdout": "test-gpu, test-driver\n"},
        "determinism_environment": {},
        "canonical_snapshot_sha256": "environment-test-hash",
    }


def fake_git() -> dict:
    return {
        "available": True,
        "commit": "a" * 40,
        "dirty": True,
        "porcelain_v1": ["?? scripts/example.py"],
        "porcelain_sha256": "git-status-test-hash",
        "note": "test snapshot",
    }


def make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    for directory in (
        root / "src" / "pkg",
        root / "scripts",
        root / "tests",
        root / "inputs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "src" / "pkg" / "env.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "scripts" / "runner.py").write_text("print('runner')\n", encoding="utf-8")
    (root / "tests" / "test_env.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "environment.yml").write_text("name: test\n", encoding="utf-8")
    checkpoint = root / "inputs" / "low-level.zip"
    checkpoint.write_bytes(b"frozen-low-level-model")
    qualification = root / "inputs" / "qualification.json"
    qualification.write_text(
        json.dumps(
            {
                "phase": "hierarchical_v11_cpu_and_baseline_qualification",
                "qualification_passed": True,
                "baseline_competence_passed": True,
                "calibration_seeds": list(CALIBRATION_SEEDS),
                "selected_reactive_arm": "reactive_mlp_64",
                "selected_design": {
                    "thermal_episode_cooling": 0.10,
                    "sensor_noise_sd": 0.02,
                    "shock_probability": 0.0005,
                    "shock_size": 0.01,
                },
                "selected_training": {
                    "learning_rate": 0.0003,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "n_steps": 64,
                    "batch_size": 256,
                    "ent_coef": 0.005,
                    "training_reward_scale": 0.02,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, checkpoint, qualification


def freeze_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    root, checkpoint, qualification = make_project(tmp_path)
    output = root / "frozen" / "FROZEN_PROTOCOL.json"
    result = freeze_protocol(
        project_root=root,
        output_path=output,
        low_level_checkpoint=checkpoint,
        qualification_path=qualification,
        environment_snapshot=fake_environment(),
        git_snapshot=fake_git(),
        created_at_utc="2026-08-27T00:00:00+00:00",
    )
    return root, output, result


def test_freeze_contains_full_source_environment_seed_and_primary_plan(tmp_path: Path):
    root, output, result = freeze_fixture(tmp_path)
    document = json.loads(output.read_text(encoding="utf-8"))

    paths = {row["path"] for row in document["source_snapshot"]["files"]}
    assert paths == {
        "src/pkg/env.py",
        "scripts/runner.py",
        "tests/test_env.py",
        "pyproject.toml",
        "environment.yml",
    }
    assert document["source_snapshot"]["file_count"] == 5
    assert document["inputs"]["low_level_checkpoint"]["sha256"]
    assert document["environment_snapshot"] == fake_environment()
    assert document["git_snapshot"]["commit"] == "a" * 40
    assert document["git_snapshot"]["dirty"] is True
    assert document["prospective_specification"]["externally_preregistered"] is False
    assert document["arms"]["task_reactive"] == {
        "identity": "reactive_mlp_64",
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "architecture": "feedforward_mlp",
        "policy_kwargs": {"net_arch": [64, 64]},
        "recurrent_state": False,
        "memory_reset": "not_applicable",
        "selection": "calibration_selected_strong_baseline",
        "selection_data": "calibration seeds 7300--7304 only",
        "reselection_after_any_heldout_result_allowed": False,
        "selection_lock": (
            "After any held-out seed result is generated, the reactive arm must not "
            "be reselected, replaced, or retuned."
        ),
    }
    assert document["arms"]["common_training_hyperparameters"]["n_steps"] == 64
    assert (
        "curriculum_lifetimes"
        not in document["arms"]["common_training_hyperparameters"]
    )
    assert document["seed_namespaces"]["calibration"]["training_pair_seeds"] == list(
        CALIBRATION_SEEDS
    )
    assert document["seed_namespaces"]["heldout"]["training_pair_seeds"] == list(
        HELDOUT_SEEDS
    )
    assert not set(
        document["seed_namespaces"]["calibration"]["evaluation_bank_seeds"]
    ) & set(document["seed_namespaces"]["heldout"]["evaluation_bank_seeds"])
    assert (
        document["primary_analysis"]["conjunction"]["per_estimand"][
            "mean_reward_per_task_at_least"
        ]
        == 0.25
    )
    assert document["primary_analysis"]["sign_flip_randomization"] == {
        "method": "Monte Carlo paired sign-flip randomization test",
        "alternative": "two-sided",
        "draws": 1_000_000,
        "rng_seed": 760_000,
        "p_value_correction": ("Phipson-Smyth add-one: (extreme + 1) / (draws + 1)"),
        "zero_handling": "retain zeros; sign changes leave zero unchanged",
    }
    assert (
        document["primary_analysis"]["secondary_sensitivity"]["method"]
        == "exact paired sign test"
    )

    report = validate_frozen_protocol(
        protocol_path=output,
        project_root=root,
        expected_protocol_sha256=result["file_sha256"],
        environment_snapshot=fake_environment(),
        git_snapshot=fake_git(),
    )
    assert report["valid"] is True
    assert report["heldout_training_seeds"] == list(HELDOUT_SEEDS)


def test_freezer_refuses_to_overwrite_even_with_identical_content(tmp_path: Path):
    root, output, _result = freeze_fixture(tmp_path)
    checkpoint = root / "inputs" / "low-level.zip"
    qualification = root / "inputs" / "qualification.json"
    original = output.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_protocol(
            project_root=root,
            output_path=output,
            low_level_checkpoint=checkpoint,
            qualification_path=qualification,
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
            created_at_utc="2026-08-27T00:00:00+00:00",
        )
    assert output.read_bytes() == original


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_validator_fails_closed_on_source_set_or_content_drift(
    tmp_path: Path, mutation: str
):
    root, output, result = freeze_fixture(tmp_path)
    source = root / "src" / "pkg" / "env.py"
    if mutation == "modify":
        source.write_text("VALUE = 2\n", encoding="utf-8")
    elif mutation == "add":
        (root / "scripts" / "late_change.py").write_text(
            "LATE = True\n", encoding="utf-8"
        )
    else:
        source.unlink()

    with pytest.raises(FreezeValidationError, match="source file set"):
        validate_frozen_protocol(
            protocol_path=output,
            project_root=root,
            expected_protocol_sha256=result["file_sha256"],
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
        )


def test_validator_fails_closed_on_protocol_checkpoint_and_environment_drift(
    tmp_path: Path,
):
    root, output, result = freeze_fixture(tmp_path)

    with pytest.raises(FreezeValidationError, match="protocol file SHA"):
        validate_frozen_protocol(
            protocol_path=output,
            project_root=root,
            expected_protocol_sha256="0" * 64,
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
        )

    (root / "inputs" / "low-level.zip").write_bytes(b"changed checkpoint")
    with pytest.raises(
        FreezeValidationError, match="input drift: low_level_checkpoint"
    ):
        validate_frozen_protocol(
            protocol_path=output,
            project_root=root,
            expected_protocol_sha256=result["file_sha256"],
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
        )

    (root / "inputs" / "low-level.zip").write_bytes(b"frozen-low-level-model")
    changed_environment = copy.deepcopy(fake_environment())
    changed_environment["cuda"]["torch_version"] = "different"
    with pytest.raises(FreezeValidationError, match="environment drift"):
        validate_frozen_protocol(
            protocol_path=output,
            project_root=root,
            expected_protocol_sha256=result["file_sha256"],
            environment_snapshot=changed_environment,
            git_snapshot=fake_git(),
        )


def test_freezer_rejects_unqualified_or_cpu_only_protocol(tmp_path: Path):
    root, checkpoint, qualification = make_project(tmp_path)
    document = json.loads(qualification.read_text(encoding="utf-8"))
    document["baseline_competence_passed"] = False
    qualification.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(FreezeError, match="baseline_competence_passed"):
        freeze_protocol(
            project_root=root,
            output_path=root / "frozen.json",
            low_level_checkpoint=checkpoint,
            qualification_path=qualification,
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
        )

    document["baseline_competence_passed"] = True
    qualification.write_text(json.dumps(document), encoding="utf-8")
    cpu_environment = copy.deepcopy(fake_environment())
    cpu_environment["cuda"]["torch_cuda_available"] = False
    cpu_environment["cuda"]["device_count"] = 0
    with pytest.raises(FreezeError, match="CUDA is unavailable"):
        freeze_protocol(
            project_root=root,
            output_path=root / "frozen.json",
            low_level_checkpoint=checkpoint,
            qualification_path=qualification,
            environment_snapshot=cpu_environment,
            git_snapshot=fake_git(),
        )


@pytest.mark.parametrize(
    ("identity", "algorithm", "policy", "net_arch"),
    [
        ("task_reset_lstm", "RecurrentPPO", "TaskResetMlpLstmPolicy", None),
        ("reactive_mlp_64", "PPO", "MlpPolicy", [64, 64]),
        ("reactive_mlp_256", "PPO", "MlpPolicy", [256, 256]),
    ],
)
def test_freezer_maps_calibration_selected_reactive_arm_to_actual_trainer(
    tmp_path: Path,
    identity: str,
    algorithm: str,
    policy: str,
    net_arch: list[int] | None,
):
    root, checkpoint, qualification = make_project(tmp_path)
    document = json.loads(qualification.read_text(encoding="utf-8"))
    document["selected_reactive_arm"] = identity
    qualification.write_text(json.dumps(document), encoding="utf-8")
    output = root / "frozen.json"
    result = freeze_protocol(
        project_root=root,
        output_path=output,
        low_level_checkpoint=checkpoint,
        qualification_path=qualification,
        environment_snapshot=fake_environment(),
        git_snapshot=fake_git(),
    )

    reactive = result["protocol"]["arms"]["task_reactive"]
    assert reactive["identity"] == identity
    assert reactive["algorithm"] == algorithm
    assert reactive["policy"] == policy
    assert reactive["policy_kwargs"].get("net_arch") == net_arch


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_reactive_arm", "heldout_best", "selected_reactive_arm"),
        ("n_steps", 32, "n_steps must equal"),
    ],
)
def test_freezer_rejects_unknown_reactive_arm_or_trainer_step_mismatch(
    tmp_path: Path, field: str, value: object, message: str
):
    root, checkpoint, qualification = make_project(tmp_path)
    document = json.loads(qualification.read_text(encoding="utf-8"))
    if field == "n_steps":
        document["selected_training"][field] = value
    else:
        document[field] = value
    qualification.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(FreezeError, match=message):
        freeze_protocol(
            project_root=root,
            output_path=root / "frozen.json",
            low_level_checkpoint=checkpoint,
            qualification_path=qualification,
            environment_snapshot=fake_environment(),
            git_snapshot=fake_git(),
        )
