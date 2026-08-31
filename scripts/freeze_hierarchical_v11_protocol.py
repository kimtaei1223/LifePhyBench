"""Create an immutable-by-convention v11 confirmatory protocol bundle.

The freezer intentionally has no experiment-running side effects.  It records
the complete Python source set, the two project environment specifications,
the selected low-level checkpoint and qualification document, and a runtime
environment fingerprint.  The companion validator must pass before a held-out
runner is allowed to start.

This is a *local prospective specification*.  A SHA-256 digest binds content,
but it does not establish a trusted time unless that digest is independently
timestamped outside the workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STUDY_ID = "lifephybench-hierarchical-thermal-v11"
CALIBRATION_SEEDS = tuple(range(7_300, 7_305))
HELDOUT_SEEDS = tuple(range(8_300, 8_330))
SOURCE_ROOTS = ("src", "scripts", "tests")
ROOT_SOURCE_FILES = ("pyproject.toml", "environment.yml")
DESIGN_KEYS = (
    "thermal_episode_cooling",
    "sensor_noise_sd",
    "shock_probability",
    "shock_size",
)
TRAINING_KEYS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "n_steps",
    "batch_size",
    "ent_coef",
    "training_reward_scale",
)
REACTIVE_ARM_SPECS: dict[str, dict[str, Any]] = {
    "task_reset_lstm": {
        "algorithm": "RecurrentPPO",
        "policy": "TaskResetMlpLstmPolicy",
        "architecture": "task_reset_lstm",
        "policy_kwargs": {},
        "recurrent_state": True,
        "memory_reset": "task_boundary",
    },
    "reactive_mlp_64": {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "architecture": "feedforward_mlp",
        "policy_kwargs": {"net_arch": [64, 64]},
        "recurrent_state": False,
        "memory_reset": "not_applicable",
    },
    "reactive_mlp_256": {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "architecture": "feedforward_mlp",
        "policy_kwargs": {"net_arch": [256, 256]},
        "recurrent_state": False,
        "memory_reset": "not_applicable",
    },
}
ALLOWED_COOLING = frozenset({0.05, 0.10, 0.15})
ALLOWED_SENSOR_NOISE = frozenset({0.01, 0.02, 0.03})
ALLOWED_SHOCKS = frozenset(
    {
        (0.0005, 0.01),
        (0.001, 0.01),
        (0.0005, 0.02),
    }
)
LOCAL_SPECIFICATION_KO = (
    "프로토콜과 분석계획은 첫 v11 확증 학습 전에 로컬에서 고정하고 "
    "SHA-256으로 결합했다. 제3자 타임스탬프나 등록 기록을 확보하지 "
    "않았으므로 이는 외부 검증 가능한 사전등록이 아니라 로컬 사전명세이며, "
    "시간 순서는 로컬 산출물과 실행 로그로만 뒷받침된다."
)
LOCAL_SPECIFICATION_EN = (
    "The protocol and analysis plan were locally frozen and SHA-256-bound "
    "before the first v11 confirmatory training run. Because no third-party "
    "timestamp or registry record was obtained, this is a local prospective "
    "specification, not an externally verifiable preregistration; the temporal "
    "ordering is supported only by local artifacts and logs."
)


class FreezeError(ValueError):
    """Raised when an input is not eligible for confirmatory freezing."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical representation used for all aggregate hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, project_root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = project_root.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise FreezeError(f"input must be inside project root: {resolved}") from error
    if path.is_symlink() or not resolved.is_file():
        raise FreezeError(f"input must be a regular non-symlink file: {path}")
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _source_paths(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in SOURCE_ROOTS:
        source_root = project_root / root_name
        if not source_root.is_dir():
            raise FreezeError(f"required source root is missing: {source_root}")
        paths.extend(source_root.rglob("*.py"))
    for name in ROOT_SOURCE_FILES:
        path = project_root / name
        if not path.is_file():
            raise FreezeError(f"required project specification is missing: {path}")
        paths.append(path)
    unique = {path.resolve(): path for path in paths}
    return sorted(
        unique.values(), key=lambda path: path.relative_to(project_root).as_posix()
    )


def build_source_snapshot(project_root: Path) -> dict[str, Any]:
    """Hash the complete declared source set, including additions and deletions."""

    root = project_root.resolve(strict=True)
    records = [file_record(path, root) for path in _source_paths(root)]
    if not records:
        raise FreezeError("source snapshot is empty")
    return {
        "selection": {
            "python_roots": list(SOURCE_ROOTS),
            "python_pattern": "**/*.py",
            "project_specifications": list(ROOT_SOURCE_FILES),
            "symlinks_allowed": False,
        },
        "files": records,
        "file_count": len(records),
        "canonical_tree_sha256": sha256_bytes(canonical_json_bytes(records)),
    }


def derive_domain_seed(domain: str, pair_seed: int) -> int:
    """Derive a 32-bit seed without reusing a training RNG namespace."""

    material = f"{STUDY_ID}|{domain}|{pair_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def build_seed_namespaces() -> dict[str, Any]:
    calibration_eval = [
        derive_domain_seed("calibration-evaluation-bank", seed)
        for seed in CALIBRATION_SEEDS
    ]
    heldout_eval = [
        derive_domain_seed("heldout-evaluation-bank", seed) for seed in HELDOUT_SEEDS
    ]
    analysis = {
        "bootstrap-interaction": derive_domain_seed(
            "analysis-bootstrap-interaction", 0
        ),
        "bootstrap-stochastic": derive_domain_seed("analysis-bootstrap-stochastic", 0),
        "monte-carlo-sign-flip": 760_000,
    }
    all_values = [
        *CALIBRATION_SEEDS,
        *HELDOUT_SEEDS,
        *calibration_eval,
        *heldout_eval,
        *analysis.values(),
    ]
    if len(all_values) != len(set(all_values)):
        raise FreezeError("derived seed namespaces collided")
    return {
        "derivation": {
            "algorithm": "SHA-256 first four bytes, unsigned big-endian",
            "material": f"{STUDY_ID}|<domain>|<pair_seed>",
            "range": "uint32",
        },
        "calibration": {
            "training_pair_seeds": list(CALIBRATION_SEEDS),
            "evaluation_bank_seeds": calibration_eval,
            "domain": "calibration-evaluation-bank",
        },
        "heldout": {
            "training_pair_seeds": list(HELDOUT_SEEDS),
            "evaluation_bank_seeds": heldout_eval,
            "domain": "heldout-evaluation-bank",
            "untouched_before_freeze": True,
        },
        "analysis_rng": analysis,
        "pairing_rule": (
            "Within a pair, both arms use the same realized task/heat schedule; "
            "training, evaluation, and analysis namespaces never share RNG state."
        ),
    }


def _run_capture(command: list[str], *, cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "command": command,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def collect_environment_snapshot(project_root: Path) -> dict[str, Any]:
    """Capture deterministic package, system, and CUDA identity fields."""

    pip_result = _run_capture(
        [sys.executable, "-m", "pip", "freeze", "--all"], cwd=project_root
    )
    if not pip_result["available"] or pip_result["returncode"] != 0:
        raise FreezeError(f"pip freeze failed: {pip_result['stderr'].strip()}")
    pip_lines = sorted(
        line.strip() for line in pip_result["stdout"].splitlines() if line.strip()
    )
    pip_text = "\n".join(pip_lines) + "\n"

    cuda: dict[str, Any]
    try:
        import torch

        available = bool(torch.cuda.is_available())
        devices = []
        if available:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "compute_capability": [properties.major, properties.minor],
                        "total_memory_bytes": properties.total_memory,
                    }
                )
        cuda = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "torch_cuda_available": available,
            "device_count": torch.cuda.device_count() if available else 0,
            "devices": devices,
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
    except (ImportError, RuntimeError) as error:
        cuda = {
            "torch_import_error": f"{type(error).__name__}: {error}",
            "torch_cuda_available": False,
            "device_count": 0,
            "devices": [],
        }
    nvidia_smi = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        cwd=project_root,
    )
    snapshot = {
        "pip_freeze": {
            "lines": pip_lines,
            "line_count": len(pip_lines),
            "sha256": sha256_bytes(pip_text.encode("utf-8")),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "cuda": cuda,
        "nvidia_smi": nvidia_smi,
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    snapshot["canonical_snapshot_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    return snapshot


def collect_git_snapshot(project_root: Path) -> dict[str, Any]:
    commit = _run_capture(["git", "rev-parse", "HEAD"], cwd=project_root)
    status = _run_capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
    )
    available = bool(
        commit["available"]
        and commit["returncode"] == 0
        and status["available"]
        and status["returncode"] == 0
    )
    status_lines = status["stdout"].splitlines() if available else []
    return {
        "available": available,
        "commit": commit["stdout"].strip() if available else None,
        "dirty": bool(status_lines) if available else None,
        "porcelain_v1": status_lines,
        "porcelain_sha256": sha256_bytes(
            ("\n".join(status_lines) + ("\n" if status_lines else "")).encode("utf-8")
        ),
        "note": (
            "The content manifest, not Git cleanliness, is authoritative for this "
            "freeze. A commit hash alone is not an external timestamp."
        ),
    }


def _finite_number(document: dict[str, Any], name: str) -> float:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeError(f"qualification {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FreezeError(f"qualification {name} must be finite")
    return number


def read_qualification(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot read qualification JSON: {path}") from error
    if document.get("qualification_passed") is not True:
        raise FreezeError("qualification_passed must be true")
    if document.get("baseline_competence_passed") is not True:
        raise FreezeError("baseline_competence_passed must be true")
    if document.get("calibration_seeds") != list(CALIBRATION_SEEDS):
        raise FreezeError("qualification calibration seeds do not match 7300--7304")

    selected_design = document.get("selected_design")
    if not isinstance(selected_design, dict) or set(selected_design) != set(
        DESIGN_KEYS
    ):
        raise FreezeError(f"selected_design must contain exactly {DESIGN_KEYS}")
    design = {key: _finite_number(selected_design, key) for key in DESIGN_KEYS}
    if design["thermal_episode_cooling"] not in ALLOWED_COOLING:
        raise FreezeError("selected cooling is outside the declared CPU-gate grid")
    if design["sensor_noise_sd"] not in ALLOWED_SENSOR_NOISE:
        raise FreezeError("selected sensor noise is outside the declared CPU-gate grid")
    shock = (design["shock_probability"], design["shock_size"])
    if shock not in ALLOWED_SHOCKS:
        raise FreezeError("selected shock pair is outside the declared CPU-gate grid")

    selected_training = document.get("selected_training")
    if not isinstance(selected_training, dict) or set(selected_training) != set(
        TRAINING_KEYS
    ):
        raise FreezeError(f"selected_training must contain exactly {TRAINING_KEYS}")
    training = {key: _finite_number(selected_training, key) for key in TRAINING_KEYS}
    for key in ("n_steps", "batch_size"):
        if not training[key].is_integer() or training[key] <= 0:
            raise FreezeError(f"selected_training {key} must be a positive integer")
        training[key] = int(training[key])
    if training["n_steps"] != 64:
        raise FreezeError(
            "selected_training n_steps must equal the v11 trainer value 64"
        )
    if not 0.0 < training["learning_rate"]:
        raise FreezeError("learning_rate must be positive")
    if not 0.0 < training["gamma"] <= 1.0:
        raise FreezeError("gamma must be in (0, 1]")
    if not 0.0 < training["gae_lambda"] <= 1.0:
        raise FreezeError("gae_lambda must be in (0, 1]")
    if training["ent_coef"] < 0.0 or training["training_reward_scale"] <= 0.0:
        raise FreezeError("entropy must be non-negative and reward scale positive")
    selected_reactive_arm = document.get("selected_reactive_arm")
    if selected_reactive_arm not in REACTIVE_ARM_SPECS:
        raise FreezeError(
            f"selected_reactive_arm must be one of {tuple(REACTIVE_ARM_SPECS)}"
        )
    return {
        "document": document,
        "design": design,
        "training": training,
        "reactive_arm": selected_reactive_arm,
    }


def _ensure_cuda_ready(environment: dict[str, Any]) -> None:
    cuda = environment.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("torch_cuda_available") is not True:
        raise FreezeError("CUDA is unavailable; refuse a CUDA confirmatory freeze")
    if not isinstance(cuda.get("device_count"), int) or cuda["device_count"] < 1:
        raise FreezeError("CUDA device_count must be at least one")
    nvidia_smi = environment.get("nvidia_smi")
    if not isinstance(nvidia_smi, dict) or nvidia_smi.get("returncode") != 0:
        raise FreezeError("nvidia-smi preflight did not succeed")


def _protocol_payload(
    *,
    project_root: Path,
    low_level_checkpoint: Path,
    qualification_path: Path,
    environment_snapshot: dict[str, Any],
    git_snapshot: dict[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    qualification = read_qualification(qualification_path)
    _ensure_cuda_ready(environment_snapshot)
    low_level = file_record(low_level_checkpoint, project_root)
    qualification_record = file_record(qualification_path, project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "phase": "hierarchical_thermal_v11_locally_frozen_protocol",
        "status": "ready_for_confirmatory_runner_preflight",
        "created_at_utc_local_clock": created_at_utc,
        "prospective_specification": {
            "externally_timestamped": False,
            "externally_preregistered": False,
            "wording_ko": LOCAL_SPECIFICATION_KO,
            "wording_en": LOCAL_SPECIFICATION_EN,
        },
        "physics": {
            "environment_id": "Pusher-v5",
            "canonical_task_seed": 811,
            "thermal_heat_rate": 0.05,
            "trip_load": 0.10,
            "low_power_scale": 0.40,
            "trip_penalty": 75.0,
            "high_power_bonus": 2.0,
            "episode_steps": 100,
            "tasks_per_lifetime": 20,
            **qualification["design"],
            "conditions": {
                "fixed": {
                    "initial_thermal_load": {"distribution": "constant", "value": 0.04},
                    "shock_probability": 0.0,
                    "shock_size": 0.0,
                },
                "stochastic": {
                    "initial_thermal_load": {
                        "distribution": "uniform",
                        "low": 0.0,
                        "high": 0.08,
                    },
                    "shock_probability": qualification["design"]["shock_probability"],
                    "shock_size": qualification["design"]["shock_size"],
                },
            },
        },
        "observation": {
            "policy_summary_exact_order": [
                "previous_mode",
                "noisy_thermal_sensor",
                "normalized_task_index",
                "previous_trip",
            ],
            "privileged_exact_thermal_load": "info_only_not_policy_observation",
            "task_boundary_observed": True,
        },
        "arms": {
            "lifetime": {
                "algorithm": "RecurrentPPO",
                "memory_reset": "lifetime_boundary_only",
            },
            "task_reactive": {
                "identity": qualification["reactive_arm"],
                **REACTIVE_ARM_SPECS[qualification["reactive_arm"]],
                "selection": "calibration_selected_strong_baseline",
                "selection_data": "calibration seeds 7300--7304 only",
                "reselection_after_any_heldout_result_allowed": False,
                "selection_lock": (
                    "After any held-out seed result is generated, the reactive arm "
                    "must not be reselected, replaced, or retuned."
                ),
            },
            "common_training_hyperparameters": qualification["training"],
            "same_capacity_where_applicable": True,
            "same_training_pair_seed_and_schedule": True,
        },
        "budgets": {
            "total_task_decisions_per_run": 100_000,
            "workers": 8,
            "torch_threads_per_process": 1,
            "evaluation_task_episodes": 4_000,
            "evaluation_complete_lifetimes": 200,
            "device": "cuda",
            "deterministic_evaluation": True,
        },
        "seed_namespaces": build_seed_namespaces(),
        "primary_analysis": {
            "unit": "independent paired training seed",
            "arm_contrast": "lifetime_minus_task_reactive",
            "co_primary_estimands": {
                "stochastic_superiority": "(lifetime - task_reactive)_stochastic",
                "inference_specificity_interaction": (
                    "(lifetime - task_reactive)_stochastic - "
                    "(lifetime - task_reactive)_fixed"
                ),
            },
            "conjunction": {
                "rule": "all co-primary criteria must pass for both estimands",
                "per_estimand": {
                    "mean_reward_per_task_at_least": 0.25,
                    "paired_seed_bootstrap_95_ci_lower_above": 0.0,
                    "monte_carlo_two_sided_sign_flip_p_below": 0.05,
                },
                "failure_interpretation": (
                    "If either co-primary estimand fails any criterion, the central "
                    "v11 claim is not confirmed."
                ),
            },
            "bootstrap": {
                "resamples": 100_000,
                "unit": "paired training-seed contrast",
                "interval": "percentile_95_percent",
                "rng_seed_key": "bootstrap-interaction or bootstrap-stochastic",
            },
            "sign_flip_randomization": {
                "method": "Monte Carlo paired sign-flip randomization test",
                "alternative": "two-sided",
                "draws": 1_000_000,
                "rng_seed": 760_000,
                "p_value_correction": (
                    "Phipson-Smyth add-one: (extreme + 1) / (draws + 1)"
                ),
                "zero_handling": "retain zeros; sign changes leave zero unchanged",
            },
            "secondary_sensitivity": {
                "method": "exact paired sign test",
                "alternative": "two-sided",
                "zero_handling": "discard exact zero differences from the binomial count",
                "confirmatory_conjunction_component": False,
            },
            "raw_rows_required": True,
            "episode_rows_are_not_independent_units": True,
        },
        "missingness_and_rerun": {
            "exclude_failed_or_unfavorable_seeds": False,
            "replacement_seeds_allowed": False,
            "partial_directory_policy": "fail_closed_no_implicit_skip",
            "infrastructure_failure": (
                "retain failure log; resume only from a hash-validated checkpoint or "
                "rerun the same frozen seed from scratch under an explicit audit record"
            ),
            "scientific_null_result": (
                "write a complete negative result and exit normally; do not treat a "
                "failed hypothesis as an integrity error"
            ),
            "post_freeze_change": "preserve original and issue a labeled amendment",
        },
        "inputs": {
            "low_level_checkpoint": low_level,
            "qualification": qualification_record,
            "qualification_phase": qualification["document"].get("phase"),
        },
        "source_snapshot": build_source_snapshot(project_root),
        "environment_snapshot": environment_snapshot,
        "git_snapshot": git_snapshot,
        "runner_preflight": {
            "require_exact_protocol_file_sha256": True,
            "require_source_file_set_and_hash_match": True,
            "require_environment_snapshot_match": True,
            "require_low_level_checkpoint_hash_match": True,
            "require_qualification_hash_match": True,
            "require_git_commit_match_when_available": True,
            "require_cuda_without_cpu_fallback": True,
            "require_exclusive_campaign_lock": True,
            "require_new_empty_output_root": True,
        },
    }


def atomic_write_new(path: Path, data: bytes) -> None:
    """Publish complete bytes atomically and never replace an existing target."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite frozen protocol: {path}"
            ) from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_protocol(
    *,
    project_root: Path,
    output_path: Path,
    low_level_checkpoint: Path,
    qualification_path: Path,
    environment_snapshot: dict[str, Any] | None = None,
    git_snapshot: dict[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    environment = environment_snapshot or collect_environment_snapshot(root)
    git = git_snapshot or collect_git_snapshot(root)
    created = created_at_utc or datetime.now(timezone.utc).isoformat()
    payload = _protocol_payload(
        project_root=root,
        low_level_checkpoint=low_level_checkpoint,
        qualification_path=qualification_path,
        environment_snapshot=environment,
        git_snapshot=git,
        created_at_utc=created,
    )
    payload["integrity"] = {
        "canonical_payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "scope": "entire document except this integrity object",
    }
    serialized = (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_new(output_path, serialized)
    return {
        "protocol": payload,
        "path": str(output_path.resolve()),
        "file_sha256": sha256_bytes(serialized),
        "bytes": len(serialized),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the locally prospective hierarchical thermal v11 protocol."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--low-level-checkpoint", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)

    def resolve_input(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    output_path = resolve_input(args.output)
    digest_path = output_path.with_suffix(output_path.suffix + ".sha256")
    if digest_path.exists():
        raise FileExistsError(f"refusing to overwrite detached digest: {digest_path}")
    result = freeze_protocol(
        project_root=root,
        output_path=output_path,
        low_level_checkpoint=resolve_input(args.low_level_checkpoint),
        qualification_path=resolve_input(args.qualification),
    )
    atomic_write_new(
        digest_path,
        f"{result['file_sha256']}  {output_path.name}\n".encode(),
    )
    result["digest_path"] = str(digest_path.resolve())
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "protocol"}, indent=2
        )
    )
    print(
        "[LOCAL PROSPECTIVE SPECIFICATION FROZEN; NOT EXTERNALLY PREREGISTERED]",
        flush=True,
    )


if __name__ == "__main__":
    main()
