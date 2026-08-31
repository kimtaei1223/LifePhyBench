"""Fail-closed validator for a frozen hierarchical thermal v11 protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from scripts.freeze_hierarchical_v11_protocol import (
        CALIBRATION_SEEDS,
        HELDOUT_SEEDS,
        LOCAL_SPECIFICATION_EN,
        LOCAL_SPECIFICATION_KO,
        REACTIVE_ARM_SPECS,
        SCHEMA_VERSION,
        STUDY_ID,
        TRAINING_KEYS,
        FreezeError,
        build_seed_namespaces,
        build_source_snapshot,
        canonical_json_bytes,
        collect_environment_snapshot,
        collect_git_snapshot,
        file_record,
        read_qualification,
        sha256_bytes,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from freeze_hierarchical_v11_protocol import (  # type: ignore[no-redef]
        CALIBRATION_SEEDS,
        HELDOUT_SEEDS,
        LOCAL_SPECIFICATION_EN,
        LOCAL_SPECIFICATION_KO,
        REACTIVE_ARM_SPECS,
        SCHEMA_VERSION,
        STUDY_ID,
        TRAINING_KEYS,
        FreezeError,
        build_seed_namespaces,
        build_source_snapshot,
        canonical_json_bytes,
        collect_environment_snapshot,
        collect_git_snapshot,
        file_record,
        read_qualification,
        sha256_bytes,
    )


class FreezeValidationError(ValueError):
    """Raised on any mismatch that must block a held-out runner."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeValidationError(message)


def _canonical_without_integrity(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("integrity", None)
    return sha256_bytes(canonical_json_bytes(payload))


def validate_protocol_semantics(document: dict[str, Any]) -> None:
    require(document.get("schema_version") == SCHEMA_VERSION, "schema version mismatch")
    require(document.get("study_id") == STUDY_ID, "study id mismatch")
    require(
        document.get("phase") == "hierarchical_thermal_v11_locally_frozen_protocol",
        "phase mismatch",
    )
    require(
        document.get("status") == "ready_for_confirmatory_runner_preflight",
        "protocol is not ready for runner preflight",
    )
    integrity = document.get("integrity")
    require(isinstance(integrity, dict), "integrity object missing")
    require(
        integrity.get("canonical_payload_sha256")
        == _canonical_without_integrity(document),
        "canonical protocol payload hash mismatch",
    )
    prospective = document.get("prospective_specification", {})
    require(
        prospective.get("externally_timestamped") is False
        and prospective.get("externally_preregistered") is False,
        "unsupported external registration claim",
    )
    require(
        prospective.get("wording_ko") == LOCAL_SPECIFICATION_KO,
        "Korean scope wording drift",
    )
    require(
        prospective.get("wording_en") == LOCAL_SPECIFICATION_EN,
        "English scope wording drift",
    )

    physics = document.get("physics", {})
    required_physics = {
        "environment_id": "Pusher-v5",
        "canonical_task_seed": 811,
        "thermal_heat_rate": 0.05,
        "trip_load": 0.10,
        "low_power_scale": 0.40,
        "trip_penalty": 75.0,
        "high_power_bonus": 2.0,
        "episode_steps": 100,
        "tasks_per_lifetime": 20,
    }
    for key, expected in required_physics.items():
        require(physics.get(key) == expected, f"frozen physics mismatch: {key}")
    require(
        physics.get("conditions", {}).get("fixed")
        == {
            "initial_thermal_load": {"distribution": "constant", "value": 0.04},
            "shock_probability": 0.0,
            "shock_size": 0.0,
        },
        "fixed condition mismatch",
    )
    stochastic = physics.get("conditions", {}).get("stochastic", {})
    require(
        stochastic.get("initial_thermal_load")
        == {"distribution": "uniform", "low": 0.0, "high": 0.08},
        "stochastic initial-load distribution mismatch",
    )
    require(
        stochastic.get("shock_probability") == physics.get("shock_probability")
        and stochastic.get("shock_size") == physics.get("shock_size"),
        "stochastic shock wiring mismatch",
    )
    require(
        document.get("observation", {}).get("policy_summary_exact_order")
        == [
            "previous_mode",
            "noisy_thermal_sensor",
            "normalized_task_index",
            "previous_trip",
        ],
        "policy observation order mismatch",
    )
    arms = document.get("arms", {})
    require(
        arms.get("lifetime", {}).get("algorithm") == "RecurrentPPO",
        "lifetime arm mismatch",
    )
    reactive = arms.get("task_reactive", {})
    reactive_identity = reactive.get("identity")
    require(reactive_identity in REACTIVE_ARM_SPECS, "task-reactive identity mismatch")
    require(
        {key: reactive.get(key) for key in REACTIVE_ARM_SPECS[reactive_identity]}
        == REACTIVE_ARM_SPECS[reactive_identity],
        "task-reactive algorithm or architecture mismatch",
    )
    require(
        reactive.get("selection_data") == "calibration seeds 7300--7304 only"
        and reactive.get("reselection_after_any_heldout_result_allowed") is False
        and reactive.get("selection_lock")
        == (
            "After any held-out seed result is generated, the reactive arm must not "
            "be reselected, replaced, or retuned."
        ),
        "held-out reactive-arm selection lock mismatch",
    )
    common_training = arms.get("common_training_hyperparameters")
    require(
        isinstance(common_training, dict)
        and set(common_training) == set(TRAINING_KEYS),
        "common training schema mismatch",
    )
    require(common_training.get("n_steps") == 64, "v11 n_steps must equal 64")
    budgets = document.get("budgets", {})
    require(
        budgets
        == {
            "total_task_decisions_per_run": 100_000,
            "workers": 8,
            "torch_threads_per_process": 1,
            "evaluation_task_episodes": 4_000,
            "evaluation_complete_lifetimes": 200,
            "device": "cuda",
            "deterministic_evaluation": True,
        },
        "budget mismatch",
    )
    require(
        document.get("seed_namespaces") == build_seed_namespaces(),
        "seed namespace mismatch",
    )
    seed_document = document["seed_namespaces"]
    require(
        seed_document["calibration"]["training_pair_seeds"] == list(CALIBRATION_SEEDS),
        "calibration seeds mismatch",
    )
    require(
        seed_document["heldout"]["training_pair_seeds"] == list(HELDOUT_SEEDS),
        "held-out seeds mismatch",
    )
    primary = document.get("primary_analysis", {})
    require(
        set(primary.get("co_primary_estimands", {}))
        == {"stochastic_superiority", "inference_specificity_interaction"},
        "co-primary estimands mismatch",
    )
    require(
        primary.get("conjunction", {}).get("per_estimand")
        == {
            "mean_reward_per_task_at_least": 0.25,
            "paired_seed_bootstrap_95_ci_lower_above": 0.0,
            "monte_carlo_two_sided_sign_flip_p_below": 0.05,
        },
        "primary conjunction mismatch",
    )
    require(
        primary.get("bootstrap", {}).get("unit") == "paired training-seed contrast"
        and primary.get("sign_flip_randomization", {}).get("alternative") == "two-sided"
        and primary.get("sign_flip_randomization", {}).get("draws") == 1_000_000
        and primary.get("sign_flip_randomization", {}).get("rng_seed") == 760_000
        and primary.get("sign_flip_randomization", {}).get("p_value_correction")
        == "Phipson-Smyth add-one: (extreme + 1) / (draws + 1)",
        "statistical unit or alternative mismatch",
    )
    require(
        primary.get("secondary_sensitivity", {}).get("method")
        == "exact paired sign test"
        and primary.get("secondary_sensitivity", {}).get(
            "confirmatory_conjunction_component"
        )
        is False,
        "secondary exact sign-test specification mismatch",
    )
    missing = document.get("missingness_and_rerun", {})
    require(
        missing.get("exclude_failed_or_unfavorable_seeds") is False,
        "seed exclusion enabled",
    )
    require(
        missing.get("replacement_seeds_allowed") is False, "replacement seeds enabled"
    )
    require(
        missing.get("partial_directory_policy") == "fail_closed_no_implicit_skip",
        "partial-run policy mismatch",
    )


def validate_frozen_protocol(
    *,
    protocol_path: Path,
    project_root: Path,
    expected_protocol_sha256: str | None,
    environment_snapshot: dict[str, Any] | None = None,
    git_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol_bytes = protocol_path.read_bytes()
    actual_protocol_sha256 = sha256_bytes(protocol_bytes)
    if expected_protocol_sha256 is not None:
        require(
            actual_protocol_sha256 == expected_protocol_sha256.lower(),
            "frozen protocol file SHA-256 mismatch",
        )
    try:
        document = json.loads(protocol_bytes)
    except json.JSONDecodeError as error:
        raise FreezeValidationError("frozen protocol is not valid JSON") from error
    validate_protocol_semantics(document)

    root = project_root.resolve(strict=True)
    try:
        current_source = build_source_snapshot(root)
    except FreezeError as error:
        raise FreezeValidationError(str(error)) from error
    require(
        current_source == document.get("source_snapshot"),
        "source file set, size, hash, or canonical tree hash drift",
    )
    inputs = document.get("inputs", {})
    for key in ("low_level_checkpoint", "qualification"):
        frozen_record = inputs.get(key)
        require(isinstance(frozen_record, dict), f"missing input record: {key}")
        candidate = root / frozen_record.get("path", "")
        try:
            current_record = file_record(candidate, root)
        except (FreezeError, OSError) as error:
            raise FreezeValidationError(
                f"cannot verify input {key}: {error}"
            ) from error
        require(current_record == frozen_record, f"input drift: {key}")

    qualification_path = root / inputs["qualification"]["path"]
    try:
        qualification = read_qualification(qualification_path)
    except FreezeError as error:
        raise FreezeValidationError(
            f"qualification no longer passes: {error}"
        ) from error
    require(
        document["arms"]["task_reactive"]["identity"] == qualification["reactive_arm"],
        "reactive-arm identity differs from qualification",
    )
    require(
        document["arms"]["common_training_hyperparameters"]
        == qualification["training"],
        "training hyperparameters differ from qualification",
    )
    for key, value in qualification["design"].items():
        require(
            document["physics"].get(key) == value,
            f"physics differs from qualification: {key}",
        )

    current_environment = environment_snapshot or collect_environment_snapshot(root)
    require(
        current_environment == document.get("environment_snapshot"),
        "runtime package/system/CUDA environment drift",
    )
    cuda = current_environment.get("cuda", {})
    require(
        cuda.get("torch_cuda_available") is True,
        "CUDA unavailable; CPU fallback forbidden",
    )
    require(cuda.get("device_count", 0) >= 1, "no CUDA device detected")
    require(
        current_environment.get("nvidia_smi", {}).get("returncode") == 0,
        "nvidia-smi validation failed",
    )

    frozen_git = document.get("git_snapshot", {})
    current_git = git_snapshot or collect_git_snapshot(root)
    if frozen_git.get("available"):
        require(current_git.get("available") is True, "Git became unavailable")
        require(
            current_git.get("commit") == frozen_git.get("commit"),
            "Git commit changed after freeze",
        )
    return {
        "valid": True,
        "study_id": STUDY_ID,
        "protocol_sha256": actual_protocol_sha256,
        "source_tree_sha256": current_source["canonical_tree_sha256"],
        "heldout_training_seeds": list(HELDOUT_SEEDS),
        "message": "runner preflight passed; this remains a local prospective specification",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a frozen v11 protocol before any held-out runner starts."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Detached SHA-256 printed by the freezer; mandatory for fail-closed launch.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.project_root.resolve(strict=True)
    protocol = args.protocol if args.protocol.is_absolute() else root / args.protocol
    report = validate_frozen_protocol(
        protocol_path=protocol,
        project_root=root,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("[V11 FROZEN PROTOCOL PREFLIGHT PASSED — NO TRAINING STARTED]", flush=True)


if __name__ == "__main__":
    main()
