#!/usr/bin/env python3
"""Audit v11 calibration cells and select the strongest task-reactive arm.

Only calibration seeds 7300--7304 are accepted.  The lifetime arm is reported
descriptively but is never used to select the task-reactive comparator.  The
selection rule ranks candidates by stochastic-condition reward, then by the
equal-condition aggregate reward, and finally by a fixed arm order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

CALIBRATION_SEEDS = tuple(range(7_300, 7_305))
CONDITIONS = ("fixed", "stochastic")
POLICY_ARMS = (
    "lifetime_lstm",
    "task_reset_lstm",
    "reactive_mlp_64",
    "reactive_mlp_256",
)
REACTIVE_ARMS = POLICY_ARMS[1:]
ARM_TIE_BREAK_ORDER = {arm: index for index, arm in enumerate(REACTIVE_ARMS)}
EXPECTED_DECISIONS = 100_000
EXPECTED_EVAL_TASKS = 4_000
MIN_BASELINE_GAIN = 0.25
MIN_MIXED_LIFETIME_RATE = 0.50
MAX_TRIP_RATE = 0.02


class CalibrationQualificationError(ValueError):
    """Raised when calibration evidence is incomplete or inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_new(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite qualification: {path}")
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite qualification: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def expected_run_name(condition: str, arm: str, seed: int) -> str:
    return f"v11-{condition}-{arm}-seed{seed}-decisions100k"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationQualificationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CalibrationQualificationError(f"{label} must be finite")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationQualificationError(f"cannot read JSON: {path}") from error
    if not isinstance(document, dict):
        raise CalibrationQualificationError(f"JSON root must be an object: {path}")
    return document


def _read_raw_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CalibrationQualificationError(
                        f"raw row {line_number} is not an object: {path}"
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise CalibrationQualificationError(f"cannot read raw rows: {path}") from error
    if len(rows) != expected:
        raise CalibrationQualificationError(
            f"expected {expected} raw rows, found {len(rows)}: {path}"
        )
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def select_reactive_arm(summaries: dict[str, dict[str, dict[str, float]]]) -> str:
    """Apply the frozen deterministic comparator-selection ordering."""

    def rank(arm: str) -> tuple[float, float, int]:
        stochastic = summaries["stochastic"][arm]["mean_task_episode_reward"]
        aggregate = _mean(
            [summaries[condition][arm]["mean_task_episode_reward"] for condition in CONDITIONS]
        )
        return stochastic, aggregate, -ARM_TIE_BREAK_ORDER[arm]

    return max(REACTIVE_ARMS, key=rank)


def assess_baseline_competence(
    *,
    selected_arm: str,
    summaries: dict[str, dict[str, dict[str, float]]],
    all_low_arms: list[str],
) -> dict[str, Any]:
    if not all_low_arms:
        raise CalibrationQualificationError(
            "no empirically verified Always-Low calibration anchor was found"
        )
    stochastic = summaries["stochastic"]
    anchor_arm = max(
        all_low_arms, key=lambda arm: stochastic[arm]["mean_task_episode_reward"]
    )
    selected = stochastic[selected_arm]
    anchor = stochastic[anchor_arm]
    gain = selected["mean_task_episode_reward"] - anchor["mean_task_episode_reward"]
    criteria = {
        "gain_over_empirical_always_low_at_least_0_25": gain >= MIN_BASELINE_GAIN,
        "mixed_mode_lifetime_rate_at_least_0_50": (
            selected["both_modes_lifetime_rate"] >= MIN_MIXED_LIFETIME_RATE
        ),
        "thermal_trip_rate_at_most_0_02": selected["thermal_trip_rate"] <= MAX_TRIP_RATE,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": {
            "minimum_reward_gain_per_task": MIN_BASELINE_GAIN,
            "minimum_both_modes_lifetime_rate": MIN_MIXED_LIFETIME_RATE,
            "maximum_thermal_trip_rate": MAX_TRIP_RATE,
        },
        "empirical_always_low_arm": anchor_arm,
        "empirical_always_low_mean_reward": anchor["mean_task_episode_reward"],
        "selected_stochastic_mean_reward": selected["mean_task_episode_reward"],
        "selected_gain_over_always_low": gain,
    }


def analyze_calibration(
    *, calibration_root: Path, cpu_qualification_path: Path
) -> dict[str, Any]:
    cpu = _read_json(cpu_qualification_path)
    if cpu.get("phase") != "hierarchical_v11_cpu_qualification" or cpu.get("passed") is not True:
        raise CalibrationQualificationError("CPU qualification is missing or did not pass")
    if cpu.get("development_seeds") != list(CALIBRATION_SEEDS):
        raise CalibrationQualificationError("CPU qualification seed set mismatch")
    selected_design = cpu.get("selected_design")
    if not isinstance(selected_design, dict):
        raise CalibrationQualificationError("CPU selected_design is missing")

    expected_names = {
        expected_run_name(condition, arm, seed)
        for seed in CALIBRATION_SEEDS
        for condition in CONDITIONS
        for arm in POLICY_ARMS
    }
    actual_names = {path.name for path in calibration_root.glob("*decisions100k") if path.is_dir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise CalibrationQualificationError(
            f"calibration cell set mismatch; missing={missing}, extra={extra}"
        )

    cells: list[dict[str, Any]] = []
    common_training: dict[str, Any] | None = None
    low_level_hash: str | None = None
    evaluation_seeds: dict[int, int] = {}
    for seed in CALIBRATION_SEEDS:
        for condition in CONDITIONS:
            for arm in POLICY_ARMS:
                run = calibration_root / expected_run_name(condition, arm, seed)
                metadata_path = run / "metadata.json"
                status_path = run / "status.json"
                model_path = run / "model.zip"
                raw_path = run / "evaluation_tasks.jsonl"
                if not all(path.is_file() for path in (metadata_path, status_path, model_path, raw_path)):
                    raise CalibrationQualificationError(f"incomplete calibration cell: {run}")
                metadata = _read_json(metadata_path)
                status = _read_json(status_path)
                arguments = metadata.get("arguments", {})
                expected_arguments = {
                    "condition": condition,
                    "policy_arm": arm,
                    "seed": seed,
                    "total_task_decisions": EXPECTED_DECISIONS,
                    "eval_task_episodes": EXPECTED_EVAL_TASKS,
                }
                actual_arguments = {key: arguments.get(key) for key in expected_arguments}
                if actual_arguments != expected_arguments:
                    raise CalibrationQualificationError(f"argument mismatch: {run}")
                if status.get("status") != "complete":
                    raise CalibrationQualificationError(f"cell status is not complete: {run}")
                if metadata.get("actual_training_device") != "cuda":
                    raise CalibrationQualificationError(f"cell did not train on CUDA: {run}")
                if metadata.get("model_sha256") != sha256(model_path):
                    raise CalibrationQualificationError(f"model hash mismatch: {run}")
                rows = _read_raw_rows(raw_path, EXPECTED_EVAL_TASKS)
                if any(
                    row.get("condition") != condition
                    or row.get("evaluation_seed") != metadata.get("evaluation_seed")
                    for row in rows
                ):
                    raise CalibrationQualificationError(f"raw evaluation wiring mismatch: {run}")
                row_reward = _mean([_finite(row.get("reward"), "raw reward") for row in rows])
                evaluation = metadata.get("evaluation", {})
                metadata_reward = _finite(
                    evaluation.get("mean_task_episode_reward"), "metadata reward"
                )
                if not math.isclose(row_reward, metadata_reward, rel_tol=0.0, abs_tol=1e-12):
                    raise CalibrationQualificationError(f"raw/metadata reward mismatch: {run}")
                for key, value in selected_design.items():
                    if not math.isclose(
                        _finite(arguments.get(key), f"argument {key}"),
                        _finite(value, f"selected design {key}"),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise CalibrationQualificationError(f"CPU design mismatch: {run}")
                training = {
                    "learning_rate": _finite(arguments.get("learning_rate"), "learning_rate"),
                    "gamma": _finite(arguments.get("gamma"), "gamma"),
                    "gae_lambda": _finite(arguments.get("gae_lambda"), "gae_lambda"),
                    "n_steps": 64,
                    "batch_size": min(256, 64 * int(arguments.get("workers"))),
                    "ent_coef": _finite(arguments.get("ent_coef"), "ent_coef"),
                    "training_reward_scale": _finite(
                        arguments.get("training_reward_scale"), "training_reward_scale"
                    ),
                }
                if common_training is None:
                    common_training = training
                elif common_training != training:
                    raise CalibrationQualificationError("training hyperparameters differ across cells")
                current_low_level_hash = metadata.get("low_level_model_sha256")
                if low_level_hash is None:
                    low_level_hash = current_low_level_hash
                elif low_level_hash != current_low_level_hash:
                    raise CalibrationQualificationError("low-level checkpoint differs across cells")
                evaluation_seed = int(metadata.get("evaluation_seed"))
                prior_evaluation_seed = evaluation_seeds.setdefault(seed, evaluation_seed)
                if prior_evaluation_seed != evaluation_seed:
                    raise CalibrationQualificationError(
                        f"paired evaluation seed mismatch for training seed {seed}"
                    )
                cells.append(
                    {
                        "condition": condition,
                        "policy_arm": arm,
                        "seed": seed,
                        "mean_task_episode_reward": metadata_reward,
                        "high_power_selection_rate": _finite(
                            evaluation.get("high_power_selection_rate"), "high rate"
                        ),
                        "thermal_trip_rate": _finite(
                            evaluation.get("thermal_trip_rate"), "trip rate"
                        ),
                        "both_modes_lifetime_rate": _finite(
                            evaluation.get("both_modes_lifetime_rate"), "mixed-mode rate"
                        ),
                        "all_actions_low": all(int(row.get("action")) == 0 for row in rows),
                        "metadata_sha256": sha256(metadata_path),
                        "raw_sha256": sha256(raw_path),
                        "model_sha256": sha256(model_path),
                    }
                )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for condition in CONDITIONS:
        summaries[condition] = {}
        for arm in POLICY_ARMS:
            selected = [
                cell for cell in cells if cell["condition"] == condition and cell["policy_arm"] == arm
            ]
            summaries[condition][arm] = {
                "mean_task_episode_reward": _mean(
                    [cell["mean_task_episode_reward"] for cell in selected]
                ),
                "sd_across_training_seeds": _sd(
                    [cell["mean_task_episode_reward"] for cell in selected]
                ),
                "high_power_selection_rate": _mean(
                    [cell["high_power_selection_rate"] for cell in selected]
                ),
                "thermal_trip_rate": _mean([cell["thermal_trip_rate"] for cell in selected]),
                "both_modes_lifetime_rate": _mean(
                    [cell["both_modes_lifetime_rate"] for cell in selected]
                ),
            }

    selected_arm = select_reactive_arm(summaries)
    all_low_arms = [
        arm
        for arm in REACTIVE_ARMS
        if all(
            cell["all_actions_low"]
            for cell in cells
            if cell["condition"] == "stochastic" and cell["policy_arm"] == arm
        )
    ]
    competence = assess_baseline_competence(
        selected_arm=selected_arm,
        summaries=summaries,
        all_low_arms=all_low_arms,
    )
    cpu_design_passed = cpu.get("passed") is True
    baseline_passed = competence["passed"] is True
    qualification_passed = cpu_design_passed and baseline_passed
    return {
        "phase": "hierarchical_v11_cpu_and_baseline_qualification",
        "status": "calibration_only_not_confirmatory_evidence",
        "qualification_passed": qualification_passed,
        "cpu_design_qualification_passed": cpu_design_passed,
        "baseline_competence_passed": baseline_passed,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "heldout_seeds_accessed": False,
        "selected_reactive_arm": selected_arm,
        "selection_rule": {
            "candidate_arms": list(REACTIVE_ARMS),
            "primary_rank": "highest mean stochastic-condition reward across calibration seeds",
            "secondary_rank": "highest equal-condition aggregate mean reward",
            "final_tie_break": list(REACTIVE_ARMS),
            "lifetime_arm_used_for_selection": False,
        },
        "baseline_competence": competence,
        "selected_design": selected_design,
        "selected_training": common_training,
        "low_level_model_sha256": low_level_hash,
        "evaluation_seed_by_training_seed": evaluation_seeds,
        "summaries": summaries,
        "calibration_descriptive_only": {
            "stochastic_lifetime_minus_selected_reactive": (
                summaries["stochastic"]["lifetime_lstm"]["mean_task_episode_reward"]
                - summaries["stochastic"][selected_arm]["mean_task_episode_reward"]
            ),
            "fixed_lifetime_minus_selected_reactive": (
                summaries["fixed"]["lifetime_lstm"]["mean_task_episode_reward"]
                - summaries["fixed"][selected_arm]["mean_task_episode_reward"]
            ),
            "not_a_confirmatory_gate": True,
        },
        "artifact_audit": {
            "expected_cells": 40,
            "validated_cells": len(cells),
            "raw_rows_per_cell": EXPECTED_EVAL_TASKS,
            "all_cuda": True,
            "cell_artifacts": cells,
        },
        "inputs": {
            "cpu_qualification": str(cpu_qualification_path.resolve()),
            "cpu_qualification_sha256": sha256(cpu_qualification_path),
            "calibration_root": str(calibration_root.resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/calibration"),
    )
    parser.add_argument(
        "--cpu-qualification",
        type=Path,
        default=Path("outputs/hierarchical_v11/CPU_QUALIFICATION.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_v11/CALIBRATION_QUALIFICATION.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_calibration(
        calibration_root=args.calibration_root,
        cpu_qualification_path=args.cpu_qualification,
    )
    atomic_write_new(args.output, report)
    print(
        json.dumps(
            {
                "qualification_passed": report["qualification_passed"],
                "baseline_competence_passed": report["baseline_competence_passed"],
                "selected_reactive_arm": report["selected_reactive_arm"],
                "selected_gain_over_always_low": report["baseline_competence"][
                    "selected_gain_over_always_low"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["qualification_passed"]:
        raise SystemExit("v11 calibration qualification failed; protocol freeze is blocked")


if __name__ == "__main__":
    main()
