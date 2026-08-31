#!/usr/bin/env python3
"""Run the fail-closed v11 held-out campaign from a frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.validate_hierarchical_v11_freeze import validate_frozen_protocol
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from validate_hierarchical_v11_freeze import validate_frozen_protocol

CONDITIONS = ("fixed", "stochastic")


class ConfirmatoryRunnerError(ValueError):
    """Raised when a held-out launch or completed cell fails validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def expected_run_name(condition: str, arm: str, seed: int, decisions: int) -> str:
    return f"v11-heldout-{condition}-{arm}-seed{seed}-decisions{decisions // 1000}k"


def validate_completed_cell(
    run_directory: Path,
    *,
    condition: str,
    arm: str,
    seed: int,
    evaluation_seed: int,
    decisions: int,
    eval_tasks: int,
    protocol_sha256: str,
) -> None:
    required = {
        "metadata": run_directory / "metadata.json",
        "status": run_directory / "status.json",
        "model": run_directory / "model.zip",
        "raw": run_directory / "evaluation_tasks.jsonl",
    }
    if not all(path.is_file() for path in required.values()):
        raise ConfirmatoryRunnerError(f"incomplete held-out cell: {run_directory}")
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    status = json.loads(required["status"].read_text(encoding="utf-8"))
    expected_arguments = {
        "condition": condition,
        "policy_arm": arm,
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "total_task_decisions": decisions,
        "eval_task_episodes": eval_tasks,
        "study_phase": "confirmatory",
        "protocol_sha256": protocol_sha256,
    }
    actual_arguments = {
        name: metadata.get("arguments", {}).get(name) for name in expected_arguments
    }
    if actual_arguments != expected_arguments:
        raise ConfirmatoryRunnerError(f"held-out argument mismatch: {run_directory}")
    if (
        metadata.get("phase") != "hierarchical_thermal_v11_heldout_confirmatory"
        or metadata.get("status") != "heldout_confirmatory_cell_complete"
        or metadata.get("actual_training_device") != "cuda"
        or status.get("status") != "complete"
        or status.get("phase") != "v11_confirmatory_heldout"
    ):
        raise ConfirmatoryRunnerError(f"held-out phase/device mismatch: {run_directory}")
    if metadata.get("model_sha256") != sha256(required["model"]):
        raise ConfirmatoryRunnerError(f"held-out model hash mismatch: {run_directory}")
    raw_rows = 0
    with required["raw"].open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("condition") != condition
                or row.get("evaluation_seed") != evaluation_seed
            ):
                raise ConfirmatoryRunnerError(f"held-out raw wiring mismatch: {run_directory}")
            raw_rows += 1
    if raw_rows != eval_tasks:
        raise ConfirmatoryRunnerError(f"held-out raw row count mismatch: {run_directory}")


def build_command(
    *,
    trainer: Path,
    output_root: Path,
    run_name: str,
    condition: str,
    arm: str,
    seed: int,
    evaluation_seed: int,
    protocol: dict[str, Any],
    protocol_sha256: str,
) -> list[str]:
    budgets = protocol["budgets"]
    physics = protocol["physics"]
    training = protocol["arms"]["common_training_hyperparameters"]
    low_level = protocol["inputs"]["low_level_checkpoint"]["path"]
    return [
        sys.executable,
        "-u",
        str(trainer),
        "--study-phase",
        "confirmatory",
        "--protocol-sha256",
        protocol_sha256,
        "--condition",
        condition,
        "--policy-arm",
        arm,
        "--workers",
        str(budgets["workers"]),
        "--total-task-decisions",
        str(budgets["total_task_decisions_per_run"]),
        "--eval-task-episodes",
        str(budgets["evaluation_task_episodes"]),
        "--seed",
        str(seed),
        "--evaluation-seed",
        str(evaluation_seed),
        "--device",
        str(budgets["device"]),
        "--torch-threads-per-process",
        str(budgets["torch_threads_per_process"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--gamma",
        str(training["gamma"]),
        "--gae-lambda",
        str(training["gae_lambda"]),
        "--ent-coef",
        str(training["ent_coef"]),
        "--training-reward-scale",
        str(training["training_reward_scale"]),
        "--episode-steps",
        str(physics["episode_steps"]),
        "--episodes-per-lifetime",
        str(physics["tasks_per_lifetime"]),
        "--canonical-task-seed",
        str(physics["canonical_task_seed"]),
        "--trip-load",
        str(physics["trip_load"]),
        "--low-power-scale",
        str(physics["low_power_scale"]),
        "--trip-penalty",
        str(physics["trip_penalty"]),
        "--high-power-bonus",
        str(physics["high_power_bonus"]),
        "--thermal-heat-rate",
        str(physics["thermal_heat_rate"]),
        "--thermal-episode-cooling",
        str(physics["thermal_episode_cooling"]),
        "--sensor-noise-sd",
        str(physics["sensor_noise_sd"]),
        "--shock-probability",
        str(physics["shock_probability"]),
        "--shock-size",
        str(physics["shock_size"]),
        "--fixed-initial-load",
        str(physics["conditions"]["fixed"]["initial_thermal_load"]["value"]),
        "--stochastic-initial-load-low",
        str(physics["conditions"]["stochastic"]["initial_thermal_load"]["low"]),
        "--stochastic-initial-load-high",
        str(physics["conditions"]["stochastic"]["initial_thermal_load"]["high"]),
        "--low-level-model",
        str(low_level),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/confirmatory"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else project_root / args.output_root
    )
    expected_digest = args.expected_protocol_sha256.lower()
    preflight = validate_frozen_protocol(
        protocol_path=protocol_path,
        project_root=project_root,
        expected_protocol_sha256=expected_digest,
    )
    if output_root.exists():
        raise SystemExit(f"confirmatory output root must be new and absent: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    lock_path = output_root / ".campaign.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.write("\n")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = preflight["protocol_sha256"]
    seeds = protocol["seed_namespaces"]["heldout"]["training_pair_seeds"]
    evaluation_seeds = protocol["seed_namespaces"]["heldout"]["evaluation_bank_seeds"]
    if len(seeds) != len(evaluation_seeds):
        raise ConfirmatoryRunnerError("held-out training/evaluation seed count mismatch")
    reactive_arm = protocol["arms"]["task_reactive"]["identity"]
    arms = ("lifetime_lstm", reactive_arm)
    budgets = protocol["budgets"]
    decisions = budgets["total_task_decisions_per_run"]
    eval_tasks = budgets["evaluation_task_episodes"]
    cells = [
        (condition, arm, seed, evaluation_seed)
        for seed, evaluation_seed in zip(seeds, evaluation_seeds, strict=True)
        for condition in CONDITIONS
        for arm in arms
    ]
    cell_names = [
        expected_run_name(condition, arm, seed, decisions)
        for condition, arm, seed, _evaluation_seed in cells
    ]
    atomic_json(
        output_root / "manifest.json",
        {
            "phase": "hierarchical_thermal_v11_heldout_confirmatory",
            "status": "prospectively_frozen_campaign_started",
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": protocol_sha256,
            "source_tree_sha256": preflight["source_tree_sha256"],
            "training_seeds": seeds,
            "evaluation_seeds": evaluation_seeds,
            "conditions": list(CONDITIONS),
            "arms": list(arms),
            "expected_cells": cell_names,
            "cell_count": len(cells),
            "calibration_selection_used": True,
            "heldout_results_accessed_before_launch": False,
        },
    )
    atomic_json(
        output_root / "progress.json",
        {
            "phase": "hierarchical_thermal_v11_heldout_confirmatory",
            "completed_cells": [],
            "remaining_cells": cell_names,
            "complete": False,
        },
    )

    completed: list[str] = []
    try:
        trainer = project_root / "scripts/train_hierarchical_thermal_v11.py"
        for index, (condition, arm, seed, evaluation_seed) in enumerate(cells, start=1):
            run_name = cell_names[index - 1]
            command = build_command(
                trainer=trainer,
                output_root=output_root,
                run_name=run_name,
                condition=condition,
                arm=arm,
                seed=seed,
                evaluation_seed=evaluation_seed,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
            )
            print(f"[START {index}/{len(cells)}] {run_name}", flush=True)
            with (output_root / f"{run_name}.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            validate_completed_cell(
                output_root / run_name,
                condition=condition,
                arm=arm,
                seed=seed,
                evaluation_seed=evaluation_seed,
                decisions=decisions,
                eval_tasks=eval_tasks,
                protocol_sha256=protocol_sha256,
            )
            completed.append(run_name)
            atomic_json(
                output_root / "progress.json",
                {
                    "phase": "hierarchical_thermal_v11_heldout_confirmatory",
                    "completed_cells": completed,
                    "remaining_cells": cell_names[index:],
                    "complete": index == len(cells),
                },
            )
            print(f"[DONE {index}/{len(cells)}] {run_name}", flush=True)
        atomic_json(
            output_root / "CAMPAIGN_COMPLETE.json",
            {
                "phase": "hierarchical_thermal_v11_heldout_confirmatory",
                "status": "all_frozen_cells_complete",
                "protocol_sha256": protocol_sha256,
                "completed_cells": completed,
            },
        )
        print("[V11 HELD-OUT CONFIRMATORY CELLS COMPLETE]", flush=True)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
