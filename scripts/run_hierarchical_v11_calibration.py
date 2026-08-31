#!/usr/bin/env python3
"""Run the predeclared v11 GPU calibration cells sequentially.

Only calibration optimization seeds are accepted.  Held-out seeds are rejected
by construction so this convenience runner cannot accidentally open the
confirmatory data before protocol freeze.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CALIBRATION_SEEDS = tuple(range(7300, 7305))
CONDITIONS = ("fixed", "stochastic")
POLICY_ARMS = (
    "lifetime_lstm",
    "task_reset_lstm",
    "reactive_mlp_64",
    "reactive_mlp_256",
)


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


def expected_cell(
    *, condition: str, policy_arm: str, seed: int, decisions: int
) -> str:
    return f"v11-{condition}-{policy_arm}-seed{seed}-decisions{decisions // 1000}k"


def validate_completed_cell(
    run_directory: Path,
    *,
    condition: str,
    policy_arm: str,
    seed: int,
    decisions: int,
    eval_tasks: int,
) -> bool:
    metadata_path = run_directory / "metadata.json"
    status_path = run_directory / "status.json"
    model_path = run_directory / "model.zip"
    raw_path = run_directory / "evaluation_tasks.jsonl"
    if not any(path.exists() for path in (metadata_path, status_path, model_path, raw_path)):
        return False
    if not all(path.is_file() for path in (metadata_path, status_path, model_path, raw_path)):
        raise RuntimeError(f"partial calibration cell requires audit: {run_directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = {
        "condition": condition,
        "policy_arm": policy_arm,
        "seed": seed,
        "total_task_decisions": decisions,
        "eval_task_episodes": eval_tasks,
    }
    actual = {name: metadata["arguments"].get(name) for name in expected}
    if actual != expected or status.get("status") != "complete":
        raise RuntimeError(
            f"existing cell does not match requested calibration: {run_directory}"
        )
    if sum(1 for _ in raw_path.open("r", encoding="utf-8")) != eval_tasks:
        raise RuntimeError(f"raw evaluation row count mismatch: {run_directory}")
    return True


def build_command(
    *,
    trainer: Path,
    output_root: Path,
    run_name: str,
    condition: str,
    policy_arm: str,
    seed: int,
    decisions: int,
    eval_tasks: int,
    workers: int,
    device: str,
    qualification: dict[str, Any] | None,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(trainer),
        "--condition",
        condition,
        "--policy-arm",
        policy_arm,
        "--workers",
        str(workers),
        "--total-task-decisions",
        str(decisions),
        "--eval-task-episodes",
        str(eval_tasks),
        "--seed",
        str(seed),
        "--evaluation-seed",
        str(720_000 + seed),
        "--device",
        device,
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]
    selected = (qualification or {}).get("selected_design")
    if selected:
        mapping = {
            "thermal_episode_cooling": "--thermal-episode-cooling",
            "sensor_noise_sd": "--sensor-noise-sd",
            "shock_probability": "--shock-probability",
            "shock_size": "--shock-size",
        }
        for name, flag in mapping.items():
            if name in selected:
                command.extend([flag, str(selected[name])])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CALIBRATION_SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--policy-arms", nargs="+", choices=POLICY_ARMS, default=list(POLICY_ARMS))
    parser.add_argument("--total-task-decisions", type=int, default=100_000)
    parser.add_argument("--eval-task-episodes", type=int, default=4_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--qualification",
        type=Path,
        default=Path("outputs/hierarchical_v11/CPU_QUALIFICATION.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/calibration"),
    )
    parser.add_argument("--allow-without-qualification", action="store_true")
    args = parser.parse_args()
    if not set(args.seeds).issubset(CALIBRATION_SEEDS):
        raise SystemExit(
            f"only predeclared calibration seeds are permitted: {CALIBRATION_SEEDS}"
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("duplicate calibration seeds are forbidden")
    if min(args.total_task_decisions, args.eval_task_episodes, args.workers) <= 0:
        raise SystemExit("training/evaluation budgets and workers must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    qualification_path = args.qualification
    if not qualification_path.is_absolute():
        qualification_path = project_root / qualification_path
    qualification = None
    if qualification_path.is_file():
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if qualification.get("passed") is not True:
            raise SystemExit("v11 CPU qualification exists but did not pass")
    elif not args.allow_without_qualification:
        raise SystemExit(f"passing v11 CPU qualification is required: {qualification_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".campaign.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise SystemExit(
            f"campaign lock already exists; verify no runner is active: {lock_path}"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.write("\n")

    cells = [
        (condition, arm, seed)
        for seed in args.seeds
        for condition in args.conditions
        for arm in args.policy_arms
    ]
    completed: list[str] = []
    try:
        trainer = project_root / "scripts/train_hierarchical_thermal_v11.py"
        for index, (condition, arm, seed) in enumerate(cells, start=1):
            run_name = expected_cell(
                condition=condition,
                policy_arm=arm,
                seed=seed,
                decisions=args.total_task_decisions,
            )
            run_directory = output_root / run_name
            if validate_completed_cell(
                run_directory,
                condition=condition,
                policy_arm=arm,
                seed=seed,
                decisions=args.total_task_decisions,
                eval_tasks=args.eval_task_episodes,
            ):
                print(f"[SKIP VERIFIED {index}/{len(cells)}] {run_name}", flush=True)
                completed.append(run_name)
                continue
            command = build_command(
                trainer=trainer,
                output_root=output_root,
                run_name=run_name,
                condition=condition,
                policy_arm=arm,
                seed=seed,
                decisions=args.total_task_decisions,
                eval_tasks=args.eval_task_episodes,
                workers=args.workers,
                device=args.device,
                qualification=qualification,
            )
            log_path = output_root / f"{run_name}.log"
            print(f"[START {index}/{len(cells)}] {run_name}", flush=True)
            with log_path.open("x", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not validate_completed_cell(
                run_directory,
                condition=condition,
                policy_arm=arm,
                seed=seed,
                decisions=args.total_task_decisions,
                eval_tasks=args.eval_task_episodes,
            ):
                raise RuntimeError(f"trainer returned without a complete cell: {run_name}")
            completed.append(run_name)
            atomic_json(
                output_root / "progress.json",
                {
                    "phase": "v11_calibration_only",
                    "completed_cells": completed,
                    "remaining_cells": [
                        expected_cell(
                            condition=other_condition,
                            policy_arm=other_arm,
                            seed=other_seed,
                            decisions=args.total_task_decisions,
                        )
                        for other_condition, other_arm, other_seed in cells[index:]
                    ],
                    "complete": index == len(cells),
                },
            )
            print(f"[DONE {index}/{len(cells)}] {run_name}", flush=True)
        print("[V11 CALIBRATION CELLS COMPLETE]", flush=True)
    finally:
        # Only this process could have created this exact exclusive lock.
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
