#!/usr/bin/env python3
"""Train the frozen monolithic recurrent thermal-aware Reacher baseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_json(path: Path, document: dict[str, Any]) -> None:
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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def quarantine(directory: Path) -> Path:
    ordinal = 1
    while True:
        target = directory.with_name(f"{directory.name}.interrupted-{ordinal}")
        if not target.exists():
            shutil.move(str(directory), str(target))
            return target
        ordinal += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("configs/reacher_cross_task_stage2_v1.json")
    )
    parser.add_argument(
        "--selection", type=Path, default=Path("outputs/reacher_replication/low_level/SELECTION.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication/monolithic_baseline")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    project_root = Path(__file__).resolve().parent.parent
    protocol_path = args.protocol if args.protocol.is_absolute() else project_root / args.protocol
    selection_path = args.selection if args.selection.is_absolute() else project_root / args.selection
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    selection = read_json(selection_path)
    selected = selection["selected"]
    if int(selected["seed"]) != int(protocol["selected_low_level_seed"]):
        raise SystemExit("low-level selection drift")
    low_level_model = Path(selected["model"])
    config = protocol["monolithic_baseline"]
    train_seeds = [int(value) for value in config["training_seeds"]]
    evaluation_seeds = [int(value) for value in config["evaluation_seeds"]]
    if len(train_seeds) != len(evaluation_seeds):
        raise SystemExit("training and evaluation seed counts differ")
    decisions = int(config["total_task_decisions_per_seed"])
    evaluation_tasks = int(config["evaluation_tasks_per_seed"])
    status_path = output_root / "status.json"
    completed: list[int] = []
    for ordinal, (seed, evaluation_seed) in enumerate(
        zip(train_seeds, evaluation_seeds, strict=True), start=1
    ):
        name = f"reacher-monolithic-lifetime-seed{seed}-decisions{decisions // 1000}k"
        directory = output_root / name
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            completed.append(seed)
            print(f"[SKIP {ordinal}/{len(train_seeds)}] {name}", flush=True)
            continue
        if directory.exists():
            retained = quarantine(directory)
            print(f"[RETAINED INCOMPLETE] {retained.name}", flush=True)
        atomic_json(
            status_path,
            {
                "status": "training",
                "current_seed": seed,
                "completed_seeds": completed,
                "total_seeds": len(train_seeds),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        command = [
            sys.executable,
            str(project_root / "scripts/train_hierarchical_thermal_v11.py"),
            "--environment-id", "Reacher-v5",
            "--condition", "stochastic",
            "--policy-arm", "lifetime_lstm",
            "--workers", str(args.workers),
            "--total-task-decisions", str(decisions),
            "--eval-task-episodes", str(evaluation_tasks),
            "--seed", str(seed),
            "--evaluation-seed", str(evaluation_seed),
            "--device", args.device,
            "--thermal-episode-cooling", "0.15",
            "--sensor-noise-sd", "0.01",
            "--shock-probability", "0.0005",
            "--shock-size", "0.01",
            "--low-level-model", str(low_level_model),
            "--output-root", str(output_root),
            "--run-name", name,
            "--study-phase", "calibration",
        ]
        print(f"[START {ordinal}/{len(train_seeds)}] {name}", flush=True)
        subprocess.run(command, check=True, cwd=project_root)
        completed.append(seed)
        print(f"[DONE {ordinal}/{len(train_seeds)}] {name}", flush=True)

    candidates = []
    for seed in train_seeds:
        name = f"reacher-monolithic-lifetime-seed{seed}-decisions{decisions // 1000}k"
        metadata = read_json(output_root / name / "metadata.json")
        evaluation = metadata["evaluation"]
        candidates.append(
            {
                "seed": seed,
                "run_name": name,
                "model": str((output_root / name / "model.zip").resolve()),
                "mean_task_episode_reward": float(evaluation["mean_task_episode_reward"]),
                "thermal_trip_rate": float(evaluation["thermal_trip_rate"]),
                "high_power_selection_rate": float(evaluation["high_power_selection_rate"]),
            }
        )
    safe = [row for row in candidates if row["thermal_trip_rate"] <= 0.02]
    if safe:
        selected_model = max(safe, key=lambda row: row["mean_task_episode_reward"])
    else:
        selected_model = min(
            candidates,
            key=lambda row: (row["thermal_trip_rate"], -row["mean_task_episode_reward"]),
        )
    result = {
        "phase": "reacher_monolithic_baseline_development_selection",
        "confirmatory_evidence": False,
        "selection_rule": config["selection_rule"],
        "candidates": candidates,
        "selected": selected_model,
    }
    atomic_json(output_root / "SELECTION.json", result)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "completed_seeds": train_seeds,
            "selected_seed": selected_model["seed"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
