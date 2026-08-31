#!/usr/bin/env python3
"""Train and select the frozen Reacher low-level controller candidates.

This is the first, development-only stage of the cross-task replication.  It
does not inspect or consume any confirmatory seed reserved by the protocol.
The runner is resumable at completed-run granularity and retains interrupted
directories instead of deleting them.
"""

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
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def quarantine_incomplete(directory: Path) -> Path:
    ordinal = 1
    while True:
        candidate = directory.with_name(f"{directory.name}.interrupted-{ordinal}")
        if not candidate.exists():
            shutil.move(str(directory), str(candidate))
            return candidate
        ordinal += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/reacher_cross_task_replication_v1.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication/low_level")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")

    project_root = Path(__file__).resolve().parent.parent
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = project_root / protocol_path
    protocol = read_json(protocol_path)
    task = protocol["task"]
    stage = protocol["low_level_development"]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    manifest = {
        "phase": "reacher_cross_task_low_level_development",
        "protocol": str(protocol_path.resolve()),
        "protocol_document": protocol,
        "workers": args.workers,
        "device": args.device,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise SystemExit("low-level manifest mismatch; use a new output root")
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)

    seeds = [int(seed) for seed in stage["seeds"]]
    timesteps = int(stage["total_timesteps_per_seed"])
    evaluations = int(stage["evaluation_task_episodes"])
    completed: list[int] = []
    for ordinal, seed in enumerate(seeds, start=1):
        name = f"reacher-static-task-seed{seed}-steps{timesteps // 1000}k"
        directory = output_root / name
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            completed.append(seed)
            print(f"[SKIP {ordinal}/{len(seeds)}] {name}", flush=True)
            continue
        if directory.exists():
            retained = quarantine_incomplete(directory)
            print(f"[RETAINED INCOMPLETE] {retained.name}", flush=True)
        atomic_json(
            status_path,
            {
                "status": "training",
                "current_seed": seed,
                "completed_seeds": completed,
                "total_seeds": len(seeds),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        command = [
            sys.executable,
            str(project_root / "scripts/train_fair_recurrent.py"),
            "--memory-mode", "task",
            "--environment-id", str(task["environment_id"]),
            "--mechanism", "thermal",
            "--degradation-mode", "exogenous_clock",
            "--workers", str(args.workers),
            "--episode-steps", str(task["episode_steps"]),
            "--episodes-per-lifetime", str(task["tasks_per_lifetime"]),
            "--total-timesteps", str(timesteps),
            "--eval-task-episodes", str(evaluations),
            "--seed", str(seed),
            "--device", args.device,
            "--thermal-exogenous-dose-per-step", "0.0",
            "--thermal-heat-rate", "0.1",
            "--thermal-cooling-rate", "0.0",
            "--thermal-episode-cooling", "0.0",
            "--canonical-task-seed", str(task["canonical_task_seed"]),
            "--output-root", str(output_root),
            "--run-name", name,
        ]
        print(f"[START {ordinal}/{len(seeds)}] {name}", flush=True)
        subprocess.run(command, check=True, cwd=project_root)
        completed.append(seed)
        print(f"[DONE {ordinal}/{len(seeds)}] {name}", flush=True)

    candidates = []
    for seed in seeds:
        name = f"reacher-static-task-seed{seed}-steps{timesteps // 1000}k"
        directory = output_root / name
        metadata = read_json(directory / "metadata.json")
        evaluation = metadata["task_episode_evaluation"]
        candidates.append(
            {
                "seed": seed,
                "run_name": name,
                "model": str((directory / "model.zip").resolve()),
                "mean_task_episode_reward": float(evaluation["mean_task_episode_reward"]),
                "std_task_episode_reward": float(evaluation["std_task_episode_reward"]),
            }
        )
    selected = max(candidates, key=lambda row: row["mean_task_episode_reward"])
    selection = {
        "phase": "reacher_cross_task_low_level_development_selection",
        "confirmatory_evidence": False,
        "selection_rule": stage["selection_rule"],
        "candidates": candidates,
        "selected": selected,
    }
    atomic_json(output_root / "SELECTION.json", selection)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "completed_seeds": seeds,
            "selected_seed": selected["seed"],
            "selected_model": selected["model"],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
