"""Run the four-cell GPU wiring pilot for the thermal commitment task."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4_999)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-task-episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--commitment-trip-penalty", type=float, default=75.0)
    parser.add_argument("--commitment-high-power-bonus", type=float, default=2.0)
    parser.add_argument(
        "--commitment-control-cost-basis",
        choices=["applied_action", "requested_action"],
        default="requested_action",
    )
    parser.add_argument(
        "--output-root", default="outputs/thermal_commitment_pilot"
    )
    args = parser.parse_args()
    if (
        min(args.seed, args.total_timesteps, args.workers, args.eval_task_episodes)
        <= 0
        or args.commitment_trip_penalty <= 0.0
        or args.commitment_high_power_bonus <= 0.0
    ):
        raise SystemExit("seed and budgets must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = (
        ("dynamic", "endogenous_action"),
        ("static", "exogenous_clock"),
    )
    manifest = {
        "phase": "thermal_commitment_gpu_wiring_pilot",
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_task_episodes": args.eval_task_episodes,
        "cells": [
            {"label": label, "degradation_mode": degradation}
            for label, degradation in cells
        ],
        "canonical_task_seed": 811,
        "thermal": {
            "heat_rate": 0.1,
            "cooling_rate": 0.0,
            "episode_cooling": 0.0,
            "exogenous_dose_per_step": 0.0,
        },
        "commitment": {
            "trip_load": 0.10,
            "low_power_scale": 0.40,
            "trip_penalty": args.commitment_trip_penalty,
            "high_power_throughput_bonus": args.commitment_high_power_bonus,
            "control_cost_basis": args.commitment_control_cost_basis,
        },
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch: {manifest_path}; use a new root")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    index = 0
    for label, degradation in cells:
        for memory in ("task", "lifetime"):
            index += 1
            run_name = (
                f"thermal-commitment-{label}-{memory}-seed{args.seed}-"
                f"steps{args.total_timesteps // 1000}k"
            )
            run_directory = output_root / run_name
            if (run_directory / "metadata.json").exists():
                print(f"[SKIP {index}/4] {run_name}", flush=True)
                continue
            if run_directory.exists():
                raise SystemExit(f"incomplete run exists: {run_directory}")
            command = [
                sys.executable,
                str(project_root / "scripts/train_fair_recurrent.py"),
                "--memory-mode",
                memory,
                "--environment-id",
                "Pusher-v5",
                "--mechanism",
                "thermal",
                "--degradation-mode",
                degradation,
                "--workers",
                str(args.workers),
                "--total-timesteps",
                str(args.total_timesteps),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--eval-task-episodes",
                str(args.eval_task_episodes),
                "--thermal-exogenous-dose-per-step",
                "0.0",
                "--thermal-heat-rate",
                "0.1",
                "--thermal-cooling-rate",
                "0.0",
                "--thermal-episode-cooling",
                "0.0",
                "--canonical-task-seed",
                "811",
                "--thermal-commitment",
                "--commitment-trip-load",
                "0.10",
                "--commitment-low-power-scale",
                "0.40",
                "--commitment-trip-penalty",
                str(args.commitment_trip_penalty),
                "--commitment-high-power-bonus",
                str(args.commitment_high_power_bonus),
                "--commitment-control-cost-basis",
                args.commitment_control_cost_basis,
                "--output-root",
                str(output_root),
                "--run-name",
                run_name,
            ]
            print(f"[START {index}/4] {run_name}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE {index}/4] {run_name}", flush=True)
    print("[PILOT COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
