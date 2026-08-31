"""Run the v6 decision-masked thermal commitment calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4_989)
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-task-episodes", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--version-label", default="v6")
    parser.add_argument("--append-previous-applied-action", action="store_true")
    parser.add_argument(
        "--output-root", default="outputs/thermal_commitment_calibration_v6"
    )
    args = parser.parse_args()
    if min(
        args.seed, args.total_timesteps, args.workers, args.eval_task_episodes
    ) <= 0:
        raise SystemExit("seed and budgets must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = (
        ("dynamic", "endogenous_action"),
        ("static", "exogenous_clock"),
    )
    manifest = {
        "phase": (
            f"thermal_commitment_{args.version_label}_decision_masked_calibration"
        ),
        "status": "calibration_not_confirmatory_evidence",
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
            "trip_penalty": 75.0,
            "high_power_throughput_bonus": 2.0,
            "control_cost_basis": "requested_action",
        },
        "training_only_curriculum": {
            "start_trip_load": 0.70,
            "duration_lifetimes_per_worker": 10,
            "evaluation_trip_load": 0.10,
        },
        "optimization": {
            "decision_only_mode_loss": True,
            "mode_coordinate": 0,
            "mode_selected_observation_index_from_end": -3,
        },
    }
    if args.append_previous_applied_action:
        manifest["representation"] = {
            "previous_applied_action_observed": True,
            "zeroed_at_every_task_boundary": True,
            "privileged_health_exposed": False,
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
                f"thermal-commitment-{args.version_label}-{label}-{memory}-"
                f"seed{args.seed}-"
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
                "75.0",
                "--commitment-high-power-bonus",
                "2.0",
                "--commitment-control-cost-basis",
                "requested_action",
                "--commitment-curriculum-start-trip-load",
                "0.70",
                "--commitment-curriculum-lifetimes",
                "10",
                "--commitment-mask-mode-loss",
                "--output-root",
                str(output_root),
                "--run-name",
                run_name,
            ]
            if args.append_previous_applied_action:
                command.insert(
                    command.index("--output-root"),
                    "--append-previous-applied-action",
                )
            print(f"[START {index}/4] {run_name}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE {index}/4] {run_name}", flush=True)
    print("[MASKED PILOT COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
