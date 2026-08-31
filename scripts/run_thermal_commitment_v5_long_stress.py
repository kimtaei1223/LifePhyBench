"""Run the frozen v5 long-budget dynamic learnability stress test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (4990, 4991, 4992, 4993)
TOTAL_TIMESTEPS = 2_000_000
EVAL_TASK_EPISODES = 1_000
WORKERS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/thermal_commitment_v5_long_stress"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "thermal_commitment_v5_long_dynamic_stress",
        "status": "calibration_not_confirmatory_evidence",
        "purpose": "distinguish 300k undertraining from structural mode collapse",
        "seeds": list(SEEDS),
        "memory_modes": ["task", "lifetime"],
        "degradation_mode": "endogenous_action",
        "total_timesteps": TOTAL_TIMESTEPS,
        "eval_task_episodes": EVAL_TASK_EPISODES,
        "workers": WORKERS,
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
        "frozen_success_rule": {
            "minimum_passing_lifetime_seeds": 3,
            "total_lifetime_seeds": 4,
            "per_lifetime_seed": {
                "minimum_cold_mode_selections": 40,
                "minimum_hot_mode_selections": 40,
                "minimum_cold_high_power_selection_rate": 0.60,
                "maximum_hot_high_power_selection_rate": 0.40,
                "maximum_thermal_trip_rate": 0.20,
            },
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

    total_runs = len(SEEDS) * 2
    index = 0
    for seed in SEEDS:
        for memory in ("task", "lifetime"):
            index += 1
            run_name = (
                f"thermal-commitment-v5-long-dynamic-{memory}-seed{seed}-"
                "steps2000k"
            )
            run_directory = output_root / run_name
            if (run_directory / "metadata.json").exists():
                print(f"[SKIP {index}/{total_runs}] {run_name}", flush=True)
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
                "endogenous_action",
                "--workers",
                str(WORKERS),
                "--total-timesteps",
                str(TOTAL_TIMESTEPS),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--eval-task-episodes",
                str(EVAL_TASK_EPISODES),
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
                "--output-root",
                str(output_root),
                "--run-name",
                run_name,
            ]
            print(f"[START {index}/{total_runs}] {run_name}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE {index}/{total_runs}] {run_name}", flush=True)
    print("[LONG STRESS TRAINING COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
