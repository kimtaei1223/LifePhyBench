#!/usr/bin/env python3
"""Train exogenous thermal controls matched to endogenous policy dose."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        default="outputs/thermal_policy_matched_dose.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/thermal_exogenous_matched",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task-episodes", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    results = calibration.get("results", [])
    if not results:
        raise SystemExit("calibration contains no policy results")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent.parent
    python = sys.executable

    for result in results:
        run_name = (
            f"thermal-exogenous_clock-{result['memory_mode']}-"
            f"seed{result['seed']}-steps{args.total_timesteps // 1000}k"
        )
        run_directory = output_root / run_name
        metadata = run_directory / "metadata.json"
        if metadata.exists():
            print(f"[SKIP] {run_name}", flush=True)
            continue
        if run_directory.exists():
            raise SystemExit(
                f"incomplete run directory exists; preserve it and choose a new "
                f"--output-root: {run_directory}"
            )
        dose = result["recommended_exogenous_dose_per_step"]
        command = [
            python,
            str(project_root / "scripts/train_recurrent_smoke.py"),
            "--environment-id",
            "Pusher-v5",
            "--mechanism",
            "thermal",
            "--degradation-mode",
            "exogenous_clock",
            "--memory-mode",
            result["memory_mode"],
            "--workers",
            str(args.workers),
            "--total-timesteps",
            str(args.total_timesteps),
            "--learning-rate",
            "0.0003",
            "--seed",
            str(result["seed"]),
            "--device",
            args.device,
            "--eval-task-episodes",
            str(args.task_episodes),
            "--thermal-exogenous-dose-per-step",
            str(dose),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
        ]
        print(
            f"[START] {run_name} dose={dose:.9f}",
            flush=True,
        )
        subprocess.run(command, check=True, cwd=project_root)
        print(f"[DONE] {run_name}", flush=True)
    print("[CAMPAIGN COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
