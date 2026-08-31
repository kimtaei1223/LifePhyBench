#!/usr/bin/env python3
"""Retest unstable exogenous policies with exact calibrated source-seed doses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def calibrated_episode_doses(path: Path) -> dict[int, float]:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    doses: dict[int, float] = {}
    for result in calibration.get("results", []):
        if result.get("memory_mode") != "episode":
            continue
        seed = int(result["seed"])
        doses[seed] = float(result["recommended_exogenous_dose_per_step"])
    return doses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration",
        default="outputs/thermal_policy_matched_dose_long.json",
    )
    parser.add_argument("--source-seeds", nargs="+", type=int, default=[1002, 1004])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[2002, 2004])
    parser.add_argument(
        "--output-root",
        default="outputs/thermal_exogenous_outlier_retest_corrected",
    )
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task-episodes", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    calibration_path = Path(args.calibration)
    doses = calibrated_episode_doses(calibration_path)
    missing = sorted(set(args.source_seeds) - set(doses))
    if missing:
        raise SystemExit(f"calibration has no episode dose for source seeds: {missing}")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "corrected_outlier_robustness_retest",
        "calibration": str(calibration_path),
        "source_seeds": args.source_seeds,
        "train_seeds": args.train_seeds,
        "source_seed_doses": {str(seed): doses[seed] for seed in args.source_seeds},
        "total_timesteps": args.total_timesteps,
        "task_episodes": args.task_episodes,
    }
    (output_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    for source_seed in args.source_seeds:
        dose = doses[source_seed]
        for train_seed in args.train_seeds:
            run_name = (
                f"thermal-exogenous_clock-episode-source{source_seed}-"
                f"train{train_seed}-steps{args.total_timesteps // 1000}k"
            )
            run_directory = output_root / run_name
            if (run_directory / "metadata.json").exists():
                print(f"[SKIP] {run_name}", flush=True)
                continue
            if run_directory.exists():
                raise SystemExit(
                    f"incomplete run exists: {run_directory}; preserve it and use "
                    "a different --output-root"
                )
            command = [
                sys.executable,
                str(project_root / "scripts/train_recurrent_smoke.py"),
                "--environment-id",
                "Pusher-v5",
                "--mechanism",
                "thermal",
                "--degradation-mode",
                "exogenous_clock",
                "--memory-mode",
                "episode",
                "--workers",
                str(args.workers),
                "--total-timesteps",
                str(args.total_timesteps),
                "--learning-rate",
                "0.0003",
                "--seed",
                str(train_seed),
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
                f"[START] {run_name} source_dose={dose:.12f}", flush=True
            )
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE] {run_name}", flush=True)
    print("[CAMPAIGN COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
