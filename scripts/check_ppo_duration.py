#!/usr/bin/env python3
"""Run sequential PPO duration checks before freezing a smoke budget."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--mechanism", default="wear")
    parser.add_argument("--degradation-mode", default="endogenous_action")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[500_000, 1_000_000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/gpu_duration_checks")
    args = parser.parse_args()
    if not args.timesteps or any(steps <= 0 for steps in args.timesteps):
        raise SystemExit("--timesteps must contain positive values")
    trainer = Path(__file__).with_name("train_sb3_smoke.py")
    for steps in args.timesteps:
        run_name = f"ppo-{steps // 1000}k-lr{args.learning_rate:g}-seed{args.seed}"
        command = [
            sys.executable,
            str(trainer),
            "--algorithm",
            "ppo",
            "--environment-id",
            args.environment_id,
            "--mechanism",
            args.mechanism,
            "--degradation-mode",
            args.degradation_mode,
            "--workers",
            str(args.workers),
            "--total-timesteps",
            str(steps),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--output-root",
            args.output_root,
            "--run-name",
            run_name,
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
