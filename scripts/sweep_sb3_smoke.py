#!/usr/bin/env python3
"""Run a small, sequential SB3 smoke sweep after individual smoke runs pass."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "sac"], required=True)
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--mechanism", default="wear")
    parser.add_argument("--degradation-mode", default="endogenous_action")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-4, 1e-4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/gpu_sweeps")
    args = parser.parse_args()
    trainer = Path(__file__).with_name("train_sb3_smoke.py")
    for index, (learning_rate, seed) in enumerate(
        product(args.learning_rates, args.seeds), start=1
    ):
        run_name = f"{args.algorithm}-trial{index}-lr{learning_rate:g}-seed{seed}"
        command = [
            sys.executable,
            str(trainer),
            "--algorithm",
            args.algorithm,
            "--environment-id",
            args.environment_id,
            "--mechanism",
            args.mechanism,
            "--degradation-mode",
            args.degradation_mode,
            "--workers",
            str(args.workers),
            "--total-timesteps",
            str(args.total_timesteps),
            "--learning-rate",
            str(learning_rate),
            "--seed",
            str(seed),
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
