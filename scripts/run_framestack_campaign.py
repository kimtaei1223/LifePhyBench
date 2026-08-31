#!/usr/bin/env python3
"""Run finite-history PPO development trials sequentially from one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--history-mode", choices=["task", "lifetime"], default="lifetime")
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--mechanism", default="wear")
    parser.add_argument("--degradation-mode", default="endogenous_action")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/framestack_campaign")
    args = parser.parse_args()
    if any(size <= 0 for size in args.stack_sizes):
        raise SystemExit("stack sizes must be positive")

    trainer = Path(__file__).with_name("train_framestack_ppo.py")
    for stack_size, seed in product(args.stack_sizes, args.seeds):
        run_name = (
            f"{args.history_mode}-stack{stack_size}-seed{seed}-"
            f"steps{args.total_timesteps // 1000}k"
        )
        command = [
            sys.executable,
            str(trainer),
            "--stack-size",
            str(stack_size),
            "--history-mode",
            args.history_mode,
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
            str(args.learning_rate),
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
