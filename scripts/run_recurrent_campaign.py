#!/usr/bin/env python3
"""Run recurrent episode/lifetime baselines sequentially from one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--mechanism", default="wear")
    parser.add_argument("--degradation-mode", default="endogenous_action")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    parser.add_argument("--memory-modes", nargs="+", default=["episode", "lifetime"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/recurrent_campaign")
    args = parser.parse_args()
    invalid_modes = set(args.memory_modes) - {"episode", "lifetime"}
    if invalid_modes:
        raise SystemExit(f"invalid memory modes: {sorted(invalid_modes)}")
    trainer = Path(__file__).with_name("train_recurrent_smoke.py")
    for memory_mode, seed in product(args.memory_modes, args.seeds):
        run_name = f"{memory_mode}-seed{seed}-steps{args.total_timesteps // 1000}k"
        command = [
            sys.executable,
            str(trainer),
            "--memory-mode",
            memory_mode,
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
