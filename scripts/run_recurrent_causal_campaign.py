#!/usr/bin/env python3
"""Run resumable endogenous then exogenous recurrent development baselines."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class CampaignRun:
    degradation_mode: str
    memory_mode: str
    seed: int
    output_root: Path
    total_timesteps: int

    @property
    def run_name(self) -> str:
        return f"{self.memory_mode}-seed{self.seed}-steps{self.total_timesteps // 1000}k"

    @property
    def run_directory(self) -> Path:
        return self.output_root / self.run_name


def make_runs(
    *,
    endogenous_seeds: list[int],
    exogenous_seeds: list[int],
    memory_modes: list[str],
    total_timesteps: int,
    endogenous_output_root: Path,
    exogenous_output_root: Path,
) -> list[CampaignRun]:
    runs: list[CampaignRun] = []
    for mode, seed in product(memory_modes, endogenous_seeds):
        runs.append(
            CampaignRun(
                degradation_mode="endogenous_action",
                memory_mode=mode,
                seed=seed,
                output_root=endogenous_output_root,
                total_timesteps=total_timesteps,
            )
        )
    for mode, seed in product(memory_modes, exogenous_seeds):
        runs.append(
            CampaignRun(
                degradation_mode="exogenous_clock",
                memory_mode=mode,
                seed=seed,
                output_root=exogenous_output_root,
                total_timesteps=total_timesteps,
            )
        )
    return runs


def run_one(
    run: CampaignRun,
    *,
    trainer: Path,
    environment_id: str,
    mechanism: str,
    workers: int,
    learning_rate: float,
    device: str,
) -> None:
    metadata = run.run_directory / "metadata.json"
    if metadata.is_file():
        print(f"skip completed run: {run.run_directory}", flush=True)
        return
    if run.run_directory.exists():
        raise SystemExit(
            "refusing to overwrite incomplete run directory: "
            f"{run.run_directory}. Inspect or rename it before resuming."
        )
    command = [
        sys.executable,
        str(trainer),
        "--memory-mode",
        run.memory_mode,
        "--environment-id",
        environment_id,
        "--mechanism",
        mechanism,
        "--degradation-mode",
        run.degradation_mode,
        "--workers",
        str(workers),
        "--total-timesteps",
        str(run.total_timesteps),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(run.seed),
        "--device",
        device,
        "--output-root",
        str(run.output_root),
        "--run-name",
        run.run_name,
    ]
    print(
        "start "
        f"mode={run.degradation_mode} memory={run.memory_mode} seed={run.seed}",
        flush=True,
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--mechanism", default="wear")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--endogenous-seeds", type=int, nargs="+", default=[1002, 1003, 1004])
    parser.add_argument(
        "--exogenous-seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004]
    )
    parser.add_argument("--memory-modes", nargs="+", default=["episode", "lifetime"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--endogenous-output-root", default="outputs/recurrent_campaign")
    parser.add_argument(
        "--exogenous-output-root", default="outputs/recurrent_exogenous_campaign"
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.total_timesteps <= 0:
        raise SystemExit("workers and total-timesteps must be positive")
    invalid_modes = set(args.memory_modes) - {"episode", "lifetime"}
    if invalid_modes:
        raise SystemExit(f"invalid memory modes: {sorted(invalid_modes)}")

    runs = make_runs(
        endogenous_seeds=args.endogenous_seeds,
        exogenous_seeds=args.exogenous_seeds,
        memory_modes=args.memory_modes,
        total_timesteps=args.total_timesteps,
        endogenous_output_root=Path(args.endogenous_output_root),
        exogenous_output_root=Path(args.exogenous_output_root),
    )
    trainer = Path(__file__).with_name("train_recurrent_smoke.py")
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}]", flush=True)
        run_one(
            run,
            trainer=trainer,
            environment_id=args.environment_id,
            mechanism=args.mechanism,
            workers=args.workers,
            learning_rate=args.learning_rate,
            device=args.device,
        )


if __name__ == "__main__":
    main()
