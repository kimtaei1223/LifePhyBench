"""Execute a GPU pilot, semantic gate, and full canonical-probe campaign in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], project_root: Path) -> None:
    print("[PIPELINE]", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=project_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot-seed", type=int, default=3999)
    parser.add_argument("--pilot-timesteps", type=int, default=50_000)
    parser.add_argument("--pilot-eval-task-episodes", type=int, default=20)
    parser.add_argument("--full-timesteps", type=int, default=2_000_000)
    parser.add_argument("--full-eval-task-episodes", type=int, default=1_000)
    parser.add_argument("--pilot-output-root", default="outputs/canonical_thermal_probe_pilot")
    parser.add_argument("--full-output-root", default="outputs/canonical_thermal_probe")
    args = parser.parse_args()
    if args.workers <= 0 or args.pilot_timesteps <= 0 or args.full_timesteps <= 0:
        raise SystemExit("workers and timestep budgets must be positive")

    project_root = Path(__file__).resolve().parent.parent
    python = sys.executable
    runner = str(project_root / "scripts/run_canonical_thermal_probe_campaign.py")
    validator = str(project_root / "scripts/validate_canonical_thermal_probe_pilot.py")
    run(
        [
            python,
            runner,
            "--seeds", str(args.pilot_seed),
            "--total-timesteps", str(args.pilot_timesteps),
            "--workers", str(args.workers),
            "--eval-task-episodes", str(args.pilot_eval_task_episodes),
            "--device", args.device,
            "--output-root", args.pilot_output_root,
        ],
        project_root,
    )
    run(
        [
            python,
            validator,
            "--output-root", args.pilot_output_root,
            "--expected-task-episodes", str(args.pilot_eval_task_episodes),
        ],
        project_root,
    )
    run(
        [
            python,
            runner,
            "--total-timesteps", str(args.full_timesteps),
            "--workers", str(args.workers),
            "--eval-task-episodes", str(args.full_eval_task_episodes),
            "--device", args.device,
            "--output-root", args.full_output_root,
        ],
        project_root,
    )
    print("[PIPELINE COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
