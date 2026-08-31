"""Run the frozen canonical-reset thermal-probe experiment design.

This runner is intentionally inert until invoked. It records the planned
dynamic and static control cells but does not itself claim that either cell has
been validated by learned-policy results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CANONICAL_TASK_SEED = 811
THERMAL_HEAT_RATE = 0.1
THERMAL_COOLING_RATE = 0.0
THERMAL_EPISODE_COOLING = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(4000, 4010)))
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-task-episodes", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/canonical_thermal_probe")
    args = parser.parse_args()
    if min(args.seeds) < 0 or args.total_timesteps <= 0 or args.workers <= 0:
        raise SystemExit("seeds, total-timesteps, and workers must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = (
        ("endogenous_action", "dynamic", 0.0),
        ("exogenous_clock", "static", 0.0),
    )
    manifest = {
        "phase": "canonical_thermal_probe_pre_registered_design",
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "eval_task_episodes": args.eval_task_episodes,
        "cells": [
            {
                "degradation_mode": degradation_mode,
                "label": label,
                "thermal_exogenous_dose_per_step": dose,
            }
            for degradation_mode, label, dose in cells
        ],
        "canonical_task_seed": CANONICAL_TASK_SEED,
        "thermal_parameters": {
            "heat_rate": THERMAL_HEAT_RATE,
            "cooling_rate": THERMAL_COOLING_RATE,
            "episode_cooling": THERMAL_EPISODE_COOLING,
        },
        "controlled_semantics": {
            "gym_and_gae_boundary": "lifetime_only_for_all_cells",
            "task_boundary_marker": "observed_by_all_cells",
            "memory_intervention": "forced_lstm_reset_at_task_boundary_only",
            "static_control": "thermal exogenous dose fixed to zero",
        },
    }
    manifest_path = output_root / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch in {manifest_path}; use a new output root")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = len(cells) * 2 * len(args.seeds)
    index = 0
    for degradation_mode, label, dose in cells:
        for memory_mode in ("task", "lifetime"):
            for seed in args.seeds:
                index += 1
                name = (
                    f"canonical-thermal-{label}-{memory_mode}-seed{seed}-"
                    f"steps{args.total_timesteps // 1000}k"
                )
                run_directory = output_root / name
                if (run_directory / "metadata.json").exists():
                    print(f"[SKIP {index}/{total}] {name}", flush=True)
                    continue
                if run_directory.exists():
                    raise SystemExit(f"incomplete run exists: {run_directory}; use a new output root")
                command = [
                    sys.executable,
                    str(project_root / "scripts/train_fair_recurrent.py"),
                    "--memory-mode", memory_mode,
                    "--environment-id", "Pusher-v5",
                    "--mechanism", "thermal",
                    "--degradation-mode", degradation_mode,
                    "--workers", str(args.workers),
                    "--total-timesteps", str(args.total_timesteps),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--eval-task-episodes", str(args.eval_task_episodes),
                    "--thermal-exogenous-dose-per-step", str(dose),
                    "--thermal-heat-rate", str(THERMAL_HEAT_RATE),
                    "--thermal-cooling-rate", str(THERMAL_COOLING_RATE),
                    "--thermal-episode-cooling", str(THERMAL_EPISODE_COOLING),
                    "--canonical-task-seed", str(CANONICAL_TASK_SEED),
                    "--output-root", str(output_root),
                    "--run-name", name,
                ]
                print(f"[START {index}/{total}] {name}", flush=True)
                subprocess.run(command, check=True, cwd=project_root)
                print(f"[DONE {index}/{total}] {name}", flush=True)
    print("[CAMPAIGN COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
