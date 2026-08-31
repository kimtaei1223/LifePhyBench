"""Run the no-degradation control for the fair recurrent-memory intervention."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(3000, 3010)))
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-task-episodes", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs/fair_static_health_control")
    args = parser.parse_args()
    if min(args.seeds) < 0 or args.total_timesteps <= 0 or args.workers <= 0:
        raise SystemExit("seeds, total-timesteps, and workers must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "fair_static_health_control",
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "eval_task_episodes": args.eval_task_episodes,
        "controlled_semantics": {
            "gym_and_gae_boundary": "lifetime_only_for_both_arms",
            "task_boundary_marker": "observed_by_both_arms",
            "memory_intervention": "forced_lstm_reset_at_task_boundary_only",
            "thermal_exogenous_dose_per_step": 0.0,
            "expected_thermal_load": 0.0,
            "expected_actuator_efficiency": 1.0,
        },
    }
    manifest_path = output_root / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch in {manifest_path}; use a new output root")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = 2 * len(args.seeds)
    index = 0
    for memory_mode in ("task", "lifetime"):
        for seed in args.seeds:
            index += 1
            name = f"thermal-static-health-{memory_mode}-seed{seed}-steps{args.total_timesteps // 1000}k"
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
                "--degradation-mode", "exogenous_clock",
                "--workers", str(args.workers),
                "--total-timesteps", str(args.total_timesteps),
                "--seed", str(seed),
                "--device", args.device,
                "--eval-task-episodes", str(args.eval_task_episodes),
                "--thermal-exogenous-dose-per-step", "0.0",
                "--output-root", str(output_root),
                "--run-name", name,
            ]
            print(f"[START {index}/{total}] {name}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE {index}/{total}] {name}", flush=True)
    print("[CAMPAIGN COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
