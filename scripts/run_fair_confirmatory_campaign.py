#!/usr/bin/env python3
"""Run a held-out 2x2 campaign with reset and dose confounds controlled."""

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
    parser.add_argument(
        "--fixed-exogenous-dose",
        type=float,
        default=0.008750098932466562,
        help="One frozen pilot-calibrated dose shared by both memory arms.",
    )
    parser.add_argument("--output-root", default="outputs/fair_confirmatory")
    args = parser.parse_args()
    if args.fixed_exogenous_dose < 0.0:
        raise SystemExit("fixed-exogenous-dose must be non-negative")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "held_out_fair_confirmatory",
        "seeds": args.seeds,
        "total_timesteps": args.total_timesteps,
        "eval_task_episodes": args.eval_task_episodes,
        "fixed_exogenous_dose": args.fixed_exogenous_dose,
        "dose_source": (
            "frozen pooled median of the ten 2M pilot policy-dose estimates; "
            "shared by task and lifetime memory arms"
        ),
        "controlled_semantics": {
            "gym_and_gae_boundary": "lifetime_only_for_all_four_cells",
            "task_boundary_marker": "observed_by_all_four_cells",
            "memory_intervention": "forced_lstm_reset_at_task_boundary_only",
        },
    }
    manifest_path = output_root / "campaign_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(
                f"manifest mismatch in {manifest_path}; use a new --output-root"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    conditions = [
        (degradation, memory)
        for degradation in ("endogenous_action", "exogenous_clock")
        for memory in ("task", "lifetime")
    ]
    total = len(conditions) * len(args.seeds)
    index = 0
    for degradation, memory in conditions:
        for seed in args.seeds:
            index += 1
            name = (
                f"thermal-{degradation}-{memory}-seed{seed}-"
                f"steps{args.total_timesteps // 1000}k"
            )
            directory = output_root / name
            if (directory / "metadata.json").exists():
                print(f"[SKIP {index}/{total}] {name}", flush=True)
                continue
            if directory.exists():
                raise SystemExit(
                    f"incomplete run exists: {directory}; preserve it and use a "
                    "different --output-root"
                )
            command = [
                sys.executable,
                str(project_root / "scripts/train_fair_recurrent.py"),
                "--memory-mode",
                memory,
                "--environment-id",
                "Pusher-v5",
                "--mechanism",
                "thermal",
                "--degradation-mode",
                degradation,
                "--workers",
                str(args.workers),
                "--total-timesteps",
                str(args.total_timesteps),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--eval-task-episodes",
                str(args.eval_task_episodes),
                "--thermal-exogenous-dose-per-step",
                str(args.fixed_exogenous_dose),
                "--output-root",
                str(output_root),
                "--run-name",
                name,
            ]
            print(f"[START {index}/{total}] {name}", flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            print(f"[DONE {index}/{total}] {name}", flush=True)
    print("[CAMPAIGN COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
