#!/usr/bin/env python3
"""Run every result-independent Reacher stage-2 task in frozen order."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/reacher_replication")
    )
    parser.add_argument("--belief-workers", type=int, default=12)
    parser.add_argument("--training-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "stage2_status.json"
    commands = [
        [
            sys.executable,
            str(project_root / "scripts/run_reacher_belief_development.py"),
            "--workers", str(args.belief_workers),
            "--device", args.device,
            "--resume",
        ],
        [
            sys.executable,
            str(project_root / "scripts/run_reacher_monolithic_baseline.py"),
            "--workers", str(args.training_workers),
            "--device", args.device,
        ],
    ]
    for ordinal, command in enumerate(commands, start=1):
        status_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "substage": ordinal,
                    "substages": len(commands),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(command, check=True, cwd=project_root)
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "substages": len(commands),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[REACHER STAGE2 COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
