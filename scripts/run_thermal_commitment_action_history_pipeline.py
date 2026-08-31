"""Run and validate the v7 action-history calibration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], project_root: Path) -> None:
    print(f"[PIPELINE] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=project_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/thermal_commitment_calibration_v7"),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    scripts = project_root / "scripts"
    run(
        [
            sys.executable,
            str(scripts / "run_thermal_commitment_masked_pilot.py"),
            "--seed",
            "4987",
            "--version-label",
            "v7",
            "--append-previous-applied-action",
            "--output-root",
            str(args.output_root),
        ],
        project_root,
    )
    run(
        [
            sys.executable,
            str(scripts / "validate_thermal_commitment_pilot.py"),
            "--input-root",
            str(args.output_root),
            "--output",
            str(args.output_root / "validation.json"),
        ],
        project_root,
    )
    run(
        [
            sys.executable,
            str(scripts / "validate_thermal_commitment_learnability.py"),
            "--input-root",
            str(args.output_root),
            "--output",
            str(args.output_root / "learnability_validation.json"),
        ],
        project_root,
    )
    print("[PIPELINE COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
