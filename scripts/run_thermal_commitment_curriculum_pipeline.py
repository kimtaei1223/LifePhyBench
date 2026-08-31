"""Run and validate the v5 thermal-commitment curriculum calibration."""

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
        default=Path("outputs/thermal_commitment_calibration_v5"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scripts = project_root / "scripts"
    output_root = args.output_root
    run(
        [
            sys.executable,
            str(scripts / "run_thermal_commitment_curriculum_pilot.py"),
            "--output-root",
            str(output_root),
        ],
        project_root,
    )
    run(
        [
            sys.executable,
            str(scripts / "validate_thermal_commitment_pilot.py"),
            "--input-root",
            str(output_root),
            "--output",
            str(output_root / "validation.json"),
        ],
        project_root,
    )
    run(
        [
            sys.executable,
            str(scripts / "validate_thermal_commitment_learnability.py"),
            "--input-root",
            str(output_root),
            "--output",
            str(output_root / "learnability_validation.json"),
        ],
        project_root,
    )
    print("[PIPELINE COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
