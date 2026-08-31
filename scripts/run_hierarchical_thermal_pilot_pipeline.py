"""Run CPU gates, the GPU pilot, validation, and then stop for review."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(command: list[str], project_root: Path) -> None:
    print("[PIPELINE]", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    scripts = project_root / "scripts"
    python = sys.executable
    run([python, str(scripts / "qualify_hierarchical_low_level_controllers.py")], project_root)
    run([python, str(scripts / "run_hierarchical_thermal_gate.py")], project_root)
    run(
        [
            python,
            str(scripts / "run_hierarchical_thermal_pilot.py"),
            "--total-task-decisions",
            "50000",
            "--eval-task-episodes",
            "400",
            "--workers",
            "8",
            "--torch-threads-per-process",
            "1",
            "--device",
            "cuda",
        ],
        project_root,
    )
    run(
        [
            python,
            str(scripts / "validate_hierarchical_thermal_pilot.py"),
            "--expected-task-decisions",
            "50000",
            "--expected-eval-task-episodes",
            "400",
        ],
        project_root,
    )
    print("[STOPPED AS DESIGNED — MANUAL REVIEW REQUIRED]", flush=True)


if __name__ == "__main__":
    main()
