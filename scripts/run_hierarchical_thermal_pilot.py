"""Run the frozen four-cell hierarchical thermal calibration pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4_999)
    parser.add_argument("--total-task-decisions", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads-per-process", type=int, default=1)
    parser.add_argument("--eval-task-episodes", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--curriculum-start-trip-load", type=float, default=None)
    parser.add_argument("--curriculum-lifetimes", type=int, default=0)
    parser.add_argument("--training-reward-scale", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--trip-load", type=float, default=0.10)
    parser.add_argument("--low-power-scale", type=float, default=0.40)
    parser.add_argument("--trip-penalty", type=float, default=75.0)
    parser.add_argument("--high-power-bonus", type=float, default=2.0)
    parser.add_argument("--thermal-heat-rate", type=float, default=0.10)
    parser.add_argument("--teacher-safe-high-load", type=float, default=None)
    parser.add_argument("--teacher-shaping", type=float, default=0.0)
    parser.add_argument("--summary-mode", choices=["full", "mode_trip"], default="full")
    parser.add_argument(
        "--low-level-model",
        default=(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument(
        "--output-root", default="outputs/hierarchical_thermal_pilot_v8"
    )
    args = parser.parse_args()
    if min(
        args.seed,
        args.total_task_decisions,
        args.workers,
        args.eval_task_episodes,
        args.torch_threads_per_process,
    ) <= 0:
        raise SystemExit("seed and budgets must be positive")
    if (args.curriculum_lifetimes == 0) != (
        args.curriculum_start_trip_load is None
    ):
        raise SystemExit("curriculum start and duration must be enabled together")
    if (
        args.training_reward_scale <= 0.0
        or args.ent_coef < 0.0
        or args.learning_rate <= 0.0
    ):
        raise SystemExit("reward scale must be positive and entropy non-negative")
    if args.teacher_shaping < 0.0 or (
        args.teacher_shaping > 0.0 and args.teacher_safe_high_load is None
    ):
        raise SystemExit("teacher shaping requires a safe-high load")

    project_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    low_level_model = Path(args.low_level_model)
    if not low_level_model.is_absolute():
        low_level_model = (project_root / low_level_model).resolve()
    if not low_level_model.exists():
        raise SystemExit(f"low-level model does not exist: {low_level_model}")

    cells = (
        ("dynamic", "endogenous_action", "task"),
        ("dynamic", "endogenous_action", "lifetime"),
        ("static", "exogenous_clock", "task"),
        ("static", "exogenous_clock", "lifetime"),
    )
    manifest = {
        "phase": "hierarchical_thermal_gpu_calibration_pilot",
        "status": "calibration_not_confirmatory_evidence",
        "seed": args.seed,
        "total_task_decisions": args.total_task_decisions,
        "eval_task_episodes": args.eval_task_episodes,
        "workers": args.workers,
        "torch_threads_per_process": args.torch_threads_per_process,
        "curriculum_start_trip_load": args.curriculum_start_trip_load,
        "curriculum_lifetimes": args.curriculum_lifetimes,
        "training_reward_scale": args.training_reward_scale,
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "physical_design": {
            "trip_load": args.trip_load,
            "low_power_scale": args.low_power_scale,
            "trip_penalty": args.trip_penalty,
            "high_power_bonus": args.high_power_bonus,
            "thermal_heat_rate": args.thermal_heat_rate,
        },
        "training_teacher": {
            "safe_high_load": args.teacher_safe_high_load,
            "shaping": args.teacher_shaping,
            "evaluation_enabled": False,
        },
        "summary_mode": args.summary_mode,
        "low_level_model": str(low_level_model),
        "low_level_model_sha256": sha256(low_level_model),
        "selection_manifest": str(
            (project_root / "outputs/hierarchical_thermal_controller_qualification.json")
            .resolve()
        ),
        "automatic_stop": "after pilot validation; manual review required",
        "cells": [
            {
                "label": label,
                "degradation_mode": degradation,
                "memory_mode": memory,
            }
            for label, degradation, memory in cells
        ],
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(f"manifest mismatch: {manifest_path}; use a new root")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    trainer = project_root / "scripts/train_hierarchical_thermal.py"
    for index, (label, degradation, memory) in enumerate(cells, start=1):
        run_name = (
            f"hierarchical-thermal-{label}-{memory}-seed{args.seed}-"
            f"decisions{args.total_task_decisions // 1000}k"
        )
        run_directory = output_root / run_name
        if (run_directory / "metadata.json").exists():
            print(f"[SKIP {index}/4] {run_name}", flush=True)
            continue
        if run_directory.exists():
            raise SystemExit(f"incomplete run exists: {run_directory}")
        command = [
            sys.executable,
            str(trainer),
            "--memory-mode",
            memory,
            "--degradation-mode",
            degradation,
            "--workers",
            str(args.workers),
            "--torch-threads-per-process",
            str(args.torch_threads_per_process),
            "--total-task-decisions",
            str(args.total_task_decisions),
            "--eval-task-episodes",
            str(args.eval_task_episodes),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--training-reward-scale",
            str(args.training_reward_scale),
            "--ent-coef",
            str(args.ent_coef),
            "--learning-rate",
            str(args.learning_rate),
            "--trip-load",
            str(args.trip_load),
            "--low-power-scale",
            str(args.low_power_scale),
            "--trip-penalty",
            str(args.trip_penalty),
            "--high-power-bonus",
            str(args.high_power_bonus),
            "--thermal-heat-rate",
            str(args.thermal_heat_rate),
            "--summary-mode",
            args.summary_mode,
            "--low-level-model",
            str(low_level_model),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
        ]
        if args.curriculum_lifetimes > 0:
            command.extend(
                [
                    "--curriculum-start-trip-load",
                    str(args.curriculum_start_trip_load),
                    "--curriculum-lifetimes",
                    str(args.curriculum_lifetimes),
                ]
            )
        if args.teacher_shaping > 0.0:
            command.extend(
                [
                    "--teacher-safe-high-load",
                    str(args.teacher_safe_high_load),
                    "--teacher-shaping",
                    str(args.teacher_shaping),
                ]
            )
        print(f"[START {index}/4] {run_name}", flush=True)
        subprocess.run(command, check=True, cwd=project_root)
        print(f"[DONE {index}/4] {run_name}", flush=True)
    print("[PILOT COMPLETE — VALIDATE BEFORE ANY CONFIRMATORY RUN]", flush=True)


if __name__ == "__main__":
    main()
