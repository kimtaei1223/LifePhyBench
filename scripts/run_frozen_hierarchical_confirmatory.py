"""Run the held-out campaign exactly as specified by FROZEN_PROTOCOL.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(command: list[str], project_root: Path) -> None:
    print("[CONFIRMATORY]", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/FROZEN_PROTOCOL.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/confirmatory"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("ready_for_confirmatory") is not True:
        raise SystemExit("frozen protocol is not confirmatory-ready")
    for relative, expected in protocol["source_sha256"].items():
        actual = sha256(project_root / relative)
        if actual != expected:
            raise SystemExit(f"source drift before confirmatory: {relative}")
    if protocol["training_strategy"]["teacher"] != 0.0:
        raise SystemExit("confirmatory protocol unexpectedly requires teacher shaping")

    calibration_metadata = sorted(
        (project_root / "outputs/hierarchical_autonomous_v10/learning_search")
        .glob("s1_curriculum_only/seed*/**/metadata.json")
    )
    if len(calibration_metadata) != 20:
        raise SystemExit("expected 20 calibration cell metadata files")
    reward_scales = {
        json.loads(path.read_text(encoding="utf-8"))["arguments"][
            "training_reward_scale"
        ]
        for path in calibration_metadata
    }
    if reward_scales != {0.02}:
        raise SystemExit(f"calibration reward-scale mismatch: {reward_scales}")

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = protocol["training_strategy"]
    design = protocol["physical_design"]
    manifest = {
        "phase": "hierarchical_thermal_held_out_confirmatory_campaign",
        "status": "frozen_before_any_confirmatory_training",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256(args.protocol.resolve()),
        "confirmatory_seeds": protocol["confirmatory_seeds_frozen"],
        "calibration_seeds_excluded": protocol["calibration_seeds_used"],
        "training_strategy": strategy,
        "physical_design": design,
        "training_reward_scale": 0.02,
        "training_reward_scale_provenance": (
            "verified invariant across all 20 frozen calibration cell metadata files"
        ),
        "evaluation_task_episodes": protocol["evaluation_task_episodes"],
        "low_level_model_sha256": protocol["low_level_model_sha256"],
        "source_sha256": protocol["source_sha256"],
        "statistical_plan": {
            "unit": "independent training seed",
            "primary_estimand": (
                "paired mean task reward: dynamic lifetime-memory minus dynamic task-reset"
            ),
            "primary_test": "one-sided one-sample t-test against zero",
            "primary_interval": "seed bootstrap 95% CI with 100000 resamples",
            "alpha": 0.05,
            "primary_success": "one-sided p < alpha and bootstrap lower bound > 0",
            "secondary": [
                "one-sided Wilcoxon signed-rank",
                "static memory effect",
                "dynamic-minus-static difference in differences",
                "cold-minus-hot high-power selection gap",
            ],
        },
        "legacy_note": (
            "per-run trainer metadata retains its calibration phase label because "
            "the frozen source hash forbids post-freeze edits; this campaign manifest "
            "is the authoritative study phase"
        ),
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"confirmatory manifest mismatch: {manifest_path}")
    else:
        write(manifest_path, manifest)
    if args.preflight_only:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print("[CONFIRMATORY PREFLIGHT COMPLETE — NO HELD-OUT TRAINING STARTED]")
        return

    runner = project_root / "scripts/run_hierarchical_thermal_pilot.py"
    seeds = protocol["confirmatory_seeds_frozen"]
    for index, seed in enumerate(seeds, start=1):
        seed_root = output_root / f"seed{seed}"
        command = [
            sys.executable, str(runner),
            "--seed", str(seed),
            "--total-task-decisions", str(strategy["decisions"]),
            "--workers", str(args.workers),
            "--torch-threads-per-process", "1",
            "--eval-task-episodes", str(protocol["evaluation_task_episodes"]),
            "--device", args.device,
            "--curriculum-start-trip-load", str(strategy["start"]),
            "--curriculum-lifetimes", str(strategy["duration"]),
            "--training-reward-scale", "0.02",
            "--ent-coef", str(strategy["entropy"]),
            "--learning-rate", str(strategy["lr"]),
            "--trip-load", str(design["trip_load"]),
            "--low-power-scale", str(design["low_power_scale"]),
            "--trip-penalty", str(design["trip_penalty"]),
            "--high-power-bonus", str(design["high_power_bonus"]),
            "--thermal-heat-rate", str(design["thermal_heat_rate"]),
            "--summary-mode", str(design["summary_mode"]),
            "--output-root", str(seed_root),
        ]
        print(f"[CONFIRMATORY SEED {index}/{len(seeds)}] {seed}", flush=True)
        run(command, project_root)
        write(
            output_root / "PROGRESS.json",
            {
                "completed_seeds": seeds[:index],
                "remaining_seeds": seeds[index:],
                "complete": index == len(seeds),
            },
        )
    run(
        [
            sys.executable,
            str(project_root / "scripts/analyze_frozen_hierarchical_confirmatory.py"),
            "--input-root",
            str(output_root),
        ],
        project_root,
    )
    print("[CONFIRMATORY CAMPAIGN AND ANALYSIS COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
