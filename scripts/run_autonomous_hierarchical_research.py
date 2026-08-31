"""Autonomously reach a frozen, confirmatory-ready hierarchical protocol.

The loop is deliberately large but finite.  Physics and observation semantics
come from the CPU oracle search and are frozen before this script trains a
policy.  Training-only strategies are tried in a declared order, screened on
one calibration seed, and promoted to five-seed replication.  Confirmatory
seeds are never run here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


CALIBRATION_SEEDS = (5300, 5301, 5302, 5303, 5304)
CONFIRMATORY_SEEDS = tuple(range(6300, 6320))
STRATEGIES = (
    {"name": "s1_curriculum_only", "decisions": 50_000, "start": 0.30, "duration": 120, "teacher": 0.0, "entropy": 0.005, "lr": 3e-4},
    {"name": "s2_light_teacher", "decisions": 100_000, "start": 0.30, "duration": 300, "teacher": 5.0, "entropy": 0.005, "lr": 3e-4},
    {"name": "s3_medium_teacher", "decisions": 100_000, "start": 0.60, "duration": 300, "teacher": 10.0, "entropy": 0.01, "lr": 3e-4},
    {"name": "s4_long_medium_teacher", "decisions": 200_000, "start": 0.60, "duration": 600, "teacher": 20.0, "entropy": 0.01, "lr": 3e-4},
    {"name": "s5_long_strong_teacher", "decisions": 200_000, "start": 1.00, "duration": 600, "teacher": 40.0, "entropy": 0.01, "lr": 3e-4},
    {"name": "s6_slow_finetune", "decisions": 400_000, "start": 1.00, "duration": 1_200, "teacher": 20.0, "entropy": 0.02, "lr": 1e-4},
    {"name": "s7_slow_strong_teacher", "decisions": 400_000, "start": 1.00, "duration": 1_200, "teacher": 40.0, "entropy": 0.02, "lr": 1e-4},
    {"name": "s8_extended_teacher", "decisions": 800_000, "start": 1.00, "duration": 2_400, "teacher": 75.0, "entropy": 0.03, "lr": 1e-4},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], project_root: Path) -> None:
    print("[AUTONOMOUS]", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {(row["label"], row["memory"]): row for row in report["rows"]}


def metrics(report: dict) -> dict:
    rows = lookup(report)
    task = rows[("dynamic", "task")]
    lifetime = rows[("dynamic", "lifetime")]
    static_task = rows[("static", "task")]
    static_lifetime = rows[("static", "lifetime")]
    cold = lifetime["cold_high_rate"]
    hot = lifetime["hot_high_rate"]
    adaptation_gap = None if cold is None or hot is None else cold - hot
    return {
        "wiring": report["wiring_passed"],
        "static_controls": static_task["high_rate"] > 0.75 and static_lifetime["high_rate"] > 0.75,
        "reward_effect": lifetime["mean_reward"] - task["mean_reward"],
        "adaptation_gap": adaptation_gap,
        "lifetime_high_rate": lifetime["high_rate"],
        "lifetime_trip_rate": lifetime["trip_rate"],
    }


def screen_passed(row: dict) -> bool:
    return bool(
        row["wiring"]
        and row["static_controls"]
        and row["adaptation_gap"] is not None
        and row["adaptation_gap"] > 0.10
        and row["reward_effect"] > 0.0
        and 0.02 < row["lifetime_high_rate"] < 0.98
    )


def robust_gate(rows: list[dict]) -> dict:
    effects = [row["reward_effect"] for row in rows]
    gaps = [row["adaptation_gap"] for row in rows if row["adaptation_gap"] is not None]
    criteria = {
        "all_wiring": all(row["wiring"] for row in rows),
        "all_static_controls": all(row["static_controls"] for row in rows),
        "adaptive_in_four_of_five": sum(
            row["adaptation_gap"] is not None and row["adaptation_gap"] > 0.10
            for row in rows
        ) >= 4,
        "positive_effect_in_four_of_five": sum(effect > 0.0 for effect in effects) >= 4,
        "nondegenerate_in_four_of_five": sum(
            0.02 < row["lifetime_high_rate"] < 0.98 for row in rows
        ) >= 4,
        "mean_effect_above_half": statistics.fmean(effects) > 0.5,
        "median_effect_positive": statistics.median(effects) > 0.0,
        "mean_adaptation_gap_above_point_one": (
            len(gaps) == 5 and statistics.fmean(gaps) > 0.10
        ),
    }
    return {
        "seed_metrics": rows,
        "mean_reward_effect": statistics.fmean(effects),
        "median_reward_effect": statistics.median(effects),
        "mean_adaptation_gap": statistics.fmean(gaps) if gaps else None,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-task-episodes", type=int, default=1_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--design-search",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/design_search.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/learning_search"),
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    design_document = json.loads(args.design_search.read_text(encoding="utf-8"))
    if not design_document["passed"]:
        raise SystemExit("oracle design search did not pass")
    design = design_document["selected_design"]
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "autonomous_hierarchical_learning_search",
        "status": "calibration_only_confirmatory_untouched",
        "design_search": str(args.design_search.resolve()),
        "design": design,
        "strategies_in_fixed_order": list(STRATEGIES),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "reserved_confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "promotion": "one-seed screen then five-seed robust gate",
        "evaluation_task_episodes": args.eval_task_episodes,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"manifest mismatch: {manifest_path}")
    else:
        write(manifest_path, manifest)

    runner = project_root / "scripts/run_hierarchical_thermal_pilot.py"
    validator = project_root / "scripts/validate_hierarchical_thermal_pilot.py"
    strategy_records = []
    for strategy in STRATEGIES:
        record = {"strategy": strategy, "screen_passed": False, "seed_results": [], "robust_gate": None}
        for seed_index, seed in enumerate(CALIBRATION_SEEDS):
            run_root = output_root / strategy["name"] / f"seed{seed}"
            command = [
                sys.executable, str(runner),
                "--seed", str(seed),
                "--total-task-decisions", str(strategy["decisions"]),
                "--workers", str(args.workers),
                "--torch-threads-per-process", "1",
                "--eval-task-episodes", str(args.eval_task_episodes),
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
                "--output-root", str(run_root),
            ]
            if strategy["teacher"] > 0.0:
                command.extend(
                    [
                        "--teacher-safe-high-load", str(design["teacher_safe_high_load"]),
                        "--teacher-shaping", str(strategy["teacher"]),
                    ]
                )
            run(command, project_root)
            run(
                [
                    sys.executable, str(validator),
                    "--input-root", str(run_root),
                    "--expected-task-decisions", str(strategy["decisions"]),
                    "--expected-eval-task-episodes", str(args.eval_task_episodes),
                ],
                project_root,
            )
            report_path = run_root / "MANUAL_REVIEW_REQUIRED.json"
            row = metrics(json.loads(report_path.read_text(encoding="utf-8")))
            record["seed_results"].append({"seed": seed, "report": str(report_path), "metrics": row})
            if seed_index == 0:
                record["screen_passed"] = screen_passed(row)
                if not record["screen_passed"]:
                    print(f"[SCREEN FAIL] {strategy['name']}", flush=True)
                    break
        if record["screen_passed"] and len(record["seed_results"]) == 5:
            record["robust_gate"] = robust_gate(
                [item["metrics"] for item in record["seed_results"]]
            )
        strategy_records.append(record)
        status = {
            "phase": "autonomous_hierarchical_learning_search",
            "complete": False,
            "selected_strategy": None,
            "strategy_records": strategy_records,
        }
        if record["robust_gate"] and record["robust_gate"]["passed"]:
            frozen = {
                "phase": "hierarchical_thermal_frozen_confirmatory_protocol",
                "ready_for_confirmatory": True,
                "physical_design": design,
                "training_strategy": strategy,
                "low_level_model": design_document["low_level_model"],
                "low_level_model_sha256": design_document["low_level_model_sha256"],
                "calibration_seeds_used": list(CALIBRATION_SEEDS),
                "confirmatory_seeds_frozen": list(CONFIRMATORY_SEEDS),
                "robust_calibration_gate": record["robust_gate"],
                "evaluation_task_episodes": args.eval_task_episodes,
                "source_sha256": {
                    name: sha256(project_root / name)
                    for name in (
                        "src/lifephybench/envs/hierarchical_thermal.py",
                        "scripts/train_hierarchical_thermal.py",
                        "scripts/run_hierarchical_thermal_pilot.py",
                        "scripts/validate_hierarchical_thermal_pilot.py",
                    )
                },
                "automatic_stop": "confirmatory campaign requires manual authorization",
            }
            write(output_root.parent / "FROZEN_PROTOCOL.json", frozen)
            status.update(
                {
                    "complete": True,
                    "selected_strategy": strategy,
                    "outcome": "ready_for_confirmatory_manual_authorization_required",
                }
            )
            write(output_root / "AUTONOMOUS_STATUS.json", status)
            print("[READY FOR CONFIRMATORY — AUTOMATIC STOP]", flush=True)
            return
        write(output_root / "AUTONOMOUS_STATUS.json", status)

    status = {
        "phase": "autonomous_hierarchical_learning_search",
        "complete": True,
        "selected_strategy": None,
        "outcome": "exhausted_declared_strategies_not_ready",
        "strategy_records": strategy_records,
    }
    write(output_root / "AUTONOMOUS_STATUS.json", status)
    print("[DECLARED SEARCH EXHAUSTED — NOT READY]", flush=True)


if __name__ == "__main__":
    main()
