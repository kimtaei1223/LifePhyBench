"""Bounded automatic calibration for hierarchical thermal learning.

This script may change training-only curriculum and optimization settings.  It
never changes environment physics, observations, rewards used for evaluation,
or pass thresholds.  Confirmatory seeds are deliberately outside this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


CANDIDATES = (
    {
        "name": "c1_mild_curriculum",
        "curriculum_start_trip_load": 0.30,
        "curriculum_lifetimes": 120,
        "training_reward_scale": 0.02,
        "ent_coef": 0.005,
    },
    {
        "name": "c2_medium_curriculum",
        "curriculum_start_trip_load": 0.60,
        "curriculum_lifetimes": 180,
        "training_reward_scale": 0.02,
        "ent_coef": 0.01,
    },
    {
        "name": "c3_long_curriculum",
        "curriculum_start_trip_load": 1.00,
        "curriculum_lifetimes": 240,
        "training_reward_scale": 0.02,
        "ent_coef": 0.01,
    },
    {
        "name": "c4_exploration_curriculum",
        "curriculum_start_trip_load": 1.00,
        "curriculum_lifetimes": 240,
        "training_reward_scale": 0.02,
        "ent_coef": 0.03,
    },
)
CALIBRATION_SEEDS = (5200, 5201, 5202)


def run(command: list[str], project_root: Path) -> None:
    print("[AUTO]", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, check=True)


def row_lookup(report: dict) -> dict[tuple[str, str], dict]:
    return {(row["label"], row["memory"]): row for row in report["rows"]}


def screen_passed(report: dict) -> bool:
    checks = report["behavior_checks"]
    return bool(
        report["wiring_passed"]
        and checks["static_task_prefers_high"]
        and checks["static_lifetime_prefers_high"]
        and checks["dynamic_lifetime_uses_high_more_when_cold"]
    )


def replication_summary(reports: list[dict]) -> dict:
    reward_differences = []
    adaptation_gaps = []
    individual = []
    for report in reports:
        rows = row_lookup(report)
        task = rows[("dynamic", "task")]
        lifetime = rows[("dynamic", "lifetime")]
        reward_difference = lifetime["mean_reward"] - task["mean_reward"]
        adaptation_gap = lifetime["cold_high_rate"] - lifetime["hot_high_rate"]
        reward_differences.append(reward_difference)
        adaptation_gaps.append(adaptation_gap)
        checks = report["behavior_checks"]
        individual.append(
            {
                "wiring_passed": report["wiring_passed"],
                "static_controls_passed": (
                    checks["static_task_prefers_high"]
                    and checks["static_lifetime_prefers_high"]
                ),
                "adaptive_lifetime": checks[
                    "dynamic_lifetime_uses_high_more_when_cold"
                ],
                "positive_memory_effect": reward_difference > 0.0,
                "reward_difference": reward_difference,
                "adaptation_gap": adaptation_gap,
            }
        )
    criteria = {
        "all_wiring_passed": all(row["wiring_passed"] for row in individual),
        "all_static_controls_passed": all(
            row["static_controls_passed"] for row in individual
        ),
        "adaptive_lifetime_in_at_least_two_seeds": sum(
            row["adaptive_lifetime"] for row in individual
        )
        >= 2,
        "positive_memory_effect_in_at_least_two_seeds": sum(
            row["positive_memory_effect"] for row in individual
        )
        >= 2,
        "mean_adaptation_gap_above_point_one": (
            statistics.fmean(adaptation_gaps) > 0.10
        ),
        "mean_memory_effect_positive": statistics.fmean(reward_differences) > 0.0,
    }
    return {
        "individual": individual,
        "mean_reward_difference": statistics.fmean(reward_differences),
        "mean_adaptation_gap": statistics.fmean(adaptation_gaps),
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def write_status(output_root: Path, payload: dict) -> None:
    (output_root / "AUTO_CALIBRATION_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-task-decisions", type=int, default=50_000)
    parser.add_argument("--eval-task-episodes", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hierarchical_thermal_auto_calibration_v9"),
    )
    args = parser.parse_args()
    if min(args.workers, args.total_task_decisions, args.eval_task_episodes) <= 0:
        raise SystemExit("budgets must be positive")

    project_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "bounded_hierarchical_thermal_auto_calibration",
        "status": "calibration_only_confirmatory_seeds_untouched",
        "fixed_environment": {
            "trip_load": 0.10,
            "low_power_scale": 0.40,
            "trip_penalty": 75.0,
            "high_power_bonus": 2.0,
            "thermal_heat_rate": 0.10,
            "observations_unchanged": True,
            "evaluation_rewards_unscaled": True,
        },
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "confirmatory_seeds": "reserved; never accessed by this script",
        "candidates_in_fixed_order": list(CANDIDATES),
        "screen_rule": (
            "wiring + both static controls + dynamic lifetime cold-hot gap > 0.10"
        ),
        "replication_rule": (
            "three calibration seeds; fixed criteria in replication_summary"
        ),
        "maximum_campaigns": len(CANDIDATES) * len(CALIBRATION_SEEDS),
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"manifest mismatch: {manifest_path}; use a new root")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    runner = project_root / "scripts/run_hierarchical_thermal_pilot.py"
    validator = project_root / "scripts/validate_hierarchical_thermal_pilot.py"
    candidate_results = []
    for candidate_index, candidate in enumerate(CANDIDATES, start=1):
        reports = []
        candidate_record = {
            "candidate": candidate,
            "screen_passed": False,
            "replication": None,
            "seed_reports": [],
        }
        for seed_index, seed in enumerate(CALIBRATION_SEEDS):
            run_root = output_root / candidate["name"] / f"seed{seed}"
            command = [
                sys.executable,
                str(runner),
                "--seed",
                str(seed),
                "--total-task-decisions",
                str(args.total_task_decisions),
                "--workers",
                str(args.workers),
                "--torch-threads-per-process",
                "1",
                "--eval-task-episodes",
                str(args.eval_task_episodes),
                "--device",
                args.device,
                "--curriculum-start-trip-load",
                str(candidate["curriculum_start_trip_load"]),
                "--curriculum-lifetimes",
                str(candidate["curriculum_lifetimes"]),
                "--training-reward-scale",
                str(candidate["training_reward_scale"]),
                "--ent-coef",
                str(candidate["ent_coef"]),
                "--output-root",
                str(run_root),
            ]
            run(command, project_root)
            run(
                [
                    sys.executable,
                    str(validator),
                    "--input-root",
                    str(run_root),
                    "--expected-task-decisions",
                    str(args.total_task_decisions),
                    "--expected-eval-task-episodes",
                    str(args.eval_task_episodes),
                ],
                project_root,
            )
            report_path = run_root / "MANUAL_REVIEW_REQUIRED.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reports.append(report)
            candidate_record["seed_reports"].append(str(report_path))
            if seed_index == 0:
                candidate_record["screen_passed"] = screen_passed(report)
                if not candidate_record["screen_passed"]:
                    print(
                        f"[AUTO {candidate_index}/{len(CANDIDATES)}] "
                        f"screen failed: {candidate['name']}",
                        flush=True,
                    )
                    break

        if candidate_record["screen_passed"] and len(reports) == 3:
            candidate_record["replication"] = replication_summary(reports)
        candidate_results.append(candidate_record)
        status = {
            "phase": "bounded_hierarchical_thermal_auto_calibration",
            "candidate_results": candidate_results,
            "selected_candidate": None,
            "complete": False,
        }
        if (
            candidate_record["replication"] is not None
            and candidate_record["replication"]["passed"]
        ):
            status.update(
                {
                    "selected_candidate": candidate,
                    "complete": True,
                    "outcome": "calibration_pass_manual_confirmatory_freeze_required",
                }
            )
            write_status(output_root, status)
            print("[AUTO CALIBRATION PASS — STOP BEFORE CONFIRMATORY]", flush=True)
            return
        write_status(output_root, status)

    final = {
        "phase": "bounded_hierarchical_thermal_auto_calibration",
        "candidate_results": candidate_results,
        "selected_candidate": None,
        "complete": True,
        "outcome": "all_predeclared_candidates_failed_structural_redesign_required",
    }
    write_status(output_root, final)
    print("[AUTO CALIBRATION EXHAUSTED — STRUCTURAL REDESIGN REQUIRED]", flush=True)


if __name__ == "__main__":
    main()
