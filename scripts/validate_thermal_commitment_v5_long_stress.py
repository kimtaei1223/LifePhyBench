"""Validate the frozen v5 long-budget dynamic learnability stress test."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def at_least(value: Any, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def at_most(value: Any, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/thermal_commitment_v5_long_stress"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input_root / "validation.json"

    manifest = json.loads(
        (args.input_root / "manifest.json").read_text(encoding="utf-8")
    )
    expected_seeds = {int(seed) for seed in manifest["seeds"]}
    paths = sorted(args.input_root.glob("*/metadata.json"))
    if len(paths) != 2 * len(expected_seeds):
        raise SystemExit(
            f"expected {2 * len(expected_seeds)} runs, found {len(paths)}"
        )

    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    wiring_checks: dict[str, bool] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        seed = int(arguments["seed"])
        memory = str(arguments["memory_mode"])
        key = (seed, memory)
        if key in indexed:
            raise SystemExit(f"duplicate stress cell: {key}")
        indexed[key] = evaluation
        wiring_checks[path.parent.name] = all(
            (
                arguments.get("thermal_commitment") is True,
                arguments.get("degradation_mode") == "endogenous_action",
                int(arguments.get("total_timesteps", 0))
                == int(manifest["total_timesteps"]),
                int(arguments.get("eval_task_episodes", 0))
                == int(manifest["eval_task_episodes"]),
                float(arguments.get("commitment_trip_load", -1.0)) == 0.10,
                float(
                    arguments.get("commitment_curriculum_start_trip_load", -1.0)
                )
                == 0.70,
                int(arguments.get("commitment_curriculum_lifetimes", -1)) == 10,
                document.get("controlled_semantics", {}).get(
                    "training_trip_load_curriculum_only"
                )
                is True,
                float(
                    document.get("controlled_semantics", {}).get(
                        "evaluation_trip_load", -1.0
                    )
                )
                == 0.10,
                (path.parent / "model.zip").exists(),
            )
        )

    expected = {
        (seed, memory)
        for seed in expected_seeds
        for memory in ("task", "lifetime")
    }
    if set(indexed) != expected:
        raise SystemExit(f"factorial mismatch: {sorted(indexed)}")

    rule = manifest["frozen_success_rule"]
    per_seed_rule = rule["per_lifetime_seed"]
    seed_rows = []
    passing_seeds = 0
    for seed in sorted(expected_seeds):
        task = indexed[(seed, "task")]
        lifetime = indexed[(seed, "lifetime")]
        criteria = {
            "cold_support": at_least(
                lifetime["cold_mode_selections"],
                per_seed_rule["minimum_cold_mode_selections"],
            ),
            "hot_support": at_least(
                lifetime["hot_mode_selections"],
                per_seed_rule["minimum_hot_mode_selections"],
            ),
            "high_when_cold": at_least(
                lifetime["cold_high_power_selection_rate"],
                per_seed_rule["minimum_cold_high_power_selection_rate"],
            ),
            "low_when_hot": at_most(
                lifetime["hot_high_power_selection_rate"],
                per_seed_rule["maximum_hot_high_power_selection_rate"],
            ),
            "avoids_excess_trips": at_most(
                lifetime["thermal_trip_rate"],
                per_seed_rule["maximum_thermal_trip_rate"],
            ),
        }
        passed = all(criteria.values())
        passing_seeds += int(passed)
        seed_rows.append(
            {
                "seed": seed,
                "passed": passed,
                "criteria": criteria,
                "task_reward": task["mean_task_episode_reward"],
                "lifetime_reward": lifetime["mean_task_episode_reward"],
                "lifetime_minus_task_reward": (
                    lifetime["mean_task_episode_reward"]
                    - task["mean_task_episode_reward"]
                ),
                "task_high_power_selection_rate": task[
                    "high_power_selection_rate"
                ],
                "lifetime_high_power_selection_rate": lifetime[
                    "high_power_selection_rate"
                ],
                "lifetime_cold_high_power_selection_rate": lifetime[
                    "cold_high_power_selection_rate"
                ],
                "lifetime_hot_high_power_selection_rate": lifetime[
                    "hot_high_power_selection_rate"
                ],
                "lifetime_cold_mode_selections": lifetime[
                    "cold_mode_selections"
                ],
                "lifetime_hot_mode_selections": lifetime["hot_mode_selections"],
                "lifetime_thermal_trip_rate": lifetime["thermal_trip_rate"],
            }
        )

    reward_differences = [row["lifetime_minus_task_reward"] for row in seed_rows]
    wiring_passed = all(wiring_checks.values())
    behavior_passed = passing_seeds >= int(rule["minimum_passing_lifetime_seeds"])
    report = {
        "phase": "thermal_commitment_v5_long_dynamic_stress_validation",
        "status": "calibration_not_confirmatory_evidence",
        "wiring_passed": wiring_passed,
        "behavior_passed": behavior_passed,
        "passed": wiring_passed and behavior_passed,
        "passing_lifetime_seeds": passing_seeds,
        "required_passing_lifetime_seeds": int(
            rule["minimum_passing_lifetime_seeds"]
        ),
        "wiring_checks": wiring_checks,
        "seed_rows": seed_rows,
        "reward_difference_summary": {
            "mean": statistics.mean(reward_differences),
            "sample_sd": statistics.stdev(reward_differences),
            "interpretation": "exploratory calibration summary; not inference",
        },
    }
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("v5 long dynamic stress gate failed")


if __name__ == "__main__":
    main()
