"""Validate the hierarchical thermal pilot and create a manual-review report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/hierarchical_thermal_pilot_v8"),
    )
    parser.add_argument("--expected-task-decisions", type=int, default=50_000)
    parser.add_argument("--expected-eval-task-episodes", type=int, default=400)
    args = parser.parse_args()
    manifest = json.loads(
        (args.input_root / "manifest.json").read_text(encoding="utf-8")
    )
    paths = sorted(args.input_root.glob("*/metadata.json"))
    if len(paths) != 4:
        raise SystemExit(f"expected four pilot metadata files, found {len(paths)}")

    rows = []
    seen = set()
    wiring_checks: dict[str, bool] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        label = (
            "dynamic"
            if arguments["degradation_mode"] == "endogenous_action"
            else "static"
        )
        key = (label, arguments["memory_mode"])
        if key in seen:
            raise SystemExit(f"duplicate pilot cell: {key}")
        seen.add(key)
        prefix = f"{label}_{arguments['memory_mode']}"
        physical_design = manifest["physical_design"]
        wiring_checks.update(
            {
                f"{prefix}_task_decisions": arguments["total_task_decisions"]
                == args.expected_task_decisions,
                f"{prefix}_eval_episodes": evaluation["task_episodes"]
                == args.expected_eval_task_episodes,
                f"{prefix}_controller_hash": document["low_level_model_sha256"]
                == manifest["low_level_model_sha256"],
                f"{prefix}_thread_cap": arguments[
                    "torch_threads_per_process"
                ]
                == 1,
                f"{prefix}_frozen_controller": document["controlled_semantics"][
                    "low_level_controller_frozen"
                ]
                is True,
                f"{prefix}_health_hidden": document["controlled_semantics"][
                    "privileged_health_exposed"
                ]
                is False,
                f"{prefix}_finite_reward": math.isfinite(
                    evaluation["mean_task_episode_reward"]
                ),
                f"{prefix}_valid_high_rate": 0.0
                <= evaluation["high_power_selection_rate"]
                <= 1.0,
                f"{prefix}_physical_design_frozen": all(
                    arguments[name] == physical_design[name]
                    for name in (
                        "trip_load",
                        "low_power_scale",
                        "trip_penalty",
                        "high_power_bonus",
                        "thermal_heat_rate",
                    )
                ),
                f"{prefix}_evaluation_reward_unscaled": document[
                    "controlled_semantics"
                ]["evaluation_reward_unscaled"]
                is True,
                f"{prefix}_summary_mode": arguments["summary_mode"]
                == manifest["summary_mode"],
            }
        )
        if label == "static":
            wiring_checks[f"{prefix}_static_load_zero"] = (
                evaluation["mean_episode_end_thermal_load"] == 0.0
            )
            wiring_checks[f"{prefix}_static_trip_zero"] = (
                evaluation["thermal_trip_rate"] == 0.0
            )
        else:
            wiring_checks[f"{prefix}_dynamic_load_positive"] = (
                evaluation["mean_episode_end_thermal_load"] > 0.0
            )
            wiring_checks[f"{prefix}_cold_and_hot_observed"] = (
                evaluation["cold_mode_selections"] > 0
                and evaluation["hot_mode_selections"] > 0
            )
        rows.append(
            {
                "label": label,
                "memory": arguments["memory_mode"],
                "mean_reward": evaluation["mean_task_episode_reward"],
                "reward_sd": evaluation["std_task_episode_reward"],
                "high_rate": evaluation["high_power_selection_rate"],
                "cold_high_rate": evaluation["cold_high_power_selection_rate"],
                "hot_high_rate": evaluation["hot_high_power_selection_rate"],
                "trip_rate": evaluation["thermal_trip_rate"],
                "terminal_load": evaluation["mean_episode_end_thermal_load"],
            }
        )

    expected = {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }
    wiring_checks["complete_factorial"] = seen == expected
    lookup = {(row["label"], row["memory"]): row for row in rows}
    dynamic_task = lookup[("dynamic", "task")]
    dynamic_lifetime = lookup[("dynamic", "lifetime")]
    static_task = lookup[("static", "task")]
    static_lifetime = lookup[("static", "lifetime")]
    behavior_checks = {
        "static_task_prefers_high": static_task["high_rate"] > 0.75,
        "static_lifetime_prefers_high": static_lifetime["high_rate"] > 0.75,
        "dynamic_lifetime_uses_high_more_when_cold": (
            dynamic_lifetime["cold_high_rate"]
            > dynamic_lifetime["hot_high_rate"] + 0.10
        ),
        "dynamic_lifetime_exceeds_task_reward": (
            dynamic_lifetime["mean_reward"] > dynamic_task["mean_reward"]
        ),
    }
    failed_wiring = [name for name, passed in wiring_checks.items() if not passed]
    report = {
        "phase": "hierarchical_thermal_gpu_pilot_manual_review",
        "status": "manual_review_required_no_confirmatory_run_started",
        "wiring_passed": not failed_wiring,
        "failed_wiring_checks": failed_wiring,
        "behavior_checks": behavior_checks,
        "behavior_ready": all(behavior_checks.values()),
        "interpretation": (
            "One calibration seed is insufficient for a scientific conclusion. "
            "Behavior checks determine whether the confirmatory design is ready, "
            "not whether the hypothesis is proven."
        ),
        "rows": rows,
    }
    output = args.input_root / "MANUAL_REVIEW_REQUIRED.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed_wiring:
        raise SystemExit(f"pilot wiring validation failed: {failed_wiring}")


if __name__ == "__main__":
    main()
