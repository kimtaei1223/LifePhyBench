"""Apply frozen behavioral entry gates to a thermal commitment calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STATIC_MINIMUM_HIGH_RATE = 0.80
DYNAMIC_MINIMUM_COLD_HIGH_RATE = 0.60
DYNAMIC_MAXIMUM_HOT_HIGH_RATE = 0.40
DYNAMIC_MAXIMUM_TRIP_RATE = 0.20
MINIMUM_CONDITIONAL_SELECTIONS = 40


def at_least(value, threshold: float) -> bool:
    return value is not None and float(value) >= threshold


def at_most(value, threshold: float) -> bool:
    return value is not None and float(value) <= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/thermal_commitment_calibration_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/thermal_commitment_calibration_v2/learnability_validation.json"
        ),
    )
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("*/metadata.json"))
    if len(paths) != 4:
        raise SystemExit(f"expected four learned-policy cells, found {len(paths)}")

    rows = []
    indexed = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        label = "dynamic" if "-dynamic-" in arguments["run_name"] else "static"
        memory = str(arguments["memory_mode"])
        row = {
            "label": label,
            "memory": memory,
            "reward": evaluation["mean_task_episode_reward"],
            "high_power_selection_rate": evaluation["high_power_selection_rate"],
            "cold_high_power_selection_rate": evaluation[
                "cold_high_power_selection_rate"
            ],
            "hot_high_power_selection_rate": evaluation[
                "hot_high_power_selection_rate"
            ],
            "cold_mode_selections": evaluation["cold_mode_selections"],
            "hot_mode_selections": evaluation["hot_mode_selections"],
            "thermal_trip_rate": evaluation["thermal_trip_rate"],
            "thermal_load": evaluation["mean_episode_end_thermal_load"],
        }
        rows.append(row)
        indexed[(label, memory)] = row

    expected = {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }
    if set(indexed) != expected:
        raise SystemExit(f"factorial mismatch: {sorted(indexed)}")
    dynamic_lifetime = indexed[("dynamic", "lifetime")]
    criteria = {
        "static_task_prefers_high": at_least(
            indexed[("static", "task")]["high_power_selection_rate"],
            STATIC_MINIMUM_HIGH_RATE,
        ),
        "static_lifetime_prefers_high": at_least(
            indexed[("static", "lifetime")]["high_power_selection_rate"],
            STATIC_MINIMUM_HIGH_RATE,
        ),
        "dynamic_lifetime_has_cold_support": at_least(
            dynamic_lifetime["cold_mode_selections"], MINIMUM_CONDITIONAL_SELECTIONS
        ),
        "dynamic_lifetime_has_hot_support": at_least(
            dynamic_lifetime["hot_mode_selections"], MINIMUM_CONDITIONAL_SELECTIONS
        ),
        "dynamic_lifetime_uses_high_when_cold": at_least(
            dynamic_lifetime["cold_high_power_selection_rate"],
            DYNAMIC_MINIMUM_COLD_HIGH_RATE,
        ),
        "dynamic_lifetime_uses_low_when_hot": at_most(
            dynamic_lifetime["hot_high_power_selection_rate"],
            DYNAMIC_MAXIMUM_HOT_HIGH_RATE,
        ),
        "dynamic_lifetime_avoids_excess_trips": at_most(
            dynamic_lifetime["thermal_trip_rate"], DYNAMIC_MAXIMUM_TRIP_RATE
        ),
    }
    report = {
        "phase": "thermal_commitment_learnability_calibration_gate",
        "status": "calibration_not_confirmatory_evidence",
        "thresholds": {
            "static_minimum_high_rate": STATIC_MINIMUM_HIGH_RATE,
            "dynamic_minimum_cold_high_rate": DYNAMIC_MINIMUM_COLD_HIGH_RATE,
            "dynamic_maximum_hot_high_rate": DYNAMIC_MAXIMUM_HOT_HIGH_RATE,
            "dynamic_maximum_trip_rate": DYNAMIC_MAXIMUM_TRIP_RATE,
            "minimum_conditional_selections": MINIMUM_CONDITIONAL_SELECTIONS,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("thermal commitment learnability gate failed")


if __name__ == "__main__":
    main()
