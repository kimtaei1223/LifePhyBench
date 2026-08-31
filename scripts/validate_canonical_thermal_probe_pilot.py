"""Validate configuration and health semantics of a canonical-probe GPU pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-task-episodes", type=int, default=20)
    args = parser.parse_args()

    metadata_paths = sorted(args.output_root.glob("*/metadata.json"))
    if len(metadata_paths) != 4:
        raise SystemExit(f"expected four pilot metadata files, found {len(metadata_paths)}")
    rows = []
    for path in metadata_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        required = {
            "canonical_task_seed": 811,
            "thermal_heat_rate": 0.1,
            "thermal_cooling_rate": 0.0,
            "thermal_episode_cooling": 0.0,
        }
        mismatched = {
            key: (arguments.get(key), expected)
            for key, expected in required.items()
            if arguments.get(key) != expected
        }
        if mismatched:
            raise SystemExit(f"canonical configuration mismatch in {path}: {mismatched}")
        if evaluation["task_episodes"] != args.expected_task_episodes:
            raise SystemExit(f"evaluation budget mismatch in {path}")
        name = str(arguments["run_name"])
        label = "dynamic" if "-dynamic-" in name else "static"
        rows.append(
            {
                "path": str(path),
                "label": label,
                "memory_mode": arguments["memory_mode"],
                "degradation_mode": arguments["degradation_mode"],
                "dose": float(arguments["thermal_exogenous_dose_per_step"]),
                "thermal_load": float(evaluation["mean_episode_end_thermal_load"]),
                "efficiency": float(evaluation["mean_episode_end_efficiency"]),
            }
        )
    labels = {(row["label"], row["memory_mode"]) for row in rows}
    expected_labels = {(label, memory) for label in ("dynamic", "static") for memory in ("task", "lifetime")}
    if labels != expected_labels:
        raise SystemExit(f"missing or duplicate pilot cells: {sorted(labels)}")
    static_rows = [row for row in rows if row["label"] == "static"]
    dynamic_rows = [row for row in rows if row["label"] == "dynamic"]
    if any(row["degradation_mode"] != "exogenous_clock" or row["dose"] != 0.0 for row in static_rows):
        raise SystemExit("static control configuration is invalid")
    if any(row["thermal_load"] != 0.0 or row["efficiency"] != 1.0 for row in static_rows):
        raise SystemExit("static control accumulated thermal health")
    if any(row["degradation_mode"] != "endogenous_action" for row in dynamic_rows):
        raise SystemExit("dynamic cell configuration is invalid")
    if not all(row["thermal_load"] > 0.0 and row["efficiency"] < 1.0 for row in dynamic_rows):
        raise SystemExit("dynamic pilot did not exhibit thermal state change")
    report = {"phase": "canonical_thermal_probe_pilot_validation", "passed": True, "rows": rows}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
