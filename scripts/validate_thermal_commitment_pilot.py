"""Validate wiring and health metrics from the thermal commitment pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/thermal_commitment_pilot")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/thermal_commitment_pilot/validation.json"),
    )
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("*/metadata.json"))
    if len(paths) != 4:
        raise SystemExit(f"expected four pilot runs, found {len(paths)}")

    manifest = json.loads((args.input_root / "manifest.json").read_text(encoding="utf-8"))
    commitment = manifest["commitment"]
    rows = []
    seen = set()
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        run_name = str(arguments["run_name"])
        label = "dynamic" if "-dynamic-" in run_name else "static"
        key = (label, arguments["memory_mode"])
        if key in seen:
            raise SystemExit(f"duplicate pilot cell: {key}")
        seen.add(key)
        checks = {
            "thermal_commitment": arguments.get("thermal_commitment") is True,
            "canonical_task_seed": arguments.get("canonical_task_seed") == 811,
            "trip_load": arguments.get("commitment_trip_load")
            == commitment["trip_load"],
            "low_power_scale": arguments.get("commitment_low_power_scale") == 0.40,
            "trip_penalty": arguments.get("commitment_trip_penalty")
            == commitment["trip_penalty"],
            "high_power_bonus": arguments.get("commitment_high_power_bonus")
            == commitment["high_power_throughput_bonus"],
            "control_cost_basis": arguments.get("commitment_control_cost_basis")
            == commitment["control_cost_basis"],
            "mode_metric": evaluation.get("high_power_selection_rate") is not None,
            "trip_metric": evaluation.get("thermal_trip_rate") is not None,
        }
        optimization = manifest.get("optimization", {})
        if optimization.get("decision_only_mode_loss") is True:
            checks.update(
                {
                    "mode_loss_cli_enabled": arguments.get(
                        "commitment_mask_mode_loss"
                    )
                    is True,
                    "mode_loss_metadata_enabled": document.get(
                        "controlled_semantics", {}
                    ).get("commitment_mode_loss_masked_after_decision")
                    is True,
                }
            )
        representation = manifest.get("representation", {})
        if representation.get("previous_applied_action_observed") is True:
            checks.update(
                {
                    "action_history_cli_enabled": arguments.get(
                        "append_previous_applied_action"
                    )
                    is True,
                    "action_history_metadata_enabled": document.get(
                        "controlled_semantics", {}
                    ).get("previous_applied_action_observed")
                    is True,
                    "action_history_boundary_zeroed": document.get(
                        "controlled_semantics", {}
                    ).get("previous_action_zeroed_at_task_boundary")
                    is True,
                }
            )
        if label == "static":
            checks.update(
                {
                    "static_thermal_zero": evaluation[
                        "mean_episode_end_thermal_load"
                    ]
                    == 0.0,
                    "static_efficiency_one": evaluation[
                        "mean_episode_end_efficiency"
                    ]
                    == 1.0,
                    "static_trip_zero": evaluation["thermal_trip_rate"] == 0.0,
                }
            )
        else:
            checks.update(
                {
                    "dynamic_thermal_positive": evaluation[
                        "mean_episode_end_thermal_load"
                    ]
                    > 0.0,
                    "dynamic_efficiency_below_one": evaluation[
                        "mean_episode_end_efficiency"
                    ]
                    < 1.0,
                }
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(f"pilot validation failed {failed}: {path}")
        rows.append(
            {
                "label": label,
                "memory": arguments["memory_mode"],
                "mean_reward": evaluation["mean_task_episode_reward"],
                "thermal_load": evaluation["mean_episode_end_thermal_load"],
                "efficiency": evaluation["mean_episode_end_efficiency"],
                "trip_rate": evaluation["thermal_trip_rate"],
                "high_power_selection_rate": evaluation[
                    "high_power_selection_rate"
                ],
            }
        )
    expected = {
        (label, memory)
        for label in ("dynamic", "static")
        for memory in ("task", "lifetime")
    }
    if seen != expected:
        raise SystemExit(f"pilot factorial mismatch: {seen}")
    report = {
        "phase": "thermal_commitment_gpu_wiring_pilot_validation",
        "passed": True,
        "interpretation": "Wiring gate only; not evidence for the learned-policy hypothesis.",
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
