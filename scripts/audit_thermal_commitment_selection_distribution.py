"""Trace boundary thermal loads and mode choices of learned commitment policies."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_fair_recurrent import make_fair_environment
from train_recurrent_smoke import require_recurrent_ppo


def make_environment(arguments: dict[str, Any], monitor: Any):
    return make_fair_environment(
        environment_id=arguments["environment_id"],
        mechanism=arguments["mechanism"],
        degradation_mode=arguments["degradation_mode"],
        episode_steps=int(arguments["episode_steps"]),
        episodes_per_lifetime=int(arguments["episodes_per_lifetime"]),
        exogenous_dose_per_step=float(arguments["exogenous_dose_per_step"]),
        thermal_exogenous_dose_per_step=float(
            arguments["thermal_exogenous_dose_per_step"]
        ),
        joint_aging_exogenous_dose_per_step=float(
            arguments["joint_aging_exogenous_dose_per_step"]
        ),
        thermal_heat_rate=float(arguments["thermal_heat_rate"]),
        thermal_cooling_rate=float(arguments["thermal_cooling_rate"]),
        thermal_episode_cooling=float(arguments["thermal_episode_cooling"]),
        canonical_task_seed=int(arguments["canonical_task_seed"]),
        thermal_commitment=True,
        commitment_trip_load=float(arguments["commitment_trip_load"]),
        commitment_low_power_scale=float(arguments["commitment_low_power_scale"]),
        commitment_trip_penalty=float(arguments["commitment_trip_penalty"]),
        commitment_high_power_bonus=float(arguments["commitment_high_power_bonus"]),
        commitment_control_cost_basis=str(
            arguments.get("commitment_control_cost_basis", "applied_action")
        ),
        monitor=monitor,
    )


def trace_policy(
    run_directory: Path,
    task_episodes: int,
    device: str,
    modules: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    arguments = metadata["arguments"]
    environment = make_environment(arguments, modules["Monitor"])
    try:
        model = modules["RecurrentPPO"].load(
            str(run_directory / "model"), device=device
        )
        observation, _ = environment.reset(seed=int(metadata["evaluation_seed"]))
        recurrent_state = None
        episode_start = np.asarray([True])
        rows = []
        current_row: dict[str, object] | None = None
        current_reward = 0.0
        while len(rows) < task_episodes:
            action, recurrent_state = model.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            observation, reward, terminated, truncated, info = environment.step(action)
            current_reward += float(reward)
            if bool(info.get("lifephy/thermal_mode_selected_now", False)):
                if current_row is not None:
                    raise RuntimeError("new mode selection before prior task boundary")
                current_row = {
                    "run_name": arguments["run_name"],
                    "label": (
                        "dynamic"
                        if "-dynamic-" in arguments["run_name"]
                        else "static"
                    ),
                    "memory": arguments["memory_mode"],
                    "seed": int(arguments["seed"]),
                    "task_number": len(rows),
                    "episode_index": int(info["lifephy/episode_index"]),
                    "thermal_load_at_selection": float(
                        info["lifephy/thermal_load_at_mode_selection"]
                    ),
                    "high_power": info["lifephy/thermal_mode"] == "high",
                    "trip": bool(info["lifephy/thermal_trip"]),
                }
            task_boundary = bool(
                info.get("lifephy/inner_task_boundary", terminated or truncated)
            )
            if task_boundary:
                if current_row is None:
                    raise RuntimeError("task boundary without a mode selection")
                current_row["task_reward"] = current_reward
                current_row["thermal_load_at_end"] = float(
                    info["lifephy/thermal_load"]
                )
                current_row["efficiency_at_end"] = float(
                    info["lifephy/actuator_efficiency"]
                )
                rows.append(current_row)
                current_row = None
                current_reward = 0.0
            gym_boundary = bool(terminated or truncated)
            if gym_boundary and len(rows) < task_episodes:
                observation, _ = environment.reset()
            episode_start = np.asarray([gym_boundary])
    finally:
        environment.close()

    loads = np.asarray([float(row["thermal_load_at_selection"]) for row in rows])
    rewards = np.asarray([float(row["task_reward"]) for row in rows])
    high = np.asarray([bool(row["high_power"]) for row in rows])
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[int(row["episode_index"])].append(row)
    quantiles = {
        str(q): float(np.quantile(loads, q))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    summary = {
        "run_name": arguments["run_name"],
        "label": "dynamic" if "-dynamic-" in arguments["run_name"] else "static",
        "memory": arguments["memory_mode"],
        "task_episodes": len(rows),
        "native_reward": float(metadata["task_episode_evaluation"]["mean_task_episode_reward"]),
        "traced_reward": float(rewards.mean()),
        "native_replay_absolute_error": abs(
            float(rewards.mean())
            - float(metadata["task_episode_evaluation"]["mean_task_episode_reward"])
        ),
        "thermal_load_quantiles": quantiles,
        "high_power_selection_rate": float(high.mean()),
        "episode_index_profiles": [
            {
                "episode_index": episode_index,
                "n": len(items),
                "mean_thermal_load": float(
                    np.mean([row["thermal_load_at_selection"] for row in items])
                ),
                "high_power_selection_rate": float(
                    np.mean([row["high_power"] for row in items])
                ),
            }
            for episode_index, items in sorted(by_episode.items())
        ],
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/thermal_commitment_calibration_v3"),
    )
    parser.add_argument("--task-episodes", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/thermal_commitment_selection_audit_v3"),
    )
    args = parser.parse_args()
    runs = sorted(path.parent for path in args.input_root.glob("*/metadata.json"))
    if len(runs) != 4:
        raise SystemExit(f"expected four policies, found {len(runs)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    modules = require_recurrent_ppo()
    summaries = []
    rows = []
    for index, run in enumerate(runs, start=1):
        print(f"[TRACE {index}/4] {run.name}", flush=True)
        summary, policy_rows = trace_policy(
            run, args.task_episodes, args.device, modules
        )
        summaries.append(summary)
        rows.extend(policy_rows)

    with (args.output_root / "boundary_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "phase": "thermal_commitment_v3_selection_distribution_audit",
        "status": "calibration_diagnostic_not_inference",
        "policies": summaries,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
