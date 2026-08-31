"""Measure forced-mode returns and calibrate the static high-power bonus."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from train_fair_recurrent import make_fair_environment
from train_recurrent_smoke import require_recurrent_ppo

from lifephybench.recurrent_evaluation import evaluate_task_episodes, evaluation_as_dict


class ForcedCommitmentModel:
    """Delegate policy inference while overriding only boundary mode choices."""

    def __init__(self, model: Any, high_power: bool) -> None:
        self.model = model
        self.high_power = high_power

    def predict(self, observation, state, episode_start, deterministic):
        action, next_state = self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        action_array = np.asarray(action).copy()
        if float(np.asarray(observation)[-1]) > 0.5:
            action_array[0] = 1.0 if self.high_power else -1.0
        return action_array, next_state


def evaluate_forced_mode(
    run_directory: Path,
    high_power: bool,
    task_episodes: int,
    device: str,
    modules: dict[str, Any],
) -> dict[str, object]:
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    arguments = metadata["arguments"]
    environment = make_fair_environment(
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
        monitor=modules["Monitor"],
    )
    try:
        model = modules["RecurrentPPO"].load(
            str(run_directory / "model"), device=device
        )
        forced = ForcedCommitmentModel(model, high_power=high_power)
        evaluation = evaluate_task_episodes(
            forced,
            environment,
            task_episodes,
            seed=int(metadata["evaluation_seed"]),
        )
    finally:
        environment.close()
    return {
        "run_name": arguments["run_name"],
        "training_label": (
            "dynamic" if "-dynamic-" in arguments["run_name"] else "static"
        ),
        "memory_mode": arguments["memory_mode"],
        "forced_mode": "high" if high_power else "low",
        "current_high_power_bonus": float(arguments["commitment_high_power_bonus"]),
        "evaluation": evaluation_as_dict(evaluation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/thermal_commitment_pilot")
    )
    parser.add_argument("--task-episodes", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/thermal_commitment_bonus_calibration.json"),
    )
    parser.add_argument("--target-static-margin", type=float, default=2.0)
    args = parser.parse_args()
    if args.task_episodes <= 0 or args.target_static_margin <= 0.0:
        raise SystemExit("evaluation budget and target margin must be positive")
    runs = sorted(path.parent for path in args.input_root.glob("*/metadata.json"))
    if len(runs) != 4:
        raise SystemExit(f"expected four pilot policies, found {len(runs)}")

    modules = require_recurrent_ppo()
    rows = []
    for index, run in enumerate(runs, start=1):
        for high_power in (False, True):
            label = "high" if high_power else "low"
            print(f"[FORCED {index}/4] {run.name} -> {label}", flush=True)
            rows.append(
                evaluate_forced_mode(
                    run, high_power, args.task_episodes, args.device, modules
                )
            )

    indexed = {
        (str(row["training_label"]), str(row["memory_mode"]), str(row["forced_mode"])): row
        for row in rows
    }
    static_calibration = []
    for memory in ("task", "lifetime"):
        low = indexed[("static", memory, "low")]
        high = indexed[("static", memory, "high")]
        low_reward = float(low["evaluation"]["mean_task_episode_reward"])
        high_reward = float(high["evaluation"]["mean_task_episode_reward"])
        current_bonus = float(high["current_high_power_bonus"])
        break_even = current_bonus - (high_reward - low_reward)
        static_calibration.append(
            {
                "memory_mode": memory,
                "forced_high_reward": high_reward,
                "forced_low_reward": low_reward,
                "high_minus_low_at_current_bonus": high_reward - low_reward,
                "break_even_high_power_bonus": break_even,
                "bonus_for_target_margin": break_even + args.target_static_margin,
            }
        )
    recommended = float(
        math.ceil(max(row["bonus_for_target_margin"] for row in static_calibration))
    )
    report = {
        "phase": "thermal_commitment_bonus_calibration_not_inference",
        "task_episodes_per_forced_evaluation": args.task_episodes,
        "target_static_high_minus_low_margin": args.target_static_margin,
        "static_calibration": static_calibration,
        "recommended_integer_bonus_for_gpu_calibration": recommended,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
