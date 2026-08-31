"""Evaluate paired one-task forced-mode returns at controlled thermal loads."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from train_fair_recurrent import make_fair_environment
from train_recurrent_smoke import require_recurrent_ppo

from lifephybench.envs.mujoco_pusher import PusherActuatorWear


THERMAL_LOADS = (0.0, 0.025, 0.05, 0.075, 0.09, 0.10, 0.125)


def health_wrapper(environment: Any) -> PusherActuatorWear:
    current = environment
    while current is not None:
        if isinstance(current, PusherActuatorWear):
            return current
        current = getattr(current, "env", None)
    raise TypeError("PusherActuatorWear not found in wrapper stack")


def make_dynamic_environment(arguments: dict[str, Any], monitor: Any):
    return make_fair_environment(
        environment_id=arguments["environment_id"],
        mechanism=arguments["mechanism"],
        degradation_mode="endogenous_action",
        episode_steps=int(arguments["episode_steps"]),
        episodes_per_lifetime=int(arguments["episodes_per_lifetime"]),
        exogenous_dose_per_step=float(arguments["exogenous_dose_per_step"]),
        thermal_exogenous_dose_per_step=0.0,
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


def evaluate_one_task(
    model: Any,
    environment: Any,
    initial_thermal_load: float,
    high_power: bool,
    seed: int,
) -> dict[str, Any]:
    observation, _ = environment.reset(seed=seed)
    base = health_wrapper(environment)
    base.set_thermal_load_for_diagnostic(initial_thermal_load)
    recurrent_state = None
    episode_start = np.asarray([True])
    total_reward = 0.0
    steps = 0
    tripped = False
    final_info: dict[str, Any] = {}
    while True:
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        action = np.asarray(action).copy()
        if steps == 0:
            action[0] = 1.0 if high_power else -1.0
        observation, reward, terminated, truncated, info = environment.step(action)
        steps += 1
        total_reward += float(reward)
        tripped = tripped or bool(info.get("lifephy/thermal_trip", False))
        final_info = dict(info)
        if bool(info.get("lifephy/inner_task_boundary", terminated or truncated)):
            break
        episode_start = np.asarray([terminated or truncated])
    return {
        "initial_thermal_load": initial_thermal_load,
        "forced_mode": "high" if high_power else "low",
        "task_reward": total_reward,
        "task_steps": steps,
        "tripped": tripped,
        "terminal_thermal_load": float(final_info["lifephy/thermal_load"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/thermal_commitment_calibration_v6"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/thermal_commitment_calibration_v6/"
            "local_mode_counterfactual.json"
        ),
    )
    args = parser.parse_args()
    runs = sorted(path.parent for path in args.input_root.glob("*/metadata.json"))
    if len(runs) != 4:
        raise SystemExit(f"expected four policies, found {len(runs)}")

    modules = require_recurrent_ppo()
    rows = []
    for run_index, run in enumerate(runs):
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        arguments = metadata["arguments"]
        model = modules["RecurrentPPO"].load(str(run / "model"), device=args.device)
        for load_index, thermal_load in enumerate(THERMAL_LOADS):
            for high_power in (False, True):
                environment = make_dynamic_environment(
                    arguments, modules["Monitor"]
                )
                try:
                    result = evaluate_one_task(
                        model,
                        environment,
                        thermal_load,
                        high_power,
                        seed=(
                            int(metadata["evaluation_seed"])
                            + 10_000 * run_index
                            + load_index
                        ),
                    )
                finally:
                    environment.close()
                result.update(
                    {
                        "run_name": arguments["run_name"],
                        "training_label": (
                            "dynamic" if "-dynamic-" in arguments["run_name"] else "static"
                        ),
                        "memory_mode": arguments["memory_mode"],
                    }
                )
                rows.append(result)
                print(
                    f"[{run.name}] load={thermal_load:.3f} "
                    f"mode={result['forced_mode']} reward={result['task_reward']:.3f} "
                    f"steps={result['task_steps']} trip={result['tripped']}",
                    flush=True,
                )

    indexed = {
        (
            row["run_name"],
            float(row["initial_thermal_load"]),
            row["forced_mode"],
        ): row
        for row in rows
    }
    comparisons = []
    for run in runs:
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        run_name = metadata["arguments"]["run_name"]
        for thermal_load in THERMAL_LOADS:
            low = indexed[(run_name, thermal_load, "low")]
            high = indexed[(run_name, thermal_load, "high")]
            comparisons.append(
                {
                    "run_name": run_name,
                    "training_label": high["training_label"],
                    "memory_mode": high["memory_mode"],
                    "initial_thermal_load": thermal_load,
                    "high_minus_low_reward": (
                        high["task_reward"] - low["task_reward"]
                    ),
                    "high_reward": high["task_reward"],
                    "low_reward": low["task_reward"],
                    "high_tripped": high["tripped"],
                    "high_steps": high["task_steps"],
                }
            )
    zero_load = [row for row in comparisons if row["initial_thermal_load"] == 0.0]
    zero_advantages = [float(row["high_minus_low_reward"]) for row in zero_load]
    report = {
        "phase": "thermal_commitment_v6_local_mode_counterfactual",
        "status": "diagnostic_not_inference",
        "protocol": {
            "thermal_loads": list(THERMAL_LOADS),
            "paired_canonical_task": True,
            "recurrent_state_at_task_start": "zero",
            "evaluation_physics": "endogenous_action_for_all_source_policies",
        },
        "cold_zero_summary": {
            "policies": len(zero_load),
            "forced_high_trip_rate": statistics.mean(
                float(row["high_tripped"]) for row in zero_load
            ),
            "mean_high_minus_low_reward": statistics.mean(zero_advantages),
            "minimum_high_minus_low_reward": min(zero_advantages),
            "maximum_high_minus_low_reward": max(zero_advantages),
            "mean_high_steps_before_boundary": statistics.mean(
                int(row["high_steps"]) for row in zero_load
            ),
        },
        "comparisons": comparisons,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["cold_zero_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
