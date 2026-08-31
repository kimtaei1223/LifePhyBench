"""CPU-only semantic audit for canonical task resets with hidden thermal state."""

from __future__ import annotations

import argparse
import json

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def run_trial(
    task_seed: int,
    heating_steps: int,
    probe_steps: int,
    thermal_heat_rate: float,
) -> dict[str, float]:
    config = ActuatorWearConfig(
        wear_rate=0.0,
        thermal_enabled=True,
        thermal_heat_rate=thermal_heat_rate,
        thermal_cooling_rate=0.0,
        thermal_episode_cooling=0.0,
        canonical_task_seed=task_seed,
    )
    cold = PusherActuatorWear.make(config, max_episode_steps=100)
    hot = PusherActuatorWear.make(config, max_episode_steps=100)
    try:
        cold.reset_lifetime(seed=10_000 + task_seed)
        hot.reset_lifetime(seed=20_000 + task_seed)
        action = np.full(cold.action_space.shape, 1.0)
        for _ in range(heating_steps):
            hot.step(action)
        hot_load_at_boundary = float(hot.thermal_load)
        cold_observation, _ = cold.reset(seed=30_000 + task_seed)
        hot_observation, _ = hot.reset(seed=40_000 + task_seed)
        cold_load_at_boundary = float(cold.thermal_load)
        boundary_observation_max_abs_difference = float(
            np.max(np.abs(cold_observation - hot_observation))
        )
        for _ in range(probe_steps):
            cold_observation, *_ = cold.step(action)
            hot_observation, *_ = hot.step(action)
        cold_response_norm = float(np.linalg.norm(cold_observation[11:18]))
        hot_response_norm = float(np.linalg.norm(hot_observation[11:18]))
        return {
            "task_seed": task_seed,
            "cold_thermal_load_at_boundary": cold_load_at_boundary,
            "hot_thermal_load_at_boundary": hot_load_at_boundary,
            "boundary_observation_max_abs_difference": boundary_observation_max_abs_difference,
            "cold_response_norm": cold_response_norm,
            "hot_response_norm": hot_response_norm,
            "response_gap": cold_response_norm - hot_response_norm,
        }
    finally:
        cold.close()
        hot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-seeds", nargs="+", type=int, default=list(range(700, 710)))
    parser.add_argument("--heating-steps", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--thermal-heat-rate", type=float, default=0.1)
    parser.add_argument("--minimum-response-gap", type=float, default=0.25)
    args = parser.parse_args()
    if (
        min(args.task_seeds) < 0
        or args.heating_steps <= 0
        or args.probe_steps <= 0
        or args.thermal_heat_rate <= 0.0
    ):
        raise SystemExit("task seeds, step counts, and thermal heat rate must be positive")

    trials = [
        run_trial(seed, args.heating_steps, args.probe_steps, args.thermal_heat_rate)
        for seed in args.task_seeds
    ]
    max_boundary_difference = max(row["boundary_observation_max_abs_difference"] for row in trials)
    min_response_gap = min(row["response_gap"] for row in trials)
    min_hot_load = min(row["hot_thermal_load_at_boundary"] for row in trials)
    passed = (
        max_boundary_difference <= 1e-12
        and min_hot_load > 0.0
        and min_response_gap >= args.minimum_response_gap
    )
    report = {
        "audit": "canonical_thermal_probe",
        "protocol": {
            "task_seeds": args.task_seeds,
            "heating_steps": args.heating_steps,
            "probe_steps": args.probe_steps,
            "thermal_heat_rate": args.thermal_heat_rate,
            "minimum_response_gap": args.minimum_response_gap,
        },
        "summary": {
            "max_boundary_observation_difference": max_boundary_difference,
            "min_hot_thermal_load_at_boundary": min_hot_load,
            "min_response_gap": min_response_gap,
            "mean_response_gap": float(np.mean([row["response_gap"] for row in trials])),
            "passed": passed,
        },
        "trials": trials,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("canonical thermal probe audit failed")


if __name__ == "__main__":
    main()
