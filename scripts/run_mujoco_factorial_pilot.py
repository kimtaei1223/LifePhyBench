#!/usr/bin/env python3
"""Run a non-learning 2x2 pilot for persistence and endogeneity."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from typing import Any

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def scripted_action(
    controller: str, env: PusherActuatorWear, rng: np.random.Generator
) -> np.ndarray:
    if controller == "zero":
        return np.zeros(env.action_space.shape, dtype=np.float64)
    if controller == "low_constant":
        return 0.25 * np.asarray(env.action_space.high, dtype=np.float64)
    if controller == "high_constant":
        return np.asarray(env.action_space.high, dtype=np.float64)
    if controller == "random_uniform":
        return rng.uniform(env.action_space.low, env.action_space.high)
    raise ValueError(f"unknown controller: {controller}")


def run_condition(
    *,
    controller: str,
    degradation_mode: str,
    persistent: bool,
    seed: int,
    episodes: int,
    episode_steps: int,
    wear_rate: float,
    exogenous_dose_per_step: float,
) -> dict[str, Any]:
    config = ActuatorWearConfig(
        wear_rate=wear_rate,
        degradation_mode=degradation_mode,
        exogenous_dose_per_step=exogenous_dose_per_step,
    )
    env = PusherActuatorWear.make(config, max_episode_steps=episode_steps)
    rng = np.random.default_rng(seed)
    episode_terminal_wear: list[float] = []
    total_reward = 0.0
    total_action_dose = 0.0
    total_health_dose = 0.0

    try:
        _observation, _info = env.reset_lifetime(seed=seed, lifetime_id=seed)
        for episode in range(episodes):
            if episode > 0:
                if persistent:
                    _observation, _info = env.reset()
                else:
                    _observation, _info = env.reset_lifetime(
                        seed=seed + episode,
                        lifetime_id=seed * episodes + episode,
                    )

            while True:
                action = scripted_action(controller, env, rng)
                _observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_action_dose += info["lifephy/action_dose"]
                total_health_dose += info["lifephy/health_dose"]
                if terminated or truncated:
                    break
            episode_terminal_wear.append(env.wear)

        return {
            "controller": controller,
            "degradation_mode": degradation_mode,
            "persistence": "persistent_lifetime" if persistent else "episode_reset",
            "seed": seed,
            "episodes": episodes,
            "episode_steps": episode_steps,
            "mean_episode_terminal_wear": statistics.mean(episode_terminal_wear),
            "final_wear": env.wear,
            "final_efficiency": env.efficiency,
            "mean_step_reward": total_reward / (episodes * episode_steps),
            "cumulative_action_dose": total_action_dose,
            "cumulative_health_dose": total_health_dose,
        }
    finally:
        env.close()


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["controller"], row["degradation_mode"], row["persistence"])
        groups.setdefault(key, []).append(row)

    metric_names = (
        "mean_episode_terminal_wear",
        "final_wear",
        "final_efficiency",
        "mean_step_reward",
        "cumulative_action_dose",
        "cumulative_health_dose",
    )
    results = []
    for key, group in sorted(groups.items()):
        result: dict[str, Any] = {
            "controller": key[0],
            "degradation_mode": key[1],
            "persistence": key[2],
            "seeds": len(group),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in group]
            result[metric] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--wear-rate", type=float, default=0.001)
    parser.add_argument("--exogenous-dose-per-step", type=float, default=0.25)
    parser.add_argument("--include-rows", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.episode_steps <= 0 or args.seeds <= 0:
        raise SystemExit("episode and seed counts must be positive")

    rows = []
    for controller, mode, persistent, seed in product(
        ("zero", "low_constant", "high_constant", "random_uniform"),
        ("endogenous_action", "exogenous_clock"),
        (False, True),
        range(args.seeds),
    ):
        rows.append(
            run_condition(
                controller=controller,
                degradation_mode=mode,
                persistent=persistent,
                seed=seed,
                episodes=args.episodes,
                episode_steps=args.episode_steps,
                wear_rate=args.wear_rate,
                exogenous_dose_per_step=args.exogenous_dose_per_step,
            )
        )

    output: dict[str, Any] = {
        "phase": "mujoco_factorial_pilot_not_for_paper",
        "configuration": vars(args),
        "aggregate": aggregate(rows),
    }
    if args.include_rows:
        output["rows"] = rows
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
