#!/usr/bin/env python3
"""Run a CPU-only factorial semantic pilot over tasks and health mechanisms."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from itertools import product
from typing import Any

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.lifetime_analysis import write_jsonl


def health_config(
    mechanism: str,
    degradation_mode: str,
    degradation_law_family: str,
) -> ActuatorWearConfig:
    common: dict[str, Any] = {
        "wear_rate": 0.0,
        "degradation_law_family": degradation_law_family,
        "stochastic_shock_probability": 0.05,
        "stochastic_shock_size": 0.01,
    }
    if mechanism == "wear":
        common.update({"wear_rate": 0.002, "degradation_mode": degradation_mode})
    elif mechanism == "thermal":
        common.update(
            {
                "thermal_enabled": True,
                "thermal_heat_rate": 0.01,
                "thermal_cooling_rate": 0.01,
                "thermal_episode_cooling": 0.10,
                "thermal_degradation_mode": degradation_mode,
            }
        )
    elif mechanism == "joint_aging":
        common.update(
            {
                "joint_aging_enabled": True,
                "joint_aging_rate": 0.002,
                "joint_aging_degradation_mode": degradation_mode,
            }
        )
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    return ActuatorWearConfig(**common)


def action_for(
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
    raise ValueError(f"unknown controller {controller!r}")


def health_value(env: PusherActuatorWear, mechanism: str) -> float:
    if mechanism == "wear":
        return env.wear
    if mechanism == "thermal":
        return env.thermal_load
    if mechanism == "joint_aging":
        return env.joint_aging
    raise ValueError(f"unknown mechanism {mechanism!r}")


def run_lifetime(
    *,
    environment_id: str,
    mechanism: str,
    degradation_mode: str,
    persistent: bool,
    controller: str,
    seed: int,
    episodes: int,
    episode_steps: int,
    degradation_law_family: str,
) -> dict[str, Any]:
    env = PusherActuatorWear.make(
        health_config(mechanism, degradation_mode, degradation_law_family),
        environment_id=environment_id,
        max_episode_steps=episode_steps,
    )
    rng = np.random.default_rng(seed)
    total_reward = 0.0
    terminal_health: list[float] = []
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
            steps = 0
            while steps < episode_steps:
                action = action_for(controller, env, rng)
                _observation, reward, terminated, truncated, _info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            terminal_health.append(health_value(env, mechanism))
        return {
            "condition": (
                "persistent_lifetime" if persistent else "episode_physics_reset"
            ),
            "environment_id": environment_id,
            "mechanism": mechanism,
            "degradation_mode": degradation_mode,
            "degradation_law_family": degradation_law_family,
            "controller": controller,
            "seed": seed,
            "episodes": episodes,
            "episode_steps": episode_steps,
            "mean_episode_return": total_reward / episodes,
            "mean_terminal_health": statistics.mean(terminal_health),
            "final_health": health_value(env, mechanism),
            "cumulative_action_dose": env.cumulative_action_dose,
        }
    finally:
        env.close()


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    key_names = (
        "environment_id",
        "mechanism",
        "degradation_mode",
        "degradation_law_family",
        "controller",
        "condition",
    )
    for record in records:
        grouped[tuple(str(record[name]) for name in key_names)].append(record)
    output = []
    for key, rows in sorted(grouped.items()):
        result = dict(zip(key_names, key, strict=True))
        result["lifetimes"] = len(rows)
        for metric in ("mean_episode_return", "mean_terminal_health", "final_health"):
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-ids", nargs="+", default=["Pusher-v5", "Reacher-v5"])
    parser.add_argument("--mechanisms", nargs="+", default=["wear", "thermal", "joint_aging"])
    parser.add_argument("--controllers", nargs="+", default=["zero", "low_constant", "high_constant", "random_uniform"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--degradation-law-family",
        choices=["power", "threshold", "stochastic_shock"],
        default="power",
    )
    parser.add_argument("--output", help="optional per-lifetime JSONL destination")
    args = parser.parse_args()
    if args.episodes <= 0 or args.episode_steps <= 0 or args.seeds <= 0:
        raise SystemExit("episodes, episode-steps, and seeds must be positive")

    records = [
        run_lifetime(
            environment_id=environment_id,
            mechanism=mechanism,
            degradation_mode=degradation_mode,
            persistent=persistent,
            controller=controller,
            seed=seed,
            episodes=args.episodes,
            episode_steps=args.episode_steps,
            degradation_law_family=args.degradation_law_family,
        )
        for environment_id, mechanism, degradation_mode, persistent, controller, seed in product(
            args.environment_ids,
            args.mechanisms,
            ("endogenous_action", "exogenous_clock"),
            (False, True),
            args.controllers,
            range(1000, 1000 + args.seeds),
        )
    ]
    if args.output:
        write_jsonl(records, args.output)
    print(
        json.dumps(
            {
                "phase": "multitask_health_factorial_pilot_not_for_paper",
                "configuration": vars(args),
                "lifetime_records": len(records),
                "aggregate": summarize(records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
