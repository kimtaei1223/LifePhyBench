#!/usr/bin/env python3
"""Exercise recoverable thermal health without training a policy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--thermal-heat-rate", type=float, default=0.005)
    parser.add_argument("--thermal-cooling-rate", type=float, default=0.01)
    parser.add_argument("--thermal-episode-cooling", type=float, default=0.10)
    args = parser.parse_args()
    if args.steps <= 0 or args.episode_steps <= 0:
        raise SystemExit("--steps and --episode-steps must be positive")

    env = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=args.thermal_heat_rate,
            thermal_cooling_rate=args.thermal_cooling_rate,
            thermal_episode_cooling=args.thermal_episode_cooling,
        ),
        max_episode_steps=args.episode_steps,
    )
    rng = np.random.default_rng(0)
    episode_thermal_loads: list[float] = []
    try:
        _observation, _info = env.reset_lifetime(seed=0, lifetime_id=0)
        for step in range(args.steps):
            action = rng.uniform(env.action_space.low, env.action_space.high)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                episode_thermal_loads.append(env.thermal_load)
                if step + 1 < args.steps:
                    _observation, _info = env.reset()
        print(
            json.dumps(
                {
                    "phase": "mujoco_thermal_pilot_not_for_paper",
                    "configuration": vars(args),
                    "terminal_thermal_loads": episode_thermal_loads,
                    "final": env.audit_state(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
