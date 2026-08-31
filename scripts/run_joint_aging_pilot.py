#!/usr/bin/env python3
"""Exercise persistent joint aging without training a policy."""

from __future__ import annotations

import argparse
import json

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--joint-aging-rate", type=float, default=0.002)
    parser.add_argument("--joint-aging-damping-multiplier", type=float, default=4.0)
    args = parser.parse_args()
    if args.steps <= 0 or args.episode_steps <= 0:
        raise SystemExit("--steps and --episode-steps must be positive")

    env = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_rate=args.joint_aging_rate,
            joint_aging_damping_multiplier=args.joint_aging_damping_multiplier,
        ),
        max_episode_steps=args.episode_steps,
    )
    rng = np.random.default_rng(0)
    terminal_joint_aging: list[float] = []
    try:
        _observation, _info = env.reset_lifetime(seed=0, lifetime_id=0)
        for step in range(args.steps):
            action = rng.uniform(env.action_space.low, env.action_space.high)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                terminal_joint_aging.append(env.joint_aging)
                if step + 1 < args.steps:
                    _observation, _info = env.reset()
        print(
            json.dumps(
                {
                    "phase": "mujoco_joint_aging_pilot_not_for_paper",
                    "configuration": vars(args),
                    "terminal_joint_aging": terminal_joint_aging,
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
