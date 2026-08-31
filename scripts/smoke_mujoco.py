#!/usr/bin/env python3
"""Exercise the first simulator-backed selective-reset environment."""

import argparse
import json

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--wear-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.steps <= 0 or args.episode_steps <= 0:
        raise SystemExit("--steps and --episode-steps must be positive")

    env = PusherActuatorWear.make(
        ActuatorWearConfig(wear_rate=args.wear_rate),
        max_episode_steps=args.episode_steps,
    )
    observation, _info = env.reset_lifetime(seed=0, lifetime_id=0)
    initial = env.audit_state()
    rng = np.random.default_rng(0)
    transitions = 0
    episode_resets = 0

    while transitions < args.steps:
        action = rng.uniform(env.action_space.low, env.action_space.high)
        observation, _reward, terminated, truncated, _info = env.step(action)
        transitions += 1
        if (terminated or truncated) and transitions < args.steps:
            observation, _info = env.reset()
            episode_resets += 1

    final = env.audit_state()
    print(
        json.dumps(
            {
                "transitions": transitions,
                "episode_resets": episode_resets,
                "observation_shape": list(np.asarray(observation).shape),
                "initial": initial,
                "final": final,
            },
            indent=2,
            sort_keys=True,
        )
    )
    env.close()


if __name__ == "__main__":
    main()
