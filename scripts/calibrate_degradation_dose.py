#!/usr/bin/env python3
"""Estimate a matched exogenous dose from a reference random-action rollout.

The endogenous and exogenous conditions must be matched on dose magnitude before
their causal difference is interpreted.  This script provides a deterministic,
pre-training reference target that can be recorded in experiment metadata.
"""

from __future__ import annotations

import argparse
import json
import statistics

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="outputs/degradation_dose_calibration.json")
    args = parser.parse_args()
    if args.episodes <= 0 or args.episode_steps <= 0:
        raise SystemExit("episodes and episode-steps must be positive")

    env = PusherActuatorWear.make(
        ActuatorWearConfig(wear_rate=0.0),
        environment_id=args.environment_id,
        max_episode_steps=args.episode_steps,
    )
    rng = np.random.default_rng(args.seed)
    doses: list[float] = []
    try:
        for episode in range(args.episodes):
            env.reset_lifetime(seed=args.seed + episode, lifetime_id=episode)
            for _ in range(args.episode_steps):
                action = rng.uniform(env.action_space.low, env.action_space.high)
                doses.append(env._normalized_action_dose(action, 2.0))
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()

    result = {
        "phase": "reference_dose_calibration",
        "environment_id": args.environment_id,
        "episodes": args.episodes,
        "episode_steps": args.episode_steps,
        "seed": args.seed,
        "action_dose_exponent": 2.0,
        "reference_action_dose_mean": statistics.mean(doses),
        "reference_action_dose_std": statistics.stdev(doses) if len(doses) > 1 else 0.0,
        "transitions": len(doses),
        "recommended_exogenous_dose_per_step": statistics.mean(doses),
    }
    output = args.output
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
