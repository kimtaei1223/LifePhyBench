#!/usr/bin/env python3
"""Validate health persistence and runtime mutations on two MuJoCo tasks."""

from __future__ import annotations

import json

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def run(environment_id: str, seed: int) -> dict[str, object]:
    env = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.001,
            thermal_enabled=True,
            thermal_heat_rate=0.01,
            thermal_cooling_rate=0.01,
            thermal_episode_cooling=0.10,
            joint_aging_enabled=True,
            joint_aging_rate=0.001,
        ),
        environment_id=environment_id,
        max_episode_steps=50,
    )
    rng = np.random.default_rng(seed)
    try:
        _observation, _info = env.reset_lifetime(seed=seed, lifetime_id=seed)
        for _ in range(50):
            action = rng.uniform(env.action_space.low, env.action_space.high)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        before_episode_reset = env.audit_state()
        _observation, _info = env.reset(seed=seed + 1)
        after_episode_reset = env.audit_state()
        _observation, _info = env.reset_lifetime(seed=seed + 2, lifetime_id=seed + 2)
        after_lifetime_reset = env.audit_state()
        return {
            "environment_id": environment_id,
            "action_shape": list(env.action_space.shape),
            "before_episode_reset": before_episode_reset,
            "after_episode_reset": after_episode_reset,
            "after_lifetime_reset": after_lifetime_reset,
        }
    finally:
        env.close()


def main() -> None:
    output = {
        "phase": "multitask_health_semantic_smoke_not_for_paper",
        "results": [run("Pusher-v5", 0), run("Reacher-v5", 1)],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
