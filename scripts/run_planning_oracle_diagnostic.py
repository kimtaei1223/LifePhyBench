#!/usr/bin/env python3
"""Compare a myopic privileged controller with a lifetime-planning anchor."""

from __future__ import annotations

import json

from lifephybench.evaluation import evaluate_many
from lifephybench.planning import (
    LifetimeDPOracleController,
    LifetimeDPPlanner,
    PlannerConfig,
)
from lifephybench.policies import MyopicStateOracleController
from lifephybench.toy_env import ToyWearConfig


def main() -> None:
    episodes = 12
    env_config = ToyWearConfig(
        horizon=3,
        wear_rate=0.025,
        wear_exponent=2.0,
        process_noise_std=0.0,
        stochastic_shock_probability=0.0,
    )
    planner_config = PlannerConfig(episodes_per_lifetime=episodes)
    results = [
        evaluate_many(MyopicStateOracleController, env_config, range(100, 101), episodes),
        evaluate_many(
            lambda: LifetimeDPOracleController(
                LifetimeDPPlanner(env_config, planner_config)
            ),
            env_config,
            range(100, 101),
            episodes,
        ),
    ]
    print(json.dumps({"diagnostic": "lifetime_planning_oracle", "results": results}, indent=2))


if __name__ == "__main__":
    main()
