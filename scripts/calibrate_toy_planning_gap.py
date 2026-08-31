"""CPU-only calibration search for a toy lifetime-planning gap.

This is a configuration-calibration tool, not an inferential experiment. A
selected setting must be frozen and separately evaluated before it is used as a
benchmark claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import product

from lifephybench.evaluation import evaluate_lifetime
from lifephybench.planning import (
    LifetimeDPOracleController,
    LifetimeDPPlanner,
    PlannerConfig,
)
from lifephybench.policies import MyopicStateOracleController
from lifephybench.toy_env import ToyWearConfig


def main() -> None:
    planner_config = PlannerConfig(
        episodes_per_lifetime=12,
        action_bins=21,
        position_resolution=0.05,
        wear_resolution=0.02,
    )
    rows = []
    for horizon, target, wear_rate, damage_cost, minimum_gain in product(
        (1, 2),
        (0.6, 0.8, 0.95),
        (0.02, 0.05, 0.1),
        (0.0, 0.1, 0.5, 1.0),
        (0.15, 0.4),
    ):
        config = ToyWearConfig(
            horizon=horizon,
            target=target,
            wear_rate=wear_rate,
            damage_cost=damage_cost,
            minimum_gain=minimum_gain,
            energy_cost=0.01,
        )
        myopic = evaluate_lifetime(MyopicStateOracleController(), config, 100, 12)
        planned = evaluate_lifetime(
            LifetimeDPOracleController(LifetimeDPPlanner(config, planner_config)),
            config,
            100,
            12,
        )
        rows.append(
            {
                "config": asdict(config),
                "mean_episode_return_gap": planned.mean_episode_return
                - myopic.mean_episode_return,
                "success_auc_gap": planned.success_auc - myopic.success_auc,
                "myopic": asdict(myopic),
                "planner": asdict(planned),
            }
        )
    rows.sort(key=lambda row: float(row["mean_episode_return_gap"]), reverse=True)
    report = {
        "phase": "toy_planning_gap_calibration_not_inference",
        "planner_config": asdict(planner_config),
        "candidate_count": len(rows),
        "top_candidates": rows[:10],
        "recommended_calibration_candidate": rows[0],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
