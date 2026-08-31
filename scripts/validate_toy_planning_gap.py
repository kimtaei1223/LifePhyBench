"""Validate the calibration-selected toy planning gap on fixed held-out variations."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from lifephybench.evaluation import evaluate_many
from lifephybench.planning import (
    LifetimeDPOracleController,
    LifetimeDPPlanner,
    PlannerConfig,
)
from lifephybench.policies import MyopicStateOracleController
from lifephybench.toy_env import ToyWearConfig

BASE_CONFIG = ToyWearConfig(
    horizon=2,
    target=0.8,
    wear_rate=0.1,
    damage_cost=0.5,
    minimum_gain=0.15,
    energy_cost=0.01,
)
HELD_OUT_VARIATIONS = {
    "target_interpolation_075": {"target": 0.75},
    "target_interpolation_085": {"target": 0.85},
    "wear_interpolation_008": {"wear_rate": 0.08},
    "wear_extrapolation_012": {"wear_rate": 0.12},
}
PLANNER_CONFIG = PlannerConfig(
    episodes_per_lifetime=12,
)
PRACTICAL_RETURN_MARGIN = 0.05
SEEDS = range(200, 210)


def main() -> None:
    rows = []
    for name, overrides in HELD_OUT_VARIATIONS.items():
        config = replace(BASE_CONFIG, **overrides)
        myopic = evaluate_many(MyopicStateOracleController, config, SEEDS, 12)
        planner_core = LifetimeDPPlanner(config, PLANNER_CONFIG)
        planner = evaluate_many(
            lambda planner_core=planner_core: LifetimeDPOracleController(planner_core),
            config,
            SEEDS,
            12,
        )
        return_gap = (
            planner["aggregate"]["mean_episode_return"]["mean"]
            - myopic["aggregate"]["mean_episode_return"]["mean"]
        )
        success_gap = (
            planner["aggregate"]["success_auc"]["mean"]
            - myopic["aggregate"]["success_auc"]["mean"]
        )
        rows.append(
            {
                "name": name,
                "config": asdict(config),
                "return_gap": return_gap,
                "success_auc_gap": success_gap,
                "passes_practical_return_margin": return_gap >= PRACTICAL_RETURN_MARGIN,
            }
        )
    passed = all(row["passes_practical_return_margin"] for row in rows)
    report = {
        "phase": "held_out_toy_planning_gap_validation",
        "interpretation": (
            "Deterministic task-variation validation. The repeated seed labels "
            "check reproducibility but are not independent stochastic samples."
        ),
        "calibration_base_config": asdict(BASE_CONFIG),
        "planner_config": asdict(PLANNER_CONFIG),
        "held_out_variations": rows,
        "practical_return_margin": PRACTICAL_RETURN_MARGIN,
        "passed": passed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("held-out planning-gap validation failed")


if __name__ == "__main__":
    main()
