from lifephybench.evaluation import evaluate_lifetime
from lifephybench.planning import (
    LifetimeDPOracleController,
    LifetimeDPPlanner,
    PlannerConfig,
)
from lifephybench.policies import MyopicStateOracleController
from lifephybench.toy_env import ToyWearConfig


def test_high_resolution_planner_clears_frozen_target_variation_margin():
    config = ToyWearConfig(
        horizon=2,
        target=0.75,
        wear_rate=0.1,
        damage_cost=0.5,
        minimum_gain=0.15,
        energy_cost=0.01,
    )
    planner = LifetimeDPPlanner(config, PlannerConfig(episodes_per_lifetime=12))
    planned = evaluate_lifetime(
        LifetimeDPOracleController(planner), config, seed=200, episodes=12
    )
    myopic = evaluate_lifetime(MyopicStateOracleController(), config, 200, 12)
    assert planned.mean_episode_return - myopic.mean_episode_return >= 0.05
