import unittest

from lifephybench.audits import run_clock_shortcut_audit
from lifephybench.evaluation import evaluate_lifetime
from lifephybench.planning import (
    LifetimeDPOracleController,
    LifetimeDPPlanner,
    PlannerConfig,
)
from lifephybench.policies import LifetimeEMAController, MyopicStateOracleController
from lifephybench.toy_env import ToyWearConfig, ToyWearEnv


class ToyLifecycleTests(unittest.TestCase):
    def test_episode_reset_preserves_wear(self):
        env = ToyWearEnv(ToyWearConfig(wear_rate=0.1))
        env.reset_lifetime(seed=1)
        env.step(1.0)
        wear_before_reset = env.state.wear
        env.reset_episode()
        self.assertEqual(env.state.position, 0.0)
        self.assertAlmostEqual(env.state.wear, wear_before_reset)

    def test_lifetime_reset_clears_wear(self):
        env = ToyWearEnv(ToyWearConfig(wear_rate=0.1))
        env.reset_lifetime(seed=1)
        env.step(1.0)
        self.assertGreater(env.state.wear, 0.0)
        env.reset_lifetime(seed=2)
        self.assertEqual(env.state.wear, 0.0)
        self.assertEqual(env.state.episode_index, 0)

    def test_larger_action_causes_more_damage(self):
        config = ToyWearConfig(wear_rate=0.1, wear_exponent=2.0)
        low = ToyWearEnv(config)
        high = ToyWearEnv(config)
        low.reset_lifetime(seed=3)
        high.reset_lifetime(seed=3)
        low.step(0.25)
        high.step(1.0)
        self.assertGreater(high.state.wear, low.state.wear)

    def test_exogenous_clock_damage_ignores_action_magnitude(self):
        config = ToyWearConfig(
            wear_rate=0.1,
            degradation_mode="exogenous_clock",
            exogenous_dose_per_step=0.5,
        )
        low = ToyWearEnv(config)
        high = ToyWearEnv(config)
        low.reset_lifetime(seed=3)
        high.reset_lifetime(seed=3)
        low.step(0.1)
        high.step(1.0)
        self.assertEqual(low.state.wear, high.state.wear)

    def test_hidden_history_changes_transition_from_same_observation(self):
        config = ToyWearConfig(process_noise_std=0.0)
        fresh = ToyWearEnv(config)
        worn = ToyWearEnv(config)
        fresh.reset_lifetime(seed=4)
        worn.reset_lifetime(seed=4)
        worn.set_wear_for_diagnostic(0.8)
        self.assertEqual(fresh.observation, worn.observation)
        fresh_next = fresh.step(0.5).observation
        worn_next = worn.step(0.5).observation
        self.assertNotEqual(fresh_next, worn_next)

    def test_seeded_evaluation_is_deterministic(self):
        config = ToyWearConfig(
            stochastic_shock_probability=0.2,
            stochastic_shock_size=0.03,
        )
        first = evaluate_lifetime(LifetimeEMAController(), config, 9, 12)
        second = evaluate_lifetime(LifetimeEMAController(), config, 9, 12)
        self.assertEqual(first, second)

    def test_episode_physics_reset_is_an_explicit_counterfactual(self):
        config = ToyWearConfig(horizon=3, wear_rate=0.03)
        persistent = evaluate_lifetime(
            LifetimeEMAController(), config, 11, 30, persistent_physics=True
        )
        resetting = evaluate_lifetime(
            LifetimeEMAController(), config, 11, 30, persistent_physics=False
        )
        self.assertEqual(persistent.physics_protocol, "persistent_lifetime")
        self.assertEqual(resetting.physics_protocol, "episode_physics_reset")
        self.assertGreater(persistent.final_wear, resetting.final_wear)
        self.assertNotEqual(
            persistent.mean_episode_return,
            resetting.mean_episode_return,
        )

    def test_invalid_step_before_reset_fails(self):
        env = ToyWearEnv()
        with self.assertRaises(RuntimeError):
            env.step(0.1)

    def test_clock_shortcut_audit_separates_causal_conditions(self):
        common = {"horizon": 20, "target": 100.0, "wear_rate": 0.002}
        endogenous = run_clock_shortcut_audit(
            ToyWearConfig(**common, degradation_mode="endogenous_action"),
            train_seeds=range(10, 14),
            test_seeds=range(14, 18),
            steps_per_lifetime=60,
        )
        exogenous = run_clock_shortcut_audit(
            ToyWearConfig(
                **common,
                degradation_mode="exogenous_clock",
                exogenous_dose_per_step=0.5,
            ),
            train_seeds=range(10, 14),
            test_seeds=range(14, 18),
            steps_per_lifetime=60,
        )
        self.assertTrue(endogenous.passed)
        self.assertTrue(exogenous.passed)

    def test_lifetime_planner_uses_privileged_wear_and_is_deterministic(self):
        config = ToyWearConfig(horizon=2, wear_rate=0.03)
        planner_config = PlannerConfig(episodes_per_lifetime=4)
        first = evaluate_lifetime(
            LifetimeDPOracleController(LifetimeDPPlanner(config, planner_config)),
            config,
            seed=7,
            episodes=4,
        )
        second = evaluate_lifetime(
            LifetimeDPOracleController(LifetimeDPPlanner(config, planner_config)),
            config,
            seed=7,
            episodes=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.controller, "lifetime_dp_oracle_discretized")

    def test_planning_oracle_is_close_to_myopic_privileged_baseline(self):
        config = ToyWearConfig(horizon=2, wear_rate=0.03)
        myopic = evaluate_lifetime(MyopicStateOracleController(), config, 7, 4)
        planned = evaluate_lifetime(
            LifetimeDPOracleController(
                LifetimeDPPlanner(config, PlannerConfig(episodes_per_lifetime=4))
            ),
            config,
            seed=7,
            episodes=4,
        )
        self.assertGreaterEqual(
            planned.mean_episode_return, myopic.mean_episode_return - 0.02
        )


if __name__ == "__main__":
    unittest.main()
