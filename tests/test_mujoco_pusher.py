import unittest

import numpy as np

try:
    from lifephybench.envs.mujoco_pusher import (
        ActuatorWearConfig,
        PusherActuatorWear,
    )
except ImportError:  # The dependency-free core remains testable without MuJoCo.
    ActuatorWearConfig = None
    PusherActuatorWear = None


@unittest.skipIf(PusherActuatorWear is None, "MuJoCo optional dependencies unavailable")
class PusherActuatorWearTests(unittest.TestCase):
    def make_env(
        self,
        wear_rate=0.1,
        expose_health=False,
        degradation_mode="endogenous_action",
        **health_kwargs,
    ):
        return PusherActuatorWear.make(
            ActuatorWearConfig(
                wear_rate=wear_rate,
                minimum_efficiency=0.2,
                expose_health=expose_health,
                degradation_mode=degradation_mode,
                **health_kwargs,
            )
        )

    def test_episode_reset_preserves_wear_and_model_gain(self):
        env = self.make_env()
        try:
            env.reset_lifetime(seed=1)
            env.step(np.full(env.action_space.shape, 2.0))
            wear_before = env.wear
            gain_before = np.array(
                env.unwrapped.model.actuator_gainprm[:, 0], copy=True
            )
            env.reset(seed=2)
            np.testing.assert_allclose(env.wear, wear_before)
            np.testing.assert_allclose(
                env.unwrapped.model.actuator_gainprm[:, 0], gain_before
            )
        finally:
            env.close()

    def test_lifetime_reset_restores_health_and_gain(self):
        env = self.make_env()
        try:
            env.reset_lifetime(seed=3)
            base_gain = np.array(env.unwrapped.model.actuator_gainprm[:, 0], copy=True)
            env.step(np.full(env.action_space.shape, 2.0))
            self.assertGreater(env.wear, 0.0)
            env.reset_lifetime(seed=3)
            self.assertEqual(env.wear, 0.0)
            np.testing.assert_allclose(
                env.unwrapped.model.actuator_gainprm[:, 0], base_gain
            )
        finally:
            env.close()

    def test_action_dose_is_endogenous(self):
        idle = self.make_env()
        loaded = self.make_env()
        try:
            idle.reset_lifetime(seed=4)
            loaded.reset_lifetime(seed=4)
            idle.step(np.zeros(idle.action_space.shape))
            loaded.step(np.full(loaded.action_space.shape, 2.0))
            self.assertEqual(idle.wear, 0.0)
            self.assertGreater(loaded.wear, idle.wear)
        finally:
            idle.close()
            loaded.close()

    def test_threshold_law_requires_suprathreshold_action_dose(self):
        config = {
            "wear_rate": 0.1,
            "degradation_law_family": "threshold",
            "threshold_dose": 0.5,
        }
        low = self.make_env(**config)
        high = self.make_env(**config)
        try:
            low.reset_lifetime(seed=35)
            high.reset_lifetime(seed=35)
            low.step(np.full(low.action_space.shape, 0.5))
            high.step(np.full(high.action_space.shape, 2.0))
            self.assertEqual(low.wear, 0.0)
            self.assertGreater(high.wear, 0.0)
        finally:
            low.close()
            high.close()

    def test_stochastic_shock_law_is_seed_deterministic(self):
        config = ActuatorWearConfig(
            wear_rate=0.001,
            degradation_law_family="stochastic_shock",
            stochastic_shock_probability=1.0,
            stochastic_shock_size=0.1,
        )
        first = PusherActuatorWear.make(config)
        second = PusherActuatorWear.make(config)
        try:
            first.reset_lifetime(seed=36)
            second.reset_lifetime(seed=36)
            action = np.zeros(first.action_space.shape)
            first.step(action)
            second.step(action)
            self.assertEqual(first.wear, second.wear)
            self.assertEqual(first.wear, 0.1)
        finally:
            first.close()
            second.close()

    def test_exogenous_control_is_action_independent(self):
        idle = self.make_env(degradation_mode="exogenous_clock")
        loaded = self.make_env(degradation_mode="exogenous_clock")
        try:
            idle.reset_lifetime(seed=40)
            loaded.reset_lifetime(seed=40)
            idle.step(np.zeros(idle.action_space.shape))
            loaded.step(np.full(loaded.action_space.shape, 2.0))
            self.assertAlmostEqual(idle.wear, loaded.wear)
            self.assertNotEqual(
                idle.cumulative_action_dose, loaded.cumulative_action_dose
            )
            self.assertEqual(idle.cumulative_health_dose, loaded.cumulative_health_dose)
        finally:
            idle.close()
            loaded.close()

    def test_health_is_hidden_by_default_and_optional_when_privileged(self):
        hidden = self.make_env(expose_health=False)
        visible = self.make_env(expose_health=True)
        try:
            hidden_observation, hidden_info = hidden.reset_lifetime(seed=5)
            visible_observation, visible_info = visible.reset_lifetime(seed=5)
            self.assertEqual(
                visible_observation.shape[0], hidden_observation.shape[0] + 2
            )
            self.assertIn("lifephy/wear", hidden_info)
            self.assertEqual(visible_observation[-2], visible_info["lifephy/wear"])
        finally:
            hidden.close()
            visible.close()

    def test_thermal_state_heats_and_partially_cools_across_episode_reset(self):
        env = self.make_env(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.4,
            thermal_episode_cooling=0.25,
        )
        try:
            env.reset_lifetime(seed=50)
            env.step(np.full(env.action_space.shape, 2.0))
            thermal_before_reset = env.thermal_load
            self.assertGreater(thermal_before_reset, 0.0)
            env.reset(seed=51)
            self.assertAlmostEqual(env.thermal_load, thermal_before_reset * 0.75)
            self.assertGreater(env.thermal_load, 0.0)
            env.reset_lifetime(seed=52)
            self.assertEqual(env.thermal_load, 0.0)
        finally:
            env.close()

    def test_thermal_derating_changes_actual_rollout(self):
        fresh = self.make_env(wear_rate=0.0, thermal_enabled=True)
        hot = self.make_env(wear_rate=0.0, thermal_enabled=True)
        try:
            fresh_observation, _ = fresh.reset_lifetime(seed=60)
            hot_observation, _ = hot.reset_lifetime(seed=60)
            np.testing.assert_allclose(fresh_observation, hot_observation)
            hot.set_thermal_load_for_diagnostic(0.9)
            action = np.full(fresh.action_space.shape, 1.0)
            for _ in range(5):
                fresh_observation, *_ = fresh.step(action)
                hot_observation, *_ = hot.step(action)
            self.assertGreater(
                np.linalg.norm(fresh_observation[11:18]),
                np.linalg.norm(hot_observation[11:18]),
            )
        finally:
            fresh.close()
            hot.close()

    def test_canonical_task_reset_hides_health_but_preserves_thermal_dynamics(self):
        config = {
            "wear_rate": 0.0,
            "thermal_enabled": True,
            "thermal_heat_rate": 0.1,
            "thermal_cooling_rate": 0.0,
            "thermal_episode_cooling": 0.0,
            "canonical_task_seed": 811,
        }
        cold = self.make_env(**config)
        hot = self.make_env(**config)
        try:
            cold.reset_lifetime(seed=61)
            hot.reset_lifetime(seed=62)
            hot_action = np.full(hot.action_space.shape, 1.0)
            for _ in range(5):
                hot.step(hot_action)
            self.assertGreater(hot.thermal_load, 0.0)
            cold_observation, _ = cold.reset(seed=63)
            hot_observation, _ = hot.reset(seed=64)
            np.testing.assert_allclose(cold_observation, hot_observation)
            self.assertEqual(cold.thermal_load, 0.0)
            action = np.full(cold.action_space.shape, 1.0)
            for _ in range(5):
                cold_observation, *_ = cold.step(action)
                hot_observation, *_ = hot.step(action)
            self.assertGreater(
                np.linalg.norm(cold_observation[11:18]),
                np.linalg.norm(hot_observation[11:18]),
            )
        finally:
            cold.close()
            hot.close()

    def test_joint_aging_persists_and_lifetime_reset_restores_damping(self):
        env = self.make_env(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_rate=0.2,
            joint_aging_damping_multiplier=4.0,
        )
        try:
            env.reset_lifetime(seed=70)
            base_damping = np.array(
                env.unwrapped.model.dof_damping[env._actuated_dof_indices], copy=True
            )
            env.step(np.full(env.action_space.shape, 2.0))
            aging_before_reset = env.joint_aging
            self.assertGreater(aging_before_reset, 0.0)
            self.assertTrue(
                np.all(
                    env.unwrapped.model.dof_damping[env._actuated_dof_indices]
                    > base_damping
                )
            )
            env.reset(seed=71)
            self.assertEqual(env.joint_aging, aging_before_reset)
            env.reset_lifetime(seed=72)
            self.assertEqual(env.joint_aging, 0.0)
            np.testing.assert_allclose(
                env.unwrapped.model.dof_damping[env._actuated_dof_indices],
                base_damping,
            )
        finally:
            env.close()

    def test_joint_aging_exogenous_control_is_action_independent(self):
        config = {
            "wear_rate": 0.0,
            "joint_aging_enabled": True,
            "joint_aging_rate": 0.1,
            "joint_aging_degradation_mode": "exogenous_clock",
            "joint_aging_exogenous_dose_per_step": 0.25,
        }
        idle = self.make_env(**config)
        loaded = self.make_env(**config)
        try:
            idle.reset_lifetime(seed=75)
            loaded.reset_lifetime(seed=75)
            idle.step(np.zeros(idle.action_space.shape))
            loaded.step(np.full(loaded.action_space.shape, 2.0))
            self.assertAlmostEqual(idle.joint_aging, loaded.joint_aging)
            self.assertNotEqual(
                idle.cumulative_action_dose, loaded.cumulative_action_dose
            )
            self.assertEqual(
                idle.cumulative_joint_aging_dose,
                loaded.cumulative_joint_aging_dose,
            )
        finally:
            idle.close()
            loaded.close()

    def test_joint_aging_damping_changes_actual_rollout(self):
        fresh = self.make_env(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_damping_multiplier=8.0,
        )
        aged = self.make_env(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_damping_multiplier=8.0,
        )
        try:
            fresh_observation, _ = fresh.reset_lifetime(seed=80)
            aged_observation, _ = aged.reset_lifetime(seed=80)
            np.testing.assert_allclose(fresh_observation, aged_observation)
            aged.set_joint_aging_for_diagnostic(0.9)
            action = np.full(fresh.action_space.shape, 1.0)
            for _ in range(5):
                fresh_observation, *_ = fresh.step(action)
                aged_observation, *_ = aged.step(action)
            self.assertGreater(
                np.linalg.norm(fresh_observation[11:18]),
                np.linalg.norm(aged_observation[11:18]),
            )
        finally:
            fresh.close()
            aged.close()

    def test_reacher_supports_all_three_health_mechanisms(self):
        env = PusherActuatorWear.make(
            ActuatorWearConfig(
                wear_rate=0.1,
                thermal_enabled=True,
                thermal_heat_rate=0.1,
                joint_aging_enabled=True,
                joint_aging_rate=0.1,
            ),
            environment_id="Reacher-v5",
        )
        try:
            env.reset_lifetime(seed=90)
            env.step(np.full(env.action_space.shape, 2.0))
            self.assertGreater(env.wear, 0.0)
            self.assertGreater(env.thermal_load, 0.0)
            self.assertGreater(env.joint_aging, 0.0)
            before_reset = (env.wear, env.thermal_load, env.joint_aging)
            env.reset(seed=91)
            self.assertEqual(env.wear, before_reset[0])
            self.assertEqual(env.joint_aging, before_reset[2])
            self.assertGreater(env.thermal_load, 0.0)
            env.reset_lifetime(seed=92)
            self.assertEqual(
                (env.wear, env.thermal_load, env.joint_aging), (0.0, 0.0, 0.0)
            )
        finally:
            env.close()

    def test_workers_have_independent_models_and_health(self):
        first = self.make_env()
        second = self.make_env()
        try:
            first.reset_lifetime(seed=6)
            second.reset_lifetime(seed=6)
            first.step(np.full(first.action_space.shape, 2.0))
            self.assertGreater(first.wear, second.wear)
            self.assertFalse(
                np.shares_memory(
                    first.unwrapped.model.actuator_gainprm,
                    second.unwrapped.model.actuator_gainprm,
                )
            )
            self.assertFalse(
                np.allclose(
                    first.unwrapped.model.actuator_gainprm[:, 0],
                    second.unwrapped.model.actuator_gainprm[:, 0],
                )
            )
        finally:
            first.close()
            second.close()

    def test_degraded_gain_changes_actual_rollout(self):
        fresh = self.make_env(wear_rate=0.0)
        worn = self.make_env(wear_rate=0.0)
        try:
            fresh_observation, _ = fresh.reset_lifetime(seed=7)
            worn_observation, _ = worn.reset_lifetime(seed=7)
            np.testing.assert_allclose(fresh_observation, worn_observation)
            worn.set_wear_for_diagnostic(0.9)
            action = np.full(fresh.action_space.shape, 1.0)
            for _ in range(5):
                fresh_observation, *_ = fresh.step(action)
                worn_observation, *_ = worn.step(action)
            self.assertGreater(
                np.linalg.norm(fresh_observation - worn_observation), 1e-6
            )
            # Pusher-v5 places the seven actuated joint velocities at [11:18].
            # Lower actuator gain must reduce their response to the same command.
            self.assertGreater(
                np.linalg.norm(fresh_observation[11:18]),
                np.linalg.norm(worn_observation[11:18]),
            )
        finally:
            fresh.close()
            worn.close()


if __name__ == "__main__":
    unittest.main()
