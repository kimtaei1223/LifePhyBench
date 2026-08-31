import numpy as np
from gymnasium import spaces

from lifephybench.envs.hierarchical_thermal import (
    HierarchicalThermalConfig,
    HierarchicalThermalModeEnv,
)


class ZeroLowLevelModel:
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64
    )
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(7,), dtype=np.float32)

    def predict(self, observation, state, episode_start, deterministic):
        assert np.asarray(observation).shape == (24,)
        assert deterministic is True
        return np.zeros(7, dtype=np.float32), state


def make_environment(degradation_mode="exogenous_clock"):
    return HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path="unused-in-test",
            degradation_mode=degradation_mode,
            episode_steps=3,
            episodes_per_lifetime=2,
        ),
        low_level_model=ZeroLowLevelModel(),
    )


def test_one_discrete_action_executes_one_complete_physical_task():
    environment = make_environment()
    try:
        observation, info = environment.reset(seed=1)
        assert observation.shape == (28,)
        np.testing.assert_array_equal(observation[-5:-1], np.zeros(4))
        assert observation[-1] == 1.0
        assert info["lifephy/hierarchical_lifetime_start"] is True

        observation, reward, terminated, truncated, info = environment.step(1)
        assert info["lifephy/hierarchical_physical_steps"] == 3
        assert info["lifephy/inner_task_boundary"] is True
        assert info["lifephy/thermal_mode"] == "high"
        assert info["lifephy/thermal_load"] == 0.0
        assert not terminated and not truncated
        assert observation[-5] == 1.0
        assert observation[-1] == 1.0
        assert np.isfinite(reward)

        _, _, terminated, truncated, info = environment.step(1)
        assert not terminated and truncated
        assert info["lifephy/lifetime_boundary"] is True
    finally:
        environment.close()


def test_dynamic_summary_contains_only_action_derived_increment():
    environment = make_environment("endogenous_action")
    try:
        observation, _ = environment.reset(seed=2)
        observation, _, _, _, info = environment.step(0)
        summary = observation[-5:-1]
        assert summary[0] == -1.0
        assert summary[1] == 0.0
        assert summary[3] == 0.0
        assert info["lifephy/thermal_load"] == 0.0
    finally:
        environment.close()


def test_invalid_discrete_mode_is_rejected():
    environment = make_environment()
    try:
        environment.reset(seed=3)
        try:
            environment.step(2)
        except ValueError as error:
            assert "invalid discrete mode" in str(error)
        else:
            raise AssertionError("invalid action was accepted")
    finally:
        environment.close()


def test_mode_trip_summary_hides_dose_and_reward_and_teacher_is_training_only():
    environment = HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path="unused-in-test",
            degradation_mode="endogenous_action",
            episode_steps=3,
            episodes_per_lifetime=2,
            summary_mode="mode_trip",
            training_teacher_safe_high_load=0.0,
            training_teacher_shaping=5.0,
        ),
        low_level_model=ZeroLowLevelModel(),
    )
    try:
        environment.reset(seed=4)
        observation, reward, _, _, info = environment.step(0)
        assert observation[-5] == -1.0
        assert observation[-4] == 0.0
        assert observation[-3] == 0.0
        assert observation[-2] == 0.0
        assert info["lifephy/training_teacher_target"] == "low"
        assert info["lifephy/training_teacher_reward"] == 5.0
        assert reward == info["lifephy/hierarchical_physical_reward"] + 5.0
    finally:
        environment.close()
