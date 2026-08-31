from dataclasses import replace

import numpy as np
import pytest
from gymnasium import spaces

from lifephybench.envs.hierarchical_thermal import (
    HierarchicalThermalConfig,
    HierarchicalThermalModeEnv,
)
from lifephybench.envs.hierarchical_thermal_v11 import (
    HierarchicalThermalV11Config,
    HierarchicalThermalV11Env,
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


class ZeroReacherLowLevelModel:
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64
    )
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def predict(self, observation, state, episode_start, deterministic):
        assert np.asarray(observation).shape == (11,)
        assert deterministic is True
        return np.zeros(2, dtype=np.float32), state


def config(condition="fixed", **overrides):
    base = HierarchicalThermalV11Config(
        condition=condition,
        low_level_model_path="injected-test-model",
        episode_steps=3,
        episodes_per_lifetime=4,
    )
    return replace(base, **overrides)


def make_environment(condition="fixed", **overrides):
    return HierarchicalThermalV11Env(
        config(condition, **overrides), low_level_model=ZeroLowLevelModel()
    )


def test_reacher_morphology_uses_generic_observation_and_action_contract():
    reacher_config = config(
        "fixed", environment_id="Reacher-v5", sensor_noise_sd=0.0
    )
    environment = HierarchicalThermalV11Env(
        reacher_config, low_level_model=ZeroReacherLowLevelModel()
    )
    try:
        observation, _ = environment.reset(seed=17)
        assert observation.shape == (15,)
        assert environment.observation_space.contains(observation)
        next_observation, reward, terminated, truncated, info = environment.step(0)
        assert next_observation.shape == (15,)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["lifephy/hierarchical_physical_steps"] > 0
    finally:
        environment.close()


def test_observation_keeps_v10_shape_with_nonprivileged_v11_summary():
    environment = make_environment(sensor_noise_sd=0.02)
    try:
        observation, info = environment.reset(seed=17)
        assert observation.shape == (28,)
        assert environment.observation_space.contains(observation)
        summary = observation[-5:-1]
        assert summary[0] == 0.0
        assert summary[1] == info["lifephy/v11_load_sensor"]
        assert summary[2] == 0.0
        assert summary[3] == 0.0
        assert observation[-1] == 1.0
        assert info["lifephy/thermal_load"] == pytest.approx(0.04)
        # The policy receives the clipped noisy sensor, not the exact audit load.
        assert summary[1] != pytest.approx(info["lifephy/thermal_load"])

        next_observation, _, _, _, step_info = environment.step(0)
        next_summary = next_observation[-5:-1]
        assert next_summary[0] == -1.0
        assert next_summary[1] == step_info["lifephy/v11_next_load_sensor"]
        assert next_summary[2] == pytest.approx(1.0 / 3.0)
        assert next_summary[3] == float(step_info["lifephy/thermal_trip"])
        # The complete four-coordinate contract leaves no coordinate for exact
        # health, task reward, or action dose (audit values may coincide with a
        # marker numerically, so membership tests would be invalid here).
        np.testing.assert_allclose(
            next_summary,
            [
                -1.0,
                step_info["lifephy/v11_next_load_sensor"],
                1.0 / 3.0,
                float(step_info["lifephy/thermal_trip"]),
            ],
        )
    finally:
        environment.close()


def test_sensor_noise_is_active_and_identically_indexed_in_both_conditions():
    fixed = make_environment("fixed", sensor_noise_sd=0.02)
    stochastic = make_environment("stochastic", sensor_noise_sd=0.02)
    try:
        fixed_observation, fixed_info = fixed.reset(seed=29)
        stochastic_observation, stochastic_info = stochastic.reset(seed=29)
        assert fixed_info["lifephy/v11_load_sensor_raw_noise"] == pytest.approx(
            stochastic_info["lifephy/v11_load_sensor_raw_noise"]
        )
        assert fixed_observation[-4] == pytest.approx(
            np.clip(
                0.04 + fixed_info["lifephy/v11_load_sensor_raw_noise"], 0.0, 1.0
            )
        )
        assert stochastic_observation[-4] == pytest.approx(
            np.clip(
                stochastic_info["lifephy/v11_initial_thermal_load"]
                + stochastic_info["lifephy/v11_load_sensor_raw_noise"],
                0.0,
                1.0,
            )
        )
    finally:
        fixed.close()
        stochastic.close()


def test_fixed_and_stochastic_physics_share_heat_and_cooling_but_not_shocks():
    common = {
        "sensor_noise_sd": 0.0,
        "thermal_heat_rate": 0.05,
        "thermal_episode_cooling": 0.10,
        "shock_probability": 1.0,
        "shock_size": 0.01,
    }
    fixed = make_environment("fixed", **common)
    stochastic_a = make_environment("stochastic", **common)
    stochastic_b = make_environment("stochastic", **common)
    try:
        _, fixed_reset = fixed.reset(seed=41)
        _, stochastic_reset_a = stochastic_a.reset(seed=41)
        _, stochastic_reset_b = stochastic_b.reset(seed=41)
        assert fixed_reset["lifephy/v11_initial_thermal_load"] == pytest.approx(0.04)
        assert 0.0 <= stochastic_reset_a["lifephy/v11_initial_thermal_load"] <= 0.08
        assert stochastic_reset_a["lifephy/v11_initial_thermal_load"] == pytest.approx(
            stochastic_reset_b["lifephy/v11_initial_thermal_load"]
        )

        _, _, _, _, fixed_info = fixed.step(0)
        stochastic_observation_a, _, _, _, stochastic_info_a = stochastic_a.step(0)
        stochastic_observation_b, _, _, _, stochastic_info_b = stochastic_b.step(0)
        assert fixed_info["lifephy/v11_task_shock_count"] == 0
        assert stochastic_info_a["lifephy/v11_task_shock_count"] == 3
        assert stochastic_info_a["lifephy/v11_task_shock_step_indices"] == [0, 1, 2]
        assert stochastic_info_a["lifephy/v11_task_shock_step_indices"] == (
            stochastic_info_b["lifephy/v11_task_shock_step_indices"]
        )
        np.testing.assert_allclose(stochastic_observation_a, stochastic_observation_b)
        # Zero low-level actions add no ordinary heat. Cooling occurs after all
        # three indexed shocks and before the next task-boundary observation.
        expected_stochastic_load = (
            stochastic_reset_a["lifephy/v11_initial_thermal_load"] + 0.03
        ) * 0.90
        assert stochastic_info_a["lifephy/thermal_load"] == pytest.approx(
            expected_stochastic_load
        )
        assert fixed_info["lifephy/thermal_load"] == pytest.approx(0.04 * 0.90)
        assert fixed.config.thermal_heat_rate == stochastic_a.config.thermal_heat_rate
        assert (
            fixed.config.thermal_episode_cooling
            == stochastic_a.config.thermal_episode_cooling
        )
    finally:
        fixed.close()
        stochastic_a.close()
        stochastic_b.close()


def test_task_index_runs_from_zero_to_one_within_each_lifetime():
    environment = make_environment(
        sensor_noise_sd=0.0, episodes_per_lifetime=3, episode_steps=1
    )
    try:
        observation, _ = environment.reset(seed=53)
        assert observation[-3] == 0.0
        observation, _, _, truncated, info = environment.step(0)
        assert not truncated
        assert info["lifephy/v11_task_index"] == 0
        assert observation[-3] == pytest.approx(0.5)
        observation, _, _, truncated, info = environment.step(0)
        assert not truncated
        assert info["lifephy/v11_task_index"] == 1
        assert observation[-3] == 1.0
        _, _, _, truncated, info = environment.step(0)
        assert truncated
        assert info["lifephy/v11_task_index"] == 2
        assert info["lifephy/v11_normalized_task_index"] == 1.0
    finally:
        environment.close()


def test_indexed_uncertainty_does_not_drift_after_an_early_trip():
    settings = {
        "episode_steps": 8,
        "episodes_per_lifetime": 3,
        "sensor_noise_sd": 0.02,
        "shock_probability": 0.5,
        "shock_size": 0.01,
    }
    early_trip = make_environment("stochastic", **settings)
    full_task = make_environment("stochastic", **settings)
    try:
        early_trip.reset(seed=73)
        full_task.reset(seed=73)
        early_trip._health.set_thermal_load_for_diagnostic(0.20)
        _, _, _, _, early_info = early_trip.step(1)
        _, _, _, _, full_info = full_task.step(0)
        assert early_info["lifephy/hierarchical_physical_steps"] == 1
        assert full_info["lifephy/hierarchical_physical_steps"] == 8
        assert early_info["lifephy/v11_next_load_sensor_raw_noise"] == pytest.approx(
            full_info["lifephy/v11_next_load_sensor_raw_noise"]
        )

        _, _, _, _, early_next = early_trip.step(0)
        _, _, _, _, full_next = full_task.step(0)
        assert early_next["lifephy/v11_task_index"] == 1
        assert full_next["lifephy/v11_task_index"] == 1
        assert early_next["lifephy/v11_task_shock_step_indices"] == full_next[
            "lifephy/v11_task_shock_step_indices"
        ]
    finally:
        early_trip.close()
        full_task.close()


def test_canonical_task_state_is_independent_of_uncertainty_seed():
    first = make_environment("stochastic")
    second = make_environment("stochastic")
    try:
        first_observation, first_info = first.reset(seed=101)
        second_observation, second_info = second.reset(seed=202)
        # Raw canonical Pusher boundary state stays seed 811; only the v11
        # summary is allowed to vary with the uncertainty seed.
        np.testing.assert_array_equal(first_observation[:-5], second_observation[:-5])
        assert first_info["lifephy/v11_uncertainty_schedule_seed"] != second_info[
            "lifephy/v11_uncertainty_schedule_seed"
        ]
        assert first_observation[-5:-1].tolist() != second_observation[-5:-1].tolist()
    finally:
        first.close()
        second.close()


def test_observation_bounds_clip_sensor_under_extreme_noise_and_heat():
    environment = make_environment(
        "stochastic",
        episode_steps=2,
        sensor_noise_sd=100.0,
        shock_probability=1.0,
        shock_size=1.0,
    )
    try:
        observation, _ = environment.reset(seed=303)
        assert environment.observation_space.contains(observation)
        observation, _, _, _, _ = environment.step(0)
        assert environment.observation_space.contains(observation)
        assert 0.0 <= observation[-4] <= 1.0
    finally:
        environment.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition", "unknown"),
        ("thermal_episode_cooling", 1.1),
        ("sensor_noise_sd", -0.1),
        ("shock_probability", 1.1),
        ("shock_size", -0.01),
        ("stochastic_initial_load_high", 1.1),
    ],
)
def test_configuration_rejects_invalid_uncertainty_values(field, value):
    valid = HierarchicalThermalV11Config(
        condition="fixed", low_level_model_path="model"
    )
    with pytest.raises(ValueError):
        replace(valid, **{field: value})


def test_v10_environment_semantics_remain_separate_and_unchanged():
    v10 = HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path="unused-in-test",
            degradation_mode="exogenous_clock",
            episode_steps=1,
            episodes_per_lifetime=2,
        ),
        low_level_model=ZeroLowLevelModel(),
    )
    v11 = make_environment(sensor_noise_sd=0.0, episode_steps=1)
    try:
        v10_observation, _ = v10.reset(seed=1)
        v11_observation, _ = v11.reset(seed=1)
        assert v10_observation.shape == v11_observation.shape == (28,)
        np.testing.assert_array_equal(v10_observation[-5:-1], np.zeros(4))
        np.testing.assert_array_equal(
            v11_observation[-5:-1], np.asarray([0.0, 0.04, 0.0, 0.0])
        )
    finally:
        v10.close()
        v11.close()
