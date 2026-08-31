import numpy as np
import pytest

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.envs.thermal_commitment import (
    ThermalCommitmentConfig,
    ThermalModeCommitment,
)


def make_environment() -> ThermalModeCommitment:
    base = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.1,
            thermal_cooling_rate=0.0,
            thermal_episode_cooling=0.0,
            canonical_task_seed=811,
        ),
        max_episode_steps=100,
    )
    return ThermalModeCommitment(base)


def test_commitment_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(trip_load=0.0)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(low_power_scale=1.0)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(trip_penalty=0.0)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(high_power_throughput_bonus=0.0)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(curriculum_start_trip_load=0.7)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(curriculum_lifetimes=10)
    with pytest.raises(ValueError):
        ThermalCommitmentConfig(
            curriculum_start_trip_load=0.05, curriculum_lifetimes=10
        )


def test_training_curriculum_anneals_by_lifetime_without_changing_target():
    base = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.1,
            canonical_task_seed=811,
        ),
        max_episode_steps=100,
    )
    environment = ThermalModeCommitment(
        base,
        ThermalCommitmentConfig(
            curriculum_start_trip_load=0.70,
            curriculum_lifetimes=10,
        ),
    )
    try:
        environment.reset_lifetime(seed=1, lifetime_id=0)
        assert environment.active_trip_load == pytest.approx(0.70)
        environment.reset_lifetime(seed=2, lifetime_id=5)
        assert environment.active_trip_load == pytest.approx(0.40)
        environment.reset_lifetime(seed=3, lifetime_id=10)
        assert environment.active_trip_load == pytest.approx(0.10)
        environment.reset_lifetime(seed=4, lifetime_id=20)
        assert environment.active_trip_load == pytest.approx(0.10)

        action = np.ones(environment.action_space.shape)
        _, _, _, _, info = environment.step(action)
        assert info["lifephy/thermal_trip_load"] == pytest.approx(0.10)
        assert info["lifephy/thermal_target_trip_load"] == pytest.approx(0.10)
    finally:
        environment.close()


def test_action_and_observation_spaces_append_commitment_coordinates():
    environment = make_environment()
    try:
        observation, _ = environment.reset_lifetime(seed=1)
        assert environment.action_space.shape == (8,)
        assert environment.observation_space.shape == (25,)
        np.testing.assert_array_equal(observation[-2:], [0.0, 0.0])

        action = np.zeros(environment.action_space.shape)
        next_observation, *_ = environment.step(action)
        np.testing.assert_array_equal(next_observation[-2:], [1.0, 1.0])
    finally:
        environment.close()


def test_canonical_boundary_hides_thermal_load():
    cold = make_environment()
    hot = make_environment()
    try:
        cold.reset_lifetime(seed=1)
        hot.reset_lifetime(seed=2)
        hot.env.set_thermal_load_for_diagnostic(0.8)
        cold_observation, _ = cold.reset(seed=3)
        hot_observation, _ = hot.reset(seed=4)
        np.testing.assert_allclose(cold_observation, hot_observation, atol=0.0)
        assert cold.env.thermal_load == 0.0
        assert hot.env.thermal_load == 0.8
    finally:
        cold.close()
        hot.close()


def test_high_power_trips_only_when_hot_and_low_power_remains_available():
    environment = make_environment()
    try:
        environment.reset_lifetime(seed=1)
        high_action = np.ones(environment.action_space.shape)
        _, _, _, cold_truncated, cold_info = environment.step(high_action)
        assert not cold_truncated
        assert not cold_info["lifephy/thermal_trip"]
        assert cold_info["lifephy/applied_power_scale"] == 1.0
        assert cold_info["lifephy/high_power_throughput_bonus"] == 2.0
        assert cold_info["lifephy/thermal_load_at_mode_selection"] == 0.0

        environment.reset(seed=2)
        environment.env.set_thermal_load_for_diagnostic(0.8)
        _, reward, _, hot_truncated, hot_info = environment.step(high_action)
        assert hot_truncated
        assert reward == -75.0
        assert hot_info["lifephy/thermal_trip"]
        assert hot_info["lifephy/thermal_load_at_mode_selection"] == 0.8

        environment.reset(seed=3)
        environment.env.set_thermal_load_for_diagnostic(0.8)
        low_action = np.ones(environment.action_space.shape)
        low_action[0] = -1.0
        _, _, _, low_truncated, low_info = environment.step(low_action)
        assert not low_truncated
        assert not low_info["lifephy/thermal_trip"]
        assert low_info["lifephy/applied_power_scale"] == 0.4
        assert low_info["lifephy/control_cost_basis"] == "requested_action"
        assert low_info["lifephy/control_cost_correction"] <= 0.0
    finally:
        environment.close()


def test_requested_action_control_cost_removes_low_power_discount():
    high = make_environment()
    low = make_environment()
    try:
        high.reset_lifetime(seed=1)
        low.reset_lifetime(seed=1)
        high_action = np.ones(high.action_space.shape)
        low_action = high_action.copy()
        low_action[0] = -1.0
        _, _, _, _, high_info = high.step(high_action)
        _, _, _, _, low_info = low.step(low_action)

        assert high_info["lifephy/applied_reward_ctrl"] == pytest.approx(-0.7)
        assert high_info["reward_ctrl"] == pytest.approx(-0.7)
        assert low_info["lifephy/applied_reward_ctrl"] == pytest.approx(-0.112)
        assert low_info["reward_ctrl"] == pytest.approx(-0.7)
        assert low_info["lifephy/control_cost_correction"] == pytest.approx(-0.588)
    finally:
        high.close()
        low.close()


def test_task_reset_clears_mode_but_lifetime_reset_also_clears_health():
    environment = make_environment()
    try:
        environment.reset_lifetime(seed=1)
        action = np.ones(environment.action_space.shape)
        environment.step(action)
        assert environment.high_power is True
        assert environment.env.thermal_load > 0.0

        observation, _ = environment.reset(seed=2)
        assert environment.high_power is None
        assert environment.env.thermal_load > 0.0
        np.testing.assert_array_equal(observation[-2:], [0.0, 0.0])

        observation, _ = environment.reset_lifetime(seed=3)
        assert environment.high_power is None
        assert environment.env.thermal_load == 0.0
        np.testing.assert_array_equal(observation[-2:], [0.0, 0.0])
    finally:
        environment.close()
