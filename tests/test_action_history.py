import numpy as np

from lifephybench.envs.action_history import PreviousAppliedActionObservation
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def make_environment():
    base = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.1,
            canonical_task_seed=811,
        ),
        max_episode_steps=100,
    )
    return PreviousAppliedActionObservation(base)


def test_previous_applied_action_is_appended_then_zeroed_on_task_reset():
    environment = make_environment()
    try:
        observation, _ = environment.reset_lifetime(seed=1)
        assert observation.shape == (30,)
        np.testing.assert_array_equal(observation[-7:], np.zeros(7))

        action = np.linspace(-1.0, 1.0, 7)
        observation, _, _, _, info = environment.step(action)
        np.testing.assert_allclose(observation[-7:], action)
        assert info["lifephy/previous_applied_action_observed"] is True
        assert environment.thermal_load > 0.0

        observation, _ = environment.reset(seed=2)
        np.testing.assert_array_equal(observation[-7:], np.zeros(7))
        assert environment.thermal_load > 0.0
    finally:
        environment.close()


def test_lifetime_reset_zeros_action_history_and_thermal_health():
    environment = make_environment()
    try:
        environment.reset_lifetime(seed=1)
        environment.step(np.ones(7))
        observation, _ = environment.reset_lifetime(seed=2, lifetime_id=7)
        np.testing.assert_array_equal(observation[-7:], np.zeros(7))
        assert environment.thermal_load == 0.0
        assert environment.lifetime_id == 7
    finally:
        environment.close()
