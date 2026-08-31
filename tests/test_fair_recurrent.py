import numpy as np
import torch
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from gymnasium import spaces

from lifephybench.envs.lifetime import LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.envs.task_boundary import TaskBoundaryObservation
from lifephybench.selective_reset_policy import (
    CommitmentModeMaskedMlpLstmPolicy,
    CommitmentModeMaskedTaskResetMlpLstmPolicy,
    DecisionMaskedDiagGaussianDistribution,
    TaskResetMlpLstmPolicy,
    task_reset_mask,
)


def test_task_boundary_marker_does_not_end_inner_task():
    base = PusherActuatorWear.make(
        ActuatorWearConfig(wear_rate=0.0), max_episode_steps=1
    )
    env = TaskBoundaryObservation(LifetimeStreamWrapper(base, 2))
    try:
        observation, _ = env.reset(seed=1)
        assert observation[-1] == 1.0
        observation, _, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape)
        )
        assert info["lifephy/inner_task_boundary"]
        assert observation[-1] == 1.0
        assert not terminated and not truncated
    finally:
        env.close()


def test_canonical_task_reset_keeps_hidden_thermal_state_across_stream_boundary():
    base = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.1,
            thermal_cooling_rate=0.0,
            thermal_episode_cooling=0.0,
            canonical_task_seed=812,
        ),
        max_episode_steps=1,
    )
    env = TaskBoundaryObservation(LifetimeStreamWrapper(base, 2))
    try:
        initial, _ = env.reset(seed=1)
        boundary, _, terminated, truncated, info = env.step(
            np.full(env.action_space.shape, 1.0)
        )
        assert info["lifephy/inner_task_boundary"]
        assert not terminated and not truncated
        np.testing.assert_allclose(initial[:-1], boundary[:-1])
        assert initial[-1] == boundary[-1] == 1.0
        assert base.thermal_load > 0.0
    finally:
        env.close()


def test_task_reset_mask_combines_marker_and_lifetime_start():
    observation = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    lifetime_starts = torch.tensor([0.0, 0.0, 1.0])
    assert torch.equal(
        task_reset_mask(observation, lifetime_starts),
        torch.tensor([1.0, 0.0, 1.0]),
    )


def test_predict_path_resets_hidden_state_when_marker_is_set():
    policy = TaskResetMlpLstmPolicy(
        spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
    )
    hidden = np.ones(policy.lstm_hidden_state_shape, dtype=np.float32)
    state = (hidden.copy(), hidden.copy())
    marked = np.asarray([0.1, -0.2, 1.0], dtype=np.float32)
    zero_state = (np.zeros_like(hidden), np.zeros_like(hidden))

    action_from_marked, state_from_marked = policy.predict(
        marked,
        state=state,
        episode_start=np.asarray([False]),
        deterministic=True,
    )
    action_from_zero, state_from_zero = policy.predict(
        marked,
        state=zero_state,
        episode_start=np.asarray([False]),
        deterministic=True,
    )

    np.testing.assert_allclose(action_from_marked, action_from_zero)
    np.testing.assert_allclose(state_from_marked[0], state_from_zero[0])
    np.testing.assert_allclose(state_from_marked[1], state_from_zero[1])


def test_decision_masked_gaussian_excludes_only_mode_coordinate():
    distribution = DecisionMaskedDiagGaussianDistribution(3)
    distribution.proba_distribution(
        torch.zeros((2, 3)), torch.zeros((2, 3))
    )
    distribution.set_mode_decision_mask(torch.tensor([1.0, 0.0]))
    actions = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    per_dimension = distribution.distribution.log_prob(actions)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()

    assert torch.allclose(log_prob[0], per_dimension[0].sum())
    assert torch.allclose(log_prob[1], per_dimension[1, 1:].sum())
    assert torch.allclose(
        entropy[0] - entropy[1], distribution.distribution.entropy()[0, 0]
    )


def test_commitment_policy_mode_log_prob_is_masked_after_selection():
    policy = CommitmentModeMaskedMlpLstmPolicy(
        spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
    )
    zeros = torch.zeros(policy.lstm_hidden_state_shape)
    states = RNNStates((zeros, zeros), (zeros, zeros))
    episode_starts = torch.zeros(1)
    actions_a = torch.zeros((1, 8))
    actions_b = actions_a.clone()
    actions_b[:, 0] = 0.75

    selected = torch.zeros((1, 26))
    selected[:, -3] = 1.0
    _, selected_a, _ = policy.evaluate_actions(
        selected, actions_a, states, episode_starts
    )
    _, selected_b, _ = policy.evaluate_actions(
        selected, actions_b, states, episode_starts
    )
    assert torch.allclose(selected_a, selected_b)

    decision = selected.clone()
    decision[:, -3] = 0.0
    _, decision_a, _ = policy.evaluate_actions(
        decision, actions_a, states, episode_starts
    )
    _, decision_b, _ = policy.evaluate_actions(
        decision, actions_b, states, episode_starts
    )
    assert not torch.allclose(decision_a, decision_b)


def test_task_reset_commitment_policy_combines_both_masks():
    policy = CommitmentModeMaskedTaskResetMlpLstmPolicy(
        spaces.Box(low=-1.0, high=1.0, shape=(26,), dtype=np.float32),
        spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
        lr_schedule=lambda _: 3e-4,
    )
    assert isinstance(policy.action_dist, DecisionMaskedDiagGaussianDistribution)
