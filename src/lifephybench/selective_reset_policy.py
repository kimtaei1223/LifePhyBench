"""Recurrent policy whose task-memory reset is independent of Gym termination."""

from __future__ import annotations

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import (
    DiagGaussianDistribution,
    Distribution,
)


def task_reset_mask(observation: th.Tensor, episode_starts: th.Tensor) -> th.Tensor:
    """Combine true lifetime starts with the appended task-boundary marker."""

    boundary = (observation[..., -1] > 0.5).to(dtype=episode_starts.dtype)
    return th.maximum(episode_starts, boundary.reshape(episode_starts.shape))


class TaskResetMlpLstmPolicy(RecurrentActorCriticPolicy):
    """Force recurrent state reset at task boundaries without ending the MDP."""

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
        return super().forward(
            obs,
            lstm_states,
            task_reset_mask(obs, episode_starts),
            deterministic,
        )

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ):
        return super().evaluate_actions(
            obs,
            actions,
            lstm_states,
            task_reset_mask(obs, episode_starts),
        )

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        return super().predict_values(
            obs,
            lstm_states,
            task_reset_mask(obs, episode_starts),
        )

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> tuple[Distribution, tuple[th.Tensor, ...]]:
        """Apply the same task reset during deterministic/stochastic inference."""

        return super().get_distribution(
            obs,
            lstm_states,
            task_reset_mask(obs, episode_starts),
        )


class DecisionMaskedDiagGaussianDistribution(DiagGaussianDistribution):
    """Exclude the latched mode coordinate outside commitment decisions."""

    def __init__(self, action_dim: int):
        super().__init__(action_dim)
        self._mode_decision_mask: th.Tensor | None = None

    def set_mode_decision_mask(self, mask: th.Tensor) -> None:
        self._mode_decision_mask = mask.reshape(-1)

    def _masked_sum(self, per_dimension: th.Tensor) -> th.Tensor:
        if self._mode_decision_mask is None:
            raise RuntimeError("mode decision mask must be set before loss evaluation")
        if per_dimension.ndim != 2 or per_dimension.shape[1] != self.action_dim:
            raise ValueError("expected per-dimension values for continuous actions")
        mask = self._mode_decision_mask.to(
            device=per_dimension.device, dtype=per_dimension.dtype
        )
        if mask.shape[0] != per_dimension.shape[0]:
            raise ValueError("mode decision mask batch does not match actions")
        weights = th.cat(
            [mask[:, None], th.ones_like(per_dimension[:, 1:])], dim=1
        )
        return (per_dimension * weights).sum(dim=1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution parameters have not been set")
        return self._masked_sum(self.distribution.log_prob(actions))

    def entropy(self) -> th.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution parameters have not been set")
        return self._masked_sum(self.distribution.entropy())


class _CommitmentModeLossMaskMixin:
    """Mask the mode-coordinate PPO loss after a task has latched its mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("commitment mode masking requires Gaussian Box actions")
        self.action_dist = DecisionMaskedDiagGaussianDistribution(
            int(self.action_space.shape[0])
        )

    def _set_commitment_mode_mask(self, obs: th.Tensor) -> None:
        # ThermalModeCommitment appends [mode_selected, mode], then the fair
        # environment appends the task-boundary marker.
        decision = obs[..., -3] < 0.5
        self.action_dist.set_mode_decision_mask(decision.to(dtype=obs.dtype))

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ):
        self._set_commitment_mode_mask(obs)
        return super().forward(obs, lstm_states, episode_starts, deterministic)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
    ):
        self._set_commitment_mode_mask(obs)
        return super().evaluate_actions(
            obs, actions, lstm_states, episode_starts
        )


class CommitmentModeMaskedMlpLstmPolicy(
    _CommitmentModeLossMaskMixin, RecurrentActorCriticPolicy
):
    """Lifetime-memory policy with decision-only mode-coordinate PPO loss."""


class CommitmentModeMaskedTaskResetMlpLstmPolicy(
    _CommitmentModeLossMaskMixin, TaskResetMlpLstmPolicy
):
    """Task-memory policy with decision-only mode-coordinate PPO loss."""
