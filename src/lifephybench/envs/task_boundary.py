"""Observation marker for task boundaries inside a continuous lifetime stream."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class TaskBoundaryObservation(gym.Wrapper):
    """Append a binary marker to observations following a task reset.

    The wrapped environment must expose inner task resets as nonterminal
    transitions, as :class:`LifetimeStreamWrapper` does. Both comparison arms
    receive exactly the same marker and environment termination semantics.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("TaskBoundaryObservation requires a Box observation")
        base = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([np.asarray(base.low), np.asarray([0.0])]),
            high=np.concatenate([np.asarray(base.high), np.asarray([1.0])]),
            dtype=base.dtype,
        )

    def _mark(self, observation: Any, boundary: bool) -> np.ndarray:
        array = np.asarray(observation, dtype=self.observation_space.dtype)
        marker = np.asarray([float(boundary)], dtype=self.observation_space.dtype)
        return np.concatenate([array, marker])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self._mark(observation, True), dict(info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        boundary = bool(info.get("lifephy/inner_task_boundary", False))
        return (
            self._mark(observation, boundary),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )
