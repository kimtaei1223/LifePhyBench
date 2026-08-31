"""Finite observation history with explicit selective-reset semantics."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np


class SelectiveFrameStack(gym.Wrapper):
    """Stack observations and optionally preserve them across task resets.

    Unlike a conventional frame stack, ``history_mode='lifetime'`` clears the
    history only when :class:`LifetimeEpisodeScheduler` starts a new physical
    lifetime. Ordinary task resets append the new initial observation while
    retaining the most recent pre-reset observations.
    """

    def __init__(self, env: gym.Env, stack_size: int, history_mode: str) -> None:
        super().__init__(env)
        if stack_size <= 0:
            raise ValueError("stack_size must be positive")
        if history_mode not in {"task", "lifetime"}:
            raise ValueError("history_mode must be 'task' or 'lifetime'")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("SelectiveFrameStack requires a Box observation space")

        self.stack_size = stack_size
        self.history_mode = history_mode
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)
        self._lifetime_index: int | None = None
        base_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base_space.low] * stack_size, axis=-1),
            high=np.concatenate([base_space.high] * stack_size, axis=-1),
            dtype=base_space.dtype,
        )

    def _clear_and_seed(self, observation: np.ndarray) -> None:
        self._frames.clear()
        zero = np.zeros_like(observation)
        for _ in range(self.stack_size - 1):
            self._frames.append(zero.copy())
        self._frames.append(observation.copy())

    def _append(self, observation: Any) -> np.ndarray:
        array = np.asarray(observation, dtype=self.observation_space.dtype)
        self._frames.append(array.copy())
        return np.concatenate(tuple(self._frames), axis=-1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        array = np.asarray(observation, dtype=self.observation_space.dtype)
        lifetime_index = info.get("lifephy/scheduler_lifetime_index")
        if lifetime_index is None:
            raise RuntimeError(
                "SelectiveFrameStack must wrap LifetimeEpisodeScheduler"
            )
        clear_history = (
            self.history_mode == "task"
            or self._lifetime_index is None
            or lifetime_index != self._lifetime_index
        )
        if clear_history:
            self._clear_and_seed(array)
        else:
            self._append(array)
        self._lifetime_index = int(lifetime_index)
        return self._stacked(), dict(info)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._append(observation),
            reward,
            terminated,
            truncated,
            dict(info),
        )

    def _stacked(self) -> np.ndarray:
        if len(self._frames) != self.stack_size:
            raise RuntimeError("reset must be called before requesting stacked frames")
        return np.concatenate(tuple(self._frames), axis=-1)
