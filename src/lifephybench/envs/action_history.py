"""Non-privileged action-history observation for recurrent health inference."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class PreviousAppliedActionObservation(gym.Wrapper):
    """Append the preceding applied action and zero it at task boundaries."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("action history requires a Box action space")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("action history requires a Box observation space")
        observation = env.observation_space
        action = env.action_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    np.asarray(observation.low),
                    np.asarray(action.low, dtype=observation.dtype),
                ]
            ),
            high=np.concatenate(
                [
                    np.asarray(observation.high),
                    np.asarray(action.high, dtype=observation.dtype),
                ]
            ),
            dtype=observation.dtype,
        )

    @property
    def thermal_load(self) -> float:
        return float(getattr(self.env, "thermal_load"))

    @property
    def lifetime_id(self) -> int:
        return int(getattr(self.env, "lifetime_id"))

    def _observation(self, observation: Any, previous_action: Any) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation, dtype=self.observation_space.dtype),
                np.asarray(previous_action, dtype=self.observation_space.dtype),
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        zeros = np.zeros(self.action_space.shape, dtype=self.observation_space.dtype)
        return self._observation(observation, zeros), dict(info)

    def reset_lifetime(
        self,
        *,
        seed: int | None = None,
        lifetime_id: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_lifetime = getattr(self.env, "reset_lifetime", None)
        if not callable(reset_lifetime):
            raise TypeError("wrapped environment must expose reset_lifetime")
        observation, info = reset_lifetime(
            seed=seed, lifetime_id=lifetime_id, options=options
        )
        zeros = np.zeros(self.action_space.shape, dtype=self.observation_space.dtype)
        return self._observation(observation, zeros), dict(info)

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=self.action_space.dtype)
        clipped = np.clip(action_array, self.action_space.low, self.action_space.high)
        observation, reward, terminated, truncated, info = self.env.step(clipped)
        result = dict(info)
        result["lifephy/previous_applied_action_observed"] = True
        return (
            self._observation(observation, clipped),
            float(reward),
            bool(terminated),
            bool(truncated),
            result,
        )
