"""Gym-compatible scheduler for finite persistent-physics lifetimes."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class LifetimeEpisodeScheduler(gym.Wrapper):
    """Start a full physical lifetime every fixed number of task episodes.

    SB3 invokes :meth:`reset` after a terminated/truncated task episode. This
    wrapper maps that call to the underlying ``reset_lifetime`` only at lifetime
    boundaries; otherwise it preserves physical state through the ordinary
    episode reset. It prevents a smoke learner from accidentally training on one
    infinite, ever-degrading robot instance.
    """

    def __init__(self, env: gym.Env, episodes_per_lifetime: int) -> None:
        super().__init__(env)
        if episodes_per_lifetime <= 0:
            raise ValueError("episodes_per_lifetime must be positive")
        if not callable(getattr(env, "reset_lifetime", None)):
            raise TypeError("wrapped environment must expose reset_lifetime")
        self.episodes_per_lifetime = episodes_per_lifetime
        self._episodes_completed = episodes_per_lifetime
        self._lifetime_index = -1
        self._base_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self._base_seed = seed
        new_lifetime = self._episodes_completed >= self.episodes_per_lifetime
        if new_lifetime:
            self._lifetime_index += 1
            self._episodes_completed = 0
            lifetime_seed = (
                None
                if self._base_seed is None
                else self._base_seed + 1_000_003 * self._lifetime_index
            )
            observation, info = self.env.reset_lifetime(
                seed=lifetime_seed,
                lifetime_id=self._lifetime_index,
                options=options,
            )
        else:
            observation, info = self.env.reset(seed=seed, options=options)
        result = dict(info)
        result.update(
            {
                "lifephy/scheduler_lifetime_index": self._lifetime_index,
                "lifephy/scheduler_episode_in_lifetime": self._episodes_completed,
            }
        )
        return observation, result

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._episodes_completed += 1
        result = dict(info)
        result.update(
            {
                "lifephy/scheduler_lifetime_index": self._lifetime_index,
                "lifephy/scheduler_episode_in_lifetime": self._episodes_completed,
                "lifephy/lifetime_boundary": bool(
                    (terminated or truncated)
                    and self._episodes_completed >= self.episodes_per_lifetime
                ),
            }
        )
        return observation, reward, terminated, truncated, result


class LifetimeStreamWrapper(gym.Wrapper):
    """Expose one full physical lifetime as one recurrent-RL episode.

    Inner task terminations reset only task state and are returned as ordinary
    nonterminal transitions. Consequently a recurrent policy receives an LSTM
    reset only after ``episodes_per_lifetime`` task episodes, while physical
    health remains continuous across the inner reset boundaries.
    """

    def __init__(self, env: gym.Env, episodes_per_lifetime: int) -> None:
        super().__init__(env)
        if episodes_per_lifetime <= 0:
            raise ValueError("episodes_per_lifetime must be positive")
        if not callable(getattr(env, "reset_lifetime", None)):
            raise TypeError("wrapped environment must expose reset_lifetime")
        self.episodes_per_lifetime = episodes_per_lifetime
        self._episodes_completed = 0
        self._lifetime_index = -1
        self._base_seed: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if seed is not None:
            self._base_seed = seed
        self._lifetime_index += 1
        self._episodes_completed = 0
        lifetime_seed = (
            None
            if self._base_seed is None
            else self._base_seed + 1_000_003 * self._lifetime_index
        )
        observation, info = self.env.reset_lifetime(
            seed=lifetime_seed,
            lifetime_id=self._lifetime_index,
            options=options,
        )
        result = dict(info)
        result.update(
            {
                "lifephy/stream_lifetime_index": self._lifetime_index,
                "lifephy/stream_episode_in_lifetime": self._episodes_completed,
            }
        )
        return observation, result

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            return observation, reward, terminated, truncated, dict(info)

        self._episodes_completed += 1
        result = dict(info)
        result.update(
            {
                "lifephy/inner_task_boundary": True,
                "lifephy/stream_lifetime_index": self._lifetime_index,
                "lifephy/stream_episode_in_lifetime": self._episodes_completed,
            }
        )
        if self._episodes_completed >= self.episodes_per_lifetime:
            result["lifephy/lifetime_boundary"] = True
            return observation, reward, False, True, result

        next_observation, next_info = self.env.reset()
        result["lifephy/lifetime_boundary"] = False
        result["lifephy/next_task_episode_index"] = next_info.get(
            "lifephy/episode_index"
        )
        return next_observation, reward, False, False, result
