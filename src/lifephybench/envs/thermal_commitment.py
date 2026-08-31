"""Health-contingent actuator-mode commitment for thermal-control tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ThermalCommitmentConfig:
    """Parameters for the first-action thermal protection decision."""

    trip_load: float = 0.10
    low_power_scale: float = 0.40
    trip_penalty: float = 75.0
    high_power_throughput_bonus: float = 2.0
    control_cost_basis: str = "requested_action"
    curriculum_start_trip_load: float | None = None
    curriculum_lifetimes: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.trip_load <= 1.0:
            raise ValueError("trip_load must be in (0, 1]")
        if not 0.0 < self.low_power_scale < 1.0:
            raise ValueError("low_power_scale must be in (0, 1)")
        if self.trip_penalty <= 0.0:
            raise ValueError("trip_penalty must be positive")
        if self.high_power_throughput_bonus <= 0.0:
            raise ValueError("high_power_throughput_bonus must be positive")
        if self.control_cost_basis not in {"applied_action", "requested_action"}:
            raise ValueError(
                "control_cost_basis must be applied_action or requested_action"
            )
        if self.curriculum_lifetimes < 0:
            raise ValueError("curriculum_lifetimes must be non-negative")
        if self.curriculum_lifetimes == 0:
            if self.curriculum_start_trip_load is not None:
                raise ValueError(
                    "curriculum_start_trip_load requires positive curriculum_lifetimes"
                )
        elif self.curriculum_start_trip_load is None:
            raise ValueError(
                "positive curriculum_lifetimes requires curriculum_start_trip_load"
            )
        elif not self.trip_load <= self.curriculum_start_trip_load <= 1.0:
            raise ValueError(
                "curriculum_start_trip_load must be in [trip_load, 1]"
            )


class ThermalModeCommitment(gym.Wrapper):
    """Require a high/low-power commitment on the first action of each task.

    The first action coordinate selects high power (non-negative) or low power
    (negative). The remaining coordinates are the wrapped robot action. High
    power preserves the nominal command. Low power scales the command and is
    always safe. Selecting high power at or above the hidden thermal trip load
    terminates the task with a protection-trip penalty.

    Two public observation coordinates report whether a mode has been selected
    and which mode is active. At every task boundary both are zero, so hidden
    thermal health remains the only persistent physical difference.
    """

    def __init__(
        self,
        env: gym.Env,
        config: ThermalCommitmentConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config or ThermalCommitmentConfig()
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("ThermalModeCommitment requires a Box action space")
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("ThermalModeCommitment requires a Box observation space")
        if not hasattr(env, "thermal_load"):
            raise TypeError("wrapped environment must expose thermal_load")

        base_action = env.action_space
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [np.asarray([-1.0], dtype=base_action.dtype), base_action.low]
            ),
            high=np.concatenate(
                [np.asarray([1.0], dtype=base_action.dtype), base_action.high]
            ),
            dtype=base_action.dtype,
        )
        base_observation = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    np.asarray(base_observation.low),
                    np.asarray([0.0, -1.0], dtype=base_observation.dtype),
                ]
            ),
            high=np.concatenate(
                [
                    np.asarray(base_observation.high),
                    np.asarray([1.0, 1.0], dtype=base_observation.dtype),
                ]
            ),
            dtype=base_observation.dtype,
        )
        self._mode_selected = False
        self._high_power = False
        self._last_observation: Any | None = None
        self._last_info: dict[str, Any] = {}

    @property
    def high_power(self) -> bool | None:
        return self._high_power if self._mode_selected else None

    @property
    def active_trip_load(self) -> float:
        """Return the training threshold for the current physical lifetime."""

        start = self.config.curriculum_start_trip_load
        duration = self.config.curriculum_lifetimes
        if start is None or duration == 0:
            return self.config.trip_load
        lifetime_id = max(0, int(getattr(self.env, "lifetime_id", 0)))
        progress = min(1.0, lifetime_id / duration)
        return float(start + progress * (self.config.trip_load - start))

    def _observation(self, observation: Any) -> np.ndarray:
        mode = 1.0 if self._high_power else -1.0 if self._mode_selected else 0.0
        marker = np.asarray(
            [float(self._mode_selected), mode], dtype=self.observation_space.dtype
        )
        return np.concatenate(
            [np.asarray(observation, dtype=self.observation_space.dtype), marker]
        )

    def _clear_mode(self) -> None:
        self._mode_selected = False
        self._high_power = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._clear_mode()
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_observation = observation
        self._last_info = dict(info)
        return self._observation(observation), dict(info)

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
        self._clear_mode()
        observation, info = reset_lifetime(
            seed=seed, lifetime_id=lifetime_id, options=options
        )
        self._last_observation = observation
        self._last_info = dict(info)
        return self._observation(observation), dict(info)

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=self.action_space.dtype)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"expected action shape {self.action_space.shape}, got {action_array.shape}"
            )
        clipped = np.clip(action_array, self.action_space.low, self.action_space.high)
        selected_now = not self._mode_selected
        if selected_now:
            self._mode_selected = True
            self._high_power = bool(clipped[0] >= 0.0)

        thermal_load = float(self.env.thermal_load)
        active_trip_load = self.active_trip_load
        if self._high_power and thermal_load >= active_trip_load:
            if self._last_observation is None:
                raise RuntimeError("reset must be called before step")
            info = dict(self._last_info)
            info.update(
                {
                    "lifephy/thermal_load": thermal_load,
                    "lifephy/thermal_mode": "high",
                    "lifephy/thermal_mode_selected_now": selected_now,
                    "lifephy/thermal_load_at_mode_selection": thermal_load,
                    "lifephy/thermal_trip": True,
                    "lifephy/thermal_trip_load": active_trip_load,
                    "lifephy/thermal_target_trip_load": self.config.trip_load,
                    "lifephy/applied_power_scale": 0.0,
                    "lifephy/control_cost_basis": self.config.control_cost_basis,
                    "lifephy/applied_reward_ctrl": 0.0,
                    "lifephy/requested_reward_ctrl": 0.0,
                    "lifephy/control_cost_correction": 0.0,
                }
            )
            return (
                self._observation(self._last_observation),
                -self.config.trip_penalty,
                False,
                True,
                info,
            )

        power_scale = 1.0 if self._high_power else self.config.low_power_scale
        base_action = clipped[1:] * power_scale
        observation, reward, terminated, truncated, info = self.env.step(base_action)
        applied_reward_ctrl = float(info.get("reward_ctrl", 0.0))
        requested_reward_ctrl = applied_reward_ctrl
        control_cost_correction = 0.0
        if self.config.control_cost_basis == "requested_action":
            if "reward_ctrl" not in info:
                raise RuntimeError(
                    "requested-action control-cost correction requires reward_ctrl info"
                )
            requested_reward_ctrl = applied_reward_ctrl / power_scale**2
            control_cost_correction = requested_reward_ctrl - applied_reward_ctrl
        throughput_bonus = (
            self.config.high_power_throughput_bonus
            if selected_now and self._high_power
            else 0.0
        )
        reward = float(reward) + control_cost_correction + throughput_bonus
        self._last_observation = observation
        self._last_info = dict(info)
        result = dict(info)
        result.update(
            {
                "lifephy/thermal_mode": "high" if self._high_power else "low",
                "lifephy/thermal_mode_selected_now": selected_now,
                "lifephy/thermal_trip": False,
                "lifephy/thermal_trip_load": active_trip_load,
                "lifephy/thermal_target_trip_load": self.config.trip_load,
                "lifephy/applied_power_scale": power_scale,
                "lifephy/high_power_throughput_bonus": throughput_bonus,
                "lifephy/control_cost_basis": self.config.control_cost_basis,
                "lifephy/applied_reward_ctrl": applied_reward_ctrl,
                "lifephy/requested_reward_ctrl": requested_reward_ctrl,
                "lifephy/control_cost_correction": control_cost_correction,
            }
        )
        result["reward_ctrl"] = requested_reward_ctrl
        if selected_now:
            result["lifephy/thermal_load_at_mode_selection"] = thermal_load
        return (
            self._observation(observation),
            reward,
            bool(terminated),
            bool(truncated),
            result,
        )
