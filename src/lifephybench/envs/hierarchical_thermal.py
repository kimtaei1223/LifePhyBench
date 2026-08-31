"""Hierarchical task-level thermal mode control with a frozen low-level policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lifephybench.envs.lifetime import LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.envs.task_boundary import TaskBoundaryObservation
from lifephybench.envs.thermal_commitment import (
    ThermalCommitmentConfig,
    ThermalModeCommitment,
)


@dataclass(frozen=True)
class HierarchicalThermalConfig:
    low_level_model_path: str
    degradation_mode: str
    episode_steps: int = 100
    episodes_per_lifetime: int = 20
    canonical_task_seed: int = 811
    trip_load: float = 0.10
    low_power_scale: float = 0.40
    trip_penalty: float = 75.0
    high_power_bonus: float = 2.0
    thermal_heat_rate: float = 0.1
    curriculum_start_trip_load: float | None = None
    curriculum_lifetimes: int = 0
    training_teacher_safe_high_load: float | None = None
    training_teacher_shaping: float = 0.0
    summary_mode: str = "full"
    low_level_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.degradation_mode not in {"endogenous_action", "exogenous_clock"}:
            raise ValueError("degradation_mode must be endogenous_action or exogenous_clock")
        if self.episode_steps <= 0 or self.episodes_per_lifetime <= 0:
            raise ValueError("task and lifetime lengths must be positive")
        if self.training_teacher_shaping < 0.0:
            raise ValueError("training_teacher_shaping must be non-negative")
        if self.training_teacher_shaping > 0.0 and (
            self.training_teacher_safe_high_load is None
            or not 0.0 <= self.training_teacher_safe_high_load <= self.trip_load
        ):
            raise ValueError(
                "positive teacher shaping requires safe high load in [0, trip_load]"
            )
        if self.summary_mode not in {"full", "mode_trip"}:
            raise ValueError("summary_mode must be full or mode_trip")


class HierarchicalThermalModeEnv(gym.Env):
    """Expose one discrete mode decision per internally controlled Pusher task.

    The frozen low-level policy executes physical actions for one complete task.
    The high-level observation contains the canonical boundary state and a
    non-privileged summary of the preceding task: selected mode, applied-action
    thermal increment, normalized return, and observed trip. The summary is
    zero at each lifetime start. No wear/thermal state is exposed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: HierarchicalThermalConfig,
        low_level_model: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        base = PusherActuatorWear.make(
            ActuatorWearConfig(
                wear_rate=0.0,
                thermal_enabled=True,
                thermal_heat_rate=config.thermal_heat_rate,
                thermal_cooling_rate=0.0,
                thermal_episode_cooling=0.0,
                thermal_degradation_mode=config.degradation_mode,
                thermal_exogenous_dose_per_step=0.0,
                canonical_task_seed=config.canonical_task_seed,
            ),
            max_episode_steps=config.episode_steps,
        )
        commitment = ThermalModeCommitment(
            base,
            ThermalCommitmentConfig(
                trip_load=config.trip_load,
                low_power_scale=config.low_power_scale,
                trip_penalty=config.trip_penalty,
                high_power_throughput_bonus=config.high_power_bonus,
                control_cost_basis="requested_action",
                curriculum_start_trip_load=config.curriculum_start_trip_load,
                curriculum_lifetimes=config.curriculum_lifetimes,
            ),
        )
        self.physical_environment = TaskBoundaryObservation(
            LifetimeStreamWrapper(commitment, config.episodes_per_lifetime)
        )
        self._health = base
        self.action_space = gym.spaces.Discrete(2)
        full_space = self.physical_environment.observation_space
        if not isinstance(full_space, gym.spaces.Box):
            raise TypeError("hierarchical environment requires Box observations")
        # Strip [mode_selected, mode, task_boundary] and append four summary
        # coordinates plus a fresh task-boundary marker.
        raw_low = np.asarray(full_space.low[:-3])
        raw_high = np.asarray(full_space.high[:-3])
        summary_low = np.asarray([-1.0, 0.0, -10.0, 0.0], dtype=full_space.dtype)
        summary_high = np.asarray([1.0, 1.0, 10.0, 1.0], dtype=full_space.dtype)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([raw_low, summary_low, [0.0]]),
            high=np.concatenate([raw_high, summary_high, [1.0]]),
            dtype=full_space.dtype,
        )
        if low_level_model is None:
            from sb3_contrib import RecurrentPPO

            model_path = Path(config.low_level_model_path)
            if not (model_path.with_suffix(".zip").exists() or model_path.exists()):
                raise FileNotFoundError(f"low-level model not found: {model_path}")
            low_level_model = RecurrentPPO.load(
                str(model_path), device=config.low_level_device
            )
        self.low_level_model = low_level_model
        expected_low_observation = len(raw_low) + 1
        if tuple(self.low_level_model.observation_space.shape) != (
            expected_low_observation,
        ):
            raise ValueError(
                "low-level observation mismatch: expected "
                f"{expected_low_observation}, got "
                f"{self.low_level_model.observation_space.shape}"
            )
        if tuple(self.low_level_model.action_space.shape) != tuple(base.action_space.shape):
            raise ValueError("low-level action space does not match Pusher")
        self._physical_observation: np.ndarray | None = None
        self._summary = np.zeros(4, dtype=full_space.dtype)

    def _low_level_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation[:-3], observation[-1:]])

    def _high_level_observation(self, observation: np.ndarray) -> np.ndarray:
        marker = np.asarray([1.0], dtype=self.observation_space.dtype)
        return np.concatenate(
            [
                np.asarray(observation[:-3], dtype=self.observation_space.dtype),
                self._summary,
                marker,
            ]
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.physical_environment.reset(
            seed=seed, options=options
        )
        self._physical_observation = np.asarray(observation)
        self._summary = np.zeros(4, dtype=self.observation_space.dtype)
        result = dict(info)
        result["lifephy/hierarchical_lifetime_start"] = True
        return self._high_level_observation(self._physical_observation), result

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._physical_observation is None:
            raise RuntimeError("reset must be called before step")
        mode = int(np.asarray(action).item())
        if not self.action_space.contains(mode):
            raise ValueError(f"invalid discrete mode: {mode}")
        high_power = mode == 1
        selection_load = float(self._health.thermal_load)
        initial_load = selection_load
        low_state = None
        low_episode_start = np.asarray([True])
        total_reward = 0.0
        total_action_thermal_increment = 0.0
        steps = 0
        tripped = False
        final_info: dict[str, Any] = {}
        active_trip_load = self.config.trip_load
        while True:
            low_observation = self._low_level_observation(
                self._physical_observation
            )
            low_action, low_state = self.low_level_model.predict(
                low_observation,
                state=low_state,
                episode_start=low_episode_start,
                deterministic=True,
            )
            combined_action = np.concatenate(
                [
                    np.asarray([1.0 if high_power else -1.0]),
                    np.asarray(low_action),
                ]
            )
            observation, reward, terminated, truncated, info = (
                self.physical_environment.step(combined_action)
            )
            self._physical_observation = np.asarray(observation)
            steps += 1
            total_reward += float(reward)
            total_action_thermal_increment += (
                self.config.thermal_heat_rate
                * float(info.get("lifephy/thermal_dose", 0.0))
            )
            tripped = tripped or bool(info.get("lifephy/thermal_trip", False))
            active_trip_load = float(
                info.get("lifephy/thermal_trip_load", active_trip_load)
            )
            final_info = dict(info)
            if bool(info.get("lifephy/inner_task_boundary", terminated or truncated)):
                break
            low_episode_start = np.asarray([False])

        summary_increment = (
            np.clip(total_action_thermal_increment, 0.0, 1.0)
            if self.config.summary_mode == "full"
            else 0.0
        )
        summary_reward = (
            np.clip(total_reward / 100.0, -10.0, 10.0)
            if self.config.summary_mode == "full"
            else 0.0
        )
        self._summary = np.asarray(
            [
                1.0 if high_power else -1.0,
                summary_increment,
                summary_reward,
                float(tripped),
            ],
            dtype=self.observation_space.dtype,
        )
        teacher_reward = 0.0
        teacher_target: str | None = None
        if self.config.training_teacher_shaping > 0.0:
            teacher_high = bool(
                selection_load < self.config.training_teacher_safe_high_load
            )
            teacher_target = "high" if teacher_high else "low"
            teacher_reward = self.config.training_teacher_shaping * (
                1.0 if high_power == teacher_high else -1.0
            )
        lifetime_boundary = bool(final_info.get("lifephy/lifetime_boundary", False))
        result = dict(final_info)
        result.update(
            {
                "lifephy/inner_task_boundary": True,
                "lifephy/hierarchical_physical_steps": steps,
                "lifephy/hierarchical_previous_summary": self._summary.copy(),
                "lifephy/thermal_mode_selected_now": True,
                "lifephy/thermal_mode": "high" if high_power else "low",
                "lifephy/thermal_load_at_mode_selection": selection_load,
                "lifephy/thermal_trip_load": active_trip_load,
                "lifephy/thermal_trip": tripped,
                "lifephy/thermal_load": float(self._health.thermal_load),
                "lifephy/actuator_efficiency": float(self._health.efficiency),
                "lifephy/hierarchical_initial_thermal_load": initial_load,
                "lifephy/hierarchical_physical_reward": total_reward,
                "lifephy/training_teacher_reward": teacher_reward,
                "lifephy/training_teacher_target": teacher_target,
                "lifephy/lifetime_boundary": lifetime_boundary,
            }
        )
        return (
            self._high_level_observation(self._physical_observation),
            total_reward + teacher_reward,
            False,
            lifetime_boundary,
            result,
        )

    def close(self) -> None:
        self.physical_environment.close()
