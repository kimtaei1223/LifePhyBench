"""Stochastic hierarchical thermal diagnostic with indexed uncertainty.

This module is intentionally separate from :mod:`hierarchical_thermal`.  The
v10 environment is part of an already completed frozen experiment and must
remain byte-for-byte reproducible.  V11 keeps the same 28-dimensional policy
interface while replacing the four high-level summary coordinates with::

    [previous mode, noisy load sensor, normalized task index, previous trip]

Exact thermal state, task return, and action dose remain audit-only values in
``info``.  Sensor noise and physical shocks are generated from domain-separated
indices, rather than a sequential RNG stream.  Consequently an early trip or a
short physical episode cannot shift the uncertainty assigned to later tasks.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from lifephybench.envs.lifetime import LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.envs.task_boundary import TaskBoundaryObservation
from lifephybench.envs.thermal_commitment import (
    ThermalCommitmentConfig,
    ThermalModeCommitment,
)

_INDEX_PERSON = b"LifePhyV11"


def _indexed_seed(domain: str, *indices: int) -> int:
    """Return a stable, domain-separated seed for one pre-indexed draw."""

    digest = hashlib.blake2b(digest_size=16, person=_INDEX_PERSON)
    digest.update(domain.encode("ascii"))
    for index in indices:
        digest.update(int(index).to_bytes(16, byteorder="little", signed=True))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _indexed_uniform(domain: str, *indices: int) -> float:
    return float(np.random.default_rng(_indexed_seed(domain, *indices)).random())


def _indexed_normal(domain: str, *indices: int) -> float:
    return float(np.random.default_rng(_indexed_seed(domain, *indices)).normal())


@dataclass(frozen=True)
class HierarchicalThermalV11Config:
    """Frozen physical and uncertainty semantics for the v11 diagnostic."""

    condition: str
    low_level_model_path: str
    environment_id: str = "Pusher-v5"
    episode_steps: int = 100
    episodes_per_lifetime: int = 20
    canonical_task_seed: int = 811
    trip_load: float = 0.10
    low_power_scale: float = 0.40
    trip_penalty: float = 75.0
    high_power_bonus: float = 2.0
    thermal_heat_rate: float = 0.05
    thermal_episode_cooling: float = 0.10
    fixed_initial_load: float = 0.04
    stochastic_initial_load_low: float = 0.0
    stochastic_initial_load_high: float = 0.08
    sensor_noise_sd: float = 0.02
    shock_probability: float = 5.0e-4
    shock_size: float = 0.01
    low_level_device: str = "cpu"

    def __post_init__(self) -> None:
        if self.condition not in {"fixed", "stochastic"}:
            raise ValueError("condition must be fixed or stochastic")
        if not self.low_level_model_path:
            raise ValueError("low_level_model_path must be non-empty")
        if not self.environment_id:
            raise ValueError("environment_id must be non-empty")
        if self.episode_steps <= 0 or self.episodes_per_lifetime <= 0:
            raise ValueError("task and lifetime lengths must be positive")
        if self.canonical_task_seed < 0:
            raise ValueError("canonical_task_seed must be non-negative")

        finite_values = {
            "trip_load": self.trip_load,
            "low_power_scale": self.low_power_scale,
            "trip_penalty": self.trip_penalty,
            "high_power_bonus": self.high_power_bonus,
            "thermal_heat_rate": self.thermal_heat_rate,
            "thermal_episode_cooling": self.thermal_episode_cooling,
            "fixed_initial_load": self.fixed_initial_load,
            "stochastic_initial_load_low": self.stochastic_initial_load_low,
            "stochastic_initial_load_high": self.stochastic_initial_load_high,
            "sensor_noise_sd": self.sensor_noise_sd,
            "shock_probability": self.shock_probability,
            "shock_size": self.shock_size,
        }
        invalid = [name for name, value in finite_values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"configuration values must be finite: {invalid}")
        if not 0.0 < self.trip_load <= 1.0:
            raise ValueError("trip_load must be in (0, 1]")
        if not 0.0 < self.low_power_scale < 1.0:
            raise ValueError("low_power_scale must be in (0, 1)")
        if self.trip_penalty <= 0.0 or self.high_power_bonus <= 0.0:
            raise ValueError("trip penalty and high-power bonus must be positive")
        if self.thermal_heat_rate < 0.0:
            raise ValueError("thermal_heat_rate must be non-negative")
        if not 0.0 <= self.thermal_episode_cooling <= 1.0:
            raise ValueError("thermal_episode_cooling must be in [0, 1]")
        if not 0.0 <= self.fixed_initial_load <= 1.0:
            raise ValueError("fixed_initial_load must be in [0, 1]")
        if not (
            0.0
            <= self.stochastic_initial_load_low
            <= self.stochastic_initial_load_high
            <= 1.0
        ):
            raise ValueError("stochastic initial-load interval must lie in [0, 1]")
        if self.sensor_noise_sd < 0.0:
            raise ValueError("sensor_noise_sd must be non-negative")
        if not 0.0 <= self.shock_probability <= 1.0:
            raise ValueError("shock_probability must be in [0, 1]")
        if self.shock_size < 0.0:
            raise ValueError("shock_size must be non-negative")


class _IndexedThermalShock(gym.Wrapper):
    """Inject additive heat after indexed physical steps and before cooling."""

    def __init__(
        self,
        env: PusherActuatorWear,
        *,
        enabled: bool,
        probability: float,
        size: float,
    ) -> None:
        super().__init__(env)
        self._health = env
        self.enabled = enabled
        self.probability = probability
        self.size = size
        self.schedule_seed = 0
        self.lifetime_index = -1
        self.task_index = 0
        self.physical_step_index = 0
        self._task_shock_steps: dict[int, list[int]] = {}
        self._task_shock_amounts: dict[int, list[float]] = {}

    @property
    def thermal_load(self) -> float:
        return float(self._health.thermal_load)

    def reset_lifetime(
        self,
        *,
        seed: int | None = None,
        lifetime_id: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.schedule_seed = 0 if seed is None else int(seed)
        self.lifetime_index = (
            self.lifetime_index + 1 if lifetime_id is None else int(lifetime_id)
        )
        self.task_index = 0
        self.physical_step_index = 0
        self._task_shock_steps = {0: []}
        self._task_shock_amounts = {0: []}
        observation, info = self._health.reset_lifetime(
            seed=seed, lifetime_id=lifetime_id, options=options
        )
        return observation, self._audit_reset_info(info)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self._health.reset(seed=seed, options=options)
        self.task_index += 1
        self.physical_step_index = 0
        self._task_shock_steps.setdefault(self.task_index, [])
        self._task_shock_amounts.setdefault(self.task_index, [])
        return observation, self._audit_reset_info(info)

    def _audit_reset_info(self, info: dict[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result.update(
            {
                "lifephy/v11_uncertainty_schedule_seed": self.schedule_seed,
                "lifephy/v11_uncertainty_lifetime_index": self.lifetime_index,
                "lifephy/v11_uncertainty_task_index": self.task_index,
            }
        )
        return result

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_index = self.physical_step_index
        observation, reward, terminated, truncated, info = self._health.step(action)
        draw = _indexed_uniform(
            "physical-shock",
            self.schedule_seed,
            self.lifetime_index,
            self.task_index,
            step_index,
        )
        shocked = bool(self.enabled and draw < self.probability)
        applied_shock = 0.0
        if shocked:
            previous = float(self._health.thermal_load)
            self._health.set_thermal_load_for_diagnostic(
                min(1.0, previous + self.size)
            )
            applied_shock = float(self._health.thermal_load) - previous
            self._task_shock_steps.setdefault(self.task_index, []).append(step_index)
            self._task_shock_amounts.setdefault(self.task_index, []).append(
                applied_shock
            )
        self.physical_step_index += 1
        result = dict(info)
        result.update(
            {
                "lifephy/thermal_load": float(self._health.thermal_load),
                "lifephy/actuator_efficiency": float(self._health.efficiency),
                "lifephy/v11_physical_step_index": step_index,
                "lifephy/v11_step_shock": shocked,
                "lifephy/v11_step_shock_amount": applied_shock,
                "lifephy/v11_shock_index_draw": draw,
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), result

    def task_shock_steps(self, task_index: int) -> tuple[int, ...]:
        return tuple(self._task_shock_steps.get(int(task_index), ()))

    def task_shock_total(self, task_index: int) -> float:
        return float(sum(self._task_shock_amounts.get(int(task_index), ())))


class HierarchicalThermalV11Env(gym.Env):
    """One stochastic thermal mode decision per frozen-controller MuJoCo task."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        config: HierarchicalThermalV11Config,
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
                thermal_episode_cooling=config.thermal_episode_cooling,
                thermal_degradation_mode="endogenous_action",
                thermal_exogenous_dose_per_step=0.0,
                # Core stochastic shocks stay disabled. V11 applies indexed
                # shocks in _IndexedThermalShock, avoiding sequential RNG drift.
                stochastic_shock_probability=0.0,
                stochastic_shock_size=0.0,
                canonical_task_seed=config.canonical_task_seed,
            ),
            environment_id=config.environment_id,
            max_episode_steps=config.episode_steps,
        )
        indexed_physics = _IndexedThermalShock(
            base,
            enabled=config.condition == "stochastic",
            probability=config.shock_probability,
            size=config.shock_size,
        )
        commitment = ThermalModeCommitment(
            indexed_physics,
            ThermalCommitmentConfig(
                trip_load=config.trip_load,
                low_power_scale=config.low_power_scale,
                trip_penalty=config.trip_penalty,
                high_power_throughput_bonus=config.high_power_bonus,
                control_cost_basis="requested_action",
            ),
        )
        self.physical_environment = TaskBoundaryObservation(
            LifetimeStreamWrapper(commitment, config.episodes_per_lifetime)
        )
        self._health = base
        self._indexed_physics = indexed_physics
        self.action_space = gym.spaces.Discrete(2)
        full_space = self.physical_environment.observation_space
        if not isinstance(full_space, gym.spaces.Box):
            raise TypeError("hierarchical environment requires Box observations")
        raw_low = np.asarray(full_space.low[:-3])
        raw_high = np.asarray(full_space.high[:-3])
        summary_low = np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=full_space.dtype)
        summary_high = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=full_space.dtype)
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
            raise ValueError(
                "low-level action space does not match " f"{config.environment_id}"
            )

        self._physical_observation: np.ndarray | None = None
        self._summary = np.zeros(4, dtype=full_space.dtype)
        self._task_index = 0
        self._initial_thermal_load = 0.0
        self._current_sensor = 0.0
        self._current_sensor_noise = 0.0

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

    def _normalized_task_index(self, task_index: int) -> float:
        denominator = self.config.episodes_per_lifetime - 1
        return 0.0 if denominator <= 0 else float(task_index / denominator)

    def _sample_initial_load(self) -> float:
        if self.config.condition == "fixed":
            return float(self.config.fixed_initial_load)
        unit = _indexed_uniform(
            "initial-load",
            self._indexed_physics.schedule_seed,
            self._indexed_physics.lifetime_index,
        )
        low = self.config.stochastic_initial_load_low
        high = self.config.stochastic_initial_load_high
        return float(low + unit * (high - low))

    def _sample_sensor(self, task_index: int) -> tuple[float, float]:
        raw_noise = self.config.sensor_noise_sd * _indexed_normal(
            "load-sensor",
            self._indexed_physics.schedule_seed,
            self._indexed_physics.lifetime_index,
            task_index,
        )
        exact_load = float(self._health.thermal_load)
        sensor = float(np.clip(exact_load + raw_noise, 0.0, 1.0))
        return sensor, float(raw_noise)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.physical_environment.reset(seed=seed, options=options)
        self._physical_observation = np.asarray(observation)
        self._task_index = 0
        self._initial_thermal_load = self._sample_initial_load()
        self._health.set_thermal_load_for_diagnostic(self._initial_thermal_load)
        self._current_sensor, self._current_sensor_noise = self._sample_sensor(0)
        self._summary = np.asarray(
            [0.0, self._current_sensor, 0.0, 0.0],
            dtype=self.observation_space.dtype,
        )
        result = dict(info)
        result.update(
            {
                "lifephy/hierarchical_lifetime_start": True,
                "lifephy/v11_condition": self.config.condition,
                "lifephy/v11_initial_thermal_load": self._initial_thermal_load,
                "lifephy/v11_lifetime_initial_thermal_load": (
                    self._initial_thermal_load
                ),
                "lifephy/v11_task_index": 0,
                "lifephy/v11_normalized_task_index": 0.0,
                "lifephy/v11_load_sensor": self._current_sensor,
                "lifephy/v11_sensor_load": self._current_sensor,
                "lifephy/v11_load_sensor_raw_noise": self._current_sensor_noise,
                "lifephy/v11_sensor_noise_sd": self.config.sensor_noise_sd,
                "lifephy/v11_shock_probability": (
                    self.config.shock_probability
                    if self.config.condition == "stochastic"
                    else 0.0
                ),
                "lifephy/v11_shock_size": (
                    self.config.shock_size
                    if self.config.condition == "stochastic"
                    else 0.0
                ),
                "lifephy/v11_exact_thermal_load": float(self._health.thermal_load),
                "lifephy/v11_thermal_load": float(self._health.thermal_load),
                "lifephy/thermal_load": float(self._health.thermal_load),
                "lifephy/actuator_efficiency": float(self._health.efficiency),
            }
        )
        return self._high_level_observation(self._physical_observation), result

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._physical_observation is None:
            raise RuntimeError("reset must be called before step")
        mode = int(np.asarray(action).item())
        if not self.action_space.contains(mode):
            raise ValueError(f"invalid discrete mode: {mode}")
        selection_task_index = self._task_index
        selection_sensor = self._current_sensor
        selection_sensor_noise = self._current_sensor_noise
        high_power = mode == 1
        selection_load = float(self._health.thermal_load)
        low_state = None
        low_episode_start = np.asarray([True])
        total_reward = 0.0
        physical_steps = 0
        tripped = False
        final_info: dict[str, Any] = {}
        while True:
            low_observation = self._low_level_observation(self._physical_observation)
            low_action, low_state = self.low_level_model.predict(
                low_observation,
                state=low_state,
                episode_start=low_episode_start,
                deterministic=True,
            )
            combined_action = np.concatenate(
                [np.asarray([1.0 if high_power else -1.0]), np.asarray(low_action)]
            )
            observation, reward, terminated, truncated, info = (
                self.physical_environment.step(combined_action)
            )
            self._physical_observation = np.asarray(observation)
            physical_steps += 1
            total_reward += float(reward)
            tripped = tripped or bool(info.get("lifephy/thermal_trip", False))
            final_info = dict(info)
            if bool(info.get("lifephy/inner_task_boundary", terminated or truncated)):
                break
            low_episode_start = np.asarray([False])

        lifetime_boundary = bool(final_info.get("lifephy/lifetime_boundary", False))
        next_task_index = (
            selection_task_index
            if lifetime_boundary
            else min(selection_task_index + 1, self.config.episodes_per_lifetime - 1)
        )
        self._task_index = next_task_index
        next_sensor, next_sensor_noise = self._sample_sensor(next_task_index)
        self._current_sensor = next_sensor
        self._current_sensor_noise = next_sensor_noise
        self._summary = np.asarray(
            [
                1.0 if high_power else -1.0,
                next_sensor,
                self._normalized_task_index(next_task_index),
                float(tripped),
            ],
            dtype=self.observation_space.dtype,
        )
        shock_steps = self._indexed_physics.task_shock_steps(selection_task_index)
        result = dict(final_info)
        result.update(
            {
                "lifephy/inner_task_boundary": True,
                "lifephy/hierarchical_physical_steps": physical_steps,
                "lifephy/hierarchical_previous_summary": self._summary.copy(),
                "lifephy/thermal_mode_selected_now": True,
                "lifephy/thermal_mode": "high" if high_power else "low",
                "lifephy/thermal_load_at_mode_selection": selection_load,
                "lifephy/thermal_trip": tripped,
                "lifephy/thermal_load": float(self._health.thermal_load),
                "lifephy/actuator_efficiency": float(self._health.efficiency),
                "lifephy/hierarchical_initial_thermal_load": self._initial_thermal_load,
                "lifephy/hierarchical_physical_reward": total_reward,
                "lifephy/lifetime_boundary": lifetime_boundary,
                "lifephy/v11_condition": self.config.condition,
                "lifephy/v11_task_index": selection_task_index,
                "lifephy/v11_task_index_at_selection": selection_task_index,
                "lifephy/v11_normalized_task_index": self._normalized_task_index(
                    selection_task_index
                ),
                "lifephy/v11_load_sensor_at_mode_selection": selection_sensor,
                "lifephy/v11_sensor_load": selection_sensor,
                "lifephy/v11_lifetime_initial_thermal_load": (
                    self._initial_thermal_load
                ),
                "lifephy/v11_load_sensor_raw_noise_at_mode_selection": (
                    selection_sensor_noise
                ),
                "lifephy/v11_next_task_index": next_task_index,
                "lifephy/v11_next_load_sensor": next_sensor,
                "lifephy/v11_next_load_sensor_raw_noise": next_sensor_noise,
                "lifephy/v11_task_shock_count": len(shock_steps),
                "lifephy/v11_task_shock_step_indices": list(shock_steps),
                "lifephy/v11_task_shock_total": (
                    self._indexed_physics.task_shock_total(selection_task_index)
                ),
                "lifephy/v11_exact_thermal_load": float(self._health.thermal_load),
                "lifephy/v11_thermal_load": float(self._health.thermal_load),
                "lifephy/v11_uncertainty_schedule_seed": (
                    self._indexed_physics.schedule_seed
                ),
                "lifephy/v11_uncertainty_lifetime_index": (
                    self._indexed_physics.lifetime_index
                ),
            }
        )
        return (
            self._high_level_observation(self._physical_observation),
            total_reward,
            False,
            lifetime_boundary,
            result,
        )

    def close(self) -> None:
        self.physical_environment.close()
