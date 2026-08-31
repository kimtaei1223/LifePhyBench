"""A dependency-free diagnostic environment for lifetime semantics.

This module deliberately stays small.  Its purpose is to make cross-episode
physical persistence, causal action-to-wear coupling, and privileged oracle
information unambiguous before integrating a robotics simulator.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ToyWearConfig:
    """Parameters of the one-dimensional wear diagnostic."""

    horizon: int = 5
    target: float = 0.8
    success_tolerance: float = 0.04
    action_limit: float = 1.0
    minimum_gain: float = 0.15
    wear_rate: float = 0.012
    wear_exponent: float = 2.0
    degradation_mode: str = "endogenous_action"
    exogenous_dose_per_step: float = 0.25
    overload_threshold: float = 0.0
    recovery_per_episode: float = 0.0
    stochastic_shock_probability: float = 0.0
    stochastic_shock_size: float = 0.0
    process_noise_std: float = 0.0
    energy_cost: float = 0.01
    damage_cost: float = 0.25

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.minimum_gain <= 1.0:
            raise ValueError("minimum_gain must be in (0, 1]")
        if self.wear_rate < 0.0 or self.recovery_per_episode < 0.0:
            raise ValueError("wear and recovery rates must be non-negative")
        if self.wear_exponent <= 0.0:
            raise ValueError("wear_exponent must be positive")
        if self.degradation_mode not in {"endogenous_action", "exogenous_clock"}:
            raise ValueError(
                "degradation_mode must be 'endogenous_action' or 'exogenous_clock'"
            )
        if self.exogenous_dose_per_step < 0.0:
            raise ValueError("exogenous_dose_per_step must be non-negative")
        if not 0.0 <= self.stochastic_shock_probability <= 1.0:
            raise ValueError("shock probability must be in [0, 1]")


@dataclass
class LifetimeState:
    lifetime_id: int = -1
    episode_index: int = -1
    step_index: int = 0
    position: float = 0.0
    wear: float = 0.0
    cumulative_load: float = 0.0


@dataclass(frozen=True)
class StepResult:
    observation: tuple[float, float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float]


class ToyWearEnv:
    """A 1-D control task whose actuator gain degrades with cumulative load.

    The observable task state is `(position, target)`.  Wear is hidden from the
    agent but included in `info` for auditing and oracle baselines.  Calling
    :meth:`reset_episode` never resets wear; only :meth:`reset_lifetime` does.
    """

    def __init__(self, config: ToyWearConfig | None = None) -> None:
        self.config = config or ToyWearConfig()
        self.state = LifetimeState()
        self._rng = random.Random(0)
        self._episode_active = False

    @property
    def actuator_gain(self) -> float:
        span = 1.0 - self.config.minimum_gain
        return max(self.config.minimum_gain, 1.0 - span * self.state.wear)

    @property
    def observation(self) -> tuple[float, float]:
        return (self.state.position, self.config.target)

    def reset_lifetime(self, seed: int, lifetime_id: int = 0) -> tuple[float, float]:
        """Start a statistically independent lifetime and clear all wear."""

        self._rng = random.Random(seed)
        self.state = LifetimeState(lifetime_id=lifetime_id)
        self._episode_active = False
        return self.reset_episode()

    def reset_episode(self) -> tuple[float, float]:
        """Reset task state while preserving the cross-episode physical state."""

        if self.state.episode_index >= 0:
            self.state.wear = max(
                0.0, self.state.wear - self.config.recovery_per_episode
            )
        self.state.episode_index += 1
        self.state.step_index = 0
        self.state.position = 0.0
        self._episode_active = True
        return self.observation

    def _damage_increment(self, action: float) -> float:
        if self.config.degradation_mode == "endogenous_action":
            dose = max(0.0, abs(action) - self.config.overload_threshold)
        else:
            dose = self.config.exogenous_dose_per_step
        damage = self.config.wear_rate * dose**self.config.wear_exponent
        if self._rng.random() < self.config.stochastic_shock_probability:
            damage += self.config.stochastic_shock_size
        return damage

    def step(self, action: float) -> StepResult:
        if not self._episode_active:
            raise RuntimeError("call reset_lifetime() or reset_episode() before step()")

        clipped_action = max(
            -self.config.action_limit, min(self.config.action_limit, float(action))
        )
        gain_before_action = self.actuator_gain
        noise = self._rng.gauss(0.0, self.config.process_noise_std)
        self.state.position += gain_before_action * clipped_action + noise

        damage = self._damage_increment(clipped_action)
        self.state.wear = min(1.0, self.state.wear + damage)
        self.state.cumulative_load += abs(clipped_action)
        self.state.step_index += 1

        error = abs(self.config.target - self.state.position)
        success = error <= self.config.success_tolerance
        truncated = self.state.step_index >= self.config.horizon and not success
        terminated = success
        self._episode_active = not (terminated or truncated)

        reward = -error
        reward -= self.config.energy_cost * clipped_action**2
        reward -= self.config.damage_cost * damage
        if success:
            reward += 1.0

        return StepResult(
            observation=self.observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "success": float(success),
                "wear": self.state.wear,
                "damage_increment": damage,
                "actuator_gain": gain_before_action,
                "position_error": error,
                "clipped_action": clipped_action,
            },
        )

    def audit_state(self) -> dict[str, object]:
        """Return privileged state for tests, logging, and explicit oracles only."""

        return {
            "config": asdict(self.config),
            "state": asdict(self.state),
            "actuator_gain": self.actuator_gain,
        }

    def set_wear_for_diagnostic(self, wear: float) -> None:
        """Set hidden wear in controlled tests; never expose this to learned agents."""

        if not math.isfinite(wear) or not 0.0 <= wear <= 1.0:
            raise ValueError("wear must be finite and in [0, 1]")
        self.state.wear = wear
