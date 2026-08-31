"""Small, transparent lifetime-planning oracle for the toy diagnostic tier.

The planner receives privileged hidden wear and solves an approximation of the
remaining lifetime by dynamic programming over discretized position and wear.
It is deliberately labelled an oracle *baseline*, not a proposed method and
not an exact continuous-control optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

from .policies import Controller, Observation
from .toy_env import ToyWearConfig


@dataclass(frozen=True)
class PlannerConfig:
    """Resolution and horizon controls for the diagnostic planning oracle."""

    episodes_per_lifetime: int
    action_bins: int = 41
    position_resolution: float = 0.02
    wear_resolution: float = 0.005

    def __post_init__(self) -> None:
        if self.episodes_per_lifetime <= 0:
            raise ValueError("episodes_per_lifetime must be positive")
        if self.action_bins < 3 or self.action_bins % 2 == 0:
            raise ValueError("action_bins must be an odd integer of at least 3")
        if self.position_resolution <= 0.0 or self.wear_resolution <= 0.0:
            raise ValueError("state resolutions must be positive")


class LifetimeDPPlanner:
    """Finite-lifetime dynamic program for deterministic :class:`ToyWearEnv`.

    State is quantized only for memoization. The policy can therefore be used as
    a reproducible planning anchor while remaining computationally small enough
    for a CPU-only semantic audit.
    """

    def __init__(self, env_config: ToyWearConfig, config: PlannerConfig) -> None:
        if env_config.process_noise_std != 0.0:
            raise ValueError("planning oracle requires zero process noise")
        if env_config.stochastic_shock_probability != 0.0:
            raise ValueError("planning oracle requires zero shock probability")
        self.env_config = env_config
        self.config = config
        self.actions = tuple(
            -env_config.action_limit
            + 2.0 * env_config.action_limit * index / (config.action_bins - 1)
            for index in range(config.action_bins)
        )
        self._cache: dict[tuple[int, int, int, int], tuple[float, float]] = {}

    def _quantize(self, value: float, resolution: float) -> int:
        return round(value / resolution)

    def _state_key(
        self, episode_index: int, step_index: int, position: float, wear: float
    ) -> tuple[int, int, int, int]:
        return (
            episode_index,
            step_index,
            self._quantize(position, self.config.position_resolution),
            self._quantize(wear, self.config.wear_resolution),
        )

    def _decode(self, key: tuple[int, int, int, int]) -> tuple[float, float]:
        return (
            key[2] * self.config.position_resolution,
            min(1.0, max(0.0, key[3] * self.config.wear_resolution)),
        )

    def _transition(
        self, position: float, wear: float, action: float
    ) -> tuple[float, float, float, bool]:
        cfg = self.env_config
        clipped = max(-cfg.action_limit, min(cfg.action_limit, action))
        gain = max(cfg.minimum_gain, 1.0 - (1.0 - cfg.minimum_gain) * wear)
        next_position = position + gain * clipped
        if cfg.degradation_mode == "endogenous_action":
            dose = max(0.0, abs(clipped) - cfg.overload_threshold)
        else:
            dose = cfg.exogenous_dose_per_step
        damage = cfg.wear_rate * dose**cfg.wear_exponent
        next_wear = min(1.0, wear + damage)
        error = abs(cfg.target - next_position)
        success = error <= cfg.success_tolerance
        reward = -error - cfg.energy_cost * clipped**2 - cfg.damage_cost * damage
        if success:
            reward += 1.0
        return next_position, next_wear, reward, success

    def _value(
        self, episode_index: int, step_index: int, position: float, wear: float
    ) -> tuple[float, float]:
        key = self._state_key(episode_index, step_index, position, wear)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        canonical_position, canonical_wear = self._decode(key)
        best_value = float("-inf")
        best_action = 0.0
        for action in self.actions:
            next_position, next_wear, reward, success = self._transition(
                canonical_position, canonical_wear, action
            )
            terminal_episode = success or step_index + 1 >= self.env_config.horizon
            continuation = 0.0
            if terminal_episode and episode_index + 1 < self.config.episodes_per_lifetime:
                recovered_wear = max(
                    0.0, next_wear - self.env_config.recovery_per_episode
                )
                continuation, _ = self._value(
                    episode_index + 1, 0, 0.0, recovered_wear
                )
            elif not terminal_episode:
                continuation, _ = self._value(
                    episode_index, step_index + 1, next_position, next_wear
                )
            total_value = reward + continuation
            if total_value > best_value:
                best_value, best_action = total_value, action
        result = (best_value, best_action)
        self._cache[key] = result
        return result

    def action(
        self, episode_index: int, step_index: int, position: float, wear: float
    ) -> float:
        """Return the first action of the optimal discretized remaining plan."""

        _, action = self._value(episode_index, step_index, position, wear)
        return action


class LifetimeDPOracleController(Controller):
    """Privileged, discretized full-lifetime planning controller."""

    name = "lifetime_dp_oracle_discretized"
    uses_privileged_state = True

    def __init__(self, planner: LifetimeDPPlanner) -> None:
        self.planner = planner
        self.episode_index = -1
        self.step_index = 0

    def reset_lifetime(self) -> None:
        self.episode_index = -1
        self.step_index = 0

    def reset_episode(self) -> None:
        self.episode_index += 1
        self.step_index = 0

    def act(
        self, observation: Observation, privileged: dict[str, float] | None = None
    ) -> float:
        if privileged is None or "wear" not in privileged:
            raise ValueError("lifetime DP oracle requires privileged wear")
        position, _ = observation
        return self.planner.action(
            self.episode_index, self.step_index, position, privileged["wear"]
        )

    def observe(
        self,
        observation: Observation,
        action: float,
        next_observation: Observation,
        info: dict[str, float],
    ) -> None:
        self.step_index += 1
