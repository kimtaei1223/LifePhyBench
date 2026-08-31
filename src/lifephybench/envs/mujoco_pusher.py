"""MuJoCo health wrapper with persistent wear, thermal health, and joint aging.

The wrapper intentionally implements one mechanism well before adding friction,
joint aging, or manipulation assets.  Gymnasium's ordinary ``reset`` is an
episode reset.  ``reset_lifetime`` is the only operation that restores health.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ActuatorWearConfig:
    wear_rate: float = 2.0e-5
    wear_exponent: float = 2.0
    minimum_efficiency: float = 0.20
    expose_health: bool = False
    degradation_mode: str = "endogenous_action"
    exogenous_dose_per_step: float = 0.25
    thermal_enabled: bool = False
    thermal_heat_rate: float = 0.0
    thermal_exponent: float = 2.0
    thermal_cooling_rate: float = 0.0
    thermal_episode_cooling: float = 0.0
    thermal_minimum_efficiency: float = 0.60
    thermal_degradation_mode: str = "endogenous_action"
    thermal_exogenous_dose_per_step: float = 0.25
    joint_aging_enabled: bool = False
    joint_aging_rate: float = 0.0
    joint_aging_exponent: float = 2.0
    joint_aging_damping_multiplier: float = 4.0
    joint_aging_degradation_mode: str = "endogenous_action"
    joint_aging_exogenous_dose_per_step: float = 0.25
    degradation_law_family: str = "power"
    threshold_dose: float = 0.50
    stochastic_shock_probability: float = 0.0
    stochastic_shock_size: float = 0.0
    canonical_task_seed: int | None = None

    def __post_init__(self) -> None:
        if self.wear_rate < 0.0:
            raise ValueError("wear_rate must be non-negative")
        if self.wear_exponent <= 0.0:
            raise ValueError("wear_exponent must be positive")
        if not 0.0 < self.minimum_efficiency <= 1.0:
            raise ValueError("minimum_efficiency must be in (0, 1]")
        if self.degradation_mode not in {"endogenous_action", "exogenous_clock"}:
            raise ValueError(
                "degradation_mode must be endogenous_action or exogenous_clock"
            )
        if self.exogenous_dose_per_step < 0.0:
            raise ValueError("exogenous_dose_per_step must be non-negative")
        if self.thermal_heat_rate < 0.0:
            raise ValueError("thermal_heat_rate must be non-negative")
        if self.thermal_exponent <= 0.0:
            raise ValueError("thermal_exponent must be positive")
        if not 0.0 <= self.thermal_cooling_rate <= 1.0:
            raise ValueError("thermal_cooling_rate must be in [0, 1]")
        if not 0.0 <= self.thermal_episode_cooling <= 1.0:
            raise ValueError("thermal_episode_cooling must be in [0, 1]")
        if not 0.0 < self.thermal_minimum_efficiency <= 1.0:
            raise ValueError("thermal_minimum_efficiency must be in (0, 1]")
        if self.thermal_degradation_mode not in {
            "endogenous_action",
            "exogenous_clock",
        }:
            raise ValueError(
                "thermal_degradation_mode must be endogenous_action or "
                "exogenous_clock"
            )
        if self.thermal_exogenous_dose_per_step < 0.0:
            raise ValueError("thermal_exogenous_dose_per_step must be non-negative")
        if self.joint_aging_rate < 0.0:
            raise ValueError("joint_aging_rate must be non-negative")
        if self.joint_aging_exponent <= 0.0:
            raise ValueError("joint_aging_exponent must be positive")
        if self.joint_aging_damping_multiplier < 0.0:
            raise ValueError("joint_aging_damping_multiplier must be non-negative")
        if self.joint_aging_degradation_mode not in {
            "endogenous_action",
            "exogenous_clock",
        }:
            raise ValueError(
                "joint_aging_degradation_mode must be endogenous_action or "
                "exogenous_clock"
            )
        if self.joint_aging_exogenous_dose_per_step < 0.0:
            raise ValueError(
                "joint_aging_exogenous_dose_per_step must be non-negative"
            )
        if self.degradation_law_family not in {
            "power",
            "threshold",
            "stochastic_shock",
        }:
            raise ValueError(
                "degradation_law_family must be power, threshold, or "
                "stochastic_shock"
            )
        if self.threshold_dose < 0.0:
            raise ValueError("threshold_dose must be non-negative")
        if not 0.0 <= self.stochastic_shock_probability <= 1.0:
            raise ValueError("stochastic_shock_probability must be in [0, 1]")
        if self.stochastic_shock_size < 0.0:
            raise ValueError("stochastic_shock_size must be non-negative")
        if self.canonical_task_seed is not None and self.canonical_task_seed < 0:
            raise ValueError("canonical_task_seed must be non-negative when set")


class PusherActuatorWear(gym.Wrapper):
    """Apply persistent health dynamics to a joint-actuated MuJoCo task.

    Wear is a persistent scalar diagnostic health state. Optional thermal load
    is a second state: it increases with dose, cools during transitions and at
    episode boundaries, and is cleared only by a lifetime reset. Optional joint
    aging is a third persistent state that increases actuated-joint damping.
    The task observation remains unchanged unless ``expose_health=True``;
    privileged health is always namespaced in ``info`` for audit/oracle use.
    """

    def __init__(
        self,
        env: gym.Env,
        config: ActuatorWearConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config or ActuatorWearConfig()
        unwrapped = self.env.unwrapped
        if not hasattr(unwrapped, "model") or not hasattr(
            unwrapped.model, "actuator_gainprm"
        ):
            raise TypeError("PusherActuatorWear requires a MuJoCo environment")
        if unwrapped.model.nactuator <= 0:
            raise ValueError("environment has no actuators")

        self._base_gain = np.array(
            unwrapped.model.actuator_gainprm[:, 0], dtype=np.float64, copy=True
        )
        if np.any(self._base_gain == 0.0):
            raise ValueError("diagnostic expects non-zero fixed actuator gains")
        self._base_damping = np.array(
            unwrapped.model.dof_damping, dtype=np.float64, copy=True
        )
        actuator_joint_ids = np.unique(unwrapped.model.actuator_trnid[:, 0])
        if np.any(actuator_joint_ids < 0):
            raise ValueError("diagnostic expects joint-transmitted actuators")
        self._actuated_dof_indices = np.asarray(
            unwrapped.model.jnt_dofadr[actuator_joint_ids], dtype=np.intp
        )
        if np.any(self._base_damping[self._actuated_dof_indices] <= 0.0):
            raise ValueError("diagnostic expects positive actuated-joint damping")

        self.wear = 0.0
        self.thermal_load = 0.0
        self.joint_aging = 0.0
        self.cumulative_action_dose = 0.0
        self.cumulative_health_dose = 0.0
        self.cumulative_thermal_dose = 0.0
        self.cumulative_joint_aging_dose = 0.0
        self._health_rng = np.random.default_rng(0)
        self.lifetime_id = -1
        self.episode_index = -1

        if self.config.expose_health:
            if not isinstance(self.observation_space, gym.spaces.Box):
                raise TypeError("health exposure currently requires a Box observation")
            health_low = [0.0, self.config.minimum_efficiency]
            health_high = [1.0, 1.0]
            if self.config.thermal_enabled:
                health_low.extend([0.0, self.config.thermal_minimum_efficiency])
                health_high.extend([1.0, 1.0])
            if self.config.joint_aging_enabled:
                health_low.extend([0.0, 1.0])
                health_high.extend([1.0, self.joint_damping_multiplier_at_max])
            low = np.concatenate(
                [np.asarray(self.observation_space.low, dtype=np.float64), health_low]
            )
            high = np.concatenate(
                [np.asarray(self.observation_space.high, dtype=np.float64), health_high]
            )
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float64
            )

        self._apply_health_to_model()

    @classmethod
    def make(
        cls,
        config: ActuatorWearConfig | None = None,
        environment_id: str = "Pusher-v5",
        **gym_make_kwargs: Any,
    ) -> PusherActuatorWear:
        return cls(gym.make(environment_id, **gym_make_kwargs), config=config)

    @property
    def efficiency(self) -> float:
        """Total instantaneous actuator efficiency from wear and heat."""

        return self.wear_efficiency * self.thermal_efficiency

    @property
    def wear_efficiency(self) -> float:
        span = 1.0 - self.config.minimum_efficiency
        return 1.0 - span * self.wear

    @property
    def thermal_efficiency(self) -> float:
        if not self.config.thermal_enabled:
            return 1.0
        span = 1.0 - self.config.thermal_minimum_efficiency
        return 1.0 - span * self.thermal_load

    @property
    def joint_damping_multiplier(self) -> float:
        return 1.0 + self.config.joint_aging_damping_multiplier * self.joint_aging

    @property
    def joint_damping_multiplier_at_max(self) -> float:
        return 1.0 + self.config.joint_aging_damping_multiplier

    def _apply_health_to_model(self) -> None:
        self.env.unwrapped.model.actuator_gainprm[:, 0] = (
            self._base_gain * self.efficiency
        )
        self.env.unwrapped.model.dof_damping[self._actuated_dof_indices] = (
            self._base_damping[self._actuated_dof_indices]
            * self.joint_damping_multiplier
        )

    def _observation(self, observation: Any) -> Any:
        if not self.config.expose_health:
            return observation
        health = [self.wear, self.wear_efficiency]
        if self.config.thermal_enabled:
            health.extend([self.thermal_load, self.thermal_efficiency])
        if self.config.joint_aging_enabled:
            health.extend([self.joint_aging, self.joint_damping_multiplier])
        return np.concatenate([np.asarray(observation, dtype=np.float64), health])

    def _audit_info(
        self,
        info: dict[str, Any],
        action_dose: float = 0.0,
        health_dose: float = 0.0,
        thermal_dose: float = 0.0,
        joint_aging_dose: float = 0.0,
    ) -> dict[str, Any]:
        result = dict(info)
        result.update(
            {
                "lifephy/wear": self.wear,
                "lifephy/thermal_load": self.thermal_load,
                "lifephy/joint_aging": self.joint_aging,
                "lifephy/actuator_efficiency": self.efficiency,
                "lifephy/wear_efficiency": self.wear_efficiency,
                "lifephy/thermal_efficiency": self.thermal_efficiency,
                "lifephy/joint_damping_multiplier": self.joint_damping_multiplier,
                "lifephy/action_dose": action_dose,
                "lifephy/health_dose": health_dose,
                "lifephy/thermal_dose": thermal_dose,
                "lifephy/joint_aging_dose": joint_aging_dose,
                "lifephy/cumulative_action_dose": self.cumulative_action_dose,
                "lifephy/cumulative_health_dose": self.cumulative_health_dose,
                "lifephy/cumulative_thermal_dose": self.cumulative_thermal_dose,
                "lifephy/cumulative_joint_aging_dose": self.cumulative_joint_aging_dose,
                "lifephy/degradation_mode": self.config.degradation_mode,
                "lifephy/thermal_degradation_mode": self.config.thermal_degradation_mode,
                "lifephy/joint_aging_degradation_mode": self.config.joint_aging_degradation_mode,
                "lifephy/degradation_law_family": self.config.degradation_law_family,
                "lifephy/lifetime_id": self.lifetime_id,
                "lifephy/episode_index": self.episode_index,
            }
        )
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset task state and preserve accumulated actuator wear."""

        if self.episode_index >= 0 and self.config.thermal_enabled:
            self.thermal_load *= 1.0 - self.config.thermal_episode_cooling
        task_seed = (
            self.config.canonical_task_seed
            if self.config.canonical_task_seed is not None
            else seed
        )
        observation, info = self.env.reset(seed=task_seed, options=options)
        self.episode_index += 1
        self._apply_health_to_model()
        return self._observation(observation), self._audit_info(info)

    def reset_lifetime(
        self,
        *,
        seed: int | None = None,
        lifetime_id: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset task and physical health to create an independent lifetime."""

        self.wear = 0.0
        self.thermal_load = 0.0
        self.joint_aging = 0.0
        self.cumulative_action_dose = 0.0
        self.cumulative_health_dose = 0.0
        self.cumulative_thermal_dose = 0.0
        self.cumulative_joint_aging_dose = 0.0
        self._health_rng = np.random.default_rng(seed)
        self.lifetime_id = (
            self.lifetime_id + 1 if lifetime_id is None else int(lifetime_id)
        )
        self.episode_index = -1
        self._apply_health_to_model()
        return self.reset(seed=seed, options=options)

    def _normalized_action_dose(self, action: Any, exponent: float) -> float:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"expected action shape {self.action_space.shape}, got {action_array.shape}"
            )
        clipped = np.clip(action_array, self.action_space.low, self.action_space.high)
        scale = np.maximum(
            np.abs(np.asarray(self.action_space.low, dtype=np.float64)),
            np.abs(np.asarray(self.action_space.high, dtype=np.float64)),
        )
        normalized = clipped / np.maximum(scale, 1e-12)
        return float(np.mean(np.abs(normalized) ** exponent))

    def _health_increment(self, rate: float, dose: float) -> float:
        """Apply a preregisterable damage-law family to one health channel."""

        if rate == 0.0:
            return 0.0
        if self.config.degradation_law_family == "threshold":
            effective_dose = max(0.0, dose - self.config.threshold_dose)
        else:
            effective_dose = dose
        increment = rate * effective_dose
        if (
            self.config.degradation_law_family == "stochastic_shock"
            and self._health_rng.random() < self.config.stochastic_shock_probability
        ):
            increment += self.config.stochastic_shock_size
        return increment

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # The action experiences health at the start of the transition.
        self._apply_health_to_model()
        observation, reward, terminated, truncated, info = self.env.step(action)

        action_dose = self._normalized_action_dose(action, self.config.wear_exponent)
        health_dose = (
            action_dose
            if self.config.degradation_mode == "endogenous_action"
            else self.config.exogenous_dose_per_step
        )
        self.cumulative_action_dose += action_dose
        self.cumulative_health_dose += health_dose
        self.wear = min(
            1.0, self.wear + self._health_increment(self.config.wear_rate, health_dose)
        )
        thermal_action_dose = self._normalized_action_dose(
            action, self.config.thermal_exponent
        )
        thermal_dose = (
            thermal_action_dose
            if self.config.thermal_degradation_mode == "endogenous_action"
            else self.config.thermal_exogenous_dose_per_step
        )
        if self.config.thermal_enabled:
            self.cumulative_thermal_dose += thermal_dose
            heated = self.thermal_load + self._health_increment(
                self.config.thermal_heat_rate, thermal_dose
            )
            self.thermal_load = min(
                1.0, max(0.0, heated * (1.0 - self.config.thermal_cooling_rate))
            )
        joint_aging_action_dose = self._normalized_action_dose(
            action, self.config.joint_aging_exponent
        )
        joint_aging_dose = (
            joint_aging_action_dose
            if self.config.joint_aging_degradation_mode == "endogenous_action"
            else self.config.joint_aging_exogenous_dose_per_step
        )
        if self.config.joint_aging_enabled:
            self.cumulative_joint_aging_dose += joint_aging_dose
            self.joint_aging = min(
                1.0,
                self.joint_aging
                + self._health_increment(self.config.joint_aging_rate, joint_aging_dose),
            )
        self._apply_health_to_model()

        return (
            self._observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._audit_info(
                info,
                action_dose=action_dose,
                health_dose=health_dose,
                thermal_dose=thermal_dose,
                joint_aging_dose=joint_aging_dose,
            ),
        )

    def set_wear_for_diagnostic(self, wear: float) -> None:
        """Set privileged health for controlled tests and oracle diagnostics."""

        wear_value = float(wear)
        if not np.isfinite(wear_value) or not 0.0 <= wear_value <= 1.0:
            raise ValueError("wear must be finite and in [0, 1]")
        self.wear = wear_value
        self._apply_health_to_model()

    def set_thermal_load_for_diagnostic(self, thermal_load: float) -> None:
        """Set privileged thermal state for controlled response-direction tests."""

        value = float(thermal_load)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("thermal_load must be finite and in [0, 1]")
        self.thermal_load = value
        self._apply_health_to_model()

    def set_joint_aging_for_diagnostic(self, joint_aging: float) -> None:
        """Set privileged joint aging for controlled response-direction tests."""

        value = float(joint_aging)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("joint_aging must be finite and in [0, 1]")
        self.joint_aging = value
        self._apply_health_to_model()

    def audit_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "wear": self.wear,
            "thermal_load": self.thermal_load,
            "joint_aging": self.joint_aging,
            "efficiency": self.efficiency,
            "wear_efficiency": self.wear_efficiency,
            "thermal_efficiency": self.thermal_efficiency,
            "joint_damping_multiplier": self.joint_damping_multiplier,
            "cumulative_action_dose": self.cumulative_action_dose,
            "cumulative_health_dose": self.cumulative_health_dose,
            "cumulative_thermal_dose": self.cumulative_thermal_dose,
            "cumulative_joint_aging_dose": self.cumulative_joint_aging_dose,
            "lifetime_id": self.lifetime_id,
            "episode_index": self.episode_index,
            "actuator_gain": np.array(
                self.env.unwrapped.model.actuator_gainprm[:, 0], copy=True
            ).tolist(),
            "actuated_dof_damping": np.array(
                self.env.unwrapped.model.dof_damping[self._actuated_dof_indices],
                copy=True,
            ).tolist(),
        }
