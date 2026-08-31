"""Transparent diagnostic controllers used before learned baselines exist."""

from __future__ import annotations

Observation = tuple[float, float]


class Controller:
    name = "controller"
    uses_privileged_state = False

    def reset_lifetime(self) -> None:
        pass

    def reset_episode(self) -> None:
        pass

    def act(
        self, observation: Observation, privileged: dict[str, float] | None = None
    ) -> float:
        raise NotImplementedError

    def observe(
        self,
        observation: Observation,
        action: float,
        next_observation: Observation,
        info: dict[str, float],
    ) -> None:
        pass


class NominalReactiveController(Controller):
    """Assumes nominal gain and has no cross-episode memory."""

    name = "nominal_reactive"

    def act(
        self, observation: Observation, privileged: dict[str, float] | None = None
    ) -> float:
        position, target = observation
        return target - position


class EpisodeEMAController(Controller):
    """Estimates gain within an episode, then deliberately forgets it."""

    name = "episode_ema"

    def __init__(self, smoothing: float = 0.5, minimum_estimate: float = 0.1) -> None:
        self.smoothing = smoothing
        self.minimum_estimate = minimum_estimate
        self.gain_estimate = 1.0

    def reset_episode(self) -> None:
        self.gain_estimate = 1.0

    def act(
        self, observation: Observation, privileged: dict[str, float] | None = None
    ) -> float:
        position, target = observation
        return (target - position) / max(self.minimum_estimate, self.gain_estimate)

    def observe(
        self,
        observation: Observation,
        action: float,
        next_observation: Observation,
        info: dict[str, float],
    ) -> None:
        if abs(action) < 1e-8:
            return
        measured = (next_observation[0] - observation[0]) / action
        measured = max(self.minimum_estimate, min(1.0, measured))
        self.gain_estimate = (
            self.smoothing * measured + (1.0 - self.smoothing) * self.gain_estimate
        )


class LifetimeEMAController(EpisodeEMAController):
    """Carries a slow gain estimate across episode boundaries."""

    name = "lifetime_ema"

    def reset_lifetime(self) -> None:
        self.gain_estimate = 1.0

    def reset_episode(self) -> None:
        # Cross-episode memory is the sole intended difference from EpisodeEMA.
        pass


class MyopicStateOracleController(Controller):
    """Myopic controller with privileged gain, not a lifetime-optimal oracle.

    It compensates immediate loss of control authority but does not plan the
    action-to-damage trade-off.  Consequently it may underperform a damage-aware
    controller over a full lifetime.
    """

    name = "myopic_state_oracle"
    uses_privileged_state = True

    def act(
        self, observation: Observation, privileged: dict[str, float] | None = None
    ) -> float:
        if privileged is None or "actuator_gain" not in privileged:
            raise ValueError("state oracle requires privileged actuator_gain")
        position, target = observation
        return (target - position) / max(1e-8, privileged["actuator_gain"])
