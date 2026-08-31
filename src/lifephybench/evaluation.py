"""Lifetime-level evaluation with no transition-level data leakage."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass

from .policies import Controller
from .toy_env import ToyWearConfig, ToyWearEnv


@dataclass(frozen=True)
class LifetimeMetrics:
    controller: str
    physics_protocol: str
    seed: int
    episodes: int
    successes: int
    success_auc: float
    mean_episode_return: float
    worst_window_success: float
    final_wear: float
    cumulative_damage: float


def _worst_window(values: Sequence[float], window: int) -> float:
    if not values:
        return 0.0
    width = min(window, len(values))
    return min(
        sum(values[i : i + width]) / width for i in range(len(values) - width + 1)
    )


def evaluate_lifetime(
    controller: Controller,
    config: ToyWearConfig,
    seed: int,
    episodes: int,
    persistent_physics: bool = True,
) -> LifetimeMetrics:
    env = ToyWearEnv(config)
    observation = env.reset_lifetime(seed=seed, lifetime_id=seed)
    controller.reset_lifetime()

    episode_successes: list[float] = []
    episode_returns: list[float] = []
    cumulative_damage = 0.0

    for episode_index in range(episodes):
        if episode_index > 0:
            observation = env.reset_episode()
            if not persistent_physics:
                # Counterfactual used to quantify conventional reset bias.  Task
                # sequence and random stream are preserved; only hidden wear is
                # reset, so this is not an independent lifetime.
                env.set_wear_for_diagnostic(0.0)
        controller.reset_episode()
        total_reward = 0.0
        success = 0.0

        while True:
            privileged = None
            if controller.uses_privileged_state:
                privileged = {"actuator_gain": env.actuator_gain, "wear": env.state.wear}
            action = controller.act(observation, privileged)
            result = env.step(action)
            controller.observe(observation, action, result.observation, result.info)
            observation = result.observation
            total_reward += result.reward
            cumulative_damage += result.info["damage_increment"]
            success = max(success, result.info["success"])
            if result.terminated or result.truncated:
                break

        episode_successes.append(success)
        episode_returns.append(total_reward)

    mean_episode_return = sum(episode_returns) / max(1, episodes)
    success_auc = sum(episode_successes) / max(1, episodes)
    return LifetimeMetrics(
        controller=controller.name,
        physics_protocol=(
            "persistent_lifetime" if persistent_physics else "episode_physics_reset"
        ),
        seed=seed,
        episodes=episodes,
        successes=int(sum(episode_successes)),
        success_auc=success_auc,
        mean_episode_return=mean_episode_return,
        worst_window_success=_worst_window(episode_successes, window=10),
        final_wear=env.state.wear,
        cumulative_damage=cumulative_damage,
    )


def evaluate_many(
    controller_factory: Callable[[], Controller],
    config: ToyWearConfig,
    seeds: Iterable[int],
    episodes: int,
    persistent_physics: bool = True,
    include_lifetimes: bool = True,
) -> dict[str, object]:
    rows = [
        evaluate_lifetime(
            controller_factory(),
            config,
            seed,
            episodes,
            persistent_physics=persistent_physics,
        )
        for seed in seeds
    ]
    if not rows:
        raise ValueError("at least one seed is required")

    metric_names = (
        "success_auc",
        "mean_episode_return",
        "worst_window_success",
        "final_wear",
        "cumulative_damage",
    )
    aggregate = {}
    for name in metric_names:
        values = [float(getattr(row, name)) for row in rows]
        aggregate[name] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    result = {
        "controller": rows[0].controller,
        "physics_protocol": rows[0].physics_protocol,
        "config": asdict(config),
        "aggregate": aggregate,
    }
    if include_lifetimes:
        result["lifetimes"] = [asdict(row) for row in rows]
    return result
