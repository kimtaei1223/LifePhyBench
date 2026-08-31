"""Fair task-episode evaluation for recurrent selective-reset baselines."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskEpisodeEvaluation:
    task_episodes: int
    completed_lifetimes: int
    mean_task_episode_reward: float
    std_task_episode_reward: float
    mean_episode_end_wear: float | None = None
    mean_episode_end_thermal_load: float | None = None
    mean_episode_end_joint_aging: float | None = None
    mean_episode_end_efficiency: float | None = None
    thermal_trip_rate: float | None = None
    high_power_selection_rate: float | None = None
    mean_thermal_load_at_mode_selection: float | None = None
    cold_high_power_selection_rate: float | None = None
    hot_high_power_selection_rate: float | None = None
    cold_mode_selections: int | None = None
    hot_mode_selections: int | None = None


def evaluate_task_episodes(
    model: Any,
    environment: Any,
    task_episodes: int,
    seed: int | None = None,
) -> TaskEpisodeEvaluation:
    """Evaluate recurrent memory while reporting equal task-episode units.

    ``LifetimeStreamWrapper`` returns inner task resets as nonterminal steps, so
    the model's recurrent state persists correctly. This evaluator uses the
    namespaced inner-boundary signal to split rewards into comparable task
    episodes without resetting that state.
    """

    if task_episodes <= 0:
        raise ValueError("task_episodes must be positive")
    observation, _info = environment.reset(seed=seed)
    recurrent_state = None
    episode_start = np.array([True])
    completed_rewards: list[float] = []
    episode_end_health: dict[str, list[float]] = {
        "wear": [],
        "thermal_load": [],
        "joint_aging": [],
        "efficiency": [],
    }
    current_reward = 0.0
    completed_lifetimes = 0
    thermal_mode_selections = 0
    high_power_selections = 0
    thermal_trips = 0
    thermal_selection_loads: list[float] = []
    cold_mode_selections = 0
    hot_mode_selections = 0
    cold_high_power_selections = 0
    hot_high_power_selections = 0

    while len(completed_rewards) < task_episodes:
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        observation, reward, terminated, truncated, info = environment.step(action)
        current_reward += float(reward)
        if bool(info.get("lifephy/thermal_mode_selected_now", False)):
            thermal_mode_selections += 1
            high_power = info.get("lifephy/thermal_mode") == "high"
            if high_power:
                high_power_selections += 1
            selection_load = float(info["lifephy/thermal_load_at_mode_selection"])
            thermal_selection_loads.append(selection_load)
            trip_load = float(info["lifephy/thermal_trip_load"])
            if selection_load < trip_load:
                cold_mode_selections += 1
                cold_high_power_selections += int(high_power)
            else:
                hot_mode_selections += 1
                hot_high_power_selections += int(high_power)
        if bool(info.get("lifephy/thermal_trip", False)):
            thermal_trips += 1
        task_boundary = bool(
            info.get("lifephy/inner_task_boundary", terminated or truncated)
        )
        if task_boundary:
            completed_rewards.append(current_reward)
            for name, key in (
                ("wear", "lifephy/wear"),
                ("thermal_load", "lifephy/thermal_load"),
                ("joint_aging", "lifephy/joint_aging"),
                ("efficiency", "lifephy/actuator_efficiency"),
            ):
                value = info.get(key)
                if value is not None:
                    episode_end_health[name].append(float(value))
            current_reward = 0.0
        gym_boundary = bool(terminated or truncated)
        if bool(info.get("lifephy/lifetime_boundary", gym_boundary)):
            completed_lifetimes += 1
        if gym_boundary and len(completed_rewards) < task_episodes:
            observation, _info = environment.reset()
        episode_start = np.array([gym_boundary])

    return TaskEpisodeEvaluation(
        task_episodes=len(completed_rewards),
        completed_lifetimes=completed_lifetimes,
        mean_task_episode_reward=statistics.mean(completed_rewards),
        std_task_episode_reward=(
            statistics.stdev(completed_rewards) if len(completed_rewards) > 1 else 0.0
        ),
        mean_episode_end_wear=(
            statistics.mean(episode_end_health["wear"])
            if episode_end_health["wear"]
            else None
        ),
        mean_episode_end_thermal_load=(
            statistics.mean(episode_end_health["thermal_load"])
            if episode_end_health["thermal_load"]
            else None
        ),
        mean_episode_end_joint_aging=(
            statistics.mean(episode_end_health["joint_aging"])
            if episode_end_health["joint_aging"]
            else None
        ),
        mean_episode_end_efficiency=(
            statistics.mean(episode_end_health["efficiency"])
            if episode_end_health["efficiency"]
            else None
        ),
        thermal_trip_rate=(
            thermal_trips / thermal_mode_selections
            if thermal_mode_selections
            else None
        ),
        high_power_selection_rate=(
            high_power_selections / thermal_mode_selections
            if thermal_mode_selections
            else None
        ),
        mean_thermal_load_at_mode_selection=(
            statistics.mean(thermal_selection_loads)
            if thermal_selection_loads
            else None
        ),
        cold_high_power_selection_rate=(
            cold_high_power_selections / cold_mode_selections
            if cold_mode_selections
            else None
        ),
        hot_high_power_selection_rate=(
            hot_high_power_selections / hot_mode_selections
            if hot_mode_selections
            else None
        ),
        cold_mode_selections=(cold_mode_selections if thermal_mode_selections else None),
        hot_mode_selections=(hot_mode_selections if thermal_mode_selections else None),
    )


def evaluation_as_dict(result: TaskEpisodeEvaluation) -> dict[str, float | int]:
    """Serialize the stable task-episode evaluation schema."""

    return asdict(result)
