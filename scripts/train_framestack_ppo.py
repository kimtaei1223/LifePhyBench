#!/usr/bin/env python3
"""Train one finite-history PPO baseline with selective reset semantics."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

from train_sb3_smoke import health_config

from lifephybench.envs.history import SelectiveFrameStack
from lifephybench.envs.lifetime import LifetimeEpisodeScheduler
from lifephybench.envs.mujoco_pusher import PusherActuatorWear


@dataclass(frozen=True)
class TaskEvaluation:
    task_episodes: int
    completed_lifetimes: int
    mean_task_episode_reward: float
    std_task_episode_reward: float


def require_sb3() -> dict[str, Any]:
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "Stable-Baselines3 is required. Install the project's rl extra first."
        ) from error
    return {
        "version": stable_baselines3.__version__,
        "PPO": PPO,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
    }


def make_environment(
    *,
    environment_id: str,
    mechanism: str,
    degradation_mode: str,
    episode_steps: int,
    episodes_per_lifetime: int,
    stack_size: int,
    history_mode: str,
    monitor: Any,
) -> Any:
    base = PusherActuatorWear.make(
        health_config(mechanism, degradation_mode),
        environment_id=environment_id,
        max_episode_steps=episode_steps,
    )
    scheduler = LifetimeEpisodeScheduler(base, episodes_per_lifetime)
    history = SelectiveFrameStack(scheduler, stack_size, history_mode)
    return monitor(history)


def evaluate_task_episodes(
    model: Any,
    environment: Any,
    task_episodes: int,
    seed: int,
) -> TaskEvaluation:
    observation, _info = environment.reset(seed=seed)
    rewards: list[float] = []
    current_reward = 0.0
    completed_lifetimes = 0
    while len(rewards) < task_episodes:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        current_reward += float(reward)
        if not (terminated or truncated):
            continue
        rewards.append(current_reward)
        current_reward = 0.0
        if info.get("lifephy/lifetime_boundary", False):
            completed_lifetimes += 1
        if len(rewards) < task_episodes:
            observation, _info = environment.reset()
    return TaskEvaluation(
        task_episodes=len(rewards),
        completed_lifetimes=completed_lifetimes,
        mean_task_episode_reward=statistics.mean(rewards),
        std_task_episode_reward=statistics.pstdev(rewards),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-size", type=int, required=True)
    parser.add_argument("--history-mode", choices=["task", "lifetime"], default="lifetime")
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument(
        "--mechanism", choices=["wear", "thermal", "joint_aging"], default="wear"
    )
    parser.add_argument(
        "--degradation-mode",
        choices=["endogenous_action", "exogenous_clock"],
        default="endogenous_action",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--episodes-per-lifetime", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-task-episodes", type=int, default=200)
    parser.add_argument("--output-root", default="outputs/framestack_campaign")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if min(
        args.stack_size,
        args.workers,
        args.episode_steps,
        args.episodes_per_lifetime,
        args.total_timesteps,
        args.eval_task_episodes,
    ) <= 0:
        raise SystemExit("positive integer arguments must all be greater than zero")

    modules = require_sb3()
    run_directory = Path(args.output_root) / args.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    environment_factory = partial(
        make_environment,
        environment_id=args.environment_id,
        mechanism=args.mechanism,
        degradation_mode=args.degradation_mode,
        episode_steps=args.episode_steps,
        episodes_per_lifetime=args.episodes_per_lifetime,
        stack_size=args.stack_size,
        history_mode=args.history_mode,
        monitor=modules["Monitor"],
    )
    if args.workers == 1:
        train_environment = modules["DummyVecEnv"]([environment_factory])
    else:
        train_environment = modules["SubprocVecEnv"](
            [environment_factory for _ in range(args.workers)], start_method="spawn"
        )
    evaluation_environment = environment_factory()
    try:
        train_environment.seed(args.seed)
        model = modules["PPO"](
            "MlpPolicy",
            train_environment,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=256,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(run_directory / "tensorboard"),
        )
        model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        model.save(str(run_directory / "model"))
        evaluation_seed = args.seed + 10_000_019
        evaluation = evaluate_task_episodes(
            model,
            evaluation_environment,
            args.eval_task_episodes,
            evaluation_seed,
        )
        metadata = {
            "phase": "framestack_ppo_development_not_final_result",
            "stable_baselines3_version": modules["version"],
            "arguments": vars(args),
            "evaluation_seed": evaluation_seed,
            "task_episode_evaluation": asdict(evaluation),
        }
        (run_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        train_environment.close()
        evaluation_environment.close()


if __name__ == "__main__":
    main()
