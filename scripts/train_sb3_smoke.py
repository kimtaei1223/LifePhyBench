#!/usr/bin/env python3
"""Train one short PPO or SAC lifetime-aware smoke baseline."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from lifephybench.envs.lifetime import LifetimeEpisodeScheduler
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear


def require_sb3() -> Any:
    try:
        import stable_baselines3
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "Stable-Baselines3 is required. Install it with: "
            "./.venv-mujoco/bin/python -m pip install stable-baselines3"
        ) from error
    return {
        "version": stable_baselines3.__version__,
        "PPO": PPO,
        "SAC": SAC,
        "evaluate_policy": evaluate_policy,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
    }


def health_config(mechanism: str, degradation_mode: str) -> ActuatorWearConfig:
    if mechanism == "wear":
        return ActuatorWearConfig(wear_rate=0.001, degradation_mode=degradation_mode)
    if mechanism == "thermal":
        return ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.005,
            thermal_cooling_rate=0.01,
            thermal_episode_cooling=0.1,
            thermal_degradation_mode=degradation_mode,
        )
    if mechanism == "joint_aging":
        return ActuatorWearConfig(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_rate=0.001,
            joint_aging_degradation_mode=degradation_mode,
        )
    raise ValueError(f"unknown health mechanism {mechanism!r}")


def make_environment(
    *,
    environment_id: str,
    mechanism: str,
    degradation_mode: str,
    episode_steps: int,
    episodes_per_lifetime: int,
    monitor: Any,
) -> Any:
    environment = PusherActuatorWear.make(
        health_config(mechanism, degradation_mode),
        environment_id=environment_id,
        max_episode_steps=episode_steps,
    )
    return monitor(LifetimeEpisodeScheduler(environment, episodes_per_lifetime))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "sac"], required=True)
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
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--output-root", default="outputs/gpu_smoke")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.total_timesteps <= 0 or args.episode_steps <= 0:
        raise SystemExit("workers, total-timesteps, and episode-steps must be positive")

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
        algorithm_class = modules["PPO"] if args.algorithm == "ppo" else modules["SAC"]
        model_kwargs: dict[str, Any] = {
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": args.device,
            "verbose": 1,
            "tensorboard_log": str(run_directory / "tensorboard"),
        }
        if args.algorithm == "ppo":
            model_kwargs.update({"n_steps": 1024, "batch_size": 256})
        else:
            model_kwargs.update({"buffer_size": 250_000, "learning_starts": 10_000})
        model = algorithm_class("MlpPolicy", train_environment, **model_kwargs)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=args.progress_bar)
        model.save(str(run_directory / "model"))
        mean_reward, reward_std = modules["evaluate_policy"](
            model,
            evaluation_environment,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )
        metadata = {
            "phase": "sb3_lifetime_smoke_not_final_result",
            "stable_baselines3_version": modules["version"],
            "arguments": vars(args),
            "mean_eval_reward": mean_reward,
            "std_eval_reward": reward_std,
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
