#!/usr/bin/env python3
"""Train an episode-reset or lifetime-persistent RecurrentPPO baseline."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any

from lifephybench.envs.lifetime import LifetimeEpisodeScheduler, LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.recurrent_evaluation import evaluate_task_episodes, evaluation_as_dict


def require_recurrent_ppo() -> Any:
    try:
        import sb3_contrib
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "sb3-contrib is required. Install it with: "
            "./.venv-mujoco/bin/python -m pip install sb3-contrib"
        ) from error
    return {
        "version": sb3_contrib.__version__,
        "RecurrentPPO": RecurrentPPO,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
    }


def health_config(
    mechanism: str,
    degradation_mode: str,
    *,
    exogenous_dose_per_step: float,
    thermal_exogenous_dose_per_step: float,
    joint_aging_exogenous_dose_per_step: float,
    thermal_heat_rate: float = 0.005,
    thermal_cooling_rate: float = 0.01,
    thermal_episode_cooling: float = 0.1,
    canonical_task_seed: int | None = None,
) -> ActuatorWearConfig:
    if mechanism == "wear":
        return ActuatorWearConfig(
            wear_rate=0.001,
            degradation_mode=degradation_mode,
            exogenous_dose_per_step=exogenous_dose_per_step,
        )
    if mechanism == "thermal":
        return ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=thermal_heat_rate,
            thermal_cooling_rate=thermal_cooling_rate,
            thermal_episode_cooling=thermal_episode_cooling,
            thermal_degradation_mode=degradation_mode,
            thermal_exogenous_dose_per_step=thermal_exogenous_dose_per_step,
            canonical_task_seed=canonical_task_seed,
        )
    if mechanism == "joint_aging":
        return ActuatorWearConfig(
            wear_rate=0.0,
            joint_aging_enabled=True,
            joint_aging_rate=0.001,
            joint_aging_degradation_mode=degradation_mode,
            joint_aging_exogenous_dose_per_step=joint_aging_exogenous_dose_per_step,
        )
    raise ValueError(f"unknown health mechanism {mechanism!r}")


def make_environment(
    *,
    environment_id: str,
    mechanism: str,
    degradation_mode: str,
    memory_mode: str,
    episode_steps: int,
    episodes_per_lifetime: int,
    exogenous_dose_per_step: float,
    thermal_exogenous_dose_per_step: float,
    joint_aging_exogenous_dose_per_step: float,
    monitor: Any,
) -> Any:
    environment = PusherActuatorWear.make(
        health_config(
            mechanism,
            degradation_mode,
            exogenous_dose_per_step=exogenous_dose_per_step,
            thermal_exogenous_dose_per_step=thermal_exogenous_dose_per_step,
            joint_aging_exogenous_dose_per_step=joint_aging_exogenous_dose_per_step,
        ),
        environment_id=environment_id,
        max_episode_steps=episode_steps,
    )
    scheduler = (
        LifetimeEpisodeScheduler(environment, episodes_per_lifetime)
        if memory_mode == "episode"
        else LifetimeStreamWrapper(environment, episodes_per_lifetime)
    )
    return monitor(scheduler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-mode", choices=["episode", "lifetime"], required=True)
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
    parser.add_argument("--exogenous-dose-per-step", type=float, default=0.25)
    parser.add_argument("--thermal-exogenous-dose-per-step", type=float, default=0.25)
    parser.add_argument("--joint-aging-exogenous-dose-per-step", type=float, default=0.25)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-task-episodes", type=int, default=200)
    parser.add_argument("--output-root", default="outputs/recurrent_smoke")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if (
        args.workers <= 0
        or args.total_timesteps <= 0
        or args.episode_steps <= 0
        or args.exogenous_dose_per_step < 0.0
        or args.thermal_exogenous_dose_per_step < 0.0
        or args.joint_aging_exogenous_dose_per_step < 0.0
    ):
        raise SystemExit("workers, total-timesteps, and episode-steps must be positive")

    modules = require_recurrent_ppo()
    run_directory = Path(args.output_root) / args.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    environment_factory = partial(
        make_environment,
        environment_id=args.environment_id,
        mechanism=args.mechanism,
        degradation_mode=args.degradation_mode,
        memory_mode=args.memory_mode,
        episode_steps=args.episode_steps,
        episodes_per_lifetime=args.episodes_per_lifetime,
        exogenous_dose_per_step=args.exogenous_dose_per_step,
        thermal_exogenous_dose_per_step=args.thermal_exogenous_dose_per_step,
        joint_aging_exogenous_dose_per_step=args.joint_aging_exogenous_dose_per_step,
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
        model = modules["RecurrentPPO"](
            "MlpLstmPolicy",
            train_environment,
            learning_rate=args.learning_rate,
            n_steps=256,
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
            seed=evaluation_seed,
        )
        metadata = {
            "phase": "recurrent_lifetime_smoke_not_final_result",
            "sb3_contrib_version": modules["version"],
            "arguments": vars(args),
            "evaluation_seed": evaluation_seed,
            "task_episode_evaluation": evaluation_as_dict(evaluation),
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
