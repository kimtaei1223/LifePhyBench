#!/usr/bin/env python3
"""Train one calibration or frozen-confirmatory v11 hidden-thermal cell.

This module is intentionally separate from the frozen v10 trainer.  It can
train recurrent lifetime memory, a same-capacity task-reset recurrent control,
or either of two feed-forward time-aware PPO controls.  Every arm receives the
same observation; the only intended difference is its memory mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from lifephybench.envs.hierarchical_thermal_v11 import (
    HierarchicalThermalV11Config,
    HierarchicalThermalV11Env,
)
from lifephybench.selective_reset_policy import TaskResetMlpLstmPolicy

POLICY_ARMS = (
    "lifetime_lstm",
    "task_reset_lstm",
    "reactive_mlp_64",
    "reactive_mlp_256",
)
STUDY_PHASES = ("calibration", "confirmatory")


class RewardScale(gym.RewardWrapper):
    """Pickle-safe training-only reward scaling."""

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, document: Any) -> None:
    """Write JSON without leaving a valid-looking partial result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def configure_cpu_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def require_training_stack() -> dict[str, Any]:
    try:
        import sb3_contrib
        import stable_baselines3
        import torch
        from sb3_contrib import RecurrentPPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "v11 requires torch, stable-baselines3, and sb3-contrib in the "
            "project virtual environment"
        ) from error
    return {
        "torch": torch,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "sb3_contrib_version": sb3_contrib.__version__,
        "PPO": PPO,
        "RecurrentPPO": RecurrentPPO,
        "Monitor": Monitor,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
    }


def make_environment(
    *,
    config_document: dict[str, Any],
    reward_scale: float,
    torch_threads_per_process: int,
    monitor: Any,
) -> gym.Env:
    configure_cpu_threads(torch_threads_per_process)
    environment: gym.Env = HierarchicalThermalV11Env(
        HierarchicalThermalV11Config(**config_document)
    )
    if reward_scale != 1.0:
        environment = RewardScale(environment, reward_scale)
    return monitor(environment)


def model_spec(policy_arm: str) -> tuple[str, Any, dict[str, Any]]:
    """Return algorithm name, policy, and policy kwargs for a declared arm."""

    if policy_arm == "lifetime_lstm":
        return "RecurrentPPO", "MlpLstmPolicy", {}
    if policy_arm == "task_reset_lstm":
        return "RecurrentPPO", TaskResetMlpLstmPolicy, {}
    if policy_arm == "reactive_mlp_64":
        return "PPO", "MlpPolicy", {"net_arch": [64, 64]}
    if policy_arm == "reactive_mlp_256":
        return "PPO", "MlpPolicy", {"net_arch": [256, 256]}
    raise ValueError(f"unknown policy arm: {policy_arm}")


def evaluate_with_raw_rows(
    model: Any,
    environment: gym.Env,
    *,
    task_episodes: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate equal task units and retain every auditable task-level row."""

    if task_episodes <= 0:
        raise ValueError("task_episodes must be positive")
    observation, reset_info = environment.reset(seed=seed)
    state = None
    episode_start = np.asarray([True])
    rows: list[dict[str, Any]] = []
    lifetime_ordinal = 0
    while len(rows) < task_episodes:
        action, state = model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        action_value = int(np.asarray(action).item())
        observation, reward, terminated, truncated, info = environment.step(action)
        task_boundary = bool(
            info.get("lifephy/inner_task_boundary", terminated or truncated)
        )
        if not task_boundary:
            raise RuntimeError("one v11 high-level action must complete exactly one task")
        rows.append(
            {
                "evaluation_seed": int(seed),
                "lifetime_ordinal": lifetime_ordinal,
                "task_index": int(
                    info.get("lifephy/v11_task_index_at_selection", len(rows) % 20)
                ),
                "action": action_value,
                "mode": info.get("lifephy/thermal_mode"),
                "reward": float(reward),
                "tripped": bool(info.get("lifephy/thermal_trip", False)),
                "sensor_load": float(info.get("lifephy/v11_sensor_load", np.nan)),
                "true_load_at_selection": float(
                    info.get("lifephy/thermal_load_at_mode_selection", np.nan)
                ),
                "true_load_after_task": float(
                    info.get("lifephy/thermal_load", np.nan)
                ),
                "initial_load": float(
                    info.get(
                        "lifephy/v11_lifetime_initial_thermal_load",
                        reset_info.get(
                            "lifephy/v11_lifetime_initial_thermal_load", np.nan
                        ),
                    )
                ),
                "condition": info.get("lifephy/v11_condition"),
                "lifetime_boundary": bool(
                    info.get("lifephy/lifetime_boundary", terminated or truncated)
                ),
            }
        )
        gym_boundary = bool(terminated or truncated)
        if gym_boundary and len(rows) < task_episodes:
            observation, reset_info = environment.reset()
            lifetime_ordinal += 1
        episode_start = np.asarray([gym_boundary])

    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    high = np.asarray([row["action"] == 1 for row in rows], dtype=np.float64)
    trips = np.asarray([row["tripped"] for row in rows], dtype=np.float64)
    lifetimes: dict[int, set[int]] = {}
    for row in rows:
        lifetimes.setdefault(int(row["lifetime_ordinal"]), set()).add(int(row["action"]))
    aggregate = {
        "task_episodes": len(rows),
        "completed_lifetimes": sum(row["lifetime_boundary"] for row in rows),
        "mean_task_episode_reward": float(rewards.mean()),
        "std_task_episode_reward": float(rewards.std(ddof=1)) if len(rows) > 1 else 0.0,
        "high_power_selection_rate": float(high.mean()),
        "thermal_trip_rate": float(trips.mean()),
        "both_modes_lifetime_rate": float(
            np.mean([len(actions) == 2 for actions in lifetimes.values()])
        ),
    }
    return aggregate, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument("--condition", choices=["fixed", "stochastic"], required=True)
    parser.add_argument("--policy-arm", choices=POLICY_ARMS, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-task-decisions", type=int, default=100_000)
    parser.add_argument("--eval-task-episodes", type=int, default=4_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads-per-process", type=int, default=1)
    parser.add_argument("--training-reward-scale", type=float, default=0.02)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--episodes-per-lifetime", type=int, default=20)
    parser.add_argument("--canonical-task-seed", type=int, default=811)
    parser.add_argument("--trip-load", type=float, default=0.10)
    parser.add_argument("--low-power-scale", type=float, default=0.40)
    parser.add_argument("--trip-penalty", type=float, default=75.0)
    parser.add_argument("--high-power-bonus", type=float, default=2.0)
    parser.add_argument("--thermal-heat-rate", type=float, default=0.05)
    parser.add_argument("--thermal-episode-cooling", type=float, default=0.10)
    parser.add_argument("--fixed-initial-load", type=float, default=0.04)
    parser.add_argument("--stochastic-initial-load-low", type=float, default=0.0)
    parser.add_argument("--stochastic-initial-load-high", type=float, default=0.08)
    parser.add_argument("--sensor-noise-sd", type=float, default=0.02)
    parser.add_argument("--shock-probability", type=float, default=5e-4)
    parser.add_argument("--shock-size", type=float, default=0.01)
    parser.add_argument(
        "--low-level-model",
        default=(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument("--output-root", default="outputs/hierarchical_v11_calibration")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--study-phase", choices=STUDY_PHASES, default="calibration")
    parser.add_argument("--protocol-sha256")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(
        args.workers,
        args.total_task_decisions,
        args.eval_task_episodes,
        args.seed,
        args.torch_threads_per_process,
        args.episode_steps,
        args.episodes_per_lifetime,
    ) <= 0:
        raise SystemExit("budgets, counts, and optimization seed must be positive")
    if args.evaluation_seed is not None and args.evaluation_seed <= 0:
        raise SystemExit("evaluation seed must be positive")
    if args.training_reward_scale <= 0.0:
        raise SystemExit("training reward scale must be positive")
    if args.ent_coef < 0.0:
        raise SystemExit("entropy coefficient must be non-negative")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("gamma and GAE lambda must be in (0, 1]")
    if args.study_phase == "confirmatory":
        digest = args.protocol_sha256 or ""
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise SystemExit("confirmatory training requires a lowercase protocol SHA-256")


def main() -> None:
    args = parse_args()
    validate_args(args)
    project_root = Path(__file__).resolve().parent.parent
    low_level_path = Path(args.low_level_model)
    if not low_level_path.is_absolute():
        low_level_path = (project_root / low_level_path).resolve()
    if not low_level_path.is_file():
        raise SystemExit(f"low-level model not found: {low_level_path}")

    run_directory = Path(args.output_root) / args.run_name
    if not run_directory.is_absolute():
        run_directory = project_root / run_directory
    run_directory.mkdir(parents=True, exist_ok=False)
    status_phase = (
        "v11_confirmatory_heldout"
        if args.study_phase == "confirmatory"
        else "v11_calibration_only"
    )
    atomic_json(
        run_directory / "status.json",
        {"status": "initializing", "phase": status_phase},
    )
    thread_count = str(args.torch_threads_per_process)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    modules = require_training_stack()
    configure_cpu_threads(args.torch_threads_per_process)
    if args.device.startswith("cuda") and not modules["torch"].cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")

    config = HierarchicalThermalV11Config(
        condition=args.condition,
        low_level_model_path=str(low_level_path),
        environment_id=args.environment_id,
        episode_steps=args.episode_steps,
        episodes_per_lifetime=args.episodes_per_lifetime,
        canonical_task_seed=args.canonical_task_seed,
        trip_load=args.trip_load,
        low_power_scale=args.low_power_scale,
        trip_penalty=args.trip_penalty,
        high_power_bonus=args.high_power_bonus,
        thermal_heat_rate=args.thermal_heat_rate,
        thermal_episode_cooling=args.thermal_episode_cooling,
        fixed_initial_load=args.fixed_initial_load,
        stochastic_initial_load_low=args.stochastic_initial_load_low,
        stochastic_initial_load_high=args.stochastic_initial_load_high,
        sensor_noise_sd=args.sensor_noise_sd,
        shock_probability=args.shock_probability,
        shock_size=args.shock_size,
        low_level_device="cpu",
    )
    config_document = asdict(config)
    common_factory = {
        "config_document": config_document,
        "torch_threads_per_process": args.torch_threads_per_process,
        "monitor": modules["Monitor"],
    }
    training_factory = partial(
        make_environment, **common_factory, reward_scale=args.training_reward_scale
    )
    evaluation_factory = partial(
        make_environment, **common_factory, reward_scale=1.0
    )
    vector_class = (
        modules["DummyVecEnv"] if args.workers == 1 else modules["SubprocVecEnv"]
    )
    vector_kwargs = {} if args.workers == 1 else {"start_method": "spawn"}
    train_environment = vector_class(
        [training_factory for _ in range(args.workers)], **vector_kwargs
    )
    evaluation_environment = evaluation_factory()
    atomic_json(
        run_directory / "status.json",
        {"status": "training", "phase": status_phase},
    )
    try:
        train_environment.seed(args.seed)
        algorithm_name, policy, policy_kwargs = model_spec(args.policy_arm)
        algorithm = modules[algorithm_name]
        rollout_size = 64 * args.workers
        model = algorithm(
            policy,
            train_environment,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=64,
            batch_size=min(256, rollout_size),
            seed=args.seed,
            device=args.device,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            verbose=1,
            tensorboard_log=str(run_directory / "tensorboard"),
        )
        actual_device = str(model.device)
        if args.device.startswith("cuda") and not actual_device.startswith("cuda"):
            raise RuntimeError(f"CUDA fallback is forbidden; actual device={actual_device}")
        model.learn(total_timesteps=args.total_task_decisions, progress_bar=False)
        model.save(str(run_directory / "model"))
        evaluation_seed = (
            args.evaluation_seed
            if args.evaluation_seed is not None
            else args.seed + 50_000_047
        )
        aggregate, raw_rows = evaluate_with_raw_rows(
            model,
            evaluation_environment,
            task_episodes=args.eval_task_episodes,
            seed=evaluation_seed,
        )
        with (run_directory / "evaluation_tasks.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for row in raw_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        model_path = run_directory / "model.zip"
        metadata = {
            "phase": (
                "hierarchical_thermal_v11_heldout_confirmatory"
                if args.study_phase == "confirmatory"
                else "hierarchical_thermal_v11_calibration"
            ),
            "status": (
                "heldout_confirmatory_cell_complete"
                if args.study_phase == "confirmatory"
                else "calibration_not_confirmatory_evidence"
            ),
            "arguments": vars(args),
            "environment_config": config_document,
            "algorithm": algorithm_name,
            "actual_training_device": actual_device,
            "versions": {
                "torch": modules["torch_version"],
                "stable_baselines3": modules["stable_baselines3_version"],
                "sb3_contrib": modules["sb3_contrib_version"],
            },
            "resolved_low_level_model": str(low_level_path),
            "low_level_model_sha256": sha256(low_level_path),
            "model_sha256": sha256(model_path),
            "evaluation_seed": evaluation_seed,
            "evaluation": aggregate,
            "observation_contract": {
                "summary": [
                    "previous_mode",
                    "noisy_thermal_sensor",
                    "normalized_task_index",
                    "previous_trip",
                ],
                "exact_thermal_load_exposed": False,
                "previous_reward_exposed": False,
                "action_derived_dose_exposed": False,
                "task_index_available_to_all_arms": True,
            },
        }
        atomic_json(run_directory / "metadata.json", metadata)
        atomic_json(
            run_directory / "status.json",
            {
                "status": "complete",
                "phase": status_phase,
                "metadata_sha256": sha256(run_directory / "metadata.json"),
            },
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except BaseException as error:
        atomic_json(
            run_directory / "status.json",
            {
                "status": "failed",
                "phase": status_phase,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        train_environment.close()
        evaluation_environment.close()


if __name__ == "__main__":
    main()
