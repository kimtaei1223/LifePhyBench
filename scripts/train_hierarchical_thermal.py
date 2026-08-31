#!/usr/bin/env python3
"""Train a task-level discrete thermal mode policy over a frozen controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from pathlib import Path

import gymnasium as gym

from train_recurrent_smoke import require_recurrent_ppo

from lifephybench.envs.hierarchical_thermal import (
    HierarchicalThermalConfig,
    HierarchicalThermalModeEnv,
)
from lifephybench.recurrent_evaluation import evaluate_task_episodes, evaluation_as_dict
from lifephybench.selective_reset_policy import TaskResetMlpLstmPolicy


def configure_cpu_threads(threads: int) -> None:
    """Prevent each spawned environment from claiming every CPU core."""
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work starts.
        pass


def make_environment(
    *,
    low_level_model_path: str,
    degradation_mode: str,
    curriculum_start_trip_load: float | None,
    curriculum_lifetimes: int,
    torch_threads_per_process: int,
    reward_scale: float,
    trip_load: float,
    low_power_scale: float,
    trip_penalty: float,
    high_power_bonus: float,
    thermal_heat_rate: float,
    teacher_safe_high_load: float | None,
    teacher_shaping: float,
    summary_mode: str,
    monitor,
):
    configure_cpu_threads(torch_threads_per_process)
    environment = HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path=low_level_model_path,
            degradation_mode=degradation_mode,
            trip_load=trip_load,
            low_power_scale=low_power_scale,
            trip_penalty=trip_penalty,
            high_power_bonus=high_power_bonus,
            thermal_heat_rate=thermal_heat_rate,
            curriculum_start_trip_load=curriculum_start_trip_load,
            curriculum_lifetimes=curriculum_lifetimes,
            training_teacher_safe_high_load=teacher_safe_high_load,
            training_teacher_shaping=teacher_shaping,
            summary_mode=summary_mode,
            low_level_device="cpu",
        )
    )
    if reward_scale != 1.0:
        environment = gym.wrappers.TransformReward(
            environment, lambda reward: reward * reward_scale
        )
    return monitor(environment)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-mode", choices=["task", "lifetime"], required=True)
    parser.add_argument(
        "--degradation-mode",
        choices=["endogenous_action", "exogenous_clock"],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-task-decisions", type=int, default=10_000)
    parser.add_argument("--eval-task-episodes", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads-per-process", type=int, default=1)
    parser.add_argument("--training-reward-scale", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--trip-load", type=float, default=0.10)
    parser.add_argument("--low-power-scale", type=float, default=0.40)
    parser.add_argument("--trip-penalty", type=float, default=75.0)
    parser.add_argument("--high-power-bonus", type=float, default=2.0)
    parser.add_argument("--thermal-heat-rate", type=float, default=0.10)
    parser.add_argument("--teacher-safe-high-load", type=float, default=None)
    parser.add_argument("--teacher-shaping", type=float, default=0.0)
    parser.add_argument("--summary-mode", choices=["full", "mode_trip"], default="full")
    parser.add_argument(
        "--low-level-model",
        default=(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument("--curriculum-start-trip-load", type=float, default=None)
    parser.add_argument("--curriculum-lifetimes", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/hierarchical_thermal")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if min(
        args.workers,
        args.total_task_decisions,
        args.eval_task_episodes,
        args.seed,
        args.torch_threads_per_process,
    ) <= 0:
        raise SystemExit("budgets and seed must be positive")
    if args.curriculum_lifetimes < 0:
        raise SystemExit("curriculum_lifetimes must be non-negative")
    if args.training_reward_scale <= 0.0:
        raise SystemExit("training_reward_scale must be positive")
    if args.ent_coef < 0.0:
        raise SystemExit("ent_coef must be non-negative")
    if not 0.0 < args.trip_load <= 1.0:
        raise SystemExit("trip_load must be in (0, 1]")
    if not 0.0 < args.low_power_scale < 1.0:
        raise SystemExit("low_power_scale must be in (0, 1)")
    if min(args.trip_penalty, args.high_power_bonus, args.thermal_heat_rate) <= 0.0:
        raise SystemExit("physical design parameters must be positive")
    if args.teacher_shaping < 0.0:
        raise SystemExit("teacher_shaping must be non-negative")
    if args.teacher_shaping > 0.0 and (
        args.teacher_safe_high_load is None
        or not 0.0 <= args.teacher_safe_high_load <= args.trip_load
    ):
        raise SystemExit("teacher shaping requires safe load in [0, trip_load]")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("gamma and gae_lambda must be in (0, 1]")
    if (args.curriculum_lifetimes == 0) != (
        args.curriculum_start_trip_load is None
    ):
        raise SystemExit("curriculum start and duration must be enabled together")
    if (
        args.curriculum_start_trip_load is not None
        and not args.trip_load <= args.curriculum_start_trip_load <= 1.0
    ):
        raise SystemExit("curriculum start must be in [0.10, 1.0]")

    project_root = Path(__file__).resolve().parent.parent
    low_level_path = Path(args.low_level_model)
    if not low_level_path.is_absolute():
        low_level_path = (project_root / low_level_path).resolve()
    if not low_level_path.exists():
        raise SystemExit(f"low-level model not found: {low_level_path}")

    thread_count = str(args.torch_threads_per_process)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    configure_cpu_threads(args.torch_threads_per_process)
    run_directory = Path(args.output_root) / args.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    modules = require_recurrent_ppo()
    common = dict(
        low_level_model_path=str(low_level_path),
        degradation_mode=args.degradation_mode,
        torch_threads_per_process=args.torch_threads_per_process,
        trip_load=args.trip_load,
        low_power_scale=args.low_power_scale,
        trip_penalty=args.trip_penalty,
        high_power_bonus=args.high_power_bonus,
        thermal_heat_rate=args.thermal_heat_rate,
        summary_mode=args.summary_mode,
        monitor=modules["Monitor"],
    )
    training_factory = partial(
        make_environment,
        **common,
        curriculum_start_trip_load=args.curriculum_start_trip_load,
        curriculum_lifetimes=args.curriculum_lifetimes,
        reward_scale=args.training_reward_scale,
        teacher_safe_high_load=args.teacher_safe_high_load,
        teacher_shaping=args.teacher_shaping,
    )
    evaluation_factory = partial(
        make_environment,
        **common,
        curriculum_start_trip_load=None,
        curriculum_lifetimes=0,
        reward_scale=1.0,
        teacher_safe_high_load=None,
        teacher_shaping=0.0,
    )
    vector_class = (
        modules["DummyVecEnv"] if args.workers == 1 else modules["SubprocVecEnv"]
    )
    vector_kwargs = {} if args.workers == 1 else {"start_method": "spawn"}
    train_environment = vector_class(
        [training_factory for _ in range(args.workers)], **vector_kwargs
    )
    evaluation_environment = evaluation_factory()
    try:
        train_environment.seed(args.seed)
        policy = (
            TaskResetMlpLstmPolicy
            if args.memory_mode == "task"
            else "MlpLstmPolicy"
        )
        rollout_size = 32 * args.workers
        model = modules["RecurrentPPO"](
            policy,
            train_environment,
            learning_rate=args.learning_rate,
            n_steps=32,
            batch_size=min(256, rollout_size),
            seed=args.seed,
            device=args.device,
            ent_coef=args.ent_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            verbose=1,
            tensorboard_log=str(run_directory / "tensorboard"),
        )
        model.learn(total_timesteps=args.total_task_decisions, progress_bar=False)
        model.save(str(run_directory / "model"))
        evaluation_seed = args.seed + 40_000_031
        evaluation = evaluate_task_episodes(
            model,
            evaluation_environment,
            args.eval_task_episodes,
            seed=evaluation_seed,
        )
        metadata = {
            "phase": "hierarchical_thermal_mode_calibration",
            "status": "calibration_not_confirmatory_evidence",
            "sb3_contrib_version": modules["version"],
            "arguments": vars(args),
            "resolved_low_level_model": str(low_level_path),
            "low_level_model_sha256": sha256(low_level_path),
            "evaluation_seed": evaluation_seed,
            "controlled_semantics": {
                "high_level_action_space": "Discrete(2): low/high",
                "one_high_level_action_per_physical_task": True,
                "physical_steps_per_nominal_task": 100,
                "tasks_per_lifetime": 20,
                "task_boundary_observed": True,
                "difference_between_memory_arms": (
                    "forced_lstm_reset_at_each_high_level_task"
                ),
                "low_level_controller_frozen": True,
                "torch_threads_per_process": args.torch_threads_per_process,
                "evaluation_reward_unscaled": True,
                "privileged_health_exposed": False,
                "summary": (
                    "previous mode, action-derived thermal increment, normalized "
                    "return, observed trip"
                    if args.summary_mode == "full"
                    else "previous mode and observed trip only; middle coordinates zero"
                ),
                "summary_mode": args.summary_mode,
                "evaluation_trip_load": args.trip_load,
                "physical_design": {
                    "trip_load": args.trip_load,
                    "low_power_scale": args.low_power_scale,
                    "trip_penalty": args.trip_penalty,
                    "high_power_bonus": args.high_power_bonus,
                    "thermal_heat_rate": args.thermal_heat_rate,
                },
                "teacher_shaping_training_only": args.teacher_shaping > 0.0,
                "training_trip_load_curriculum_only": (
                    args.curriculum_lifetimes > 0
                ),
            },
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
