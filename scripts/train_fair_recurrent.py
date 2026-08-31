#!/usr/bin/env python3
"""Train a recurrent baseline with termination semantics held constant."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from train_recurrent_smoke import health_config, require_recurrent_ppo

from lifephybench.envs.action_history import PreviousAppliedActionObservation
from lifephybench.envs.lifetime import LifetimeStreamWrapper
from lifephybench.envs.mujoco_pusher import PusherActuatorWear
from lifephybench.envs.task_boundary import TaskBoundaryObservation
from lifephybench.envs.thermal_commitment import (
    ThermalCommitmentConfig,
    ThermalModeCommitment,
)
from lifephybench.recurrent_evaluation import evaluate_task_episodes, evaluation_as_dict
from lifephybench.selective_reset_policy import (
    CommitmentModeMaskedMlpLstmPolicy,
    CommitmentModeMaskedTaskResetMlpLstmPolicy,
    TaskResetMlpLstmPolicy,
)


def make_fair_environment(
    *,
    environment_id: str,
    mechanism: str,
    degradation_mode: str,
    episode_steps: int,
    episodes_per_lifetime: int,
    exogenous_dose_per_step: float,
    thermal_exogenous_dose_per_step: float,
    joint_aging_exogenous_dose_per_step: float,
    thermal_heat_rate: float,
    thermal_cooling_rate: float,
    thermal_episode_cooling: float,
    canonical_task_seed: int | None,
    thermal_commitment: bool = False,
    commitment_trip_load: float = 0.10,
    commitment_low_power_scale: float = 0.40,
    commitment_trip_penalty: float = 75.0,
    commitment_high_power_bonus: float = 2.0,
    commitment_control_cost_basis: str = "requested_action",
    commitment_curriculum_start_trip_load: float | None = None,
    commitment_curriculum_lifetimes: int = 0,
    append_previous_applied_action: bool = False,
    monitor,
):
    base = PusherActuatorWear.make(
        health_config(
            mechanism,
            degradation_mode,
            exogenous_dose_per_step=exogenous_dose_per_step,
            thermal_exogenous_dose_per_step=thermal_exogenous_dose_per_step,
            joint_aging_exogenous_dose_per_step=joint_aging_exogenous_dose_per_step,
            thermal_heat_rate=thermal_heat_rate,
            thermal_cooling_rate=thermal_cooling_rate,
            thermal_episode_cooling=thermal_episode_cooling,
            canonical_task_seed=canonical_task_seed,
        ),
        environment_id=environment_id,
        max_episode_steps=episode_steps,
    )
    physical_environment = (
        PreviousAppliedActionObservation(base)
        if append_previous_applied_action
        else base
    )
    environment = (
        ThermalModeCommitment(
            physical_environment,
            ThermalCommitmentConfig(
                trip_load=commitment_trip_load,
                low_power_scale=commitment_low_power_scale,
                trip_penalty=commitment_trip_penalty,
                high_power_throughput_bonus=commitment_high_power_bonus,
                control_cost_basis=commitment_control_cost_basis,
                curriculum_start_trip_load=commitment_curriculum_start_trip_load,
                curriculum_lifetimes=commitment_curriculum_lifetimes,
            ),
        )
        if thermal_commitment
        else physical_environment
    )
    stream = LifetimeStreamWrapper(environment, episodes_per_lifetime)
    return monitor(TaskBoundaryObservation(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-mode", choices=["task", "lifetime"], required=True)
    parser.add_argument("--environment-id", default="Pusher-v5")
    parser.add_argument(
        "--mechanism", choices=["wear", "thermal", "joint_aging"], default="thermal"
    )
    parser.add_argument(
        "--degradation-mode",
        choices=["endogenous_action", "exogenous_clock"],
        required=True,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--episodes-per-lifetime", type=int, default=20)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-task-episodes", type=int, default=1_000)
    parser.add_argument("--exogenous-dose-per-step", type=float, default=0.25)
    parser.add_argument(
        "--thermal-exogenous-dose-per-step", type=float, default=0.008750098932466562
    )
    parser.add_argument(
        "--joint-aging-exogenous-dose-per-step", type=float, default=0.25
    )
    parser.add_argument("--thermal-heat-rate", type=float, default=0.005)
    parser.add_argument("--thermal-cooling-rate", type=float, default=0.01)
    parser.add_argument("--thermal-episode-cooling", type=float, default=0.1)
    parser.add_argument("--canonical-task-seed", type=int, default=None)
    parser.add_argument("--thermal-commitment", action="store_true")
    parser.add_argument("--commitment-trip-load", type=float, default=0.10)
    parser.add_argument("--commitment-low-power-scale", type=float, default=0.40)
    parser.add_argument("--commitment-trip-penalty", type=float, default=75.0)
    parser.add_argument("--commitment-high-power-bonus", type=float, default=2.0)
    parser.add_argument(
        "--commitment-control-cost-basis",
        choices=["applied_action", "requested_action"],
        default="requested_action",
    )
    parser.add_argument(
        "--commitment-curriculum-start-trip-load", type=float, default=None
    )
    parser.add_argument(
        "--commitment-curriculum-lifetimes", type=int, default=0
    )
    parser.add_argument("--commitment-mask-mode-loss", action="store_true")
    parser.add_argument("--append-previous-applied-action", action="store_true")
    parser.add_argument("--output-root", default="outputs/fair_recurrent")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    numeric = [
        args.workers,
        args.episode_steps,
        args.episodes_per_lifetime,
        args.total_timesteps,
        args.eval_task_episodes,
    ]
    if (
        min(numeric) <= 0
        or args.thermal_heat_rate < 0.0
        or not 0.0 <= args.thermal_cooling_rate <= 1.0
        or not 0.0 <= args.thermal_episode_cooling <= 1.0
        or (args.canonical_task_seed is not None and args.canonical_task_seed < 0)
        or not 0.0 < args.commitment_trip_load <= 1.0
        or not 0.0 < args.commitment_low_power_scale < 1.0
        or args.commitment_trip_penalty <= 0.0
        or args.commitment_high_power_bonus <= 0.0
        or args.commitment_curriculum_lifetimes < 0
        or (
            args.commitment_curriculum_lifetimes == 0
            and args.commitment_curriculum_start_trip_load is not None
        )
        or (
            args.commitment_curriculum_lifetimes > 0
            and (
                args.commitment_curriculum_start_trip_load is None
                or not args.commitment_trip_load
                <= args.commitment_curriculum_start_trip_load
                <= 1.0
            )
        )
        or (args.commitment_mask_mode_loss and not args.thermal_commitment)
    ):
        raise SystemExit("positive integer arguments must be greater than zero")

    modules = require_recurrent_ppo()
    run_directory = Path(args.output_root) / args.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    environment_arguments = dict(
        environment_id=args.environment_id,
        mechanism=args.mechanism,
        degradation_mode=args.degradation_mode,
        episode_steps=args.episode_steps,
        episodes_per_lifetime=args.episodes_per_lifetime,
        exogenous_dose_per_step=args.exogenous_dose_per_step,
        thermal_exogenous_dose_per_step=args.thermal_exogenous_dose_per_step,
        joint_aging_exogenous_dose_per_step=args.joint_aging_exogenous_dose_per_step,
        thermal_heat_rate=args.thermal_heat_rate,
        thermal_cooling_rate=args.thermal_cooling_rate,
        thermal_episode_cooling=args.thermal_episode_cooling,
        canonical_task_seed=args.canonical_task_seed,
        thermal_commitment=args.thermal_commitment,
        commitment_trip_load=args.commitment_trip_load,
        commitment_low_power_scale=args.commitment_low_power_scale,
        commitment_trip_penalty=args.commitment_trip_penalty,
        commitment_high_power_bonus=args.commitment_high_power_bonus,
        commitment_control_cost_basis=args.commitment_control_cost_basis,
        append_previous_applied_action=args.append_previous_applied_action,
        monitor=modules["Monitor"],
    )
    training_factory = partial(
        make_fair_environment,
        **environment_arguments,
        commitment_curriculum_start_trip_load=(
            args.commitment_curriculum_start_trip_load
        ),
        commitment_curriculum_lifetimes=args.commitment_curriculum_lifetimes,
    )
    evaluation_factory = partial(
        make_fair_environment,
        **environment_arguments,
        commitment_curriculum_start_trip_load=None,
        commitment_curriculum_lifetimes=0,
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
        if args.commitment_mask_mode_loss:
            policy = (
                CommitmentModeMaskedTaskResetMlpLstmPolicy
                if args.memory_mode == "task"
                else CommitmentModeMaskedMlpLstmPolicy
            )
        else:
            policy = (
                TaskResetMlpLstmPolicy
                if args.memory_mode == "task"
                else "MlpLstmPolicy"
            )
        model = modules["RecurrentPPO"](
            policy,
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
        evaluation_seed = args.seed + 30_000_019
        evaluation = evaluate_task_episodes(
            model, evaluation_environment, args.eval_task_episodes, seed=evaluation_seed
        )
        metadata = {
            "phase": "fair_selective_memory_confirmatory",
            "sb3_contrib_version": modules["version"],
            "arguments": vars(args),
            "evaluation_seed": evaluation_seed,
            "controlled_semantics": {
                "gym_horizon_steps": args.episode_steps * args.episodes_per_lifetime,
                "task_boundary_observed": True,
                "gae_boundary": "lifetime_only",
                "difference_between_memory_arms": "forced_lstm_reset_at_task_boundary",
                "training_trip_load_curriculum_only": (
                    args.commitment_curriculum_lifetimes > 0
                ),
                "evaluation_trip_load": args.commitment_trip_load,
                "commitment_mode_loss_masked_after_decision": (
                    args.commitment_mask_mode_loss
                ),
                "previous_applied_action_observed": (
                    args.append_previous_applied_action
                ),
                "previous_action_zeroed_at_task_boundary": (
                    args.append_previous_applied_action
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
