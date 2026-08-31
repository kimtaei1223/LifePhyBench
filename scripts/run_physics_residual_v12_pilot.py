#!/usr/bin/env python3
"""Train and evaluate the v12 physics-plus-learned-residual estimator.

The old v11 held-out rows are development data after the completed v11 study.
This pilot keeps a seed-level train/validation/test split, trains a supervised
GRU residual on top of the identified thermal transition, and evaluates a
non-privileged hybrid supervisor on fresh in-domain and OOD development seeds.
Nothing produced by this script is confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

try:
    from scripts.qualify_hierarchical_v11 import (
        QualificationDesign,
        evaluate_policy,
        make_default_environment_factory,
    )
    from scripts.qualify_physics_belief_v12 import (
        CurrentSensorPolicy,
        EmaPolicy,
        PhysicsBeliefPolicy,
        PrivilegedOraclePolicy,
        TransitionModel,
        _iter_stochastic_runs,
        fit_transition_model,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from qualify_hierarchical_v11 import (  # type: ignore[no-redef]
        QualificationDesign,
        evaluate_policy,
        make_default_environment_factory,
    )
    from qualify_physics_belief_v12 import (  # type: ignore[no-redef]
        CurrentSensorPolicy,
        EmaPolicy,
        PhysicsBeliefPolicy,
        PrivilegedOraclePolicy,
        TransitionModel,
        _iter_stochastic_runs,
        fit_transition_model,
    )


FEATURE_NAMES = (
    "sensor",
    "previous_action",
    "has_previous_action",
    "normalized_task_index",
    "physics_belief_mean",
    "physics_belief_std",
)


class ResidualGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, next_hidden = self.gru(features, hidden)
        return self.head(sequence), next_hidden


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_torch_save(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(document, temporary_name)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _feature(
    *,
    sensor: float,
    previous_action: int | None,
    task_index: int,
    physics_mean: float,
    physics_variance: float,
) -> list[float]:
    return [
        float(sensor),
        0.0 if previous_action is None else float(previous_action),
        float(previous_action is not None),
        float(task_index / 19.0),
        float(physics_mean),
        float(math.sqrt(max(0.0, physics_variance))),
    ]


def build_sequences(
    runs: list[tuple[int, list[dict[str, Any]]]],
    seeds: set[int],
    transition: TransitionModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[list[float]]] = []
    residuals: list[list[float]] = []
    exact_loads: list[list[float]] = []
    for seed, rows in runs:
        if seed not in seeds:
            continue
        lifetimes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            lifetimes[int(row["lifetime_ordinal"])].append(row)
        for lifetime_rows in lifetimes.values():
            ordered = sorted(lifetime_rows, key=lambda value: int(value["task_index"]))
            if len(ordered) != 20:
                raise ValueError("v12 residual training requires complete 20-task lifetimes")
            physics = PhysicsBeliefPolicy(
                transition, cutoff=1.0, uncertainty_multiplier=0.0
            )
            lifetime_features: list[list[float]] = []
            lifetime_targets: list[float] = []
            lifetime_exact: list[float] = []
            for row in ordered:
                task_index = int(row["task_index"])
                sensor = float(row["sensor_load"])
                previous_action = physics.previous_action
                mean, variance = physics.update(task_index=task_index, sensor=sensor)
                exact = float(row["true_load_at_selection"])
                lifetime_features.append(
                    _feature(
                        sensor=sensor,
                        previous_action=previous_action,
                        task_index=task_index,
                        physics_mean=mean,
                        physics_variance=variance,
                    )
                )
                lifetime_targets.append(exact - mean)
                lifetime_exact.append(exact)
                physics.previous_action = int(row["action"])
            features.append(lifetime_features)
            residuals.append(lifetime_targets)
            exact_loads.append(lifetime_exact)
    if not features:
        raise ValueError("no residual sequences matched the requested seeds")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        np.asarray(exact_loads, dtype=np.float32),
    )


def _predict_sequences(
    model: ResidualGRU,
    features: np.ndarray,
    *,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    target_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                (features[start : start + batch_size] - input_mean) / input_std,
                device=device,
            )
            output, _hidden = model(batch)
            normalized_mean = output[..., 0]
            normalized_log_variance = output[..., 1].clamp(-7.0, 4.0)
            means.append((normalized_mean * target_std).cpu().numpy())
            stds.append(
                (
                    torch.exp(0.5 * normalized_log_variance) * target_std
                ).cpu().numpy()
            )
    return np.concatenate(means), np.concatenate(stds)


def train_residual(
    *,
    runs: list[tuple[int, list[dict[str, Any]]]],
    transition: TransitionModel,
    train_seeds: set[int],
    validation_seeds: set[int],
    test_seeds: set[int],
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    rng_seed: int,
    mean_loss_weight: float = 0.05,
    nll_loss_weight: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(rng_seed)
    train_x, train_y, _train_exact = build_sequences(runs, train_seeds, transition)
    validation_x, validation_y, _validation_exact = build_sequences(
        runs, validation_seeds, transition
    )
    test_x, test_y, test_exact = build_sequences(runs, test_seeds, transition)
    input_mean = train_x.mean(axis=(0, 1), keepdims=True)
    input_std = train_x.std(axis=(0, 1), keepdims=True)
    input_std = np.maximum(input_std, 1e-6)
    target_std = max(float(train_y.std()), 1e-6)
    train_x_normalized = (train_x - input_mean) / input_std
    train_y_normalized = train_y / target_std
    validation_x_normalized = (validation_x - input_mean) / input_std
    validation_y_normalized = validation_y / target_std

    model = ResidualGRU(train_x.shape[-1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=1e-5
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_rmse = math.inf
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    batch_size = min(512, len(train_x))
    generator = np.random.default_rng(rng_seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(train_x))
        train_loss_sum = 0.0
        examples = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = torch.as_tensor(train_x_normalized[indices], device=device)
            batch_y = torch.as_tensor(train_y_normalized[indices], device=device)
            prediction, _hidden = model(batch_x)
            mean = prediction[..., 0]
            log_variance = prediction[..., 1].clamp(-7.0, 4.0)
            error = batch_y - mean
            nll = 0.5 * (error.square() * torch.exp(-log_variance) + log_variance)
            loss = (
                nll_loss_weight * nll.mean()
                + mean_loss_weight * error.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(indices)
            examples += len(indices)

        model.eval()
        with torch.no_grad():
            validation_input = torch.as_tensor(
                validation_x_normalized, device=device
            )
            validation_prediction, _hidden = model(validation_input)
            validation_residual = (
                validation_prediction[..., 0].cpu().numpy() * target_std
            )
        validation_rmse = float(
            np.sqrt(np.mean((validation_residual - validation_y) ** 2))
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / examples,
                "validation_residual_rmse": validation_rmse,
            }
        )
        if validation_rmse < best_validation_rmse - 1e-8:
            best_validation_rmse = validation_rmse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 15:
                break
    if best_state is None:
        raise RuntimeError("residual training produced no checkpoint")
    model.load_state_dict(best_state)
    residual_mean, residual_std = _predict_sequences(
        model,
        test_x,
        input_mean=input_mean,
        input_std=input_std,
        target_std=target_std,
        device=device,
    )
    physics_mean = test_x[..., FEATURE_NAMES.index("physics_belief_mean")]
    raw_sensor = test_x[..., FEATURE_NAMES.index("sensor")]
    hybrid_mean = physics_mean + residual_mean

    def rmse(prediction: np.ndarray) -> float:
        return float(np.sqrt(np.mean((prediction - test_exact) ** 2)))

    physics_rmse = rmse(physics_mean)
    hybrid_rmse = rmse(hybrid_mean)
    coverage = float(
        np.mean(np.abs(hybrid_mean - test_exact) <= 1.96 * residual_std)
    )
    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": int(train_x.shape[-1]),
        "hidden_dim": hidden_dim,
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "target_std": target_std,
        "transition_model": asdict(transition),
        "feature_names": FEATURE_NAMES,
        "rng_seed": rng_seed,
        "mean_loss_weight": mean_loss_weight,
        "nll_loss_weight": nll_loss_weight,
    }
    report = {
        "actual_training_device": str(device),
        "mean_loss_weight": mean_loss_weight,
        "nll_loss_weight": nll_loss_weight,
        "train_sequences": len(train_x),
        "validation_sequences": len(validation_x),
        "test_sequences": len(test_x),
        "epochs_completed": len(history),
        "best_validation_residual_rmse": best_validation_rmse,
        "test": {
            "raw_sensor_rmse": rmse(raw_sensor),
            "physics_belief_rmse": physics_rmse,
            "hybrid_belief_rmse": hybrid_rmse,
            "hybrid_improvement_over_physics_fraction": (
                1.0 - hybrid_rmse / physics_rmse
            ),
            "nominal_95_interval_coverage": coverage,
        },
        "history": history,
    }
    return checkpoint, report


def load_residual_checkpoint(
    path: Path, device: torch.device
) -> tuple[ResidualGRU, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ResidualGRU(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


class HybridBeliefPolicy:
    def __init__(
        self,
        model: ResidualGRU,
        checkpoint: dict[str, Any],
        *,
        residual_scale: float,
        cutoff: float,
        uncertainty_multiplier: float,
    ) -> None:
        self.model = model
        self.transition = TransitionModel(**checkpoint["transition_model"])
        self.physics = PhysicsBeliefPolicy(
            self.transition, cutoff=1.0, uncertainty_multiplier=0.0
        )
        self.input_mean = torch.as_tensor(checkpoint["input_mean"])
        self.input_std = torch.as_tensor(checkpoint["input_std"])
        self.target_std = float(checkpoint["target_std"])
        self.residual_scale = float(residual_scale)
        self.cutoff = float(cutoff)
        self.uncertainty_multiplier = float(uncertainty_multiplier)
        self.hidden: torch.Tensor | None = None
        self.last_physics_mean: float | None = None
        self.last_corrected_mean: float | None = None
        self.last_total_std: float | None = None
        self.last_residual: float | None = None

    def act(self, *, task_index: int, sensor: float, exact_load: float) -> int:
        del exact_load
        previous_action = self.physics.previous_action
        mean, variance = self.physics.update(task_index=task_index, sensor=sensor)
        features = torch.as_tensor(
            _feature(
                sensor=sensor,
                previous_action=previous_action,
                task_index=task_index,
                physics_mean=mean,
                physics_variance=variance,
            )
        ).reshape(1, 1, -1)
        normalized = (features - self.input_mean) / self.input_std
        with torch.no_grad():
            output, self.hidden = self.model(normalized, self.hidden)
        residual = float(output[0, 0, 0]) * self.target_std
        residual_std = math.exp(0.5 * float(output[0, 0, 1].clamp(-7.0, 4.0))) * self.target_std
        corrected = mean + self.residual_scale * float(np.clip(residual, -0.03, 0.03))
        total_std = math.sqrt(max(0.0, variance) + residual_std**2)
        self.last_physics_mean = float(mean)
        self.last_corrected_mean = float(corrected)
        self.last_total_std = float(total_std)
        self.last_residual = float(residual)
        action = int(
            corrected + self.uncertainty_multiplier * total_std < self.cutoff
        )
        self.physics.previous_action = action
        return action


_WORKER_FACTORY: Callable[[QualificationDesign, int], Any] | None = None
_WORKER_MODEL: ResidualGRU | None = None
_WORKER_CHECKPOINT: dict[str, Any] | None = None


def _initialize_evaluation_worker(
    low_level_model_path: str, residual_checkpoint_path: str
) -> None:
    global _WORKER_CHECKPOINT, _WORKER_FACTORY, _WORKER_MODEL
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _WORKER_FACTORY = make_default_environment_factory(Path(low_level_model_path))
    _WORKER_MODEL, _WORKER_CHECKPOINT = load_residual_checkpoint(
        Path(residual_checkpoint_path), torch.device("cpu")
    )


def _evaluation_policy_factory(spec: dict[str, Any]) -> Callable[[], Any]:
    if _WORKER_MODEL is None or _WORKER_CHECKPOINT is None:
        raise RuntimeError("residual evaluation worker was not initialized")
    policy_type = spec["type"]
    if policy_type == "current_sensor":
        return lambda: CurrentSensorPolicy(float(spec["cutoff"]))
    if policy_type == "ema":
        return lambda: EmaPolicy(float(spec["alpha"]), float(spec["cutoff"]))
    if policy_type == "physics_belief":
        transition = TransitionModel(**_WORKER_CHECKPOINT["transition_model"])
        return lambda: PhysicsBeliefPolicy(
            transition,
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    if policy_type == "hybrid_belief":
        return lambda: HybridBeliefPolicy(
            _WORKER_MODEL,
            _WORKER_CHECKPOINT,
            residual_scale=float(spec["residual_scale"]),
            cutoff=float(spec["cutoff"]),
            uncertainty_multiplier=float(spec["uncertainty_multiplier"]),
        )
    if policy_type == "privileged_oracle":
        return lambda: PrivilegedOraclePolicy(float(spec["cutoff"]))
    raise ValueError(f"unknown pilot policy type: {policy_type}")


def _evaluate_job(
    job: tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]
) -> tuple[str, str, dict[str, Any]]:
    if _WORKER_FACTORY is None:
        raise RuntimeError("residual evaluation worker was not initialized")
    condition_name, spec, design_document, seeds = job
    design = QualificationDesign(**design_document)
    summary = evaluate_policy(
        design,
        _WORKER_FACTORY,
        _evaluation_policy_factory(spec),
        seeds=seeds,
        tasks_per_lifetime=20,
        require_physical_rollouts=True,
    )
    return condition_name, str(spec["name"]), {"spec": spec, "summary": summary}


def evaluate_jobs(
    jobs: list[tuple[str, dict[str, Any], dict[str, float], tuple[int, ...]]],
    *,
    workers: int,
    low_level_model: Path,
    checkpoint: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_initialize_evaluation_worker,
        initargs=(str(low_level_model.resolve()), str(checkpoint.resolve())),
    ) as pool:
        rows = pool.map(_evaluate_job, jobs)
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for condition, policy, summary in rows:
        result[condition][policy] = summary
    return dict(result)


def _designs() -> dict[str, QualificationDesign]:
    return {
        "in_domain": QualificationDesign(0.15, 0.01, 5e-4, 0.01),
        "ood_sensor_noise": QualificationDesign(0.15, 0.03, 5e-4, 0.01),
        "ood_cooling": QualificationDesign(0.10, 0.01, 5e-4, 0.01),
        "ood_shocks": QualificationDesign(0.15, 0.01, 1e-3, 0.02),
        "ood_combined": QualificationDesign(0.10, 0.03, 1e-3, 0.02),
    }


def summarize_pilot(
    validation: dict[str, dict[str, dict[str, Any]]],
    offline: dict[str, Any],
) -> dict[str, Any]:
    condition_rows: dict[str, Any] = {}
    ood_wins = 0
    for condition, policies in validation.items():
        summaries = {name: row["summary"] for name, row in policies.items()}
        hybrid = summaries["hybrid_belief"]
        physics = summaries["physics_belief"]
        best_nonprivileged = max(
            summaries[name]["mean_reward_per_task"]
            for name in ("current_sensor", "ema_history", "physics_belief")
        )
        row = {
            "hybrid_minus_physics_reward": (
                hybrid["mean_reward_per_task"] - physics["mean_reward_per_task"]
            ),
            "hybrid_minus_best_nonprivileged_reward": (
                hybrid["mean_reward_per_task"] - best_nonprivileged
            ),
            "hybrid_minus_physics_trip_rate": (
                hybrid["trip_rate"] - physics["trip_rate"]
            ),
            "hybrid_trip_rate": hybrid["trip_rate"],
        }
        if condition != "in_domain" and row["hybrid_minus_physics_reward"] >= 0.25:
            ood_wins += 1
        condition_rows[condition] = row
    criteria = {
        "offline_hybrid_rmse_improves_physics_by_0_05": (
            offline["test"]["hybrid_improvement_over_physics_fraction"] >= 0.05
        ),
        "in_domain_reward_noninferior_within_0_25": (
            condition_rows["in_domain"]["hybrid_minus_physics_reward"] >= -0.25
        ),
        "in_domain_trip_rate_at_most_0_02": (
            condition_rows["in_domain"]["hybrid_trip_rate"] <= 0.02
        ),
        "at_least_two_ood_reward_wins_of_0_25": ood_wins >= 2,
        "all_ood_hybrid_trip_rates_at_most_0_02": all(
            row["hybrid_trip_rate"] <= 0.02
            for name, row in condition_rows.items()
            if name != "in_domain"
        ),
    }
    return {
        "conditions": condition_rows,
        "ood_reward_wins_at_least_0_25": ood_wins,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v11-input-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/confirmatory"),
    )
    parser.add_argument(
        "--low-level-model",
        type=Path,
        default=Path(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/physics_residual_v12_pilot")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rng-seed", type=int, default=12012)
    parser.add_argument("--mean-loss-weight", type=float, default=0.05)
    parser.add_argument("--nll-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--controller-selection-seeds",
        type=int,
        nargs="+",
        default=list(range(9400, 9405)),
    )
    parser.add_argument(
        "--controller-validation-seeds",
        type=int,
        nargs="+",
        default=list(range(9410, 9420)),
    )
    parser.add_argument("--selection-max-trip-rate", type=float, default=0.02)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.hidden_dim <= 0 or args.epochs <= 0:
        raise SystemExit("workers, hidden dimension, and epochs must be positive")
    if args.mean_loss_weight <= 0.0 or args.nll_loss_weight < 0.0:
        raise SystemExit("mean loss must be positive and NLL loss non-negative")
    selection_seeds = tuple(args.controller_selection_seeds)
    validation_pilot_seeds = tuple(args.controller_validation_seeds)
    if (
        not selection_seeds
        or not validation_pilot_seeds
        or set(selection_seeds).intersection(validation_pilot_seeds)
        or len(set(selection_seeds)) != len(selection_seeds)
        or len(set(validation_pilot_seeds)) != len(validation_pilot_seeds)
    ):
        raise SystemExit("controller selection and validation seeds must be unique and disjoint")
    if not 0.0 <= args.selection_max_trip_rate <= 1.0:
        raise SystemExit("selection maximum trip rate must be in [0, 1]")
    output_root = args.output_root.resolve()
    final_path = output_root / "PILOT_RESULTS.json"
    checkpoint_path = output_root / "residual_model.pt"
    training_path = output_root / "TRAINING_RESULTS.json"
    status_path = output_root / "status.json"
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root exists; use --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        print(final_path.read_text(encoding="utf-8"))
        return
    atomic_json(
        status_path,
        {"status": "initializing", "phase": "physics_residual_v12_development_pilot"},
    )
    runs = _iter_stochastic_runs(args.v11_input_root)
    train_seeds = set(range(8300, 8315))
    validation_seeds = set(range(8315, 8322))
    test_seeds = set(range(8322, 8330))
    transition = fit_transition_model(runs, train_seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    if checkpoint_path.exists() and training_path.exists() and args.resume:
        training_report = json.loads(training_path.read_text(encoding="utf-8"))
    else:
        atomic_json(
            status_path,
            {"status": "training_residual", "phase": "physics_residual_v12_development_pilot"},
        )
        checkpoint, training_report = train_residual(
            runs=runs,
            transition=transition,
            train_seeds=train_seeds,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            device=device,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            rng_seed=args.rng_seed,
            mean_loss_weight=args.mean_loss_weight,
            nll_loss_weight=args.nll_loss_weight,
        )
        atomic_torch_save(checkpoint_path, checkpoint)
        atomic_json(training_path, training_report)

    designs = _designs()
    physics_spec = {
        "name": "physics_belief",
        "type": "physics_belief",
        "cutoff": 0.06,
        "uncertainty_multiplier": 0.0,
    }
    hybrid_specs = [
        {
            "name": f"hybrid-c{cutoff:.3f}-s{scale:.2f}-z{z:.1f}",
            "type": "hybrid_belief",
            "cutoff": cutoff,
            "residual_scale": scale,
            "uncertainty_multiplier": z,
        }
        for cutoff in (0.050, 0.055, 0.060, 0.065)
        for scale in (0.50, 1.00, 1.50)
        for z in (0.0, 0.5, 1.0, 1.5, 2.0)
    ]
    selection_jobs = []
    for condition, design in designs.items():
        selection_jobs.append(
            (condition, physics_spec, asdict(design), selection_seeds)
        )
        selection_jobs.extend(
            (condition, spec, asdict(design), selection_seeds)
            for spec in hybrid_specs
        )
    atomic_json(
        status_path,
        {"status": "selecting_hybrid_controller", "phase": "physics_residual_v12_development_pilot"},
    )
    selection = evaluate_jobs(
        selection_jobs,
        workers=args.workers,
        low_level_model=args.low_level_model,
        checkpoint=checkpoint_path,
    )

    def selection_key(spec: dict[str, Any]) -> tuple[float, float, float]:
        advantages = []
        trip_rates = []
        for condition in designs:
            hybrid = selection[condition][spec["name"]]["summary"]
            physics = selection[condition]["physics_belief"]["summary"]
            advantages.append(
                hybrid["mean_reward_per_task"] - physics["mean_reward_per_task"]
            )
            trip_rates.append(hybrid["trip_rate"])
        admissible = float(max(trip_rates) <= args.selection_max_trip_rate)
        return admissible, float(np.mean(advantages)), float(min(advantages))

    selected_hybrid = max(hybrid_specs, key=selection_key)
    validation_specs = [
        {"name": "current_sensor", "type": "current_sensor", "cutoff": 0.055},
        {
            "name": "ema_history",
            "type": "ema",
            "alpha": 0.60,
            "cutoff": 0.060,
        },
        physics_spec,
        {**selected_hybrid, "name": "hybrid_belief"},
        {
            "name": "privileged_oracle",
            "type": "privileged_oracle",
            "cutoff": 0.060,
        },
    ]
    validation_jobs = [
        (condition, spec, asdict(design), validation_pilot_seeds)
        for condition, design in designs.items()
        for spec in validation_specs
    ]
    atomic_json(
        status_path,
        {"status": "validating_in_domain_and_ood", "phase": "physics_residual_v12_development_pilot"},
    )
    validation = evaluate_jobs(
        validation_jobs,
        workers=args.workers,
        low_level_model=args.low_level_model,
        checkpoint=checkpoint_path,
    )
    gate = summarize_pilot(validation, training_report)
    report = {
        "phase": "physics_residual_v12_development_pilot",
        "status": "pilot_gate_passed" if gate["passed"] else "pilot_gate_failed",
        "confirmatory_evidence": False,
        "v11_results_reclassified_as_development_only": True,
        "actual_training_device": training_report["actual_training_device"],
        "seed_split": {
            "residual_train": sorted(train_seeds),
            "residual_validation": sorted(validation_seeds),
            "residual_test": sorted(test_seeds),
            "controller_selection": list(selection_seeds),
            "controller_validation": list(validation_pilot_seeds),
        },
        "transition_model": asdict(transition),
        "training": training_report,
        "designs": {name: asdict(design) for name, design in designs.items()},
        "selected_hybrid_spec": selected_hybrid,
        "selection_max_trip_rate": args.selection_max_trip_rate,
        "selection_results": selection,
        "validation_results": validation,
        "gate": gate,
    }
    atomic_json(final_path, report)
    atomic_json(
        status_path,
        {
            "status": "complete",
            "phase": "physics_residual_v12_development_pilot",
            "passed": gate["passed"],
        },
    )
    print(
        json.dumps(
            {
                "passed": gate["passed"],
                "selected_hybrid_spec": selected_hybrid,
                "offline_test": training_report["test"],
                "gate": gate,
                "output": str(final_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
