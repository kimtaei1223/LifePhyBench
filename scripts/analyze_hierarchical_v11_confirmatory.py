#!/usr/bin/env python3
"""Analyze the frozen v11 held-out campaign at the paired-seed level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("fixed", "stochastic")


class ConfirmatoryAnalysisError(ValueError):
    """Raised when frozen evidence is incomplete or fails a wiring audit."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_new(path: Path, document: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite confirmatory result: {path}")
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite confirmatory result: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def bootstrap_mean_ci(
    values: list[float], *, resamples: int, seed: int
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ConfirmatoryAnalysisError("bootstrap values must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        count = min(10_000, remaining)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        chunks.append(array[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(chunks)
    return [float(value) for value in np.quantile(distribution, [0.025, 0.975])]


def monte_carlo_sign_flip_p(
    values: list[float], *, draws: int, seed: int
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ConfirmatoryAnalysisError("sign-flip values must be a finite nonempty vector")
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 8.0
    while remaining:
        count = min(50_000, remaining)
        signs = rng.integers(0, 2, size=(count, len(array)), dtype=np.int8)
        signs = signs * 2 - 1
        randomized = np.abs((signs * array).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - tolerance))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def exact_two_sided_sign_p(values: list[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    nonzero = positive + negative
    if nonzero == 0:
        p_value = 1.0
    else:
        tail = min(positive, negative)
        probability = sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2**nonzero)
        p_value = min(1.0, 2.0 * probability)
    return {
        "positive": positive,
        "negative": negative,
        "zeros_discarded": len(values) - nonzero,
        "nonzero_n": nonzero,
        "two_sided_p": float(p_value),
    }


def summarize_estimand(
    values: list[float],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    sign_flip_draws: int,
    sign_flip_seed: int,
    criteria: dict[str, float],
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    interval = bootstrap_mean_ci(
        values, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    sign_flip_p = monte_carlo_sign_flip_p(
        values, draws=sign_flip_draws, seed=sign_flip_seed
    )
    mean = float(array.mean())
    passed = {
        "mean_reward_per_task_at_least": mean
        >= criteria["mean_reward_per_task_at_least"],
        "paired_seed_bootstrap_95_ci_lower_above": interval[0]
        > criteria["paired_seed_bootstrap_95_ci_lower_above"],
        "monte_carlo_two_sided_sign_flip_p_below": sign_flip_p
        < criteria["monte_carlo_two_sided_sign_flip_p_below"],
    }
    return {
        "n": len(values),
        "values": values,
        "mean": mean,
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "paired_seed_bootstrap_95_ci": interval,
        "monte_carlo_two_sided_sign_flip_p": sign_flip_p,
        "exact_two_sided_sign_test": exact_two_sided_sign_p(values),
        "criteria_passed": passed,
        "estimand_passed": all(passed.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmatoryAnalysisError(f"cannot read JSON: {path}") from error
    if not isinstance(document, dict):
        raise ConfirmatoryAnalysisError(f"JSON root is not an object: {path}")
    return document


def _read_raw_mean(
    path: Path, *, expected_rows: int, condition: str, evaluation_seed: int
) -> tuple[float, str]:
    rewards: list[float] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                reward = float(row.get("reward"))
                if (
                    not math.isfinite(reward)
                    or row.get("condition") != condition
                    or row.get("evaluation_seed") != evaluation_seed
                ):
                    raise ConfirmatoryAnalysisError(f"raw wiring mismatch: {path}")
                rewards.append(reward)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ConfirmatoryAnalysisError):
            raise
        raise ConfirmatoryAnalysisError(f"cannot read raw evaluation: {path}") from error
    if len(rewards) != expected_rows:
        raise ConfirmatoryAnalysisError(
            f"expected {expected_rows} raw rows, found {len(rewards)}: {path}"
        )
    return float(np.mean(np.asarray(rewards, dtype=np.float64))), sha256(path)


def analyze_confirmatory(
    *, input_root: Path, protocol_path: Path, expected_protocol_sha256: str
) -> dict[str, Any]:
    protocol_digest = sha256(protocol_path)
    if protocol_digest != expected_protocol_sha256.lower():
        raise ConfirmatoryAnalysisError("protocol file SHA-256 mismatch")
    protocol = _read_json(protocol_path)
    manifest = _read_json(input_root / "manifest.json")
    progress = _read_json(input_root / "progress.json")
    complete = _read_json(input_root / "CAMPAIGN_COMPLETE.json")
    if progress.get("complete") is not True or progress.get("remaining_cells") != []:
        raise ConfirmatoryAnalysisError("held-out campaign is not complete")
    if any(
        document.get("protocol_sha256") != protocol_digest
        for document in (manifest, complete)
    ):
        raise ConfirmatoryAnalysisError("campaign/protocol hash mismatch")

    budgets = protocol["budgets"]
    expected_rows = budgets["evaluation_task_episodes"]
    decisions = budgets["total_task_decisions_per_run"]
    seeds = protocol["seed_namespaces"]["heldout"]["training_pair_seeds"]
    evaluation_seeds = protocol["seed_namespaces"]["heldout"]["evaluation_bank_seeds"]
    reactive_arm = protocol["arms"]["task_reactive"]["identity"]
    arms = ("lifetime_lstm", reactive_arm)
    if manifest.get("training_seeds") != seeds or manifest.get("evaluation_seeds") != evaluation_seeds:
        raise ConfirmatoryAnalysisError("manifest held-out seed mismatch")
    expected_cell_names = [
        f"v11-heldout-{condition}-{arm}-seed{seed}-decisions{decisions // 1000}k"
        for seed in seeds
        for condition in CONDITIONS
        for arm in arms
    ]
    if progress.get("completed_cells") != expected_cell_names:
        raise ConfirmatoryAnalysisError("completed cell ordering/set mismatch")

    cells: dict[tuple[int, str, str], dict[str, Any]] = {}
    artifact_hashes: list[dict[str, Any]] = []
    low_level_hash = protocol["inputs"]["low_level_checkpoint"]["sha256"]
    for seed, evaluation_seed in zip(seeds, evaluation_seeds, strict=True):
        for condition in CONDITIONS:
            for arm in arms:
                name = (
                    f"v11-heldout-{condition}-{arm}-seed{seed}-"
                    f"decisions{decisions // 1000}k"
                )
                run = input_root / name
                metadata_path = run / "metadata.json"
                status_path = run / "status.json"
                model_path = run / "model.zip"
                raw_path = run / "evaluation_tasks.jsonl"
                if not all(
                    path.is_file()
                    for path in (metadata_path, status_path, model_path, raw_path)
                ):
                    raise ConfirmatoryAnalysisError(f"incomplete held-out cell: {run}")
                metadata = _read_json(metadata_path)
                status = _read_json(status_path)
                arguments = metadata.get("arguments", {})
                expected_arguments = {
                    "condition": condition,
                    "policy_arm": arm,
                    "seed": seed,
                    "evaluation_seed": evaluation_seed,
                    "total_task_decisions": decisions,
                    "eval_task_episodes": expected_rows,
                    "study_phase": "confirmatory",
                    "protocol_sha256": protocol_digest,
                }
                if {
                    key: arguments.get(key) for key in expected_arguments
                } != expected_arguments:
                    raise ConfirmatoryAnalysisError(f"held-out argument mismatch: {run}")
                if (
                    metadata.get("phase")
                    != "hierarchical_thermal_v11_heldout_confirmatory"
                    or metadata.get("status") != "heldout_confirmatory_cell_complete"
                    or metadata.get("actual_training_device") != "cuda"
                    or metadata.get("low_level_model_sha256") != low_level_hash
                    or status.get("status") != "complete"
                    or status.get("phase") != "v11_confirmatory_heldout"
                    or metadata.get("model_sha256") != sha256(model_path)
                ):
                    raise ConfirmatoryAnalysisError(f"held-out artifact mismatch: {run}")
                raw_mean, raw_hash = _read_raw_mean(
                    raw_path,
                    expected_rows=expected_rows,
                    condition=condition,
                    evaluation_seed=evaluation_seed,
                )
                evaluation = metadata.get("evaluation", {})
                metadata_mean = float(evaluation.get("mean_task_episode_reward"))
                if not math.isclose(raw_mean, metadata_mean, rel_tol=0.0, abs_tol=1e-12):
                    raise ConfirmatoryAnalysisError(f"raw/metadata mean mismatch: {run}")
                cells[(seed, condition, arm)] = {
                    "mean_task_episode_reward": metadata_mean,
                    "high_power_selection_rate": float(
                        evaluation.get("high_power_selection_rate")
                    ),
                    "thermal_trip_rate": float(evaluation.get("thermal_trip_rate")),
                    "both_modes_lifetime_rate": float(
                        evaluation.get("both_modes_lifetime_rate")
                    ),
                }
                artifact_hashes.append(
                    {
                        "run_name": name,
                        "metadata_sha256": sha256(metadata_path),
                        "model_sha256": sha256(model_path),
                        "raw_sha256": raw_hash,
                    }
                )

    seed_rows: list[dict[str, Any]] = []
    stochastic_effects: list[float] = []
    fixed_effects: list[float] = []
    interactions: list[float] = []
    for seed in seeds:
        stochastic_lifetime = cells[(seed, "stochastic", "lifetime_lstm")]
        stochastic_reactive = cells[(seed, "stochastic", reactive_arm)]
        fixed_lifetime = cells[(seed, "fixed", "lifetime_lstm")]
        fixed_reactive = cells[(seed, "fixed", reactive_arm)]
        stochastic_effect = (
            stochastic_lifetime["mean_task_episode_reward"]
            - stochastic_reactive["mean_task_episode_reward"]
        )
        fixed_effect = (
            fixed_lifetime["mean_task_episode_reward"]
            - fixed_reactive["mean_task_episode_reward"]
        )
        interaction = stochastic_effect - fixed_effect
        stochastic_effects.append(stochastic_effect)
        fixed_effects.append(fixed_effect)
        interactions.append(interaction)
        seed_rows.append(
            {
                "seed": seed,
                "stochastic_lifetime_reward": stochastic_lifetime[
                    "mean_task_episode_reward"
                ],
                "stochastic_reactive_reward": stochastic_reactive[
                    "mean_task_episode_reward"
                ],
                "stochastic_lifetime_minus_reactive": stochastic_effect,
                "fixed_lifetime_reward": fixed_lifetime["mean_task_episode_reward"],
                "fixed_reactive_reward": fixed_reactive["mean_task_episode_reward"],
                "fixed_lifetime_minus_reactive": fixed_effect,
                "inference_specificity_interaction": interaction,
            }
        )

    primary = protocol["primary_analysis"]
    criteria = primary["conjunction"]["per_estimand"]
    bootstrap_resamples = primary["bootstrap"]["resamples"]
    analysis_rng = protocol["seed_namespaces"]["analysis_rng"]
    sign_flip = primary["sign_flip_randomization"]
    stochastic_summary = summarize_estimand(
        stochastic_effects,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=analysis_rng["bootstrap-stochastic"],
        sign_flip_draws=sign_flip["draws"],
        sign_flip_seed=sign_flip["rng_seed"],
        criteria=criteria,
    )
    interaction_summary = summarize_estimand(
        interactions,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=analysis_rng["bootstrap-interaction"],
        sign_flip_draws=sign_flip["draws"],
        sign_flip_seed=sign_flip["rng_seed"],
        criteria=criteria,
    )
    confirmatory_passed = bool(
        stochastic_summary["estimand_passed"]
        and interaction_summary["estimand_passed"]
    )
    return {
        "phase": "hierarchical_thermal_v11_heldout_confirmatory_analysis",
        "status": "final_heldout_result",
        "protocol_sha256": protocol_digest,
        "selected_reactive_arm": reactive_arm,
        "inferential_unit": "independent paired training seed",
        "episode_rows_are_not_independent_units": True,
        "wiring_passed": True,
        "co_primary": {
            "stochastic_superiority": stochastic_summary,
            "inference_specificity_interaction": interaction_summary,
        },
        "secondary_fixed_lifetime_minus_reactive": {
            "values": fixed_effects,
            "mean": float(np.mean(np.asarray(fixed_effects, dtype=np.float64))),
            "sd": float(np.std(np.asarray(fixed_effects), ddof=1)),
        },
        "confirmatory_passed": confirmatory_passed,
        "central_claim_confirmed": confirmatory_passed,
        "scientific_null_is_normal_completion": not confirmatory_passed,
        "seed_rows": seed_rows,
        "artifact_hashes": artifact_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/hierarchical_v11/confirmatory"),
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input_root / "CONFIRMATORY_RESULTS.json"
    report = analyze_confirmatory(
        input_root=args.input_root,
        protocol_path=args.protocol,
        expected_protocol_sha256=args.expected_protocol_sha256,
    )
    atomic_write_new(output, report)
    print(
        json.dumps(
            {
                "confirmatory_passed": report["confirmatory_passed"],
                "stochastic_mean": report["co_primary"]["stochastic_superiority"][
                    "mean"
                ],
                "interaction_mean": report["co_primary"][
                    "inference_specificity_interaction"
                ]["mean"],
                "output": str(output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
