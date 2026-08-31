"""Analyze the frozen hierarchical confirmatory campaign at the seed level."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    batch_size = 10_000
    means = []
    remaining = samples
    while remaining:
        count = min(batch_size, remaining)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= count
    distribution = np.concatenate(means)
    return [float(value) for value in np.quantile(distribution, [0.025, 0.975])]


def one_sample_summary(values: list[float], *, bootstrap_seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if np.all(array == 0.0):
        t_p = 1.0
    else:
        t_p = float(
            stats.ttest_1samp(array, popmean=0.0, alternative="greater").pvalue
        )
    try:
        if np.all(array == 0.0):
            wilcoxon_p = 1.0
        else:
            wilcoxon = stats.wilcoxon(
                array, alternative="greater", zero_method="wilcox"
            )
            wilcoxon_p = float(wilcoxon.pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n": len(values),
        "values": values,
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "median": float(np.median(array)),
        "positive_seeds": int(np.sum(array > 0.0)),
        "bootstrap_95_ci": bootstrap_mean_ci(
            values, samples=100_000, seed=bootstrap_seed
        ),
        "one_sided_t_p": t_p,
        "one_sided_wilcoxon_p": wilcoxon_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/confirmatory"),
    )
    args = parser.parse_args()
    manifest = json.loads((args.input_root / "manifest.json").read_text(encoding="utf-8"))
    expected_seeds = manifest["confirmatory_seeds"]
    expected_cells = {
        ("endogenous_action", "task"),
        ("endogenous_action", "lifetime"),
        ("exogenous_clock", "task"),
        ("exogenous_clock", "lifetime"),
    }
    by_seed: dict[int, dict[tuple[str, str], dict]] = {}
    wiring_checks = []
    for seed in expected_seeds:
        paths = sorted((args.input_root / f"seed{seed}").glob("*/metadata.json"))
        if len(paths) != 4:
            raise SystemExit(f"seed {seed}: expected four metadata files, found {len(paths)}")
        cells = {}
        for path in paths:
            document = json.loads(path.read_text(encoding="utf-8"))
            arguments = document["arguments"]
            evaluation = document["task_episode_evaluation"]
            key = (arguments["degradation_mode"], arguments["memory_mode"])
            if key in cells:
                raise SystemExit(f"seed {seed}: duplicate cell {key}")
            cells[key] = evaluation
            wiring_checks.append(
                bool(
                    arguments["seed"] == seed
                    and arguments["total_task_decisions"]
                    == manifest["training_strategy"]["decisions"]
                    and arguments["eval_task_episodes"]
                    == manifest["evaluation_task_episodes"]
                    and arguments["summary_mode"]
                    == manifest["physical_design"]["summary_mode"]
                    and arguments["teacher_shaping"] == 0.0
                    and document["low_level_model_sha256"]
                    == manifest["low_level_model_sha256"]
                    and math.isfinite(evaluation["mean_task_episode_reward"])
                )
            )
        if set(cells) != expected_cells:
            raise SystemExit(f"seed {seed}: factorial mismatch {set(cells)}")
        by_seed[seed] = cells

    rows = []
    dynamic_effects = []
    static_effects = []
    interactions = []
    adaptation_gaps = []
    for seed in expected_seeds:
        cells = by_seed[seed]
        dynamic_task = cells[("endogenous_action", "task")]
        dynamic_lifetime = cells[("endogenous_action", "lifetime")]
        static_task = cells[("exogenous_clock", "task")]
        static_lifetime = cells[("exogenous_clock", "lifetime")]
        dynamic_effect = (
            dynamic_lifetime["mean_task_episode_reward"]
            - dynamic_task["mean_task_episode_reward"]
        )
        static_effect = (
            static_lifetime["mean_task_episode_reward"]
            - static_task["mean_task_episode_reward"]
        )
        interaction = dynamic_effect - static_effect
        cold = dynamic_lifetime["cold_high_power_selection_rate"]
        hot = dynamic_lifetime["hot_high_power_selection_rate"]
        adaptation_gap = None if cold is None or hot is None else cold - hot
        dynamic_effects.append(dynamic_effect)
        static_effects.append(static_effect)
        interactions.append(interaction)
        if adaptation_gap is not None:
            adaptation_gaps.append(adaptation_gap)
        rows.append(
            {
                "seed": seed,
                "dynamic_task_reward": dynamic_task["mean_task_episode_reward"],
                "dynamic_lifetime_reward": dynamic_lifetime["mean_task_episode_reward"],
                "dynamic_memory_effect": dynamic_effect,
                "static_memory_effect": static_effect,
                "interaction": interaction,
                "dynamic_lifetime_high_rate": dynamic_lifetime[
                    "high_power_selection_rate"
                ],
                "dynamic_lifetime_adaptation_gap": adaptation_gap,
                "dynamic_lifetime_trip_rate": dynamic_lifetime["thermal_trip_rate"],
                "static_task_high_rate": static_task["high_power_selection_rate"],
                "static_lifetime_high_rate": static_lifetime[
                    "high_power_selection_rate"
                ],
            }
        )

    primary = one_sample_summary(dynamic_effects, bootstrap_seed=20_260_826)
    static = one_sample_summary(static_effects, bootstrap_seed=20_260_827)
    interaction = one_sample_summary(interactions, bootstrap_seed=20_260_828)
    static_control_passed = all(
        row["static_task_high_rate"] >= 0.95
        and row["static_lifetime_high_rate"] >= 0.95
        for row in rows
    )
    primary_confirmed = bool(
        primary["one_sided_t_p"] < manifest["statistical_plan"]["alpha"]
        and primary["bootstrap_95_ci"][0] > 0.0
    )
    report = {
        "phase": "hierarchical_thermal_held_out_confirmatory_analysis",
        "status": "final_held_out_result",
        "seed_is_unit_of_analysis": True,
        "wiring_passed": all(wiring_checks),
        "static_control_passed": static_control_passed,
        "primary_dynamic_lifetime_minus_task": primary,
        "secondary_static_lifetime_minus_task": static,
        "secondary_difference_in_differences": interaction,
        "mean_lifetime_adaptation_gap": (
            statistics.fmean(adaptation_gaps) if adaptation_gaps else None
        ),
        "primary_confirmed": primary_confirmed,
        "confirmatory_passed": bool(
            all(wiring_checks) and static_control_passed and primary_confirmed
        ),
        "rows": rows,
    }
    write_path = args.input_root / "CONFIRMATORY_RESULTS.json"
    write_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all(wiring_checks):
        raise SystemExit("confirmatory wiring validation failed")


if __name__ == "__main__":
    main()
