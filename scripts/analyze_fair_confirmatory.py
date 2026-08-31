"""Analyze the held-out, confound-controlled fair memory campaign."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lifephybench-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon


CONDITIONS = (
    ("endogenous_action", "task"),
    ("endogenous_action", "lifetime"),
    ("exogenous_clock", "task"),
    ("exogenous_clock", "lifetime"),
)
STATIC_MEMORY_MODES = ("task", "lifetime")


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def paired_contrast(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    result: dict[str, object] = {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "bootstrap_ci95": bootstrap_mean_ci(values, rng),
        "t_test_p": float(ttest_1samp(values, 0.0).pvalue),
        "wilcoxon_p": float(wilcoxon(values).pvalue),
        "seed_values": values.tolist(),
    }
    result["cohen_dz"] = float(values.mean() / values.std(ddof=1))
    return result


def load_records(input_roots: list[Path]) -> list[dict[str, object]]:
    records = []
    for root in input_roots:
        manifest_path = root / "campaign_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"missing campaign manifest: {manifest_path}")
        for path in sorted(root.glob("*/metadata.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            arguments = document["arguments"]
            evaluation = document["task_episode_evaluation"]
            record = {
                "source": str(path),
                "seed": int(arguments["seed"]),
                "degradation": arguments["degradation_mode"],
                "memory": arguments["memory_mode"],
                "reward": float(evaluation["mean_task_episode_reward"]),
                "reward_sd_within_policy": float(evaluation["std_task_episode_reward"]),
                "thermal_load": float(evaluation["mean_episode_end_thermal_load"]),
                "efficiency": float(evaluation["mean_episode_end_efficiency"]),
                "task_episodes": int(evaluation["task_episodes"]),
                "completed_lifetimes": int(evaluation["completed_lifetimes"]),
                "total_timesteps": int(arguments["total_timesteps"]),
                "device": arguments["device"],
                "fixed_exogenous_dose": float(arguments["thermal_exogenous_dose_per_step"]),
            }
            records.append(record)
    return records


def validate(records: list[dict[str, object]]) -> list[int]:
    expected_conditions = set(CONDITIONS)
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        condition = (str(record["degradation"]), str(record["memory"]))
        if condition not in expected_conditions:
            raise SystemExit(f"unexpected condition: {condition}")
        key = (*condition, int(record["seed"]))
        if key in seen:
            raise SystemExit(f"duplicate result: {key}")
        seen.add(key)
        if record["task_episodes"] != 1000 or record["completed_lifetimes"] != 50:
            raise SystemExit(f"evaluation protocol mismatch: {record['source']}")
        if record["total_timesteps"] != 2_000_000:
            raise SystemExit(f"training budget mismatch: {record['source']}")

    seeds = sorted({int(record["seed"]) for record in records})
    expected = {(degradation, memory, seed) for degradation, memory in CONDITIONS for seed in seeds}
    if seen != expected:
        missing = sorted(expected - seen)
        raise SystemExit(f"incomplete factorial campaign; missing={missing}")
    if len(seeds) != 10:
        raise SystemExit(f"expected 10 held-out seeds, found {seeds}")
    doses = {float(record["fixed_exogenous_dose"]) for record in records}
    if len(doses) != 1:
        raise SystemExit(f"fixed exogenous dose differs across runs: {sorted(doses)}")
    return seeds


def validate_static_control(
    records: list[dict[str, object]], expected_seeds: list[int]
) -> None:
    seen: set[tuple[str, int]] = set()
    for record in records:
        memory = str(record["memory"])
        seed = int(record["seed"])
        if memory not in STATIC_MEMORY_MODES:
            raise SystemExit(f"unexpected static-control memory mode: {memory}")
        if (memory, seed) in seen:
            raise SystemExit(f"duplicate static-control result: {(memory, seed)}")
        seen.add((memory, seed))
        if record["task_episodes"] != 1000 or record["completed_lifetimes"] != 50:
            raise SystemExit(f"static-control evaluation mismatch: {record['source']}")
        if record["total_timesteps"] != 2_000_000:
            raise SystemExit(f"static-control training budget mismatch: {record['source']}")
        if float(record["fixed_exogenous_dose"]) != 0.0:
            raise SystemExit(f"static-control dose is not zero: {record['source']}")
        if float(record["thermal_load"]) != 0.0 or float(record["efficiency"]) != 1.0:
            raise SystemExit(f"static-control health changed: {record['source']}")
    expected = {(memory, seed) for memory in STATIC_MEMORY_MODES for seed in expected_seeds}
    if seen != expected:
        raise SystemExit(f"incomplete static control; missing={sorted(expected - seen)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=[Path("outputs/fair_confirmatory_batch_a"), Path("outputs/fair_confirmatory_batch_b")],
    )
    parser.add_argument(
        "--static-root",
        type=Path,
        default=Path("outputs/fair_static_health_control"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fair_confirmatory_analysis"))
    args = parser.parse_args()

    records = load_records(args.input_roots)
    seeds = validate(records)
    static_records = load_records([args.static_root])
    validate_static_control(static_records, seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    with (args.output_root / "per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: (row["degradation"], row["memory"], row["seed"])))
    with (args.output_root / "static_control_per_seed_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(static_records[0]))
        writer.writeheader()
        writer.writerows(sorted(static_records, key=lambda row: (row["memory"], row["seed"])))

    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["degradation"]), str(record["memory"]))][int(record["seed"])] = record

    rng = np.random.default_rng(20260816)
    groups = {}
    for condition in CONDITIONS:
        rewards = np.asarray([float(grouped[condition][seed]["reward"]) for seed in seeds])
        groups["_".join(condition)] = {
            "n": len(rewards),
            "reward_mean": float(rewards.mean()),
            "reward_sd": float(rewards.std(ddof=1)),
            "reward_bootstrap_ci95": bootstrap_mean_ci(rewards, rng),
            "seed_rewards": rewards.tolist(),
        }

    endogenous = np.asarray([
        float(grouped[("endogenous_action", "lifetime")][seed]["reward"])
        - float(grouped[("endogenous_action", "task")][seed]["reward"])
        for seed in seeds
    ])
    exogenous = np.asarray([
        float(grouped[("exogenous_clock", "lifetime")][seed]["reward"])
        - float(grouped[("exogenous_clock", "task")][seed]["reward"])
        for seed in seeds
    ])
    contrasts = {
        "endogenous_lifetime_minus_task": paired_contrast(endogenous, rng),
        "exogenous_lifetime_minus_task": paired_contrast(exogenous, rng),
        "interaction_endogenous_minus_exogenous": paired_contrast(endogenous - exogenous, rng),
    }
    static_grouped: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for record in static_records:
        static_grouped[str(record["memory"])][int(record["seed"])] = record
    static_groups = {}
    for memory in STATIC_MEMORY_MODES:
        rewards = np.asarray([float(static_grouped[memory][seed]["reward"]) for seed in seeds])
        static_groups[memory] = {
            "n": len(rewards),
            "reward_mean": float(rewards.mean()),
            "reward_sd": float(rewards.std(ddof=1)),
            "reward_bootstrap_ci95": bootstrap_mean_ci(rewards, rng),
            "seed_rewards": rewards.tolist(),
        }
    static_effect = np.asarray([
        float(static_grouped["lifetime"][seed]["reward"])
        - float(static_grouped["task"][seed]["reward"])
        for seed in seeds
    ])
    static_robustness = {
        "status": "post_hoc_robustness_not_primary_confirmatory",
        "semantics": {
            "thermal_exogenous_dose_per_step": 0.0,
            "thermal_load": 0.0,
            "actuator_efficiency": 1.0,
        },
        "groups": static_groups,
        "contrasts": {
            "static_lifetime_minus_task": paired_contrast(static_effect, rng),
            "endogenous_minus_static_effect": paired_contrast(endogenous - static_effect, rng),
            "exogenous_minus_static_effect": paired_contrast(exogenous - static_effect, rng),
        },
    }
    report = {
        "phase": "held_out_fair_confirmatory_analysis",
        "analysis_population": {"seeds": seeds, "n": len(seeds), "exclusions": []},
        "controlled_semantics": {
            "gym_and_gae_boundary": "lifetime_only_for_all_four_cells",
            "task_boundary_marker": "observed_by_all_four_cells",
            "memory_intervention": "forced_lstm_reset_at_task_boundary_only",
            "fixed_exogenous_dose": float(records[0]["fixed_exogenous_dose"]),
        },
        "evaluation_protocol": "2M training; 1000 task episodes (50 completed lifetimes) per policy",
        "groups": groups,
        "contrasts": contrasts,
        "static_health_robustness": static_robustness,
    }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = [
        "# Held-out fair confirmatory analysis",
        "",
        "All ten held-out seeds (3000--3009) are retained. Each policy was trained for 2M steps and evaluated over 1,000 task episodes (50 physical lifetimes).",
        "",
        "| Condition | Mean reward | SD | 95% bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    for name, value in groups.items():
        ci = value["reward_bootstrap_ci95"]
        markdown.append(f"| {name} | {value['reward_mean']:.3f} | {value['reward_sd']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    markdown += ["", "| Paired contrast | Mean | SD | dz | 95% bootstrap CI | t-test p | Wilcoxon p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in contrasts.items():
        ci = value["bootstrap_ci95"]
        markdown.append(f"| {name} | {value['mean']:.3f} | {value['sd']:.3f} | {value['cohen_dz']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {value['t_test_p']:.4f} | {value['wilcoxon_p']:.4f} |")
    markdown += [
        "",
        "## Static-health robustness (post hoc)",
        "",
        "The thermal wrapper remains active, but its exogenous dose is fixed to zero. All evaluations recorded thermal load 0.0 and actuator efficiency 1.0.",
        "",
        "| Static condition | Mean reward | SD | 95% bootstrap CI |",
        "|---|---:|---:|---:|",
    ]
    for memory, value in static_groups.items():
        ci = value["reward_bootstrap_ci95"]
        markdown.append(f"| static_{memory} | {value['reward_mean']:.3f} | {value['reward_sd']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    markdown += ["", "| Static-related paired contrast | Mean | SD | dz | 95% bootstrap CI | t-test p | Wilcoxon p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, value in static_robustness["contrasts"].items():
        ci = value["bootstrap_ci95"]
        markdown.append(f"| {name} | {value['mean']:.3f} | {value['sd']:.3f} | {value['cohen_dz']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {value['t_test_p']:.4f} | {value['wilcoxon_p']:.4f} |")
    (args.output_root / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    colors = {"task": "#D55E00", "lifetime": "#0072B2"}
    titles = {"endogenous_action": "Endogenous thermal degradation", "exogenous_clock": "Exogenous clock control"}
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for axis, degradation in zip(axes[:2], ("endogenous_action", "exogenous_clock")):
        for index, memory in enumerate(("task", "lifetime")):
            values = [float(grouped[(degradation, memory)][seed]["reward"]) for seed in seeds]
            axis.scatter(np.full(len(seeds), index), values, color=colors[memory], s=42, zorder=3)
        for seed in seeds:
            axis.plot([0, 1], [float(grouped[(degradation, "task")][seed]["reward"]), float(grouped[(degradation, "lifetime")][seed]["reward"])], color="0.72", linewidth=1, zorder=1)
        axis.set_xticks([0, 1], ["Task reset", "Lifetime memory"])
        axis.set_title(titles[degradation])
        axis.grid(axis="y", alpha=0.25)
    static_axis = axes[2]
    for index, memory in enumerate(STATIC_MEMORY_MODES):
        values = [float(static_grouped[memory][seed]["reward"]) for seed in seeds]
        static_axis.scatter(np.full(len(seeds), index), values, color=colors[memory], s=42, zorder=3)
    for seed in seeds:
        static_axis.plot([0, 1], [float(static_grouped["task"][seed]["reward"]), float(static_grouped["lifetime"][seed]["reward"])], color="0.72", linewidth=1, zorder=1)
    static_axis.set_xticks([0, 1], ["Task reset", "Lifetime memory"])
    static_axis.set_title("Static-health control")
    static_axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean task-episode reward (higher is better)")
    figure.tight_layout()
    figure.savefig(figures / "paired_memory_effect.png", dpi=240, bbox_inches="tight")
    figure.savefig(figures / "paired_memory_effect.pdf", bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
