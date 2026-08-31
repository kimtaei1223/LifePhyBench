#!/usr/bin/env python3
"""Build paper-ready tables and figures for the 2M thermal experiments."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lifephybench-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "thermal_results_package"
FIGURES = OUTPUT / "figures"
SEEDS = ["1000", "1001", "1002", "1003", "1004"]


def read_evaluations(pattern: str) -> dict[str, dict[str, Any]]:
    result = {}
    for raw_path in glob.glob(str(ROOT / pattern)):
        path = Path(raw_path)
        seed = path.parent.name.split("seed")[1].split("-")[0]
        document = json.loads(path.read_text(encoding="utf-8"))
        evaluation = document["task_episode_evaluation"]
        if evaluation["task_episodes"] != 1000:
            raise ValueError(f"expected 1000 task episodes in {path}")
        result[seed] = evaluation
    if sorted(result) != SEEDS:
        raise ValueError(
            f"expected seeds {SEEDS} for {pattern}, found {sorted(result)}"
        )
    return result


def read_calibrated_doses() -> dict[str, dict[str, float]]:
    path = ROOT / "outputs" / "thermal_policy_matched_dose_long.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {"episode": {}, "lifetime": {}}
    for item in document["results"]:
        result[item["memory_mode"]][str(item["seed"])] = float(
            item["recommended_exogenous_dose_per_step"]
        )
    return result


def read_corrected_retests() -> list[dict[str, Any]]:
    rows = []
    pattern = (
        ROOT / "outputs/thermal_exogenous_outlier_retest_corrected/*/metadata.json"
    )
    for raw_path in sorted(glob.glob(str(pattern))):
        document = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        arguments = document["arguments"]
        name = arguments["run_name"]
        evaluation = document["task_episode_evaluation"]
        rows.append(
            {
                "source_seed": name.split("source")[1].split("-")[0],
                "train_seed": str(arguments["seed"]),
                "dose": float(arguments["thermal_exogenous_dose_per_step"]),
                "reward": float(evaluation["mean_task_episode_reward"]),
                "reward_sd": float(evaluation["std_task_episode_reward"]),
                "thermal_load": float(evaluation["mean_episode_end_thermal_load"]),
            }
        )
    if len(rows) != 4:
        raise ValueError(f"expected four corrected retests, found {len(rows)}")
    return rows


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def tensorboard_curve(run_directory: Path) -> tuple[np.ndarray, np.ndarray]:
    event_files = sorted(run_directory.glob("tensorboard/*/events.*"))
    if len(event_files) != 1:
        raise ValueError(f"expected one TensorBoard event file in {run_directory}")
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    events = accumulator.Scalars("rollout/ep_rew_mean")
    return np.asarray([event.step for event in events]), np.asarray(
        [event.value for event in events]
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260813)
    groups = {
        "endogenous_episode": read_evaluations(
            "outputs/thermal_endogenous_long/*episode-seed*-steps2000k/"
            "task_episode_evaluation.json"
        ),
        "endogenous_lifetime": read_evaluations(
            "outputs/thermal_endogenous_long/*lifetime-seed*-steps2000k/"
            "task_episode_evaluation.json"
        ),
        "matched_exogenous_episode": read_evaluations(
            "outputs/thermal_exogenous_matched_long/*episode-seed*-steps2000k/"
            "task_episode_evaluation.json"
        ),
        "matched_exogenous_lifetime": read_evaluations(
            "outputs/thermal_exogenous_matched_long/*lifetime-seed*-steps2000k/"
            "task_episode_evaluation.json"
        ),
    }
    doses = read_calibrated_doses()
    retests = read_corrected_retests()

    rows = []
    for group, evaluations in groups.items():
        degradation, memory = group.rsplit("_", 1)
        for seed in SEEDS:
            item = evaluations[seed]
            rows.append(
                {
                    "group": group,
                    "degradation": degradation,
                    "memory": memory,
                    "seed": seed,
                    "reward": item["mean_task_episode_reward"],
                    "reward_sd_within_policy": item["std_task_episode_reward"],
                    "thermal_load": item["mean_episode_end_thermal_load"],
                    "efficiency": item["mean_episode_end_efficiency"],
                    "calibrated_dose": (
                        doses[memory][seed]
                        if degradation == "matched_exogenous"
                        else ""
                    ),
                }
            )
    with (OUTPUT / "primary_seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "corrected_retests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retests[0]))
        writer.writeheader()
        writer.writerows(retests)

    contrasts = []
    end_delta = np.asarray(
        [
            groups["endogenous_lifetime"][s]["mean_task_episode_reward"]
            - groups["endogenous_episode"][s]["mean_task_episode_reward"]
            for s in SEEDS
        ]
    )
    exo_delta = np.asarray(
        [
            groups["matched_exogenous_lifetime"][s]["mean_task_episode_reward"]
            - groups["matched_exogenous_episode"][s]["mean_task_episode_reward"]
            for s in SEEDS
        ]
    )
    values = [end_delta, exo_delta, end_delta - exo_delta]
    names = ["endogenous", "matched_exogenous", "interaction"]
    raw_p = [float(ttest_1samp(value, 0).pvalue) for value in values]
    adjusted_p = holm_adjust(raw_p)
    for name, value, p_value, p_adjusted in zip(names, values, raw_p, adjusted_p):
        contrasts.append(
            {
                "contrast": name,
                "mean": float(value.mean()),
                "sd": float(value.std(ddof=1)),
                "cohen_dz": float(value.mean() / value.std(ddof=1)),
                "bootstrap_ci95": bootstrap_ci(value, rng),
                "t_test_p": p_value,
                "holm_adjusted_p": p_adjusted,
                "wilcoxon_p": float(wilcoxon(value).pvalue),
                "seed_values": value.tolist(),
            }
        )

    episode_dose = np.asarray([doses["episode"][seed] for seed in SEEDS])
    episode_reward = np.asarray(
        [
            groups["matched_exogenous_episode"][seed]["mean_task_episode_reward"]
            for seed in SEEDS
        ]
    )
    correlation = {
        "n": len(SEEDS),
        "pearson_r": float(pearsonr(episode_dose, episode_reward).statistic),
        "pearson_p": float(pearsonr(episode_dose, episode_reward).pvalue),
        "spearman_rho": float(spearmanr(episode_dose, episode_reward).statistic),
        "spearman_p": float(spearmanr(episode_dose, episode_reward).pvalue),
        "interpretation": "exploratory; n=5 and dose is policy-derived",
    }
    summary = {
        "evaluation_protocol": "2M training and 1000-task-episode evaluation",
        "primary_seed_count": 5,
        "contrasts": contrasts,
        "matched_exogenous_episode_dose_reward_correlation": correlation,
        "corrected_retests": retests,
        "invalidated_retest_directory": "outputs/thermal_exogenous_outlier_retest",
    }
    (OUTPUT / "statistical_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    colors = {"episode": "#D55E00", "lifetime": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for axis, degradation, title in zip(
        axes,
        ["endogenous", "matched_exogenous"],
        ["Endogenous thermal degradation", "Policy-matched exogenous control"],
    ):
        for position, memory in enumerate(["episode", "lifetime"]):
            rewards = np.asarray(
                [
                    groups[f"{degradation}_{memory}"][s]["mean_task_episode_reward"]
                    for s in SEEDS
                ]
            )
            axis.scatter(
                np.full(len(SEEDS), position),
                rewards,
                color=colors[memory],
                s=45,
                zorder=3,
            )
        for seed in SEEDS:
            axis.plot(
                [0, 1],
                [
                    groups[f"{degradation}_episode"][seed]["mean_task_episode_reward"],
                    groups[f"{degradation}_lifetime"][seed]["mean_task_episode_reward"],
                ],
                color="0.72",
                linewidth=1,
                zorder=1,
            )
        axis.set_xticks([0, 1], ["Episode", "Lifetime"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean task-episode reward (higher is better)")
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_memory_effect.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "paired_memory_effect.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.8))
    axis.scatter(
        episode_dose, episode_reward, s=65, color="#0072B2", label="Primary seeds"
    )
    for seed, x, y in zip(SEEDS, episode_dose, episode_reward):
        axis.annotate(
            seed, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    for index, item in enumerate(retests):
        axis.scatter(
            item["dose"],
            item["reward"],
            marker="x",
            s=70,
            color="#D55E00",
            label="Corrected retests" if index == 0 else None,
        )
    axis.set_xlabel("Calibrated exogenous dose per step")
    axis.set_ylabel("Mean task-episode reward")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "dose_reward_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "dose_reward_robustness.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    primary_root = ROOT / "outputs/thermal_exogenous_matched_long"
    for seed in SEEDS:
        run = primary_root / f"thermal-exogenous_clock-episode-seed{seed}-steps2000k"
        steps, reward = tensorboard_curve(run)
        axis.plot(
            steps,
            reward,
            linewidth=1.2 if seed not in {"1002", "1004"} else 2.1,
            alpha=0.7 if seed not in {"1002", "1004"} else 1.0,
            label=f"seed {seed}",
        )
    axis.set_xlabel("Training timesteps")
    axis.set_ylabel("Rolling episode reward")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "exogenous_episode_learning_curves.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "exogenous_episode_learning_curves.png", dpi=240, bbox_inches="tight"
    )
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
