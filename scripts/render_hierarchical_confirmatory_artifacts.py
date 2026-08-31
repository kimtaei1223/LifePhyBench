#!/usr/bin/env python3
"""Render paper-ready artifacts from the frozen v10 confirmatory result.

The input is the final seed-level analysis JSON.  This script does not rerun
statistical tests or inspect training data: it validates the stored paired
contrasts and renders them into copy-ready tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lifephybench-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "hierarchical_autonomous_v10"
    / "confirmatory"
    / "CONFIRMATORY_RESULTS.json"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "paper_artifacts"
DEFAULT_CAMPAIGN_MANIFEST = DEFAULT_INPUT.parent / "manifest.json"
DEFAULT_DESIGN_SEARCH = DEFAULT_INPUT.parents[1] / "design_search.json"
DEFAULT_FROZEN_PROTOCOL = DEFAULT_INPUT.parents[1] / "FROZEN_PROTOCOL.json"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#707070"
LIGHT_GRAY = "#B8B8B8"
SVG_HASH_SALT = "lifephybench-hierarchical-confirmatory-v10-paper-artifacts"

CELL_LABELS = {
    ("endogenous_action", "task"): ("Dynamic", "Task-reset"),
    ("endogenous_action", "lifetime"): ("Dynamic", "Lifetime"),
    ("exogenous_clock", "task"): ("Static control", "Task-reset"),
    ("exogenous_clock", "lifetime"): ("Static control", "Lifetime"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Fail closed if the stored seed-level analysis is internally inconsistent."""

    require(
        document.get("phase") == "hierarchical_thermal_held_out_confirmatory_analysis",
        "unexpected analysis phase",
    )
    require(document.get("status") == "final_held_out_result", "result is not final")
    require(
        document.get("seed_is_unit_of_analysis") is True, "seed is not analysis unit"
    )
    require(document.get("wiring_passed") is True, "wiring validation did not pass")
    require(
        document.get("confirmatory_passed") is True, "confirmatory gate did not pass"
    )

    rows = document.get("rows")
    require(isinstance(rows, list) and rows, "rows must be a non-empty list")
    rows = sorted(rows, key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in rows]
    require(len(seeds) == len(set(seeds)), "duplicate seeds in result")

    primary = document["primary_dynamic_lifetime_minus_task"]
    static = document["secondary_static_lifetime_minus_task"]
    interaction = document["secondary_difference_in_differences"]
    require(primary["n"] == len(rows), "primary n does not match rows")
    require(static["n"] == len(rows), "static n does not match rows")
    require(interaction["n"] == len(rows), "interaction n does not match rows")

    dynamic_values: list[float] = []
    static_values: list[float] = []
    interaction_values: list[float] = []
    for row in rows:
        numeric_fields = [
            "dynamic_task_reward",
            "dynamic_lifetime_reward",
            "dynamic_memory_effect",
            "static_memory_effect",
            "interaction",
            "dynamic_lifetime_high_rate",
            "dynamic_lifetime_adaptation_gap",
            "dynamic_lifetime_trip_rate",
            "static_task_high_rate",
            "static_lifetime_high_rate",
        ]
        require(
            all(math.isfinite(float(row[field])) for field in numeric_fields),
            f"seed {row['seed']} contains non-finite values",
        )
        dynamic = float(row["dynamic_lifetime_reward"]) - float(
            row["dynamic_task_reward"]
        )
        static_effect = float(row["static_memory_effect"])
        did = dynamic - static_effect
        require(
            close(dynamic, row["dynamic_memory_effect"]),
            f"seed {row['seed']} dynamic contrast mismatch",
        )
        require(close(did, row["interaction"]), f"seed {row['seed']} DiD mismatch")
        dynamic_values.append(dynamic)
        static_values.append(static_effect)
        interaction_values.append(did)

    for label, observed, stored in [
        ("primary", dynamic_values, primary["values"]),
        ("static", static_values, static["values"]),
        ("interaction", interaction_values, interaction["values"]),
    ]:
        require(len(observed) == len(stored), f"{label} values length mismatch")
        require(
            all(close(a, b) for a, b in zip(observed, stored)),
            f"{label} stored values do not match rows",
        )
        require(
            close(
                np.mean(observed),
                document[
                    {
                        "primary": "primary_dynamic_lifetime_minus_task",
                        "static": "secondary_static_lifetime_minus_task",
                        "interaction": "secondary_difference_in_differences",
                    }[label]
                ]["mean"],
            ),
            f"{label} stored mean mismatch",
        )

    adaptation_mean = float(
        np.mean([row["dynamic_lifetime_adaptation_gap"] for row in rows])
    )
    require(
        close(adaptation_mean, document["mean_lifetime_adaptation_gap"]),
        "mean adaptation gap mismatch",
    )
    return rows


def crosscheck_result_rows(
    result_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]
) -> None:
    """Recompute every result-row field available from the 80 run metadata."""

    by_seed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for cell in cell_rows:
        key = (cell["degradation_mode"], cell["memory_mode"])
        require(key in CELL_LABELS, f"unexpected cell key {key}")
        seed = int(cell["seed"])
        require(
            key not in by_seed.setdefault(seed, {}), f"duplicate seed/cell {seed} {key}"
        )
        by_seed[seed][key] = cell

    result_by_seed = {int(row["seed"]): row for row in result_rows}
    require(
        set(result_by_seed) == set(by_seed),
        "result-row and metadata seed sets differ",
    )
    for seed, row in result_by_seed.items():
        cells = by_seed[seed]
        require(set(cells) == set(CELL_LABELS), f"seed {seed} is missing a cell")
        dynamic_task = cells[("endogenous_action", "task")]
        dynamic_lifetime = cells[("endogenous_action", "lifetime")]
        static_task = cells[("exogenous_clock", "task")]
        static_lifetime = cells[("exogenous_clock", "lifetime")]
        dynamic_effect = dynamic_lifetime["reward"] - dynamic_task["reward"]
        static_effect = static_lifetime["reward"] - static_task["reward"]
        adaptation_gap = (
            dynamic_lifetime["cold_high_rate"] - dynamic_lifetime["hot_high_rate"]
        )
        expected = {
            "dynamic_task_reward": dynamic_task["reward"],
            "dynamic_lifetime_reward": dynamic_lifetime["reward"],
            "dynamic_memory_effect": dynamic_effect,
            "static_memory_effect": static_effect,
            "interaction": dynamic_effect - static_effect,
            "dynamic_lifetime_high_rate": dynamic_lifetime["high_rate"],
            "dynamic_lifetime_adaptation_gap": adaptation_gap,
            "dynamic_lifetime_trip_rate": dynamic_lifetime["trip_rate"],
            "static_task_high_rate": static_task["high_rate"],
            "static_lifetime_high_rate": static_lifetime["high_rate"],
        }
        for field, value in expected.items():
            require(
                close(row[field], value),
                f"seed {seed} result/metadata mismatch for {field}",
            )


def validate_campaign(
    campaign_root: Path,
    campaign: dict[str, Any],
    frozen: dict[str, Any],
    frozen_path: Path,
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the campaign manifest and all 80 per-run metadata documents."""

    require(
        campaign.get("phase") == "hierarchical_thermal_held_out_confirmatory_campaign",
        "unexpected confirmatory campaign phase",
    )
    require(
        campaign.get("status") == "frozen_before_any_confirmatory_training",
        "campaign was not locally frozen before held-out training",
    )
    require(
        campaign.get("protocol_sha256") == sha256(frozen_path),
        "campaign protocol hash does not match frozen protocol",
    )
    require(
        campaign.get("source_sha256") == frozen.get("source_sha256"),
        "campaign and frozen source hashes differ",
    )
    require(
        campaign.get("physical_design") == frozen.get("physical_design"),
        "campaign and frozen physical designs differ",
    )
    require(
        campaign.get("training_strategy") == frozen.get("training_strategy"),
        "campaign and frozen training strategies differ",
    )
    require(
        campaign.get("low_level_model_sha256") == frozen.get("low_level_model_sha256"),
        "campaign and frozen low-level model hashes differ",
    )
    require(
        campaign.get("evaluation_task_episodes")
        == frozen.get("evaluation_task_episodes"),
        "campaign and frozen evaluation sizes differ",
    )
    seeds = [int(seed) for seed in campaign["confirmatory_seeds"]]
    require(len(seeds) == len(set(seeds)) == 20, "expected 20 unique held-out seeds")
    require(
        seeds == [int(seed) for seed in frozen["confirmatory_seeds_frozen"]],
        "campaign seed order differs from frozen protocol",
    )
    calibration = {int(seed) for seed in campaign["calibration_seeds_excluded"]}
    require(
        calibration == {int(seed) for seed in frozen["calibration_seeds_used"]},
        "excluded calibration seeds differ from frozen protocol",
    )
    require(not calibration.intersection(seeds), "calibration/confirmatory seed leak")

    seed_directories = sorted(
        path.name
        for path in campaign_root.iterdir()
        if path.is_dir() and path.name.startswith("seed")
    )
    require(
        seed_directories == sorted(f"seed{seed}" for seed in seeds),
        "campaign seed directories do not exactly match the manifest",
    )
    all_metadata = sorted(campaign_root.glob("seed*/*/metadata.json"))
    require(
        len(all_metadata) == 80,
        f"expected 80 metadata files, found {len(all_metadata)}",
    )

    strategy = campaign["training_strategy"]
    design = campaign["physical_design"]
    cell_rows: list[dict[str, Any]] = []
    evaluation_seeds: set[int] = set()
    for seed in seeds:
        paths = sorted((campaign_root / f"seed{seed}").glob("*/metadata.json"))
        require(len(paths) == 4, f"seed {seed}: expected four metadata files")
        seed_evaluation_seeds: set[int] = set()
        seen_cells: set[tuple[str, str]] = set()
        for path in paths:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            arguments = metadata["arguments"]
            evaluation = metadata["task_episode_evaluation"]
            key = (arguments["degradation_mode"], arguments["memory_mode"])
            require(key in CELL_LABELS, f"seed {seed}: unexpected cell {key}")
            require(key not in seen_cells, f"seed {seed}: duplicate cell {key}")
            seen_cells.add(key)
            require(arguments["seed"] == seed, f"seed {seed}: argument seed mismatch")
            require(
                arguments["run_name"] == path.parent.name,
                f"seed {seed}: run-directory name mismatch",
            )
            require(
                (path.parent / "model.zip").is_file(),
                f"seed {seed} {key}: model.zip missing",
            )
            expected_arguments = {
                "total_task_decisions": strategy["decisions"],
                "curriculum_lifetimes": strategy["duration"],
                "ent_coef": strategy["entropy"],
                "learning_rate": strategy["lr"],
                "curriculum_start_trip_load": strategy["start"],
                "teacher_shaping": strategy["teacher"],
                "eval_task_episodes": campaign["evaluation_task_episodes"],
                "summary_mode": design["summary_mode"],
                "thermal_heat_rate": design["thermal_heat_rate"],
                "trip_load": design["trip_load"],
                "low_power_scale": design["low_power_scale"],
                "high_power_bonus": design["high_power_bonus"],
                "trip_penalty": design["trip_penalty"],
                "training_reward_scale": campaign["training_reward_scale"],
                "torch_threads_per_process": 1,
            }
            for field, expected in expected_arguments.items():
                observed = arguments[field]
                if isinstance(expected, (int, float)) and not isinstance(
                    expected, bool
                ):
                    matched = close(observed, expected)
                else:
                    matched = observed == expected
                require(matched, f"seed {seed} {key}: argument mismatch for {field}")
            require(
                metadata["low_level_model_sha256"]
                == campaign["low_level_model_sha256"],
                f"seed {seed} {key}: low-level model hash mismatch",
            )
            require(
                metadata.get("phase") == "hierarchical_thermal_mode_calibration"
                and metadata.get("status") == "calibration_not_confirmatory_evidence",
                f"seed {seed} {key}: unexpected legacy trainer label",
            )
            semantics = metadata["controlled_semantics"]
            required_semantics = {
                "evaluation_reward_unscaled": True,
                "low_level_controller_frozen": True,
                "one_high_level_action_per_physical_task": True,
                "privileged_health_exposed": False,
                "task_boundary_observed": True,
                "teacher_shaping_training_only": False,
                "training_trip_load_curriculum_only": True,
                "tasks_per_lifetime": 20,
                "summary_mode": design["summary_mode"],
                "evaluation_trip_load": design["trip_load"],
                "torch_threads_per_process": 1,
            }
            for field, expected in required_semantics.items():
                require(
                    semantics[field] == expected,
                    f"seed {seed} {key}: semantic mismatch for {field}",
                )
            require(
                semantics["physical_design"]
                == {
                    field: design[field]
                    for field in [
                        "high_power_bonus",
                        "low_power_scale",
                        "thermal_heat_rate",
                        "trip_load",
                        "trip_penalty",
                    ]
                },
                f"seed {seed} {key}: controlled physical design mismatch",
            )
            require(
                evaluation["task_episodes"] == campaign["evaluation_task_episodes"],
                f"seed {seed} {key}: evaluation episode count mismatch",
            )
            require(
                evaluation["completed_lifetimes"]
                == campaign["evaluation_task_episodes"]
                // semantics["tasks_per_lifetime"],
                f"seed {seed} {key}: completed lifetime count mismatch",
            )
            metrics = {
                "reward": float(evaluation["mean_task_episode_reward"]),
                "high_rate": float(evaluation["high_power_selection_rate"]),
                "trip_rate": float(evaluation["thermal_trip_rate"]),
            }
            require(
                all(math.isfinite(value) for value in metrics.values()),
                f"seed {seed} {key}: non-finite evaluation metric",
            )
            require(
                0.0 <= metrics["high_rate"] <= 1.0
                and 0.0 <= metrics["trip_rate"] <= 1.0,
                f"seed {seed} {key}: invalid evaluation rate",
            )
            cold = evaluation["cold_high_power_selection_rate"]
            hot = evaluation["hot_high_power_selection_rate"]
            cell_rows.append(
                {
                    "seed": seed,
                    "degradation_mode": key[0],
                    "memory_mode": key[1],
                    "condition": CELL_LABELS[key][0],
                    "memory": CELL_LABELS[key][1],
                    **metrics,
                    "cold_high_rate": None if cold is None else float(cold),
                    "hot_high_rate": None if hot is None else float(hot),
                    "evaluation_seed": int(metadata["evaluation_seed"]),
                    "metadata_path": str(path),
                }
            )
            seed_evaluation_seeds.add(int(metadata["evaluation_seed"]))
        require(seen_cells == set(CELL_LABELS), f"seed {seed}: factorial mismatch")
        require(
            len(seed_evaluation_seeds) == 1,
            f"seed {seed}: paired cells use different evaluation seeds",
        )
        evaluation_seed = next(iter(seed_evaluation_seeds))
        require(
            evaluation_seed not in evaluation_seeds,
            f"seed {seed}: evaluation seed reused across training seeds",
        )
        evaluation_seeds.add(evaluation_seed)

    crosscheck_result_rows(result_rows, cell_rows)
    return cell_rows


def calibration_reference(
    design_document: dict[str, Any], frozen_document: dict[str, Any]
) -> dict[str, float]:
    """Validate and extract calibration-only analytic comparators."""

    require(
        design_document.get("phase") == "hierarchical_thermal_oracle_design_search",
        "unexpected design-search phase",
    )
    require(
        design_document.get("status") == "calibration_only_not_learned_policy_evidence",
        "design search is not marked calibration-only",
    )
    require(design_document.get("passed") is True, "design search did not pass")
    require(
        frozen_document.get("phase")
        == "hierarchical_thermal_frozen_confirmatory_protocol",
        "unexpected frozen-protocol phase",
    )

    selected = design_document["selected_design"]
    frozen_design = frozen_document["physical_design"]
    design_keys = [
        "thermal_heat_rate",
        "trip_load",
        "low_power_scale",
        "high_power_bonus",
        "trip_penalty",
    ]
    require(
        all(close(selected[key], frozen_design[key]) for key in design_keys),
        "selected and frozen physical designs differ",
    )
    matches = [
        row
        for row in design_document["rows"]
        if all(close(row["design"][key], selected[key]) for key in design_keys)
    ]
    require(len(matches) == 1, "could not uniquely identify the frozen design row")
    row = matches[0]
    all_low = row["all_low"]
    oracle = row["best_prefix_schedule"]
    reactive = row["best_task_reactive"]
    task_count = len(reactive["actions"])
    require(task_count > 0, "task-reactive schedule is empty")
    reactive_mean = float(reactive["lifetime_reward"]) / task_count
    require(
        close(
            oracle["lifetime_reward"] - all_low["lifetime_reward"],
            frozen_design["oracle_improvement_over_all_low"],
        ),
        "frozen all-Low oracle improvement mismatch",
    )
    require(
        close(
            oracle["lifetime_reward"] - reactive["lifetime_reward"],
            frozen_design["oracle_improvement_over_best_task_reactive"],
        ),
        "frozen task-reactive oracle improvement mismatch",
    )
    return {
        "analytic_all_low_mean_task_reward": float(all_low["mean_task_reward"]),
        "analytic_best_task_reactive_mean_task_reward": reactive_mean,
        "analytic_lifetime_oracle_mean_task_reward": float(oracle["mean_task_reward"]),
        "analytic_best_task_reactive_lifetime_reward": float(
            reactive["lifetime_reward"]
        ),
        "analytic_lifetime_oracle_lifetime_reward": float(oracle["lifetime_reward"]),
    }


def write_seed_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "seed",
        "dynamic_task_reward",
        "dynamic_lifetime_reward",
        "dynamic_memory_effect",
        "static_memory_effect",
        "difference_in_differences",
        "dynamic_lifetime_high_rate",
        "dynamic_lifetime_cold_minus_hot_high_rate",
        "dynamic_lifetime_trip_rate",
        "static_task_high_rate",
        "static_lifetime_high_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "seed": row["seed"],
                    "dynamic_task_reward": row["dynamic_task_reward"],
                    "dynamic_lifetime_reward": row["dynamic_lifetime_reward"],
                    "dynamic_memory_effect": row["dynamic_memory_effect"],
                    "static_memory_effect": row["static_memory_effect"],
                    "difference_in_differences": row["interaction"],
                    "dynamic_lifetime_high_rate": row["dynamic_lifetime_high_rate"],
                    "dynamic_lifetime_cold_minus_hot_high_rate": row[
                        "dynamic_lifetime_adaptation_gap"
                    ],
                    "dynamic_lifetime_trip_rate": row["dynamic_lifetime_trip_rate"],
                    "static_task_high_rate": row["static_task_high_rate"],
                    "static_lifetime_high_rate": row["static_lifetime_high_rate"],
                }
            )


def summary_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = [
        (
            "Primary: dynamic Lifetime - Task",
            document["primary_dynamic_lifetime_minus_task"],
        ),
        (
            "Negative control: static Lifetime - Task",
            document["secondary_static_lifetime_minus_task"],
        ),
        (
            "Difference-in-differences",
            document["secondary_difference_in_differences"],
        ),
    ]
    result = []
    for contrast, summary in mapping:
        result.append(
            {
                "contrast": contrast,
                "n_seeds": summary["n"],
                "mean": summary["mean"],
                "sd": summary["sd"],
                "median": summary["median"],
                "bootstrap_ci95_low": summary["bootstrap_95_ci"][0],
                "bootstrap_ci95_high": summary["bootstrap_95_ci"][1],
                "one_sided_t_p": summary["one_sided_t_p"],
                "one_sided_wilcoxon_p": summary["one_sided_wilcoxon_p"],
                "positive_seeds": summary["positive_seeds"],
            }
        )
    return result


def write_summary_csv(document: dict[str, Any], path: Path) -> None:
    rows = summary_rows(document)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_calibration_reference_csv(
    reference: dict[str, float],
    confirmatory_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    task_mean = float(
        np.mean([row["dynamic_task_reward"] for row in confirmatory_rows])
    )
    lifetime_mean = float(
        np.mean([row["dynamic_lifetime_reward"] for row in confirmatory_rows])
    )
    rows = [
        {
            "policy": "Learned Task-reset",
            "mean_task_reward": task_mean,
            "evidence_scope": "held-out confirmatory; learned policy",
        },
        {
            "policy": "Learned Lifetime",
            "mean_task_reward": lifetime_mean,
            "evidence_scope": "held-out confirmatory; learned policy",
        },
        {
            "policy": "Analytic all-Low",
            "mean_task_reward": reference["analytic_all_low_mean_task_reward"],
            "evidence_scope": "calibration-only diagnostic",
        },
        {
            "policy": "Analytic best task-reactive",
            "mean_task_reward": reference[
                "analytic_best_task_reactive_mean_task_reward"
            ],
            "evidence_scope": "calibration-only diagnostic",
        },
        {
            "policy": "Analytic lifetime oracle",
            "mean_task_reward": reference["analytic_lifetime_oracle_mean_task_reward"],
            "evidence_scope": "calibration-only diagnostic",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_cells(cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the four factorial cells across independent training seeds."""

    result = []
    for (degradation_mode, memory_mode), (condition, memory) in CELL_LABELS.items():
        selected = [
            row
            for row in cell_rows
            if row["degradation_mode"] == degradation_mode
            and row["memory_mode"] == memory_mode
        ]
        require(selected, f"no rows for cell {(degradation_mode, memory_mode)}")
        seeds = {int(row["seed"]) for row in selected}
        require(
            len(seeds) == len(selected),
            f"duplicate seed in cell {(degradation_mode, memory_mode)}",
        )
        summary: dict[str, Any] = {
            "condition": condition,
            "memory": memory,
            "degradation_mode": degradation_mode,
            "memory_mode": memory_mode,
            "n_seeds": len(selected),
        }
        for source, label in [
            ("reward", "reward"),
            ("high_rate", "high_selection_rate"),
            ("trip_rate", "trip_rate"),
        ]:
            values = np.asarray([row[source] for row in selected], dtype=float)
            require(len(values) >= 2, f"sample SD needs two or more seeds: {label}")
            summary[f"{label}_mean"] = float(np.mean(values))
            summary[f"{label}_sample_sd"] = (
                0.0 if np.all(values == values[0]) else float(np.std(values, ddof=1))
            )
        result.append(summary)
    return result


def write_cell_summary_csv(cell_rows: list[dict[str, Any]], path: Path) -> None:
    rows = summarize_cells(cell_rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": SVG_HASH_SALT,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(png, dpi=320, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return [svg, png]


def render_confirmatory_effects(
    document: dict[str, Any],
    rows: list[dict[str, Any]],
    reference: dict[str, float],
    output: Path,
) -> list[Path]:
    task = np.asarray([row["dynamic_task_reward"] for row in rows], dtype=float)
    lifetime = np.asarray([row["dynamic_lifetime_reward"] for row in rows], dtype=float)
    dynamic = np.asarray([row["dynamic_memory_effect"] for row in rows], dtype=float)
    static = np.asarray([row["static_memory_effect"] for row in rows], dtype=float)
    primary = document["primary_dynamic_lifetime_minus_task"]
    control = document["secondary_static_lifetime_minus_task"]

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))

    axis = axes[0]
    for before, after in zip(task, lifetime):
        axis.plot([0, 1], [before, after], color=LIGHT_GRAY, linewidth=0.8, zorder=1)
    axis.scatter(
        np.zeros(len(rows)),
        task,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.35,
        s=28,
        zorder=3,
        label="Task-reset memory",
    )
    axis.scatter(
        np.ones(len(rows)),
        lifetime,
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        s=28,
        zorder=3,
        label="Lifetime memory",
    )
    reactive_mean = reference["analytic_best_task_reactive_mean_task_reward"]
    axis.axhline(
        reactive_mean,
        color=GREEN,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )
    axis.text(
        0.02,
        reactive_mean + 0.05,
        "Calibration-only analytic task-reactive reference",
        color=GREEN,
        fontsize=6.9,
        ha="left",
        va="bottom",
    )
    axis.set_xticks([0, 1], ["Task-reset", "Lifetime"])
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylabel("Mean task reward (higher is better)")
    axis.set_title("a  Dynamic thermal condition", loc="left", fontweight="bold")
    axis.grid(axis="y", color="0.88", linewidth=0.65)

    axis = axes[1]
    jitter = np.linspace(-0.11, 0.11, len(rows))
    axis.scatter(
        jitter,
        dynamic,
        color=BLUE,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
        s=27,
        zorder=3,
    )
    axis.scatter(
        1 + jitter,
        static,
        color=GRAY,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
        s=27,
        zorder=3,
    )
    for x, summary, color in [(0, primary, BLUE), (1, control, GRAY)]:
        low, high = summary["bootstrap_95_ci"]
        axis.errorbar(
            x,
            summary["mean"],
            yerr=[[summary["mean"] - low], [high - summary["mean"]]],
            fmt="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6.2,
            capsize=3,
            linewidth=1.5,
            zorder=5,
        )
    axis.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
    axis.set_xticks([0, 1], ["Dynamic", "Static control"])
    axis.set_xlim(-0.38, 1.38)
    axis.set_ylabel("Lifetime - Task reward")
    axis.set_title("b  Seed-level memory effect", loc="left", fontweight="bold")
    axis.grid(axis="y", color="0.88", linewidth=0.65)
    axis.text(
        0.97,
        0.97,
        f"one-sided paired t: p={primary['one_sided_t_p']:.2g}\n"
        f"bootstrap 95% CI [{primary['bootstrap_95_ci'][0]:.2f}, "
        f"{primary['bootstrap_95_ci'][1]:.2f}]",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
    )

    fig.tight_layout(w_pad=2.2)
    return save_figure(fig, output, "confirmatory_memory_effect")


def render_policy_adaptation(
    document: dict[str, Any], rows: list[dict[str, Any]], output: Path
) -> list[Path]:
    seeds = np.asarray([row["seed"] for row in rows], dtype=int)
    effects = np.asarray([row["dynamic_memory_effect"] for row in rows], dtype=float)
    gaps = np.asarray(
        [row["dynamic_lifetime_adaptation_gap"] for row in rows], dtype=float
    )
    high_rates = np.asarray(
        [row["dynamic_lifetime_high_rate"] for row in rows], dtype=float
    )
    trips = np.asarray([row["dynamic_lifetime_trip_rate"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))

    axis = axes[0]
    axis.scatter(
        seeds,
        gaps,
        color=GREEN,
        edgecolor="white",
        linewidth=0.35,
        s=29,
        zorder=3,
        label="Cold - hot High rate",
    )
    axis.scatter(
        seeds,
        high_rates,
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        s=25,
        zorder=3,
        label="Overall High rate",
    )
    axis.axhline(
        document["mean_lifetime_adaptation_gap"],
        color=GREEN,
        linestyle="--",
        linewidth=1,
        label="Mean adaptation gap",
    )
    axis.set_xlabel("Held-out training seed")
    axis.set_ylabel("Selection-rate difference / rate")
    axis.set_ylim(-0.05, 1.06)
    axis.set_title("a  Lifetime-policy behavior", loc="left", fontweight="bold")
    axis.grid(axis="y", color="0.88", linewidth=0.65)
    axis.legend(frameon=False, loc="upper left")

    axis = axes[1]
    grouped: dict[tuple[float, float, bool], int] = {}
    for gap, effect, trip in zip(gaps, effects, trips):
        key = (float(gap), float(effect), bool(trip > 0))
        grouped[key] = grouped.get(key, 0) + 1
    for trip_observed, color, marker, label in [
        (False, BLUE, "o", "No thermal trip"),
        (True, ORANGE, "X", "Trip observed"),
    ]:
        points = [
            (gap, effect, count)
            for (gap, effect, trip), count in grouped.items()
            if trip == trip_observed
        ]
        if not points:
            continue
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.4,
            s=[34 + 10 * (point[2] - 1) for point in points],
            label=label,
            zorder=4,
        )
        for gap, effect, count in points:
            if count > 1:
                peers = [point[1] for point in points if close(point[0], gap)]
                offset_y = -12 if len(peers) > 1 and effect == min(peers) else 4
                axis.annotate(
                    f"n={count}",
                    (gap, effect),
                    xytext=(5, offset_y),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
    axis.axhline(0, color="0.25", linewidth=0.8, linestyle="--")
    axis.set_xlabel("Cold - hot High-selection rate")
    axis.set_ylabel("Dynamic Lifetime - Task reward")
    axis.set_title(
        "b  Adaptation and effect (exploratory)", loc="left", fontweight="bold"
    )
    axis.grid(color="0.88", linewidth=0.65)
    axis.legend(frameon=False, loc="lower right")

    fig.tight_layout(w_pad=2.2)
    return save_figure(fig, output, "lifetime_policy_adaptation")


def format_p(value: float) -> str:
    return f"{value:.3f}" if value >= 0.001 else f"{value:.2e}"


def write_markdown(
    document: dict[str, Any],
    rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    reference: dict[str, float],
    input_path: Path,
    campaign_path: Path,
    design_path: Path,
    frozen_path: Path,
    output_path: Path,
) -> None:
    primary = document["primary_dynamic_lifetime_minus_task"]
    static = document["secondary_static_lifetime_minus_task"]
    interaction = document["secondary_difference_in_differences"]
    task_mean = float(np.mean([row["dynamic_task_reward"] for row in rows]))
    lifetime_mean = float(np.mean([row["dynamic_lifetime_reward"] for row in rows]))
    mean_high = float(np.mean([row["dynamic_lifetime_high_rate"] for row in rows]))
    mean_trip = float(np.mean([row["dynamic_lifetime_trip_rate"] for row in rows]))
    reactive_mean = reference["analytic_best_task_reactive_mean_task_reward"]
    descriptive_reactive_gap = lifetime_mean - reactive_mean
    cells = summarize_cells(cell_rows)
    cell_table = "\n".join(
        "| {condition} | {memory} | {reward_mean:.4f} ({reward_sample_sd:.4f}) | "
        "{high_selection_rate_mean:.4f} ({high_selection_rate_sample_sd:.4f}) | "
        "{trip_rate_mean:.4f} ({trip_rate_sample_sd:.4f}) |".format(**cell)
        for cell in cells
    )

    text = f"""# Held-out hierarchical thermal confirmatory results

## Provenance and validity

- Source: `{input_path.as_posix()}`
- Source SHA-256: `{sha256(input_path)}`
- Locally frozen campaign manifest: `{campaign_path.as_posix()}` (SHA-256 `{sha256(campaign_path)}`)
- Calibration design source: `{design_path.as_posix()}` (SHA-256 `{sha256(design_path)}`)
- Frozen protocol source: `{frozen_path.as_posix()}` (SHA-256 `{sha256(frozen_path)}`)
- Unit of analysis: training seed (`n = {primary["n"]}`)
- Status: `{document["status"]}`
- Wiring, static-control, and final confirmatory gates: **passed**
- The analysis plan was specified in the local campaign manifest before any held-out training. It was **not externally preregistered**.
- The inferential values below are copied from the frozen confirmatory result; this renderer does not rerun or alter the statistical analysis.

## Main result

Lifetime memory improved reward under dynamic thermal degradation by **{primary["mean"]:.4f} reward/task** (SD {primary["sd"]:.4f}; median {primary["median"]:.4f}; seed-bootstrap 95% CI [{primary["bootstrap_95_ci"][0]:.4f}, {primary["bootstrap_95_ci"][1]:.4f}]). The locally prespecified one-sided paired t-test gave *p* = {format_p(primary["one_sided_t_p"])}, and the one-sided Wilcoxon signed-rank sensitivity analysis gave *p* = {format_p(primary["one_sided_wilcoxon_p"])}. Effects were positive for {primary["positive_seeds"]}/{primary["n"]} held-out seeds.

The mean dynamic reward changed from {task_mean:.4f} (task-reset memory) to {lifetime_mean:.4f} (lifetime memory), an absolute gain of {primary["mean"]:.4f} reward/task. In the static negative control, the paired memory effect was exactly {static["mean"]:.1f} (95% CI [{static["bootstrap_95_ci"][0]:.1f}, {static["bootstrap_95_ci"][1]:.1f}]), so the difference-in-differences was {interaction["mean"]:.4f} (95% CI [{interaction["bootstrap_95_ci"][0]:.4f}, {interaction["bootstrap_95_ci"][1]:.4f}]).

## Copy-ready statistical table

| Contrast | n | Mean (SD) | Median | Seed-bootstrap 95% CI | One-sided paired t *p* | One-sided Wilcoxon *p* | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic: Lifetime - Task | {primary["n"]} | {primary["mean"]:.4f} ({primary["sd"]:.4f}) | {primary["median"]:.4f} | [{primary["bootstrap_95_ci"][0]:.4f}, {primary["bootstrap_95_ci"][1]:.4f}] | {format_p(primary["one_sided_t_p"])} | {format_p(primary["one_sided_wilcoxon_p"])} | {primary["positive_seeds"]}/{primary["n"]} |
| Static control: Lifetime - Task | {static["n"]} | {static["mean"]:.4f} ({static["sd"]:.4f}) | {static["median"]:.4f} | [{static["bootstrap_95_ci"][0]:.4f}, {static["bootstrap_95_ci"][1]:.4f}] | {format_p(static["one_sided_t_p"])} | {format_p(static["one_sided_wilcoxon_p"])} | {static["positive_seeds"]}/{static["n"]} |
| Difference-in-differences | {interaction["n"]} | {interaction["mean"]:.4f} ({interaction["sd"]:.4f}) | {interaction["median"]:.4f} | [{interaction["bootstrap_95_ci"][0]:.4f}, {interaction["bootstrap_95_ci"][1]:.4f}] | {format_p(interaction["one_sided_t_p"])} | {format_p(interaction["one_sided_wilcoxon_p"])} | {interaction["positive_seeds"]}/{interaction["n"]} |

## Four-cell descriptives across training seeds

Values are mean (sample SD), with the independent training seed as the unit (`n = 20` per cell).

| Condition | Memory | Reward | High-selection rate | Thermal-trip rate |
|---|---|---:|---:|---:|
{cell_table}

Both static arms converged to the deterministic always-High ceiling. Their exact-zero contrast is a wiring/negative-control check, not independent replication of the dynamic effect; therefore, the numerically identical difference-in-differences does not provide independent corroborating evidence.

## Calibration-only analytic references

| Policy | Mean task reward | Evidence scope |
|---|---:|---|
| Learned Task-reset | {task_mean:.4f} | Held-out confirmatory; learned policy |
| Learned Lifetime | {lifetime_mean:.4f} | Held-out confirmatory; learned policy |
| Analytic all-Low | {reference["analytic_all_low_mean_task_reward"]:.4f} | Calibration-only diagnostic |
| Analytic best task-reactive | {reactive_mean:.4f} | Calibration-only diagnostic |
| Analytic lifetime oracle | {reference["analytic_lifetime_oracle_mean_task_reward"]:.4f} | Calibration-only diagnostic |

The learned Task-reset arm coincided with the analytic all-Low solution. The learned Lifetime mean was descriptively {descriptive_reactive_gap:+.4f} reward/task relative to the calibration-only best task-reactive reference. **No held-out paired test was defined for that analytic reference, so the results do not establish that learned Lifetime memory significantly outperforms the strongest task-reactive rule.** The confirmed primary claim is limited to the prespecified learned Lifetime-versus-learned Task-reset contrast.

## Policy-behavior checks

- Mean cold-minus-hot High-selection gap: {document["mean_lifetime_adaptation_gap"]:.4f}
- Mean overall High-selection rate: {mean_high:.4f}
- Mean thermal-trip rate: {mean_trip:.4f}
- Static Task and Lifetime policies selected High at rate 1.0 for every seed.

The policy-behavior association is secondary and descriptive. It should not be presented as a causal mediation analysis.

The fixed 20-task lifetime and deterministic task boundaries permit elapsed-task counting as a cue. This experiment does not separate counting-based health inference from inference based on degradation-linked observations.

## Figure captions

**Figure 1 — Confirmatory memory effect.** (a) Paired held-out-seed rewards for task-reset and lifetime memory under dynamic thermal degradation. Lines join policies trained with the same seed. The green dashed line is the calibration-only analytic best task-reactive reference and is not an inferential confirmatory comparator. (b) Seed-level Lifetime-minus-Task effects under dynamic degradation and the static negative control. Diamonds show means; error bars show the stored 100,000-resample seed-bootstrap 95% confidence intervals. Higher reward is better.

**Figure 2 — Lifetime-policy adaptation.** (a) Overall High-action selection and cold-minus-hot High-selection differences for each held-out seed; the dashed line is the mean adaptation gap. (b) Descriptive relationship between adaptation gap and paired reward effect. Crosses indicate seeds with at least one observed thermal trip; marker area and `n` labels show coincident seeds. This panel is exploratory.

## Files

- `confirmatory_seed_results.csv`: one row per held-out training seed
- `confirmatory_statistical_table.csv`: paper-table values
- `confirmatory_cell_summary.csv`: four-cell reward/High/trip mean and sample SD
- `calibration_analytic_references.csv`: clearly scoped diagnostic references
- `confirmatory_memory_effect.svg` / `.png`: primary confirmatory figure
- `lifetime_policy_adaptation.svg` / `.png`: secondary behavior figure
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render paper-ready v10 held-out confirmatory artifacts."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--campaign-manifest", type=Path, default=DEFAULT_CAMPAIGN_MANIFEST
    )
    parser.add_argument("--design-search", type=Path, default=DEFAULT_DESIGN_SEARCH)
    parser.add_argument("--frozen-protocol", type=Path, default=DEFAULT_FROZEN_PROTOCOL)
    args = parser.parse_args()

    input_path = args.input.resolve()
    campaign_path = args.campaign_manifest.resolve()
    design_path = args.design_search.resolve()
    frozen_path = args.frozen_protocol.resolve()
    renderer_path = Path(__file__).resolve()
    output = args.output.resolve()
    require(input_path.is_file(), f"input does not exist: {input_path}")
    require(
        campaign_path.is_file(), f"campaign manifest does not exist: {campaign_path}"
    )
    require(design_path.is_file(), f"design search does not exist: {design_path}")
    require(frozen_path.is_file(), f"frozen protocol does not exist: {frozen_path}")
    document = json.loads(input_path.read_text(encoding="utf-8"))
    campaign_document = json.loads(campaign_path.read_text(encoding="utf-8"))
    design_document = json.loads(design_path.read_text(encoding="utf-8"))
    frozen_document = json.loads(frozen_path.read_text(encoding="utf-8"))
    rows = validate(document)
    cell_rows = validate_campaign(
        campaign_path.parent,
        campaign_document,
        frozen_document,
        frozen_path,
        rows,
    )
    reference = calibration_reference(design_document, frozen_document)
    output.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    generated = [
        output / "confirmatory_seed_results.csv",
        output / "confirmatory_statistical_table.csv",
        output / "calibration_analytic_references.csv",
        output / "confirmatory_cell_summary.csv",
    ]
    write_seed_csv(rows, generated[0])
    write_summary_csv(document, generated[1])
    write_calibration_reference_csv(reference, rows, generated[2])
    write_cell_summary_csv(cell_rows, generated[3])
    generated.extend(render_confirmatory_effects(document, rows, reference, output))
    generated.extend(render_policy_adaptation(document, rows, output))
    markdown = output / "RESULTS_SUMMARY.md"
    write_markdown(
        document,
        rows,
        cell_rows,
        reference,
        input_path,
        campaign_path,
        design_path,
        frozen_path,
        markdown,
    )
    generated.append(markdown)

    manifest = {
        "phase": "hierarchical_thermal_confirmatory_paper_artifacts",
        "sources": {
            "confirmatory_result": {
                "path": str(input_path),
                "sha256": sha256(input_path),
            },
            "confirmatory_campaign_manifest": {
                "path": str(campaign_path),
                "sha256": sha256(campaign_path),
            },
            "calibration_design": {
                "path": str(design_path),
                "sha256": sha256(design_path),
            },
            "frozen_protocol": {
                "path": str(frozen_path),
                "sha256": sha256(frozen_path),
            },
            "artifact_renderer": {
                "path": str(renderer_path),
                "sha256": sha256(renderer_path),
            },
        },
        "validated_seed_count": len(rows),
        "validated_metadata_count": len(cell_rows),
        "confirmatory_passed": document["confirmatory_passed"],
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        },
    }
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
