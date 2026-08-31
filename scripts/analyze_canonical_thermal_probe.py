"""Analyze the frozen canonical-reset thermal-probe campaign."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lifephybench-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon


LABELS = ("dynamic", "static")
MEMORY_MODES = ("task", "lifetime")
EXPECTED_SEEDS = list(range(4000, 4010))


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, draws: int = 50_000
) -> list[float]:
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def exact_sign_flip_p(values: np.ndarray) -> float:
    """Two-sided paired randomization test, exact for the ten seed contrasts."""
    observed = abs(float(values.mean()))
    means = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def paired_contrast(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    sd = float(values.std(ddof=1))
    sem = sd / np.sqrt(len(values))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    mean = float(values.mean())
    try:
        wilcoxon_p = float(wilcoxon(values).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "cohen_dz": mean / sd if sd else None,
        "parametric_ci95": [mean - critical * sem, mean + critical * sem],
        "bootstrap_ci95": bootstrap_mean_ci(values, rng),
        "paired_t_test_p": float(ttest_1samp(values, 0.0).pvalue) if sd else 1.0,
        "wilcoxon_p": wilcoxon_p,
        "exact_sign_flip_p": exact_sign_flip_p(values),
        "seed_values": values.tolist(),
    }


def load_records(input_root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(input_root.glob("*/metadata.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        arguments = document["arguments"]
        evaluation = document["task_episode_evaluation"]
        run_name = str(arguments["run_name"])
        label = "dynamic" if "-dynamic-" in run_name else "static" if "-static-" in run_name else ""
        records.append(
            {
                "source": str(path),
                "run_name": run_name,
                "label": label,
                "memory": str(arguments["memory_mode"]),
                "seed": int(arguments["seed"]),
                "degradation_mode": str(arguments["degradation_mode"]),
                "reward": float(evaluation["mean_task_episode_reward"]),
                "reward_sd_within_policy": float(evaluation["std_task_episode_reward"]),
                "thermal_load": float(evaluation["mean_episode_end_thermal_load"]),
                "efficiency": float(evaluation["mean_episode_end_efficiency"]),
                "task_episodes": int(evaluation["task_episodes"]),
                "completed_lifetimes": int(evaluation["completed_lifetimes"]),
                "total_timesteps": int(arguments["total_timesteps"]),
                "canonical_task_seed": arguments.get("canonical_task_seed"),
                "thermal_heat_rate": arguments.get("thermal_heat_rate"),
                "thermal_cooling_rate": arguments.get("thermal_cooling_rate"),
                "thermal_episode_cooling": arguments.get("thermal_episode_cooling"),
                "thermal_exogenous_dose_per_step": arguments.get(
                    "thermal_exogenous_dose_per_step"
                ),
            }
        )
    return records


def validate(records: list[dict[str, object]]) -> None:
    expected = {
        (label, memory, seed)
        for label in LABELS
        for memory in MEMORY_MODES
        for seed in EXPECTED_SEEDS
    }
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        key = (str(record["label"]), str(record["memory"]), int(record["seed"]))
        if key in seen:
            raise SystemExit(f"duplicate result: {key}")
        seen.add(key)
        expected_degradation = "endogenous_action" if record["label"] == "dynamic" else "exogenous_clock"
        checks = {
            "degradation_mode": record["degradation_mode"] == expected_degradation,
            "task_episodes": record["task_episodes"] == 1_000,
            "completed_lifetimes": record["completed_lifetimes"] == 50,
            "total_timesteps": record["total_timesteps"] == 2_000_000,
            "canonical_task_seed": record["canonical_task_seed"] == 811,
            "thermal_heat_rate": record["thermal_heat_rate"] == 0.1,
            "thermal_cooling_rate": record["thermal_cooling_rate"] == 0.0,
            "thermal_episode_cooling": record["thermal_episode_cooling"] == 0.0,
            "thermal_exogenous_dose_per_step": record["thermal_exogenous_dose_per_step"] == 0.0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise SystemExit(f"protocol mismatch {failed}: {record['source']}")
        if record["label"] == "static" and (
            record["thermal_load"] != 0.0 or record["efficiency"] != 1.0
        ):
            raise SystemExit(f"static health changed: {record['source']}")
        if record["label"] == "dynamic" and (
            record["thermal_load"] <= 0.0 or record["efficiency"] >= 1.0
        ):
            raise SystemExit(f"dynamic health did not change: {record['source']}")
    if seen != expected:
        raise SystemExit(
            f"incomplete factorial campaign: missing={sorted(expected - seen)}, "
            f"unexpected={sorted(seen - expected)}"
        )


def group_summary(values: np.ndarray, rng: np.random.Generator) -> dict[str, object]:
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "bootstrap_ci95": bootstrap_mean_ci(values, rng),
        "seed_values": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("outputs/canonical_thermal_probe")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/canonical_thermal_probe_analysis"),
    )
    args = parser.parse_args()

    records = load_records(args.input_root)
    validate(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    with (args.output_root / "per_seed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            sorted(records, key=lambda row: (row["label"], row["memory"], row["seed"]))
        )

    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["label"]), str(record["memory"]))][int(record["seed"])] = record

    rng = np.random.default_rng(20260823)
    group_results = {}
    for label in LABELS:
        for memory in MEMORY_MODES:
            key = (label, memory)
            rows = [grouped[key][seed] for seed in EXPECTED_SEEDS]
            group_results[f"{label}_{memory}"] = {
                "reward": group_summary(
                    np.asarray([float(row["reward"]) for row in rows]), rng
                ),
                "thermal_load": group_summary(
                    np.asarray([float(row["thermal_load"]) for row in rows]), rng
                ),
                "efficiency": group_summary(
                    np.asarray([float(row["efficiency"]) for row in rows]), rng
                ),
            }

    effects = {}
    for label in LABELS:
        effects[label] = np.asarray(
            [
                float(grouped[(label, "lifetime")][seed]["reward"])
                - float(grouped[(label, "task")][seed]["reward"])
                for seed in EXPECTED_SEEDS
            ]
        )
    interaction = effects["dynamic"] - effects["static"]
    contrasts = {
        "dynamic_lifetime_minus_task": paired_contrast(effects["dynamic"], rng),
        "static_lifetime_minus_task": paired_contrast(effects["static"], rng),
        "primary_dynamic_minus_static_interaction": paired_contrast(interaction, rng),
    }
    dynamic_thermal_effect = np.asarray(
        [
            float(grouped[("dynamic", "lifetime")][seed]["thermal_load"])
            - float(grouped[("dynamic", "task")][seed]["thermal_load"])
            for seed in EXPECTED_SEEDS
        ]
    )
    dynamic_efficiency_effect = np.asarray(
        [
            float(grouped[("dynamic", "lifetime")][seed]["efficiency"])
            - float(grouped[("dynamic", "task")][seed]["efficiency"])
            for seed in EXPECTED_SEEDS
        ]
    )
    auxiliary_physical_contrasts = {
        "dynamic_thermal_load_lifetime_minus_task": paired_contrast(
            dynamic_thermal_effect, rng
        ),
        "dynamic_efficiency_lifetime_minus_task": paired_contrast(
            dynamic_efficiency_effect, rng
        ),
    }
    primary = contrasts["primary_dynamic_minus_static_interaction"]
    primary_ci = primary["parametric_ci95"]
    supports_claim = bool(primary["mean"] > 0 and primary_ci[0] > 0)

    report = {
        "phase": "canonical_thermal_probe_confirmatory_analysis",
        "analysis_population": {
            "seeds": EXPECTED_SEEDS,
            "n_per_cell": 10,
            "runs": len(records),
            "exclusions": [],
        },
        "protocol_validation": {
            "passed": True,
            "training_timesteps_per_policy": 2_000_000,
            "evaluation_task_episodes_per_policy": 1_000,
            "canonical_task_seed": 811,
            "static_health_unchanged": True,
            "dynamic_health_changed_in_every_run": True,
        },
        "groups": group_results,
        "contrasts": contrasts,
        "auxiliary_physical_contrasts": auxiliary_physical_contrasts,
        "decision": {
            "criterion": "primary interaction mean > 0 and two-sided parametric 95% CI excludes zero",
            "supports_physical_state_specific_memory_advantage": supports_claim,
            "interpretation": (
                "Dynamic degradation amplifies the lifetime-memory advantage beyond the static control."
                if supports_claim
                else "The experiment does not resolve a dynamic-specific lifetime-memory advantage beyond the static control."
            ),
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Canonical thermal-probe analysis",
        "",
        "All 40 frozen-design runs passed protocol validation; no seed was excluded.",
        "",
        "| Cell | Reward mean ± SD | Thermal load mean | Efficiency mean |",
        "|---|---:|---:|---:|",
    ]
    for label in LABELS:
        for memory in MEMORY_MODES:
            value = group_results[f"{label}_{memory}"]
            lines.append(
                f"| {label}_{memory} | {value['reward']['mean']:.3f} ± {value['reward']['sd']:.3f} "
                f"| {value['thermal_load']['mean']:.4f} | {value['efficiency']['mean']:.4f} |"
            )
    lines += [
        "",
        "| Paired contrast | Mean | SD | dz | Parametric 95% CI | Bootstrap 95% CI | t p | Wilcoxon p | Exact sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in contrasts.items():
        pci = value["parametric_ci95"]
        bci = value["bootstrap_ci95"]
        dz = value["cohen_dz"]
        lines.append(
            f"| {name} | {value['mean']:.3f} | {value['sd']:.3f} | "
            f"{dz:.3f} | [{pci[0]:.3f}, {pci[1]:.3f}] | "
            f"[{bci[0]:.3f}, {bci[1]:.3f}] | {value['paired_t_test_p']:.4f} | "
            f"{value['wilcoxon_p']:.4f} | {value['exact_sign_flip_p']:.4f} |"
        )
    lines += [
        "",
        "### Auxiliary physical-state checks",
        "",
        "| Paired dynamic-cell contrast | Mean | 95% parametric CI | t p |",
        "|---|---:|---:|---:|",
    ]
    for name, value in auxiliary_physical_contrasts.items():
        pci = value["parametric_ci95"]
        lines.append(
            f"| {name} | {value['mean']:.4f} | [{pci[0]:.4f}, {pci[1]:.4f}] "
            f"| {value['paired_t_test_p']:.4f} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        report["decision"]["interpretation"],
        "",
        (
            "The primary estimand is (dynamic lifetime − dynamic task) − "
            "(static lifetime − static task). Positive reward values are better."
        ),
    ]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(EXPECTED_SEEDS))
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.plot(x, effects["dynamic"], "o-", label="Dynamic: lifetime - task")
    axis.plot(x, effects["static"], "s-", label="Static: lifetime - task")
    axis.set_xticks(x, [str(seed) for seed in EXPECTED_SEEDS], rotation=45)
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Paired reward difference")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "paired_memory_effects.png", dpi=180)
    figure.savefig(figures / "paired_memory_effects.pdf")
    plt.close(figure)

    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
