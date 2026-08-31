"""Analyze crossed train/evaluation physics for canonical thermal policies."""

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

from analyze_canonical_thermal_probe import (
    EXPECTED_SEEDS,
    bootstrap_mean_ci,
    paired_contrast,
)


LABELS = ("dynamic", "static")
MEMORY_MODES = ("task", "lifetime")


def load_records(input_root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(input_root.glob("*--eval-*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        evaluation = document["task_episode_evaluation"]
        records.append(
            {
                "source": str(path),
                "run_name": document["run_name"],
                "training_label": document["training_label"],
                "evaluation_label": document["evaluation_label"],
                "memory": document["memory_mode"],
                "seed": int(document["seed"]),
                "reward": float(evaluation["mean_task_episode_reward"]),
                "reward_sd_within_policy": float(
                    evaluation["std_task_episode_reward"]
                ),
                "thermal_load": float(evaluation["mean_episode_end_thermal_load"]),
                "efficiency": float(evaluation["mean_episode_end_efficiency"]),
                "task_episodes": int(evaluation["task_episodes"]),
                "completed_lifetimes": int(evaluation["completed_lifetimes"]),
                "is_native": bool(document["is_native_evaluation"]),
                "native_replay_absolute_error": document[
                    "native_replay_absolute_error"
                ],
            }
        )
    return records


def validate(records: list[dict[str, object]]) -> None:
    expected = {
        (train, evaluation, memory, seed)
        for train in LABELS
        for evaluation in LABELS
        for memory in MEMORY_MODES
        for seed in EXPECTED_SEEDS
    }
    seen: set[tuple[str, str, str, int]] = set()
    for record in records:
        key = (
            str(record["training_label"]),
            str(record["evaluation_label"]),
            str(record["memory"]),
            int(record["seed"]),
        )
        if key in seen:
            raise SystemExit(f"duplicate counterfactual result: {key}")
        seen.add(key)
        if record["task_episodes"] != 1_000 or record["completed_lifetimes"] != 50:
            raise SystemExit(f"evaluation budget mismatch: {record['source']}")
        native_expected = record["training_label"] == record["evaluation_label"]
        if record["is_native"] != native_expected:
            raise SystemExit(f"native label mismatch: {record['source']}")
        if record["evaluation_label"] == "static" and (
            record["thermal_load"] != 0.0 or record["efficiency"] != 1.0
        ):
            raise SystemExit(f"static counterfactual changed health: {record['source']}")
        if record["evaluation_label"] == "dynamic" and (
            record["thermal_load"] <= 0.0 or record["efficiency"] >= 1.0
        ):
            raise SystemExit(f"dynamic counterfactual did not change health: {record['source']}")
    if seen != expected:
        raise SystemExit(
            f"incomplete crossed evaluation: missing={sorted(expected - seen)}, "
            f"unexpected={sorted(seen - expected)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/canonical_thermal_counterfactual"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/canonical_thermal_counterfactual_analysis"),
    )
    args = parser.parse_args()
    records = load_records(args.input_root)
    validate(records)
    args.output_root.mkdir(parents=True, exist_ok=True)
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    with (args.output_root / "per_seed_crossed_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda row: (
                    row["training_label"],
                    row["evaluation_label"],
                    row["memory"],
                    row["seed"],
                ),
            )
        )

    grouped: dict[tuple[str, str, str], dict[int, dict[str, object]]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["training_label"]),
            str(record["evaluation_label"]),
            str(record["memory"]),
        )
        grouped[key][int(record["seed"])] = record

    rng = np.random.default_rng(20260823)
    groups = {}
    for train in LABELS:
        for evaluation in LABELS:
            for memory in MEMORY_MODES:
                values = np.asarray(
                    [
                        float(grouped[(train, evaluation, memory)][seed]["reward"])
                        for seed in EXPECTED_SEEDS
                    ]
                )
                groups[f"train_{train}__eval_{evaluation}__{memory}"] = {
                    "n": len(values),
                    "reward_mean": float(values.mean()),
                    "reward_sd": float(values.std(ddof=1)),
                    "reward_bootstrap_ci95": bootstrap_mean_ci(values, rng),
                    "seed_rewards": values.tolist(),
                }

    memory_effects: dict[tuple[str, str], np.ndarray] = {}
    for train in LABELS:
        for evaluation in LABELS:
            memory_effects[(train, evaluation)] = np.asarray(
                [
                    float(grouped[(train, evaluation, "lifetime")][seed]["reward"])
                    - float(grouped[(train, evaluation, "task")][seed]["reward"])
                    for seed in EXPECTED_SEEDS
                ]
            )

    train_dynamic_eval_interaction = (
        memory_effects[("dynamic", "dynamic")]
        - memory_effects[("dynamic", "static")]
    )
    train_static_eval_interaction = (
        memory_effects[("static", "dynamic")]
        - memory_effects[("static", "static")]
    )
    native_interaction = (
        memory_effects[("dynamic", "dynamic")]
        - memory_effects[("static", "static")]
    )
    triple_interaction = train_dynamic_eval_interaction - train_static_eval_interaction
    contrasts = {
        **{
            f"train_{train}__eval_{evaluation}__lifetime_minus_task": paired_contrast(
                memory_effects[(train, evaluation)], rng
            )
            for train in LABELS
            for evaluation in LABELS
        },
        "dynamic_trained__dynamic_minus_static_eval_memory_interaction": paired_contrast(
            train_dynamic_eval_interaction, rng
        ),
        "static_trained__dynamic_minus_static_eval_memory_interaction": paired_contrast(
            train_static_eval_interaction, rng
        ),
        "native_dynamic_minus_static_memory_interaction": paired_contrast(
            native_interaction, rng
        ),
        "three_way_train_x_eval_x_memory_interaction": paired_contrast(
            triple_interaction, rng
        ),
    }

    native_errors = [
        float(record["native_replay_absolute_error"])
        for record in records
        if record["is_native"] and record["native_replay_absolute_error"] is not None
    ]
    report = {
        "phase": "canonical_thermal_policy_counterfactual_analysis",
        "status": "post_hoc_diagnostic_not_new_confirmatory_test",
        "analysis_population": {
            "seeds": EXPECTED_SEEDS,
            "policies": 40,
            "evaluations": len(records),
            "exclusions": [],
        },
        "protocol_validation": {
            "passed": True,
            "native_replays": len(native_errors),
            "maximum_native_replay_absolute_error": max(native_errors),
            "static_health_unchanged": True,
            "dynamic_health_changed_in_every_evaluation": True,
        },
        "groups": groups,
        "contrasts": contrasts,
        "interpretation_guardrail": (
            "Cross-environment evaluation is diagnostic and distribution-shifted; "
            "it cannot replace the frozen native interaction test."
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Canonical thermal-policy counterfactual analysis",
        "",
        "This is a post-hoc diagnostic. Every frozen policy is evaluated in both dynamic and static physics; no policy is retrained.",
        "",
        f"Native replay maximum absolute reward error: {max(native_errors):.8f}.",
        "",
        "| Training physics | Evaluation physics | Memory | Reward mean ± SD |",
        "|---|---|---|---:|",
    ]
    for train in LABELS:
        for evaluation in LABELS:
            for memory in MEMORY_MODES:
                value = groups[f"train_{train}__eval_{evaluation}__{memory}"]
                lines.append(
                    f"| {train} | {evaluation} | {memory} | "
                    f"{value['reward_mean']:.3f} ± {value['reward_sd']:.3f} |"
                )
    lines += [
        "",
        "| Paired contrast | Mean | 95% parametric CI | t p | Exact sign-flip p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in contrasts.items():
        ci = value["parametric_ci95"]
        lines.append(
            f"| {name} | {value['mean']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{value['paired_t_test_p']:.4f} | {value['exact_sign_flip_p']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        report["interpretation_guardrail"],
    ]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    x = np.arange(len(EXPECTED_SEEDS))
    for axis, train in zip(axes, LABELS):
        axis.axhline(0.0, color="black", linewidth=0.8)
        for evaluation, marker in (("dynamic", "o"), ("static", "s")):
            axis.plot(
                x,
                memory_effects[(train, evaluation)],
                marker=marker,
                label=f"Evaluate {evaluation}",
            )
        axis.set_title(f"Trained in {train}")
        axis.set_xticks(x, [str(seed) for seed in EXPECTED_SEEDS], rotation=45)
        axis.set_xlabel("Training seed")
    axes[0].set_ylabel("Lifetime − task reward")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "crossed_memory_effects.png", dpi=180)
    figure.savefig(figures / "crossed_memory_effects.pdf")
    plt.close(figure)

    print(json.dumps(report["protocol_validation"], indent=2))


if __name__ == "__main__":
    main()
