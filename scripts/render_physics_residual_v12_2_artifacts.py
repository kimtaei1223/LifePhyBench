#!/usr/bin/env python3
"""Validate v12.2 evidence and render publication-ready tables and figures."""

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
DEFAULT_ROOT = ROOT / "outputs" / "physics_residual_v12_2_scoped_confirmatory"
DEFAULT_OUTPUT = DEFAULT_ROOT / "paper_artifacts"
CONDITION_ORDER = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_combined",
    "ood_shocks",
)
CONDITION_LABELS = {
    "in_domain": "In-domain",
    "ood_sensor_noise": "Sensor noise",
    "ood_cooling": "Cooling shift",
    "ood_combined": "Combined shift",
    "ood_shocks": "Physical shocks",
}
POLICY_LABELS = {
    "current_sensor": "Current sensor",
    "ema_history": "EMA history",
    "physics_belief": "Physics belief",
    "hybrid_belief": "Physics + residual",
    "privileged_oracle": "Privileged oracle",
}
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#B8B8B8"
SVG_HASH_SALT = "lifephybench-physics-residual-v12-2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(left: float, right: float, *, atol: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"JSON root must be an object: {path}")
    return document


def paired_effect_rows(
    cells: dict[str, dict[str, dict[str, Any]]], seeds: list[int]
) -> list[dict[str, Any]]:
    by_condition: dict[str, dict[int, float]] = {}
    for condition in CONDITION_ORDER:
        policies = cells[condition]

        def rewards(policy: str) -> dict[int, float]:
            return {
                int(row["seed"]): float(row["mean_reward_per_task"])
                for row in policies[policy]["summary"]["lifetime_rows"]
            }

        hybrid = rewards("hybrid_belief")
        physics = rewards("physics_belief")
        require(set(hybrid) == set(seeds), f"{condition}: hybrid seed mismatch")
        require(set(physics) == set(seeds), f"{condition}: physics seed mismatch")
        by_condition[condition] = {
            seed: hybrid[seed] - physics[seed] for seed in seeds
        }
    rows = []
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed}
        for condition in CONDITION_ORDER:
            row[f"{condition}_hybrid_minus_physics"] = by_condition[condition][seed]
        row["target_ood_aggregate_hybrid_minus_physics"] = float(
            np.mean(
                [
                    by_condition[condition][seed]
                    for condition in (
                        "ood_sensor_noise",
                        "ood_cooling",
                        "ood_combined",
                    )
                ]
            )
        )
        rows.append(row)
    return rows


def validate(
    protocol: dict[str, Any],
    result: dict[str, Any],
    cells: dict[str, dict[str, dict[str, Any]]],
    *,
    protocol_path: Path,
    protocol_hash_path: Path,
) -> list[dict[str, Any]]:
    require(
        protocol.get("phase") == "physics_residual_v12_2_scoped_frozen_protocol",
        "unexpected protocol phase",
    )
    require(
        protocol.get("status") == "frozen_before_any_v12_2_heldout_evaluation",
        "protocol was not frozen before held-out evaluation",
    )
    stored_protocol_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
    require(sha256(protocol_path) == stored_protocol_hash, "protocol hash mismatch")
    require(
        result.get("protocol_sha256") == stored_protocol_hash,
        "result/protocol hash mismatch",
    )
    require(
        result.get("phase") == "physics_residual_v12_2_scoped_heldout_confirmation",
        "unexpected result phase",
    )
    require(result.get("status") == "final_heldout_result", "result is not final")
    require(result.get("wiring_passed") is True, "wiring validation failed")
    require(result.get("confirmatory_passed") is True, "confirmatory gate failed")
    require(all(result["criteria"].values()), "not every frozen criterion passed")
    require(
        tuple(result["primary_scope"]) == (
            "ood_sensor_noise",
            "ood_cooling",
            "ood_combined",
        ),
        "primary scope changed",
    )
    require(
        tuple(result["secondary_boundary_conditions"]) == ("ood_shocks",),
        "secondary scope changed",
    )

    seeds = [int(seed) for seed in protocol["heldout_evaluation_seeds"]]
    require(len(seeds) == len(set(seeds)) == 100, "expected 100 unique held-out seeds")
    require(set(cells) == set(CONDITION_ORDER), "condition set mismatch")
    expected_policies = {spec["name"] for spec in protocol["policy_specs"]}
    require(expected_policies == set(POLICY_LABELS), "frozen policy set mismatch")

    for condition in CONDITION_ORDER:
        require(set(cells[condition]) == expected_policies, f"{condition}: policy set mismatch")
        for policy, record in cells[condition].items():
            summary = record["summary"]
            rows = summary["lifetime_rows"]
            row_seeds = [int(row["seed"]) for row in rows]
            require(len(rows) == 100, f"{condition}/{policy}: expected 100 rows")
            require(set(row_seeds) == set(seeds), f"{condition}/{policy}: seed mismatch")
            require(summary["lifetimes"] == 100, f"{condition}/{policy}: lifetime count mismatch")
            require(summary["tasks"] == 2000, f"{condition}/{policy}: task count mismatch")
            require(
                close(
                    np.mean([float(row["mean_reward_per_task"]) for row in rows]),
                    summary["mean_reward_per_task"],
                ),
                f"{condition}/{policy}: reward summary mismatch",
            )
            require(
                close(
                    np.mean([float(row["trip_rate"]) for row in rows]),
                    summary["trip_rate"],
                ),
                f"{condition}/{policy}: trip summary mismatch",
            )

    paired = paired_effect_rows(cells, seeds)
    for condition in CONDITION_ORDER:
        observed = [
            float(row[f"{condition}_hybrid_minus_physics"]) for row in paired
        ]
        stored = result["condition_effects"][condition]
        require(
            all(close(left, right) for left, right in zip(observed, stored["values"])),
            f"{condition}: stored paired effects mismatch",
        )
        require(close(np.mean(observed), stored["mean"]), f"{condition}: mean mismatch")
        require(
            close(np.std(observed, ddof=1), stored["sd"]),
            f"{condition}: sample SD mismatch",
        )
        require(
            close(
                cells[condition]["hybrid_belief"]["summary"]["trip_rate"],
                stored["hybrid_trip_rate"],
            ),
            f"{condition}: hybrid trip-rate mismatch",
        )

    aggregate_values = [
        float(row["target_ood_aggregate_hybrid_minus_physics"]) for row in paired
    ]
    aggregate = result["target_ood_aggregate_hybrid_minus_physics"]
    require(
        all(close(left, right) for left, right in zip(aggregate_values, aggregate["values"])),
        "target OOD aggregate values mismatch",
    )
    require(close(np.mean(aggregate_values), aggregate["mean"]), "aggregate mean mismatch")
    return paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.hashsalt": SVG_HASH_SALT,
        }
    )


def render_main_figure(result: dict[str, Any], output: Path) -> None:
    configure_plot_style()
    figure, (effect_axis, trip_axis) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    positions = np.arange(len(CONDITION_ORDER))[::-1]
    colors = [GRAY, BLUE, BLUE, BLUE, ORANGE]
    for position, condition, color in zip(positions, CONDITION_ORDER, colors):
        row = result["condition_effects"][condition]
        low, high = row["bootstrap_95_ci"]
        effect_axis.errorbar(
            row["mean"],
            position,
            xerr=[[row["mean"] - low], [high - row["mean"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
    effect_axis.axvline(0.0, color=LIGHT_GRAY, linewidth=1, linestyle="--")
    effect_axis.set_yticks(positions, [CONDITION_LABELS[name] for name in CONDITION_ORDER])
    effect_axis.set_xlabel("Reward/task: hybrid − physics")
    effect_axis.set_title("(a) Held-out paired effects (95% bootstrap CI)")
    effect_axis.grid(axis="x", alpha=0.2)

    x = np.arange(len(CONDITION_ORDER))
    hybrid = [result["condition_effects"][name]["hybrid_trip_rate"] * 100 for name in CONDITION_ORDER]
    physics = [result["condition_effects"][name]["physics_trip_rate"] * 100 for name in CONDITION_ORDER]
    width = 0.36
    trip_axis.bar(x - width / 2, physics, width, color=LIGHT_GRAY, label="Physics belief")
    trip_axis.bar(x + width / 2, hybrid, width, color=GREEN, label="Physics + residual")
    trip_axis.axhline(2.0, color=ORANGE, linewidth=1, linestyle="--", label="2% safety gate")
    trip_axis.set_xticks(x, ["ID", "Noise", "Cooling", "Combined", "Shocks"], rotation=25)
    trip_axis.set_ylabel("Thermal-trip rate (%)")
    trip_axis.set_title("(b) Safety outcomes")
    trip_axis.legend(frameon=False)
    trip_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "figure_v12_2_effects_and_safety.svg", metadata={"Date": None})
    figure.savefig(output / "figure_v12_2_effects_and_safety.png", metadata={"Software": "LifePhyBench"})
    plt.close(figure)


def render_seed_figure(result: dict[str, Any], output: Path) -> None:
    configure_plot_style()
    aggregate = result["target_ood_aggregate_hybrid_minus_physics"]
    values = np.sort(np.asarray(aggregate["values"], dtype=np.float64))
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    colors = np.where(values > 0.0, GREEN, ORANGE)
    axis.scatter(np.arange(1, len(values) + 1), values, c=colors, s=18, alpha=0.85)
    axis.axhline(0.0, color=GRAY, linewidth=1, linestyle="--")
    axis.axhline(aggregate["mean"], color=BLUE, linewidth=1.5, label=f"Mean = {aggregate['mean']:.3f}")
    low, high = aggregate["bootstrap_95_ci"]
    axis.axhspan(low, high, color=BLUE, alpha=0.12, label=f"95% CI [{low:.3f}, {high:.3f}]")
    axis.set_xlabel("Held-out seed rank (sorted by paired effect)")
    axis.set_ylabel("Target-OOD reward/task: hybrid − physics")
    axis.set_title("Held-out target-OOD aggregate across 100 paired seeds")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "figure_v12_2_seed_effects.svg", metadata={"Date": None})
    figure.savefig(output / "figure_v12_2_seed_effects.png", metadata={"Software": "LifePhyBench"})
    plt.close(figure)


def format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def write_summaries(result: dict[str, Any], output: Path) -> None:
    aggregate = result["target_ood_aggregate_hybrid_minus_physics"]
    effects = result["condition_effects"]
    korean = f"""# v12.2 확증 결과 — 논문 삽입용 요약

사전에 동결한 100개의 독립 held-out lifetime seed에서 physics-plus-residual 정책은 physics-belief 기준선보다 세 가지 목표 OOD 조건의 평균 보상을 태스크당 **{aggregate['mean']:.3f}** 향상시켰다(seed-bootstrap 95% CI [{aggregate['bootstrap_95_ci'][0]:.3f}, {aggregate['bootstrap_95_ci'][1]:.3f}], paired sign-flip p {format_p(aggregate['sign_flip_two_sided_p'])}). 센서 노이즈, 냉각 변화, 복합 변화에서의 개선은 각각 **{effects['ood_sensor_noise']['mean']:.3f}**, **{effects['ood_cooling']['mean']:.3f}**, **{effects['ood_combined']['mean']:.3f}** reward/task였으며, 세 신뢰구간 모두 0보다 높았다. 정상 환경 효과는 {effects['in_domain']['mean']:+.3f}로 사전 비열등성 기준을 만족했다. 모든 평가 조건에서 hybrid 정책의 thermal-trip rate는 2% 이하로 유지되었다.

물리 충격 조건의 효과는 {effects['ood_shocks']['mean']:+.3f} (95% CI [{effects['ood_shocks']['bootstrap_95_ci'][0]:.3f}, {effects['ood_shocks']['bootstrap_95_ci'][1]:.3f}])로 불확실했다. 이 조건은 프로토콜에서 보조 경계조건으로 지정되었으므로 순간적인 외부 충격에 대한 성능 개선을 주장하지 않는다.

분석 단위는 개별 태스크가 아니라 독립 lifetime seed이며, 모든 비교는 동일 seed의 hybrid와 physics 정책을 짝지어 계산했다. 프로토콜은 held-out 평가 전에 로컬 파일로 동결되었으나 외부 사전등록은 아니다.
"""
    english = f"""# v12.2 confirmatory result — manuscript-ready summary

Across 100 independently held-out lifetime seeds, the physics-plus-residual policy improved reward over the physics-belief baseline by **{aggregate['mean']:.3f} reward/task** when averaged across the three prespecified target shifts (seed-bootstrap 95% CI [{aggregate['bootstrap_95_ci'][0]:.3f}, {aggregate['bootstrap_95_ci'][1]:.3f}]; paired sign-flip p {format_p(aggregate['sign_flip_two_sided_p'])}). Improvements under sensor-noise, cooling-model, and combined shifts were **{effects['ood_sensor_noise']['mean']:.3f}**, **{effects['ood_cooling']['mean']:.3f}**, and **{effects['ood_combined']['mean']:.3f} reward/task**, respectively, with all three confidence intervals above zero. The in-domain effect was {effects['in_domain']['mean']:+.3f}, satisfying the frozen non-inferiority rule. Hybrid thermal-trip rates remained at or below 2% in every evaluated condition.

The physical-shock effect was uncertain ({effects['ood_shocks']['mean']:+.3f}; 95% CI [{effects['ood_shocks']['bootstrap_95_ci'][0]:.3f}, {effects['ood_shocks']['bootstrap_95_ci'][1]:.3f}]). This was a protocol-designated secondary boundary condition, so no improvement claim is made for abrupt external shocks.

The independent lifetime seed—not the individual task—was the unit of analysis, and all contrasts paired hybrid and physics policies by seed. The protocol was frozen locally before held-out evaluation but was not externally preregistered.
"""
    (output / "PAPER_RESULTS_KO.md").write_text(korean, encoding="utf-8")
    (output / "PAPER_RESULTS_EN.md").write_text(english, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = input_root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = input_root / "FROZEN_PROTOCOL.sha256"
    result_path = input_root / "CONFIRMATORY_RESULTS.json"
    cells_path = input_root / "CONFIRMATORY_CELLS.json"
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    cells = read_json(cells_path)
    paired = validate(
        protocol,
        result,
        cells,
        protocol_path=protocol_path,
        protocol_hash_path=protocol_hash_path,
    )

    condition_rows = []
    for condition in CONDITION_ORDER:
        row = result["condition_effects"][condition]
        condition_rows.append(
            {
                "condition": condition,
                "label": CONDITION_LABELS[condition],
                "role": "primary" if condition in result["primary_scope"] else ("noninferiority" if condition == "in_domain" else "secondary_boundary"),
                "n_seeds": row["n"],
                "hybrid_minus_physics_mean": row["mean"],
                "sample_sd": row["sd"],
                "bootstrap_95_ci_low": row["bootstrap_95_ci"][0],
                "bootstrap_95_ci_high": row["bootstrap_95_ci"][1],
                "sign_flip_two_sided_p": row["sign_flip_two_sided_p"],
                "hybrid_trip_rate": row["hybrid_trip_rate"],
                "physics_trip_rate": row["physics_trip_rate"],
            }
        )
    policy_rows = []
    for condition in CONDITION_ORDER:
        for policy in POLICY_LABELS:
            summary = cells[condition][policy]["summary"]
            policy_rows.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "n_seeds": summary["lifetimes"],
                    "mean_reward_per_task": summary["mean_reward_per_task"],
                    "trip_rate": summary["trip_rate"],
                    "mixed_mode_lifetime_rate": summary["mixed_mode_lifetime_rate"],
                }
            )
    write_csv(output / "condition_effects.csv", condition_rows)
    write_csv(output / "policy_summaries.csv", policy_rows)
    write_csv(output / "paired_seed_effects.csv", paired)
    render_main_figure(result, output)
    render_seed_figure(result, output)
    write_summaries(result, output)

    artifact_paths = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "phase": "physics_residual_v12_2_publication_artifacts",
        "status": "complete",
        "validation_passed": True,
        "confirmatory_passed": True,
        "protocol_sha256": sha256(protocol_path),
        "inputs": {
            "protocol": sha256(protocol_path),
            "result": sha256(result_path),
            "cells": sha256(cells_path),
        },
        "artifacts": {path.name: sha256(path) for path in artifact_paths},
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
