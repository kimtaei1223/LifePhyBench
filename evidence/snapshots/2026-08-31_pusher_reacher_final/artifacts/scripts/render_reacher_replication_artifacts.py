#!/usr/bin/env python3
"""Validate and render the final Reacher replication and margin extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lifephybench-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["svg.hashsalt"] = "lifephybench-reacher-replication-final"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "reacher_replication"
DEFAULT_OUTPUT = ROOT / "paper_artifacts" / "reacher_replication"
TARGET_OOD = ("ood_sensor_noise", "ood_cooling", "ood_combined")
CONDITION_LABELS = {
    "in_domain": "In-domain",
    "ood_sensor_noise": "Sensor noise",
    "ood_cooling": "Cooling shift",
    "ood_combined": "Combined shift",
    "ood_shocks": "Physical shocks",
}
POLICY_LABELS = {
    "physics_z0": "Physics, z=0",
    "inherited_physics_z1_5": "Inherited, z=1.5",
    "selected_calibrated_margin": "Calibrated, z=2.0",
    "hybrid_z1_5": "Hybrid, z=1.5",
}
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRAY = "#666666"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    confirmatory = root / "confirmatory"
    extension = root / "margin_extension"
    confirm_result = read_json(confirmatory / "CONFIRMATORY_RESULTS.json")
    confirm_cells = read_json(confirmatory / "CONFIRMATORY_CELLS.json")
    fresh_result = read_json(extension / "FRESH_RESULTS.json")
    fresh_cells = read_json(extension / "FRESH_CELLS.json")
    selection = read_json(extension / "CALIBRATED_MARGIN_SELECTION.json")

    confirm_hash = (confirmatory / "FROZEN_PROTOCOL.sha256").read_text().strip()
    fresh_hash = (extension / "FROZEN_FRESH_PROTOCOL.sha256").read_text().strip()
    require(
        sha256(confirmatory / "FROZEN_PROTOCOL.json") == confirm_hash,
        "confirmatory protocol hash mismatch",
    )
    require(
        sha256(extension / "FROZEN_FRESH_PROTOCOL.json") == fresh_hash,
        "extension protocol hash mismatch",
    )
    require(confirm_result["protocol_sha256"] == confirm_hash, "confirmatory binding")
    require(fresh_result["protocol_sha256"] == fresh_hash, "extension binding")
    require(confirm_result["wiring_passed"] is True, "confirmatory wiring failed")
    require(confirm_result["primary_success"] is False, "original result changed")
    require(
        confirm_result["primary_criteria"][
            "maximum_trip_rate_each_condition_at_most_0_02"
        ]
        is False,
        "original safety-gate result changed",
    )
    require(fresh_result["original_inherited_margin_result_unchanged"] is True,
            "extension overwrote original interpretation")
    require(fresh_result["extension_success"] is True, "extension did not pass")
    require(all(fresh_result["criteria"].values()), "extension criterion failed")
    require(selection["confirmatory_evidence"] is False, "calibration mislabeled")
    require(selection["selected"] == fresh_result["selected_spec"], "selection drift")

    confirm_seeds = set(
        confirm_result["target_ood_aggregate_contrasts"]["primary_uncertainty"][
            "seeds"
        ]
    )
    fresh_seeds = set(
        fresh_result["target_ood_aggregate_contrasts"]["selected_vs_physics_z0"][
            "seeds"
        ]
    )
    require(len(confirm_seeds) == len(fresh_seeds) == 100, "expected 100 seeds")
    require(confirm_seeds.isdisjoint(fresh_seeds), "confirmatory/extension seed overlap")

    for name, cells in (("confirmatory", confirm_cells), ("extension", fresh_cells)):
        for condition, policies in cells.items():
            for policy, document in policies.items():
                rows = document["summary"]["lifetime_rows"]
                require(len(rows) == 100, f"{name}/{condition}/{policy}: row count")
                require(
                    len({int(row["seed"]) for row in rows}) == 100,
                    f"{name}/{condition}/{policy}: duplicate seed",
                )

    return {
        "confirm_result": confirm_result,
        "confirm_cells": confirm_cells,
        "fresh_result": fresh_result,
        "fresh_cells": fresh_cells,
        "selection": selection,
        "confirm_hash": confirm_hash,
        "fresh_hash": fresh_hash,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contrast_row(phase: str, name: str, value: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": phase,
        "contrast": name,
        "mean_reward_per_task": value["mean"],
        "bootstrap_95_ci_lower": value["bootstrap_95_ci"][0],
        "bootstrap_95_ci_upper": value["bootstrap_95_ci"][1],
        "sign_flip_two_sided_p": value["sign_flip_two_sided_p"],
        "exact_sign_two_sided_p": value["exact_sign_two_sided_p"],
        "positive_lifetimes": value["positive"],
        "negative_lifetimes": value["negative"],
        "lifetimes": value["n"],
    }


def make_tables(data: dict[str, Any], output: Path) -> None:
    confirm = data["confirm_result"]
    fresh = data["fresh_result"]
    contrast_rows = []
    for name, value in confirm["target_ood_aggregate_contrasts"].items():
        contrast_rows.append(contrast_row("inherited_confirmatory", name, value))
    for name, value in fresh["target_ood_aggregate_contrasts"].items():
        contrast_rows.append(contrast_row("post_confirmatory_extension", name, value))
    write_csv(output / "table_reacher_contrasts.csv", contrast_rows)

    condition_rows = []
    for condition, policies in data["fresh_cells"].items():
        for policy in POLICY_LABELS:
            summary = policies[policy]["summary"]
            condition_rows.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "mean_reward_per_task": summary["mean_reward_per_task"],
                    "thermal_trip_rate": summary["trip_rate"],
                    "high_power_rate": summary["high_rate"],
                    "lifetimes": summary["lifetimes"],
                    "tasks": summary["tasks"],
                }
            )
    write_csv(output / "table_reacher_condition_results.csv", condition_rows)

    frontier_rows = []
    for item in data["selection"]["frontier"]:
        frontier_rows.append(
            {
                "policy": item["spec"]["name"],
                "cutoff": item["spec"]["cutoff"],
                "uncertainty_multiplier": item["spec"]["uncertainty_multiplier"],
                "target_ood_mean_reward_per_task": item[
                    "target_ood_mean_reward_per_task"
                ],
                "maximum_trip_rate": item["maximum_trip_rate"],
                "buffered_safe": item["buffered_safe"],
                "selected": item["spec"] == data["selection"]["selected"],
            }
        )
    write_csv(output / "table_reacher_calibration_frontier.csv", frontier_rows)


def make_figure(data: dict[str, Any], output: Path) -> None:
    confirm = data["confirm_result"]
    fresh = data["fresh_result"]
    cells = data["fresh_cells"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    contrasts = [
        ("Inherited z=1.5", confirm["target_ood_aggregate_contrasts"]["primary_uncertainty"], ORANGE),
        ("Calibrated z=2.0", fresh["target_ood_aggregate_contrasts"]["selected_vs_physics_z0"], GREEN),
        ("Hybrid z=1.5", fresh["target_ood_aggregate_contrasts"]["hybrid_vs_physics_z0"], BLUE),
    ]
    for index, (label, value, color) in enumerate(contrasts):
        lo, hi = value["bootstrap_95_ci"]
        axes[0].errorbar(
            value["mean"], index,
            xerr=[[value["mean"] - lo], [hi - value["mean"]]],
            fmt="o", color=color, capsize=4, markersize=7,
        )
    axes[0].axvline(0, color=GRAY, linewidth=1, linestyle="--")
    axes[0].set_yticks(range(len(contrasts)), [item[0] for item in contrasts])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Target-OOD reward difference / task")
    axes[0].set_title("A. Paired mean effects (95% CI)")

    policies = ["physics_z0", "inherited_physics_z1_5", "selected_calibrated_margin", "hybrid_z1_5"]
    max_trips = [max(cells[c][p]["summary"]["trip_rate"] for c in cells) for p in policies]
    colors = [GRAY, ORANGE, GREEN, BLUE]
    axes[1].bar(range(len(policies)), np.asarray(max_trips) * 100, color=colors)
    axes[1].axhline(2.0, color="black", linestyle="--", linewidth=1, label="Frozen 2% gate")
    axes[1].set_xticks(range(len(policies)), ["z=0", "Inherited\nz=1.5", "Calibrated\nz=2.0", "Hybrid\nz=1.5"])
    axes[1].set_ylabel("Maximum trip rate (%)")
    axes[1].set_title("B. Worst-condition safety")
    axes[1].legend(frameon=False)

    frontier = data["selection"]["frontier"]
    safe = [x for x in frontier if x["buffered_safe"]]
    unsafe = [x for x in frontier if not x["buffered_safe"]]
    axes[2].scatter(
        [x["maximum_trip_rate"] * 100 for x in safe],
        [x["target_ood_mean_reward_per_task"] for x in safe],
        color=GREEN, label="Buffered-safe", s=42,
    )
    axes[2].scatter(
        [x["maximum_trip_rate"] * 100 for x in unsafe],
        [x["target_ood_mean_reward_per_task"] for x in unsafe],
        color=ORANGE, label="Rejected", marker="x", s=50,
    )
    selected = next(x for x in frontier if x["spec"] == data["selection"]["selected"])
    axes[2].scatter(
        [selected["maximum_trip_rate"] * 100],
        [selected["target_ood_mean_reward_per_task"]],
        facecolors="none", edgecolors="black", linewidths=1.5, s=130, label="Selected",
    )
    axes[2].axvline(1.5, color=GRAY, linestyle="--", linewidth=1, label="1.5% selection buffer")
    axes[2].set_xlabel("Calibration maximum trip rate (%)")
    axes[2].set_ylabel("Calibration target-OOD reward / task")
    axes[2].set_title("C. Development-only margin selection")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Reacher replication: inherited failure retained, calibrated safety restored", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        path = output / f"figure_reacher_replication_summary.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else None
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            ) + "\n"
            path.write_text(normalized, encoding="utf-8")
    plt.close(fig)


def fmt_ci(value: dict[str, Any]) -> str:
    lo, hi = value["bootstrap_95_ci"]
    return f"{value['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def make_summaries(data: dict[str, Any], output: Path) -> None:
    confirm = data["confirm_result"]
    fresh = data["fresh_result"]
    inherited = confirm["target_ood_aggregate_contrasts"]["primary_uncertainty"]
    selected = fresh["target_ood_aggregate_contrasts"]["selected_vs_physics_z0"]
    selected_vs_inherited = fresh["target_ood_aggregate_contrasts"]["selected_vs_inherited"]
    hybrid = confirm["target_ood_aggregate_contrasts"]["hybrid_vs_physics_z0"]
    mono = confirm["target_ood_aggregate_contrasts"]["monolithic_vs_physics_z0"]

    en = f"""# Reacher cross-task replication and calibrated-margin extension

Status: final results. The extension is post-confirmatory and does not replace the inherited-margin failure.

## Results

| Analysis | Target-OOD paired reward difference (95% bootstrap CI) | Maximum trip rate | Decision |
|---|---:|---:|---|
| Inherited physics margin `z=1.5` vs `z=0` | {fmt_ci(inherited)} | {confirm['maximum_physics_z1_5_trip_rate'] * 100:.1f}% | Original primary gate failed |
| Reacher-calibrated margin `z=2.0` vs `z=0` | {fmt_ci(selected)} | {fresh['maximum_selected_trip_rate'] * 100:.1f}% | Post-confirmatory extension passed |
| Calibrated `z=2.0` vs inherited `z=1.5` | {fmt_ci(selected_vs_inherited)} | -- | No mean reward improvement established |
| Hybrid `z=1.5` vs `z=0` | {fmt_ci(hybrid)} | 1.4% | Secondary result |
| Monolithic RecurrentPPO vs `z=0` | {fmt_ci(mono)} | -- | Strong OOD failure in this tested family |

The inherited margin improved expected target-OOD utility but missed its frozen safety gate (2.3% versus 2.0%). A cutoff/margin pair selected only on development seeds chose cutoff 0.06 and `z=2.0`. On 100 new lifetimes, it retained a positive mean effect and reduced the maximum trip rate to 1.6%. Its mean reward did not differ clearly from the inherited setting ({fmt_ci(selected_vs_inherited)}), so the extension supports task-specific safety calibration without detectable expected-utility loss, not a new reward gain.

Only {selected['positive']} of {selected['n']} paired lifetime effects were positive (exact sign test p={selected['exact_sign_two_sided_p']:.3f}). The supported claim concerns mean expected utility, not majority-lifetime improvement. Both tasks remain in one simulator with the same phenomenological thermal law; no real-hardware or formal-safety claim is supported.

Protocol hashes: inherited `{data['confirm_hash']}`; extension `{data['fresh_hash']}`.
"""
    ko = f"""# Reacher 교차 과제 재현 및 보정 여유 확장 결과

상태: 최종 결과. 사후 확장 실험은 최초 확증 실패를 대체하거나 성공으로 재분류하지 않는다.

## 결과

| 분석 | Target-OOD 대응표본 보상 차이 (bootstrap 95% CI) | 최대 trip 비율 | 판정 |
|---|---:|---:|---|
| 상속된 물리 여유 `z=1.5` 대 `z=0` | {fmt_ci(inherited)} | {confirm['maximum_physics_z1_5_trip_rate'] * 100:.1f}% | 최초 주 기준 실패 |
| Reacher 보정 여유 `z=2.0` 대 `z=0` | {fmt_ci(selected)} | {fresh['maximum_selected_trip_rate'] * 100:.1f}% | 사후 확장 성공 |
| 보정 `z=2.0` 대 상속 `z=1.5` | {fmt_ci(selected_vs_inherited)} | -- | 평균 보상 개선 불확실 |
| Hybrid `z=1.5` 대 `z=0` | {fmt_ci(hybrid)} | 1.4% | 이차 결과 |
| Monolithic RecurrentPPO 대 `z=0` | {fmt_ci(mono)} | -- | 시험한 계열에서 강한 OOD 실패 |

상속된 여유는 target-OOD 기대 효용을 높였지만 고정된 안전 기준을 통과하지 못했다(2.3% 대 2.0%). 개발 시드만으로 cutoff 0.06과 `z=2.0`을 선택한 뒤 새로운 lifetime 100개에서 한 번 평가했다. 보정 설정은 양의 평균 효과를 유지하면서 최대 trip 비율을 1.6%로 낮췄다. 상속 설정과 보정 설정의 평균 보상 차이는 명확하지 않았으므로({fmt_ci(selected_vs_inherited)}), 이 결과는 추가 보상 향상이 아니라 검출 가능한 기대 효용 손실 없는 과제별 안전 보정을 지지한다.

대응 lifetime 중 양의 효과는 {selected['n']}개 중 {selected['positive']}개였고 exact sign test p={selected['exact_sign_two_sided_p']:.3f}이다. 따라서 주장은 다수 lifetime의 일관된 개선이 아니라 평균 기대 효용 개선으로 제한한다. 두 과제는 같은 시뮬레이터와 현상론적 열화 법칙을 공유하므로 실제 하드웨어 일반화나 형식적 안전 보장은 주장하지 않는다.

프로토콜 해시: 최초 확증 `{data['confirm_hash']}`; 확장 `{data['fresh_hash']}`.
"""
    (output / "PAPER_RESULTS_EN.md").write_text(en, encoding="utf-8")
    (output / "PAPER_RESULTS_KO.md").write_text(ko, encoding="utf-8")


def make_manifest(data: dict[str, Any], root: Path, output: Path) -> None:
    inputs = [
        root / "confirmatory" / "CONFIRMATORY_RESULTS.json",
        root / "confirmatory" / "CONFIRMATORY_CELLS.json",
        root / "confirmatory" / "FROZEN_PROTOCOL.json",
        root / "margin_extension" / "CALIBRATED_MARGIN_SELECTION.json",
        root / "margin_extension" / "FRESH_RESULTS.json",
        root / "margin_extension" / "FRESH_CELLS.json",
        root / "margin_extension" / "FROZEN_FRESH_PROTOCOL.json",
    ]
    products = sorted(path for path in output.iterdir() if path.name != "MANIFEST.json")
    manifest = {
        "phase": "reacher_replication_final_publication_artifacts",
        "status": "validated",
        "confirmatory_protocol_sha256": data["confirm_hash"],
        "extension_protocol_sha256": data["fresh_hash"],
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "artifacts": {path.name: sha256(path) for path in products},
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data = validate(args.input_root)
    make_tables(data, args.output_root)
    make_figure(data, args.output_root)
    make_summaries(data, args.output_root)
    make_manifest(data, args.input_root, args.output_root)
    print(json.dumps({
        "status": "complete",
        "output_root": str(args.output_root),
        "confirmatory_primary_success": data["confirm_result"]["primary_success"],
        "extension_success": data["fresh_result"]["extension_success"],
    }, indent=2))


if __name__ == "__main__":
    main()
