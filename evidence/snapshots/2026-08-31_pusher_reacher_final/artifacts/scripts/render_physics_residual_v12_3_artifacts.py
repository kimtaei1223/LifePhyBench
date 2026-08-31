#!/usr/bin/env python3
"""Validate v12.3 evidence and render factorial and reward-sensitivity artifacts."""

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

try:
    from scripts.run_physics_residual_v12_3_factorial_ablation import (
        ALL_CONDITIONS,
        COMPONENT_FIELDS,
        CONTRASTS,
        TARGET_OOD,
    )
    from scripts.run_physics_residual_v12_confirmatory_pipeline import bootstrap_ci
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from run_physics_residual_v12_3_factorial_ablation import (  # type: ignore[no-redef]
        ALL_CONDITIONS,
        COMPONENT_FIELDS,
        CONTRASTS,
        TARGET_OOD,
    )
    from run_physics_residual_v12_confirmatory_pipeline import (  # type: ignore[no-redef]
        bootstrap_ci,
    )

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "physics_residual_v12_3_factorial_ablation"
DEFAULT_OUTPUT = ROOT / "paper_artifacts" / "physics_residual_v12_3"
POLICY_ORDER = ("physics_z0", "physics_z1_5", "hybrid_z0", "hybrid_z1_5")
POLICY_LABELS = {
    "physics_z0": "Physics, z=0",
    "physics_z1_5": "Physics, z=1.5",
    "hybrid_z0": "Physics + residual, z=0",
    "hybrid_z1_5": "Physics + residual, z=1.5",
}
CONTRAST_ORDER = (
    "residual_at_z0",
    "residual_at_z1_5",
    "uncertainty_without_residual",
    "uncertainty_with_residual",
    "interaction",
)
CONTRAST_LABELS = {
    "residual_at_z0": "Residual at z=0",
    "residual_at_z1_5": "Residual at z=1.5",
    "uncertainty_without_residual": "Uncertainty, residual off",
    "uncertainty_with_residual": "Uncertainty, residual on",
    "interaction": "Residual x uncertainty",
}
CONDITION_LABELS = {
    "in_domain": "In-domain",
    "ood_sensor_noise": "Sensor noise",
    "ood_cooling": "Cooling shift",
    "ood_combined": "Combined shift",
    "ood_shocks": "Physical shocks",
}
BONUS_VALUES = (0.0, 1.0, 2.0, 3.0, 4.0)
TRIP_PENALTY_VALUES = (25.0, 50.0, 75.0, 100.0, 150.0)
ORIGINAL_BONUS = 2.0
ORIGINAL_TRIP_PENALTY = 75.0
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#B8B8B8"
SVG_HASH_SALT = "lifephybench-physics-residual-v12-3"


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


def alternative_reward(
    row: dict[str, Any], *, throughput_bonus: float, trip_penalty: float
) -> float:
    """Reweight recorded components without changing the evaluated trajectories."""
    return float(
        float(row["mean_base_task_return"])
        + throughput_bonus / ORIGINAL_BONUS * float(row["mean_throughput_bonus"])
        + trip_penalty / ORIGINAL_TRIP_PENALTY * float(row["mean_trip_penalty"])
    )


def _rows_by_seed(
    cells: dict[str, dict[str, dict[str, Any]]], condition: str, policy: str
) -> dict[int, dict[str, Any]]:
    return {
        int(row["seed"]): row
        for row in cells[condition][policy]["summary"]["lifetime_rows"]
    }


def aggregate_alternative_contrast(
    cells: dict[str, dict[str, dict[str, Any]]],
    *,
    treatment: str,
    control: str,
    throughput_bonus: float,
    trip_penalty: float,
) -> tuple[list[int], np.ndarray]:
    condition_values: list[np.ndarray] = []
    seed_reference: list[int] | None = None
    for condition in TARGET_OOD:
        treated = _rows_by_seed(cells, condition, treatment)
        controlled = _rows_by_seed(cells, condition, control)
        require(set(treated) == set(controlled), f"{condition}: seed mismatch")
        seeds = sorted(treated)
        if seed_reference is None:
            seed_reference = seeds
        else:
            require(seeds == seed_reference, "target-OOD seed mismatch")
        condition_values.append(
            np.asarray(
                [
                    alternative_reward(
                        treated[seed],
                        throughput_bonus=throughput_bonus,
                        trip_penalty=trip_penalty,
                    )
                    - alternative_reward(
                        controlled[seed],
                        throughput_bonus=throughput_bonus,
                        trip_penalty=trip_penalty,
                    )
                    for seed in seeds
                ],
                dtype=np.float64,
            )
        )
    require(seed_reference is not None, "no target-OOD seeds")
    return seed_reference, np.mean(np.stack(condition_values), axis=0)


def aggregate_component_contrast(
    cells: dict[str, dict[str, dict[str, Any]]],
    *,
    treatment: str,
    control: str,
    field: str,
) -> np.ndarray:
    condition_values = []
    seed_reference: list[int] | None = None
    for condition in TARGET_OOD:
        treated = _rows_by_seed(cells, condition, treatment)
        controlled = _rows_by_seed(cells, condition, control)
        seeds = sorted(treated)
        require(set(treated) == set(controlled), f"{condition}: seed mismatch")
        if seed_reference is None:
            seed_reference = seeds
        else:
            require(seeds == seed_reference, "component seed mismatch")
        condition_values.append(
            np.asarray(
                [
                    float(treated[seed][field]) - float(controlled[seed][field])
                    for seed in seeds
                ],
                dtype=np.float64,
            )
        )
    return np.mean(np.stack(condition_values), axis=0)


def validate(
    protocol: dict[str, Any],
    result: dict[str, Any],
    cells: dict[str, dict[str, dict[str, Any]]],
    *,
    protocol_path: Path,
    protocol_hash_path: Path,
) -> None:
    require(
        protocol.get("phase") == "physics_residual_v12_3_frozen_factorial_protocol",
        "unexpected protocol phase",
    )
    require(
        protocol.get("status") == "frozen_before_any_v12_3_factorial_evaluation",
        "protocol was not frozen before evaluation",
    )
    stored_hash = protocol_hash_path.read_text(encoding="utf-8").strip()
    require(sha256(protocol_path) == stored_hash, "protocol hash mismatch")
    require(result.get("protocol_sha256") == stored_hash, "result hash mismatch")
    require(
        result.get("phase") == "physics_residual_v12_3_factorial_mechanism_ablation",
        "unexpected result phase",
    )
    require(
        result.get("status") == "final_fresh_seed_ablation_result",
        "result is not final",
    )
    require(result.get("wiring_passed") is True, "wiring validation failed")
    seeds = [int(seed) for seed in protocol["seeds"]]
    require(len(seeds) == len(set(seeds)) == 100, "expected 100 fresh seeds")
    require(set(cells) == set(ALL_CONDITIONS), "condition set mismatch")

    for condition in ALL_CONDITIONS:
        require(set(cells[condition]) == set(POLICY_ORDER), "policy set mismatch")
        for policy in POLICY_ORDER:
            summary = cells[condition][policy]["summary"]
            rows = summary["lifetime_rows"]
            require(len(rows) == 100, f"{condition}/{policy}: row count")
            require(
                {int(row["seed"]) for row in rows} == set(seeds),
                f"{condition}/{policy}: seed set",
            )
            for row in rows:
                reconstructed = sum(float(row[field]) for field in COMPONENT_FIELDS)
                require(
                    close(reconstructed, row["mean_reward_per_task"]),
                    f"{condition}/{policy}: component identity",
                )

    for name, (treatment, control) in CONTRASTS.items():
        _, values = aggregate_alternative_contrast(
            cells,
            treatment=treatment,
            control=control,
            throughput_bonus=ORIGINAL_BONUS,
            trip_penalty=ORIGINAL_TRIP_PENALTY,
        )
        stored = result["target_ood_aggregate_contrasts"][name]
        require(close(values.mean(), stored["mean"]), f"{name}: mean mismatch")
        require(
            np.allclose(values, stored["values"], rtol=1e-10, atol=1e-10),
            f"{name}: paired values mismatch",
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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


def render_factorial_figure(result: dict[str, Any], output: Path) -> None:
    configure_plot_style()
    figure, (contrast_axis, decomposition_axis) = plt.subplots(
        1, 2, figsize=(11.2, 4.6)
    )
    contrasts = result["target_ood_aggregate_contrasts"]
    positions = np.arange(len(CONTRAST_ORDER))[::-1]
    colors = [GREEN, GREEN, BLUE, BLUE, ORANGE]
    for position, name, color in zip(positions, CONTRAST_ORDER, colors):
        row = contrasts[name]
        low, high = row["bootstrap_95_ci"]
        contrast_axis.errorbar(
            row["mean"],
            position,
            xerr=[[row["mean"] - low], [high - row["mean"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
    contrast_axis.axvline(0.0, color=LIGHT_GRAY, linewidth=1, linestyle="--")
    contrast_axis.set_yticks(
        positions, [CONTRAST_LABELS[name] for name in CONTRAST_ORDER]
    )
    contrast_axis.set_xlabel("Target-OOD reward/task contrast")
    contrast_axis.set_title("(a) Fresh-seed 2 x 2 factorial contrasts")
    contrast_axis.grid(axis="x", alpha=0.2)

    decomposition = result["target_ood_reward_decomposition"]
    component_names = (
        "mean_base_task_return",
        "mean_throughput_bonus",
        "mean_trip_penalty",
        "total_hybrid_z1_5_minus_physics_z0",
    )
    component_labels = ("Base task", "Throughput", "Avoided trips", "Total")
    values = [decomposition[name]["mean"] for name in component_names]
    bar_colors = [ORANGE if value < 0 else BLUE for value in values[:-1]] + [GREEN]
    bars = decomposition_axis.bar(component_labels, values, color=bar_colors)
    decomposition_axis.axhline(0.0, color=GRAY, linewidth=1)
    decomposition_axis.set_ylabel("Reward/task contribution")
    decomposition_axis.set_title("(b) Hybrid z=1.5 minus physics z=0")
    decomposition_axis.grid(axis="y", alpha=0.2)
    decomposition_axis.tick_params(axis="x", rotation=20)
    decomposition_axis.set_ylim(min(values) - 0.40, max(values) + 0.35)
    for bar, value in zip(bars, values):
        offset = 0.06 if value >= 0 else -0.12
        decomposition_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    figure.tight_layout()
    figure.savefig(output / "figure_v12_3_factorial.svg", metadata={"Date": None})
    figure.savefig(
        output / "figure_v12_3_factorial.png",
        metadata={"Software": "LifePhyBench"},
    )
    plt.close(figure)


def sensitivity_rows(
    cells: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for penalty_index, trip_penalty in enumerate(TRIP_PENALTY_VALUES):
        for bonus_index, throughput_bonus in enumerate(BONUS_VALUES):
            _, total = aggregate_alternative_contrast(
                cells,
                treatment="hybrid_z1_5",
                control="physics_z0",
                throughput_bonus=throughput_bonus,
                trip_penalty=trip_penalty,
            )
            _, residual = aggregate_alternative_contrast(
                cells,
                treatment="hybrid_z1_5",
                control="physics_z1_5",
                throughput_bonus=throughput_bonus,
                trip_penalty=trip_penalty,
            )
            _, uncertainty = aggregate_alternative_contrast(
                cells,
                treatment="physics_z1_5",
                control="physics_z0",
                throughput_bonus=throughput_bonus,
                trip_penalty=trip_penalty,
            )
            low, high = bootstrap_ci(
                total,
                seed=81000 + 100 * penalty_index + bonus_index,
                resamples=20_000,
            )
            rows.append(
                {
                    "throughput_bonus": throughput_bonus,
                    "trip_penalty": trip_penalty,
                    "total_hybrid_z1_5_minus_physics_z0": float(total.mean()),
                    "total_bootstrap_95_ci_low": low,
                    "total_bootstrap_95_ci_high": high,
                    "total_positive_seed_fraction": float(np.mean(total > 0.0)),
                    "residual_at_z1_5": float(residual.mean()),
                    "uncertainty_without_residual": float(uncertainty.mean()),
                }
            )
    return rows


def break_even_rows(
    cells: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, float]]:
    base = aggregate_component_contrast(
        cells,
        treatment="hybrid_z1_5",
        control="physics_z0",
        field="mean_base_task_return",
    )
    throughput = aggregate_component_contrast(
        cells,
        treatment="hybrid_z1_5",
        control="physics_z0",
        field="mean_throughput_bonus",
    )
    avoided_trips = aggregate_component_contrast(
        cells,
        treatment="hybrid_z1_5",
        control="physics_z0",
        field="mean_trip_penalty",
    )
    require(
        avoided_trips.mean() > 0.0, "mean avoided-trip contribution is not positive"
    )
    rows = []
    for throughput_bonus in BONUS_VALUES:
        numerator = -(
            float(base.mean())
            + throughput_bonus / ORIGINAL_BONUS * float(throughput.mean())
        )
        threshold = ORIGINAL_TRIP_PENALTY * numerator / float(avoided_trips.mean())
        rows.append(
            {
                "throughput_bonus": throughput_bonus,
                "mean_break_even_trip_penalty": threshold,
            }
        )
    return rows


def render_sensitivity_figure(rows: list[dict[str, Any]], output: Path) -> None:
    configure_plot_style()
    matrix = np.empty((len(TRIP_PENALTY_VALUES), len(BONUS_VALUES)))
    lookup = {
        (float(row["trip_penalty"]), float(row["throughput_bonus"])): float(
            row["total_hybrid_z1_5_minus_physics_z0"]
        )
        for row in rows
    }
    for y, penalty in enumerate(TRIP_PENALTY_VALUES):
        for x, bonus in enumerate(BONUS_VALUES):
            matrix[y, x] = lookup[(penalty, bonus)]
    bound = float(np.max(np.abs(matrix)))
    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-bound,
        vmax=bound,
    )
    axis.contour(
        matrix,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
        linestyles="--",
    )
    axis.set_xticks(np.arange(len(BONUS_VALUES)), BONUS_VALUES)
    axis.set_yticks(np.arange(len(TRIP_PENALTY_VALUES)), TRIP_PENALTY_VALUES)
    axis.set_xlabel("Counterfactual high-power bonus")
    axis.set_ylabel("Counterfactual thermal-trip penalty")
    axis.set_title("Post-hoc utility sensitivity on fixed trajectories")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            color = "white" if abs(matrix[y, x]) > 0.55 * bound else "black"
            axis.text(
                x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", color=color
            )
    original_x = BONUS_VALUES.index(ORIGINAL_BONUS)
    original_y = TRIP_PENALTY_VALUES.index(ORIGINAL_TRIP_PENALTY)
    axis.scatter(
        [original_x],
        [original_y],
        marker="s",
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        s=800,
    )
    axis.text(
        original_x,
        original_y - 0.30,
        "original",
        ha="center",
        va="center",
        fontsize=8,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Hybrid z=1.5 minus physics z=0 reward/task")
    figure.tight_layout()
    figure.savefig(
        output / "figure_v12_3_reward_sensitivity.svg", metadata={"Date": None}
    )
    figure.savefig(
        output / "figure_v12_3_reward_sensitivity.png",
        metadata={"Software": "LifePhyBench"},
    )
    plt.close(figure)


def format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def write_summaries(
    result: dict[str, Any], sensitivity: list[dict[str, Any]], output: Path
) -> None:
    contrasts = result["target_ood_aggregate_contrasts"]
    residual = contrasts["residual_at_z1_5"]
    uncertainty_off = contrasts["uncertainty_without_residual"]
    uncertainty_on = contrasts["uncertainty_with_residual"]
    interaction = contrasts["interaction"]
    decomposition = result["target_ood_reward_decomposition"]
    original = next(
        row
        for row in sensitivity
        if row["throughput_bonus"] == ORIGINAL_BONUS
        and row["trip_penalty"] == ORIGINAL_TRIP_PENALTY
    )
    low_penalty = next(
        row
        for row in sensitivity
        if row["throughput_bonus"] == ORIGINAL_BONUS and row["trip_penalty"] == 25.0
    )
    medium_penalty = next(
        row
        for row in sensitivity
        if row["throughput_bonus"] == ORIGINAL_BONUS and row["trip_penalty"] == 50.0
    )
    high_penalty = next(
        row
        for row in sensitivity
        if row["throughput_bonus"] == ORIGINAL_BONUS and row["trip_penalty"] == 100.0
    )
    threshold = break_even_rows_from_result(decomposition, ORIGINAL_BONUS)

    korean = f"""# v12.3 factorial 및 보상 민감도 — 논문 삽입용 요약

독립적인 fresh lifetime seed 100개를 사용한 2 x 2 분석에서, 동일한 불확실성 마진 z=1.5에서 residual의 효과는 **{residual["mean"]:+.3f} reward/task**였다(95% bootstrap CI [{residual["bootstrap_95_ci"][0]:.3f}, {residual["bootstrap_95_ci"][1]:.3f}], paired sign-flip p={format_p(residual["sign_flip_two_sided_p"])}). 반면 residual이 없는 경우 불확실성 마진의 효과는 **{uncertainty_off["mean"]:+.3f}**(95% CI [{uncertainty_off["bootstrap_95_ci"][0]:.3f}, {uncertainty_off["bootstrap_95_ci"][1]:.3f}]), residual이 있는 경우에는 **{uncertainty_on["mean"]:+.3f}**(95% CI [{uncertainty_on["bootstrap_95_ci"][0]:.3f}, {uncertainty_on["bootstrap_95_ci"][1]:.3f}])였다. 상호작용은 {interaction["mean"]:+.3f} (95% CI [{interaction["bootstrap_95_ci"][0]:.3f}, {interaction["bootstrap_95_ci"][1]:.3f}])로 음수였다. 따라서 전체 이득을 residual의 독립 효과로 귀속하지 않으며, 주된 확증 결론은 불확실성 마진을 포함한 belief supervision의 효과이다.

원래의 hybrid z=1.5 대 physics z=0 차이 {original["total_hybrid_z1_5_minus_physics_z0"]:+.3f}은 base task return {decomposition["mean_base_task_return"]["mean"]:+.3f}, throughput bonus {decomposition["mean_throughput_bonus"]["mean"]:+.3f}, avoided-trip contribution {decomposition["mean_trip_penalty"]["mean"]:+.3f}으로 분해되었다. 즉, 평균 효용 이득은 주로 thermal trip 감소에서 발생하며 즉시 task 성능의 희생을 포함한다.

동일 궤적을 사후 재가중한 민감도 분석에서 원래 throughput bonus 2를 유지할 때 평균 break-even trip penalty는 약 **{threshold:.1f}**였다. penalty 25에서는 총 효과가 {low_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, 50에서는 {medium_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, 75에서는 {original["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, 100에서는 {high_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f}이었다. 따라서 효용 개선은 thermal trip에 중간 이상의 비용을 부여하는 응용에서 성립하며, trip을 거의 중요하지 않게 취급하는 목적함수에는 일반화되지 않는다. 이 분석은 정책을 다른 reward로 재학습한 결과가 아니라 고정된 평가 궤적의 회계적 민감도 분석이다.
"""
    english = f"""# v12.3 factorial and reward sensitivity — manuscript-ready summary

In a 2 x 2 analysis using 100 independent fresh lifetime seeds, the residual effect at the matched uncertainty margin z=1.5 was **{residual["mean"]:+.3f} reward/task** (95% bootstrap CI [{residual["bootstrap_95_ci"][0]:.3f}, {residual["bootstrap_95_ci"][1]:.3f}]; paired sign-flip p={format_p(residual["sign_flip_two_sided_p"])}). The uncertainty-margin effect was **{uncertainty_off["mean"]:+.3f}** without the residual (95% CI [{uncertainty_off["bootstrap_95_ci"][0]:.3f}, {uncertainty_off["bootstrap_95_ci"][1]:.3f}]) and **{uncertainty_on["mean"]:+.3f}** with it (95% CI [{uncertainty_on["bootstrap_95_ci"][0]:.3f}, {uncertainty_on["bootstrap_95_ci"][1]:.3f}]). The interaction was negative ({interaction["mean"]:+.3f}; 95% CI [{interaction["bootstrap_95_ci"][0]:.3f}, {interaction["bootstrap_95_ci"][1]:.3f}]). We therefore do not attribute the full benefit to an independent residual effect; the confirmatory conclusion concerns belief supervision with an uncertainty margin.

The original hybrid-z=1.5 versus physics-z=0 difference of {original["total_hybrid_z1_5_minus_physics_z0"]:+.3f} decomposed into {decomposition["mean_base_task_return"]["mean"]:+.3f} base-task return, {decomposition["mean_throughput_bonus"]["mean"]:+.3f} throughput bonus, and {decomposition["mean_trip_penalty"]["mean"]:+.3f} avoided-trip contribution. The mean utility improvement was therefore driven primarily by fewer thermal trips and included a sacrifice in immediate task performance.

In a post-hoc reweighting of the same trajectories, the mean break-even trip penalty was approximately **{threshold:.1f}** when the original throughput bonus of 2 was retained. The total effect was {low_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, {medium_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, {original["total_hybrid_z1_5_minus_physics_z0"]:+.3f}, and {high_penalty["total_hybrid_z1_5_minus_physics_z0"]:+.3f} at trip penalties 25, 50, 75, and 100, respectively. The utility claim thus applies when thermal trips carry moderate or high cost, not when the objective nearly ignores them. This is an accounting sensitivity analysis on fixed evaluation trajectories, not retraining under alternative rewards.
"""
    (output / "PAPER_RESULTS_KO.md").write_text(korean, encoding="utf-8")
    (output / "PAPER_RESULTS_EN.md").write_text(english, encoding="utf-8")


def break_even_rows_from_result(
    decomposition: dict[str, Any], throughput_bonus: float
) -> float:
    base = float(decomposition["mean_base_task_return"]["mean"])
    bonus = float(decomposition["mean_throughput_bonus"]["mean"])
    avoided_trips = float(decomposition["mean_trip_penalty"]["mean"])
    require(avoided_trips > 0.0, "avoided-trip contribution must be positive")
    return (
        ORIGINAL_TRIP_PENALTY
        * -(base + throughput_bonus / ORIGINAL_BONUS * bonus)
        / avoided_trips
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = input_root / "FROZEN_PROTOCOL.json"
    protocol_hash_path = input_root / "FROZEN_PROTOCOL.sha256"
    result_path = input_root / "FACTORIAL_RESULTS.json"
    cells_path = input_root / "FACTORIAL_CELLS.json"
    protocol = read_json(protocol_path)
    result = read_json(result_path)
    cells = read_json(cells_path)
    validate(
        protocol,
        result,
        cells,
        protocol_path=protocol_path,
        protocol_hash_path=protocol_hash_path,
    )

    contrast_rows = []
    for scope in (*ALL_CONDITIONS, "target_ood_aggregate"):
        rows = (
            result["target_ood_aggregate_contrasts"]
            if scope == "target_ood_aggregate"
            else result["condition_contrasts"][scope]
        )
        for name in CONTRAST_ORDER:
            row = rows[name]
            contrast_rows.append(
                {
                    "scope": scope,
                    "scope_label": CONDITION_LABELS.get(scope, "Target OOD aggregate"),
                    "contrast": name,
                    "contrast_label": CONTRAST_LABELS[name],
                    "n_seeds": row["n"],
                    "mean": row["mean"],
                    "sample_sd": row["sd"],
                    "bootstrap_95_ci_low": row["bootstrap_95_ci"][0],
                    "bootstrap_95_ci_high": row["bootstrap_95_ci"][1],
                    "sign_flip_two_sided_p": row.get("sign_flip_two_sided_p", ""),
                }
            )

    policy_rows = []
    for condition in ALL_CONDITIONS:
        for policy in POLICY_ORDER:
            summary = cells[condition][policy]["summary"]
            policy_rows.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "n_seeds": summary["lifetimes"],
                    "mean_reward_per_task": summary["mean_reward_per_task"],
                    "mean_base_task_return": summary["mean_base_task_return"],
                    "mean_throughput_bonus": summary["mean_throughput_bonus"],
                    "mean_trip_penalty": summary["mean_trip_penalty"],
                    "trip_rate": summary["trip_rate"],
                    "high_rate": summary["high_rate"],
                }
            )

    paired_rows = []
    seeds = result["target_ood_aggregate_contrasts"]["residual_at_z0"]["seeds"]
    for index, seed in enumerate(seeds):
        paired_rows.append(
            {
                "seed": seed,
                **{
                    name: result["target_ood_aggregate_contrasts"][name]["values"][
                        index
                    ]
                    for name in CONTRAST_ORDER
                },
            }
        )

    sensitivity = sensitivity_rows(cells)
    thresholds = break_even_rows(cells)
    write_csv(output / "factorial_contrasts.csv", contrast_rows)
    write_csv(output / "policy_summaries.csv", policy_rows)
    write_csv(output / "paired_seed_contrasts.csv", paired_rows)
    write_csv(output / "reward_sensitivity.csv", sensitivity)
    write_csv(output / "reward_break_even.csv", thresholds)
    render_factorial_figure(result, output)
    render_sensitivity_figure(sensitivity, output)
    write_summaries(result, sensitivity, output)

    artifact_paths = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "phase": "physics_residual_v12_3_publication_artifacts",
        "status": "complete",
        "validation_passed": True,
        "analysis_unit": "independent lifetime seed",
        "reward_sensitivity": {
            "status": "post_hoc_fixed_trajectory_accounting_sensitivity",
            "throughput_bonus_values": list(BONUS_VALUES),
            "trip_penalty_values": list(TRIP_PENALTY_VALUES),
            "policy_retraining_performed": False,
        },
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
