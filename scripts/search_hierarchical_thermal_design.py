"""Search for a non-degenerate lifetime control design before GPU learning.

Only calibration physics are inspected here.  A design is eligible when a
full-lifetime prefix schedule uses both modes, safely beats always-low, and the
static control still favors high power.  The first eligible point in the
predeclared grid is selected; rewards never rank eligible designs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from lifephybench.envs.hierarchical_thermal import (
    HierarchicalThermalConfig,
    HierarchicalThermalModeEnv,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_environment(model, design: dict, dynamic: bool = True):
    return HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path="injected-frozen-controller",
            degradation_mode=("endogenous_action" if dynamic else "exogenous_clock"),
            trip_load=design["trip_load"],
            low_power_scale=design["low_power_scale"],
            trip_penalty=design["trip_penalty"],
            high_power_bonus=design["high_power_bonus"],
            thermal_heat_rate=design["thermal_heat_rate"],
            summary_mode="mode_trip",
        ),
        low_level_model=model,
    )


def prefix_schedule(environment, high_tasks: int, seed: int = 31) -> dict:
    environment.reset(seed=seed)
    task_rewards = []
    selection_loads = []
    trips = 0
    for task_index in range(20):
        high = task_index < high_tasks
        _, reward, _, truncated, info = environment.step(int(high))
        task_rewards.append(float(info["lifephy/hierarchical_physical_reward"]))
        selection_loads.append(float(info["lifephy/thermal_load_at_mode_selection"]))
        trips += int(info["lifephy/thermal_trip"])
        if task_index < 19 and truncated:
            raise RuntimeError("lifetime ended before 20 task decisions")
        if task_index == 19 and not truncated:
            raise RuntimeError("lifetime did not end after 20 task decisions")
    return {
        "high_tasks": high_tasks,
        "lifetime_reward": float(sum(task_rewards)),
        "mean_task_reward": float(np.mean(task_rewards)),
        "trip_count": trips,
        "terminal_load": float(environment._health.thermal_load),
        "selection_loads": selection_loads,
        "task_rewards": task_rewards,
    }


def reactive_schedule(environment, rule: dict[str, int], seed: int = 31) -> dict:
    """Evaluate a task-reset policy using only start/previous-mode/trip."""
    environment.reset(seed=seed)
    task_rewards = []
    actions = []
    trips = 0
    previous_mode = None
    previous_trip = False
    for task_index in range(20):
        if task_index == 0:
            key = "start"
        elif previous_trip:
            key = "trip"
        else:
            key = "high" if previous_mode == 1 else "low"
        action = int(rule[key])
        _, _, _, truncated, info = environment.step(action)
        actions.append(action)
        task_rewards.append(float(info["lifephy/hierarchical_physical_reward"]))
        previous_mode = action
        previous_trip = bool(info["lifephy/thermal_trip"])
        trips += int(previous_trip)
        if task_index == 19 and not truncated:
            raise RuntimeError("reactive lifetime did not terminate")
    return {
        "rule": rule,
        "actions": actions,
        "lifetime_reward": float(sum(task_rewards)),
        "trip_count": trips,
        "terminal_load": float(environment._health.thermal_load),
    }


def static_control(model, design: dict, high: bool) -> dict:
    environment = make_environment(model, design, dynamic=False)
    try:
        environment.reset(seed=31)
        _, reward, _, _, info = environment.step(int(high))
        return {
            "reward": float(info["lifephy/hierarchical_physical_reward"]),
            "tripped": bool(info["lifephy/thermal_trip"]),
            "physical_steps": int(info["lifephy/hierarchical_physical_steps"]),
        }
    finally:
        environment.close()


def safe_high_cutoff(model, design: dict) -> dict:
    rows = []
    for initial_load in np.linspace(0.0, design["trip_load"], 21):
        high_environment = make_environment(model, design)
        low_environment = make_environment(model, design)
        try:
            high_environment.reset(seed=41)
            low_environment.reset(seed=41)
            high_environment._health.set_thermal_load_for_diagnostic(initial_load)
            low_environment._health.set_thermal_load_for_diagnostic(initial_load)
            _, high_reward, _, _, high_info = high_environment.step(1)
            _, low_reward, _, _, low_info = low_environment.step(0)
            safe_and_useful = bool(
                not high_info["lifephy/thermal_trip"]
                and high_info["lifephy/hierarchical_physical_steps"] == 100
                and high_reward > low_reward
            )
            rows.append(
                {
                    "initial_load": float(initial_load),
                    "high_reward": float(high_reward),
                    "low_reward": float(low_reward),
                    "high_tripped": bool(high_info["lifephy/thermal_trip"]),
                    "safe_and_useful": safe_and_useful,
                }
            )
        finally:
            high_environment.close()
            low_environment.close()
    eligible_loads = [row["initial_load"] for row in rows if row["safe_and_useful"]]
    return {
        "grid_rows": rows,
        "safe_high_load": max(eligible_loads) if eligible_loads else None,
    }


def evaluate_design(model, design: dict) -> dict:
    environment = make_environment(model, design)
    try:
        schedules = [prefix_schedule(environment, high_tasks) for high_tasks in range(21)]
        reactive = [
            reactive_schedule(
                environment,
                dict(zip(("start", "low", "high", "trip"), actions)),
            )
            for actions in itertools.product((0, 1), repeat=4)
        ]
    finally:
        environment.close()
    best = max(schedules, key=lambda row: row["lifetime_reward"])
    all_low = schedules[0]
    all_high = schedules[-1]
    best_reactive = max(reactive, key=lambda row: row["lifetime_reward"])
    static_low = static_control(model, design, False)
    static_high = static_control(model, design, True)
    cutoff = safe_high_cutoff(model, design)
    criteria = {
        "interior_schedule_uses_at_least_two_each": 2 <= best["high_tasks"] <= 18,
        "best_schedule_has_no_trip": best["trip_count"] == 0,
        "best_beats_all_low_by_five": (
            best["lifetime_reward"] - all_low["lifetime_reward"] >= 5.0
        ),
        "memory_oracle_beats_best_reactive_by_five": (
            best["lifetime_reward"] - best_reactive["lifetime_reward"] >= 5.0
        ),
        "static_high_beats_low_by_five": (
            static_high["reward"] - static_low["reward"] >= 5.0
        ),
        "static_tasks_complete": (
            static_high["physical_steps"] == 100
            and static_low["physical_steps"] == 100
        ),
        "safe_high_region_exists": cutoff["safe_high_load"] is not None,
        "lifetime_reaches_hidden_hot_region": (
            best["terminal_load"] >= design["trip_load"]
        ),
    }
    return {
        "design": design,
        "criteria": criteria,
        "eligible": all(criteria.values()),
        "best_prefix_schedule": best,
        "all_low": all_low,
        "all_high": all_high,
        "best_task_reactive": best_reactive,
        "static_low": static_low,
        "static_high": static_high,
        "safe_high_scan": cutoff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trip-loads", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--thermal-heat-rates", type=float, nargs="+", default=[0.10, 0.075, 0.05, 0.03])
    parser.add_argument("--low-power-scales", type=float, nargs="+", default=[0.40, 0.50])
    parser.add_argument("--trip-penalty", type=float, default=75.0)
    parser.add_argument("--high-power-bonus", type=float, default=2.0)
    parser.add_argument("--max-designs", type=int, default=0)
    parser.add_argument(
        "--low-level-model",
        type=Path,
        default=Path(
            "outputs/canonical_thermal_probe/"
            "canonical-thermal-static-task-seed4003-steps2000k/model.zip"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hierarchical_autonomous_v10/design_search.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_path = args.low_level_model.resolve()
    model = RecurrentPPO.load(str(model_path), device="cpu")
    grid = [
        {
            "trip_load": trip_load,
            "thermal_heat_rate": heat_rate,
            "low_power_scale": low_scale,
            "trip_penalty": args.trip_penalty,
            "high_power_bonus": args.high_power_bonus,
        }
        for trip_load, heat_rate, low_scale in itertools.product(
            args.trip_loads, args.thermal_heat_rates, args.low_power_scales
        )
    ]
    if args.max_designs > 0:
        grid = grid[: args.max_designs]
    rows = []
    selected = None
    for index, design in enumerate(grid, start=1):
        print(f"[DESIGN {index}/{len(grid)}] {design}", flush=True)
        row = evaluate_design(model, design)
        rows.append(row)
        if row["eligible"]:
            selected = {
                **design,
                "teacher_safe_high_load": row["safe_high_scan"]["safe_high_load"],
                "oracle_high_tasks": row["best_prefix_schedule"]["high_tasks"],
                "oracle_improvement_over_all_low": (
                    row["best_prefix_schedule"]["lifetime_reward"]
                    - row["all_low"]["lifetime_reward"]
                ),
                "oracle_improvement_over_best_task_reactive": (
                    row["best_prefix_schedule"]["lifetime_reward"]
                    - row["best_task_reactive"]["lifetime_reward"]
                ),
                "summary_mode": "mode_trip",
            }
            break
    report = {
        "phase": "hierarchical_thermal_oracle_design_search",
        "status": "calibration_only_not_learned_policy_evidence",
        "selection_rule": "first eligible design in the predeclared ordered grid",
        "low_level_model": str(model_path),
        "low_level_model_sha256": sha256(model_path),
        "searched_designs": len(rows),
        "selected_design": selected,
        "passed": selected is not None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))
    if selected is None:
        raise SystemExit("no feasible design in predeclared grid")


if __name__ == "__main__":
    main()
