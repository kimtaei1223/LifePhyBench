"""CPU semantic gates for hierarchical discrete thermal mode control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lifephybench.envs.hierarchical_thermal import (
    HierarchicalThermalConfig,
    HierarchicalThermalModeEnv,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_environment(model: Path, static: bool = False):
    return HierarchicalThermalModeEnv(
        HierarchicalThermalConfig(
            low_level_model_path=str(model),
            degradation_mode=("exogenous_clock" if static else "endogenous_action"),
        )
    )


def one_task(model: Path, load: float, high: bool, static: bool = False):
    environment = make_environment(model, static=static)
    try:
        observation, _ = environment.reset(seed=81)
        environment._health.set_thermal_load_for_diagnostic(0.0 if static else load)
        _, reward, _, truncated, info = environment.step(1 if high else 0)
        return {
            "initial_load": 0.0 if static else load,
            "mode": "high" if high else "low",
            "reward": float(reward),
            "tripped": bool(info["lifephy/thermal_trip"]),
            "physical_steps": int(info["lifephy/hierarchical_physical_steps"]),
            "terminal_load": float(info["lifephy/thermal_load"]),
            "lifetime_boundary": bool(truncated),
            "observation_shape": list(observation.shape),
        }
    finally:
        environment.close()


def boundary_equivalence(model: Path):
    cold = make_environment(model)
    hot = make_environment(model)
    try:
        cold_observation, _ = cold.reset(seed=90)
        hot_observation, _ = hot.reset(seed=90)
        hot._health.set_thermal_load_for_diagnostic(0.10)
        return {
            "max_abs_observation_difference": float(
                np.max(np.abs(cold_observation - hot_observation))
            ),
            "summary_at_lifetime_start": cold_observation[-5:-1].tolist(),
        }
    finally:
        cold.close()
        hot.close()


def main() -> None:
    parser = argparse.ArgumentParser()
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
        default=Path("outputs/cpu_semantic_gates/hierarchical_thermal_v8.json"),
    )
    args = parser.parse_args()
    model = args.low_level_model.resolve()
    cold_low = one_task(model, 0.0, high=False)
    cold_high = one_task(model, 0.0, high=True)
    hot_low = one_task(model, 0.10, high=False)
    hot_high = one_task(model, 0.10, high=True)
    static_low = one_task(model, 0.0, high=False, static=True)
    static_high = one_task(model, 0.0, high=True, static=True)
    boundary = boundary_equivalence(model)
    criteria = {
        "boundary_hides_current_health": (
            boundary["max_abs_observation_difference"] == 0.0
        ),
        "lifetime_summary_starts_zero": boundary["summary_at_lifetime_start"]
        == [0.0, 0.0, 0.0, 0.0],
        "cold_high_is_safe": (
            not cold_high["tripped"] and cold_high["physical_steps"] == 100
        ),
        "cold_high_beats_low": cold_high["reward"] > cold_low["reward"],
        "hot_low_is_safe": not hot_low["tripped"],
        "hot_high_trips": hot_high["tripped"],
        "hot_low_beats_high": hot_low["reward"] > hot_high["reward"],
        "static_high_beats_low": static_high["reward"] > static_low["reward"],
        "one_decision_per_task": all(
            row["physical_steps"] == 100
            for row in (cold_low, cold_high, hot_low, static_low, static_high)
        ),
    }
    report = {
        "phase": "hierarchical_thermal_cpu_semantic_gate",
        "status": "design_validation_not_learned_policy_evidence",
        "low_level_model": str(model),
        "low_level_model_sha256": digest(model),
        "boundary": boundary,
        "counterfactuals": {
            "cold_low": cold_low,
            "cold_high": cold_high,
            "hot_low": hot_low,
            "hot_high": hot_high,
            "static_low": static_low,
            "static_high": static_high,
        },
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("hierarchical thermal semantic gate failed")


if __name__ == "__main__":
    main()
