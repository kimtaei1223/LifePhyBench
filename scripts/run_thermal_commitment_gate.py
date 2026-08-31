"""CPU semantic gates for the health-contingent thermal commitment task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lifephybench.envs.mujoco_pusher import ActuatorWearConfig, PusherActuatorWear
from lifephybench.envs.thermal_commitment import (
    ThermalCommitmentConfig,
    ThermalModeCommitment,
)

CANONICAL_TASK_SEED = 811
HEALTH_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)


def make_environment(static: bool = False) -> ThermalModeCommitment:
    base = PusherActuatorWear.make(
        ActuatorWearConfig(
            wear_rate=0.0,
            thermal_enabled=True,
            thermal_heat_rate=0.1,
            thermal_cooling_rate=0.0,
            thermal_episode_cooling=0.0,
            thermal_degradation_mode=(
                "exogenous_clock" if static else "endogenous_action"
            ),
            thermal_exogenous_dose_per_step=0.0,
            canonical_task_seed=CANONICAL_TASK_SEED,
        ),
        max_episode_steps=100,
    )
    return ThermalModeCommitment(base, ThermalCommitmentConfig())


def boundary_equivalence() -> dict[str, float]:
    cold = make_environment()
    hot = make_environment()
    try:
        cold.reset_lifetime(seed=1)
        hot.reset_lifetime(seed=2)
        hot.env.set_thermal_load_for_diagnostic(0.8)
        cold_observation, _ = cold.reset(seed=3)
        hot_observation, _ = hot.reset(seed=4)
        return {
            "cold_thermal_load": float(cold.env.thermal_load),
            "hot_thermal_load": float(hot.env.thermal_load),
            "boundary_observation_max_abs_difference": float(
                np.max(np.abs(cold_observation - hot_observation))
            ),
        }
    finally:
        cold.close()
        hot.close()


def response_separation(probe_steps: int = 5) -> dict[str, float]:
    rows = {}
    for label, decision in (("high", 1.0), ("low", -1.0)):
        environment = make_environment()
        try:
            observation, _ = environment.reset_lifetime(seed=10)
            action = np.ones(environment.action_space.shape)
            action[0] = decision
            total_reward = 0.0
            for _ in range(probe_steps):
                observation, reward, terminated, truncated, _ = environment.step(
                    action
                )
                total_reward += reward
                if terminated or truncated:
                    break
            rows[label] = {
                "response_norm": float(np.linalg.norm(observation[11:18])),
                "total_reward": total_reward,
                "terminal_thermal_load": float(environment.env.thermal_load),
            }
        finally:
            environment.close()
    return {
        "probe_steps": probe_steps,
        "high_response_norm": rows["high"]["response_norm"],
        "low_response_norm": rows["low"]["response_norm"],
        "response_norm_gap": (
            rows["high"]["response_norm"] - rows["low"]["response_norm"]
        ),
        "high_terminal_thermal_load": rows["high"]["terminal_thermal_load"],
        "low_terminal_thermal_load": rows["low"]["terminal_thermal_load"],
    }


def health_response_separation(
    hot_load: float = 0.10, probe_steps: int = 5
) -> dict[str, float]:
    rows = {}
    for label, thermal_load in (("cold", 0.0), ("hot", hot_load)):
        base = PusherActuatorWear.make(
            ActuatorWearConfig(
                wear_rate=0.0,
                thermal_enabled=True,
                thermal_heat_rate=0.1,
                thermal_cooling_rate=0.0,
                thermal_episode_cooling=0.0,
                canonical_task_seed=CANONICAL_TASK_SEED,
            ),
            max_episode_steps=100,
        )
        try:
            observation, _ = base.reset_lifetime(seed=15)
            base.set_thermal_load_for_diagnostic(thermal_load)
            action = np.ones(base.action_space.shape)
            for _ in range(probe_steps):
                observation, *_ = base.step(action)
            rows[label] = float(np.linalg.norm(observation[11:18]))
        finally:
            base.close()
    return {
        "hot_load": hot_load,
        "probe_steps": probe_steps,
        "cold_response_norm": rows["cold"],
        "hot_response_norm": rows["hot"],
        "cold_minus_hot_response_gap": rows["cold"] - rows["hot"],
    }


def one_step_return(thermal_load: float, high: bool, static: bool = False) -> float:
    environment = make_environment(static=static)
    try:
        environment.reset_lifetime(seed=20)
        environment.env.set_thermal_load_for_diagnostic(
            0.0 if static else thermal_load
        )
        action = np.zeros(environment.action_space.shape)
        action[0] = 1.0 if high else -1.0
        _, reward, _, _, _ = environment.step(action)
        return float(reward)
    finally:
        environment.close()


def decision_relevance(static: bool = False) -> dict[str, object]:
    rows = []
    for load in HEALTH_GRID:
        high_return = one_step_return(load, high=True, static=static)
        low_return = one_step_return(load, high=False, static=static)
        rows.append(
            {
                "thermal_load": load,
                "high_return": high_return,
                "low_return": low_return,
                "oracle_return": max(high_return, low_return),
                "oracle_mode": "high" if high_return > low_return else "low",
            }
        )
    oracle = float(np.mean([row["oracle_return"] for row in rows]))
    blind_high = float(np.mean([row["high_return"] for row in rows]))
    blind_low = float(np.mean([row["low_return"] for row in rows]))
    best_blind = max(blind_high, blind_low)
    return {
        "static": static,
        "health_grid": list(HEALTH_GRID),
        "oracle_mean_return": oracle,
        "blind_high_mean_return": blind_high,
        "blind_low_mean_return": blind_low,
        "best_blind_mean_return": best_blind,
        "oracle_minus_best_blind": oracle - best_blind,
        "rows": rows,
    }


def history_identifiability(histories: int = 20, steps: int = 5) -> dict[str, float]:
    true_loads = []
    history_predictions = []
    boundary_observations = []
    for index, scale in enumerate(np.linspace(0.0, 0.9, histories)):
        environment = PusherActuatorWear.make(
            ActuatorWearConfig(
                wear_rate=0.0,
                thermal_enabled=True,
                thermal_heat_rate=0.1,
                thermal_cooling_rate=0.0,
                thermal_episode_cooling=0.0,
                canonical_task_seed=CANONICAL_TASK_SEED,
            ),
            max_episode_steps=100,
        )
        try:
            environment.reset_lifetime(seed=100 + index)
            action = np.full(environment.action_space.shape, 2.0 * scale)
            integrated_dose = 0.0
            for _ in range(steps):
                environment.step(action)
                integrated_dose += float(scale**2)
            observation, _ = environment.reset(seed=200 + index)
            true_loads.append(float(environment.thermal_load))
            history_predictions.append(0.1 * integrated_dose)
            boundary_observations.append(observation)
        finally:
            environment.close()
    true = np.asarray(true_loads)
    history = np.asarray(history_predictions)
    clock = np.full_like(true, true.mean())
    observations = np.asarray(boundary_observations)
    return {
        "histories": histories,
        "steps_per_history": steps,
        "thermal_load_range": float(true.max() - true.min()),
        "history_model_rmse": float(np.sqrt(np.mean((history - true) ** 2))),
        "clock_only_rmse": float(np.sqrt(np.mean((clock - true) ** 2))),
        "boundary_observation_max_range": float(
            np.max(np.ptp(observations, axis=0))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/cpu_semantic_gates/thermal_commitment_gate_v4.json"),
    )
    args = parser.parse_args()

    boundary = boundary_equivalence()
    response = response_separation()
    health_response = health_response_separation()
    history = history_identifiability()
    dynamic_decision = decision_relevance(static=False)
    static_decision = decision_relevance(static=True)
    criteria = {
        "canonical_boundary": boundary[
            "boundary_observation_max_abs_difference"
        ]
        <= 1e-12,
        "physical_response_gap": response["response_norm_gap"] >= 1.0,
        "health_response_gap": health_response[
            "cold_minus_hot_response_gap"
        ]
        >= 0.20,
        "history_identifiability": (
            history["history_model_rmse"] <= 1e-7
            and history["clock_only_rmse"] >= 0.1
            and history["boundary_observation_max_range"] <= 1e-12
        ),
        "dynamic_decision_relevance": dynamic_decision[
            "oracle_minus_best_blind"
        ]
        >= 1.0,
        "static_negative_control": abs(
            float(static_decision["oracle_minus_best_blind"])
        )
        <= 1e-12,
    }
    report = {
        "phase": "thermal_commitment_cpu_semantic_gate",
        "status": "design_validation_not_learned_policy_evidence",
        "configuration": {
            "canonical_task_seed": CANONICAL_TASK_SEED,
            "health_grid": list(HEALTH_GRID),
            "commitment": ThermalCommitmentConfig().__dict__,
        },
        "boundary_equivalence": boundary,
        "response_separation": response,
        "health_response_separation": health_response,
        "history_identifiability": history,
        "dynamic_decision_relevance": dynamic_decision,
        "static_decision_relevance": static_decision,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("thermal commitment CPU semantic gate failed")


if __name__ == "__main__":
    main()
