"""Audit mode-coordinate causal support in the commitment rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_fair_recurrent import make_fair_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/cpu_semantic_gates/thermal_commitment_credit_mask_v6.json"
        ),
    )
    args = parser.parse_args()
    environment = make_fair_environment(
        environment_id="Pusher-v5",
        mechanism="thermal",
        degradation_mode="exogenous_clock",
        episode_steps=100,
        episodes_per_lifetime=20,
        exogenous_dose_per_step=0.25,
        thermal_exogenous_dose_per_step=0.0,
        joint_aging_exogenous_dose_per_step=0.25,
        thermal_heat_rate=0.1,
        thermal_cooling_rate=0.0,
        thermal_episode_cooling=0.0,
        canonical_task_seed=811,
        thermal_commitment=True,
        monitor=lambda env: env,
    )
    transitions = 0
    decisions = 0
    inner_tasks = 0
    try:
        observation, _ = environment.reset(seed=91)
        while True:
            action = np.zeros(environment.action_space.shape)
            observation, _, terminated, truncated, info = environment.step(action)
            transitions += 1
            decisions += int(info.get("lifephy/thermal_mode_selected_now", False))
            inner_tasks += int(info.get("lifephy/inner_task_boundary", False))
            if terminated or truncated:
                break
    finally:
        environment.close()

    report = {
        "phase": "thermal_commitment_credit_assignment_cpu_audit",
        "status": "optimization_semantics_not_learned_policy_evidence",
        "transitions": transitions,
        "task_episodes": inner_tasks,
        "causal_mode_decisions": decisions,
        "legacy_mode_loss_transitions": transitions,
        "masked_mode_loss_transitions": decisions,
        "causal_decision_fraction": decisions / transitions,
        "legacy_irrelevant_to_causal_ratio": (transitions - decisions) / decisions,
        "criteria": {
            "canonical_rollout_length": transitions == 2_000,
            "twenty_task_decisions": decisions == inner_tasks == 20,
            "decision_fraction_one_percent": decisions / transitions == 0.01,
            "masked_loss_matches_causal_support": decisions == inner_tasks,
        },
    }
    report["passed"] = all(report["criteria"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("commitment credit-assignment audit failed")


if __name__ == "__main__":
    main()
