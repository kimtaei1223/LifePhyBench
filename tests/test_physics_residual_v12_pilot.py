from __future__ import annotations

import numpy as np
import torch

from scripts.qualify_physics_belief_v12 import TransitionModel
from scripts.run_physics_residual_v12_pilot import (
    FEATURE_NAMES,
    ResidualGRU,
    build_sequences,
    summarize_pilot,
)


def test_residual_gru_preserves_batch_and_sequence_axes():
    model = ResidualGRU(len(FEATURE_NAMES), 8)
    output, hidden = model(torch.zeros(3, 20, len(FEATURE_NAMES)))
    assert output.shape == (3, 20, 2)
    assert hidden.shape == (1, 3, 8)


def test_build_sequences_uses_nonprivileged_features_and_residual_target():
    rows = []
    for task in range(20):
        rows.append(
            {
                "lifetime_ordinal": 0,
                "task_index": task,
                "sensor_load": 0.02,
                "true_load_at_selection": 0.025,
                "action": task % 2,
            }
        )
    transition = TransitionModel(0.0, 1.0, 0.0, 1e-6, 1e-4, 100, 0.99)
    features, targets, exact = build_sequences([(1, rows)], {1}, transition)
    assert features.shape == (1, 20, len(FEATURE_NAMES))
    assert targets.shape == exact.shape == (1, 20)
    assert np.isfinite(features).all()
    assert targets[0, 0] == np.float32(0.005)


def test_pilot_gate_requires_ood_benefit_and_safety():
    def policy(reward, trip):
        return {"summary": {"mean_reward_per_task": reward, "trip_rate": trip}}

    conditions = {}
    for name in (
        "in_domain",
        "ood_sensor_noise",
        "ood_cooling",
        "ood_shocks",
        "ood_combined",
    ):
        conditions[name] = {
            "current_sensor": policy(-42.0, 0.01),
            "ema_history": policy(-41.8, 0.01),
            "physics_belief": policy(-41.5, 0.01),
            "hybrid_belief": policy(-41.1, 0.01),
            "privileged_oracle": policy(-40.8, 0.0),
        }
    offline = {"test": {"hybrid_improvement_over_physics_fraction": 0.10}}
    assert summarize_pilot(conditions, offline)["passed"] is True
    conditions["ood_combined"]["hybrid_belief"] = policy(-41.1, 0.03)
    assert summarize_pilot(conditions, offline)["passed"] is False
