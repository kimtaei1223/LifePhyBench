from __future__ import annotations

import pytest

from scripts.qualify_physics_belief_v12 import (
    PhysicsBeliefPolicy,
    TransitionModel,
    assess_gate,
    fit_transition_model,
)


def test_transition_model_recovers_action_dependent_dynamics():
    rows = []
    load = 0.02
    for task_index, action in enumerate((0, 1, 0, 1, 0, 1)):
        rows.append(
            {
                "lifetime_ordinal": 0,
                "task_index": task_index,
                "true_load_at_selection": load,
                "sensor_load": load + 0.001 * (-1) ** task_index,
                "action": action,
            }
        )
        load = 0.01 + 0.8 * load + 0.02 * action
    model = fit_transition_model([(1, rows)], {1})
    assert model.intercept == pytest.approx(0.01)
    assert model.load_coefficient == pytest.approx(0.8)
    assert model.high_power_increment == pytest.approx(0.02)


def test_belief_policy_uses_transition_and_never_reads_exact_load():
    model = TransitionModel(0.01, 0.8, 0.02, 1e-6, 1e-4, 100, 0.99)
    policy = PhysicsBeliefPolicy(model, cutoff=0.06, uncertainty_multiplier=0.0)
    first = policy.act(task_index=0, sensor=0.02, exact_load=0.99)
    second = policy.act(task_index=1, sensor=0.03, exact_load=0.99)
    assert first == 1
    assert second == 1
    assert policy.mean is not None and policy.mean < 0.06


def test_gate_requires_estimation_reward_and_safety():
    def row(reward, trip):
        return {"summary": {"mean_reward_per_task": reward, "trip_rate": trip}}

    validation = {
        "current_sensor": row(-42.0, 0.01),
        "ema_history": row(-41.6, 0.01),
        "physics_belief": row(-41.3, 0.01),
        "privileged_oracle": row(-41.0, 0.005),
    }
    estimator = {
        "belief_improvement_over_raw_fraction": 0.5,
        "belief_improvement_over_ema_fraction": 0.3,
    }
    result = assess_gate(estimator, validation)
    assert result["passed"] is True
    validation["physics_belief"] = row(-42.1, 0.03)
    assert assess_gate(estimator, validation)["passed"] is False
