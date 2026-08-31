from __future__ import annotations

from scripts.run_physics_residual_v12_2_scoped_confirmatory import (
    analyze_scoped_confirmatory,
    development_scope_is_supported,
)


CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_combined",
    "ood_shocks",
)


def _rows(seeds, reward, trip):
    return {
        "summary": {
            "trip_rate": trip,
            "lifetime_rows": [
                {"seed": seed, "mean_reward_per_task": reward + seed * 1e-6}
                for seed in seeds
            ],
        }
    }


def _cells(*, trip=0.01, shock_gain=-0.5):
    seeds = list(range(100))
    cells = {}
    for condition in CONDITIONS:
        gain = -0.05 if condition == "in_domain" else 0.75
        if condition == "ood_shocks":
            gain = shock_gain
        cells[condition] = {
            "physics_belief": _rows(seeds, -42.0, 0.03),
            "hybrid_belief": _rows(seeds, -42.0 + gain, trip),
        }
    return cells


def test_scoped_confirmation_passes_despite_secondary_shock_null():
    report = analyze_scoped_confirmatory(_cells(shock_gain=0.0))
    assert report["confirmatory_passed"] is True
    assert "ood_shocks" in report["secondary_boundary_conditions"]


def test_scoped_confirmation_keeps_global_safety_gate():
    report = analyze_scoped_confirmatory(_cells(trip=0.03))
    assert report["confirmatory_passed"] is False
    assert report["criteria"]["all_condition_hybrid_trip_rates_at_most_0_02"] is False


def test_development_scope_excludes_shock_from_primary_gate():
    effects = {
        name: {
            "mean": 0.75,
            "bootstrap_95_ci": [0.25, 1.25],
            "hybrid_trip_rate": 0.01,
        }
        for name in CONDITIONS
    }
    effects["in_domain"]["mean"] = -0.05
    effects["ood_shocks"]["mean"] = -1.0
    audit = {"effects": effects}
    assert development_scope_is_supported(audit) is True
