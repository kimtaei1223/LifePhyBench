from __future__ import annotations

from scripts.run_physics_residual_v12_confirmatory_pipeline import (
    analyze_confirmatory,
    analyze_development,
)


def _policy_rows(seeds, reward, trip):
    return {
        "summary": {
            "trip_rate": trip,
            "lifetime_rows": [
                {"seed": seed, "mean_reward_per_task": reward + seed * 1e-6}
                for seed in seeds
            ],
        }
    }


def _result():
    seeds = list(range(100))
    result = {}
    for condition in (
        "in_domain",
        "ood_sensor_noise",
        "ood_cooling",
        "ood_shocks",
        "ood_combined",
    ):
        hybrid_gain = -0.05 if condition == "in_domain" else 0.60
        result[condition] = {
            "physics_belief": _policy_rows(seeds, -42.0, 0.015),
            "hybrid_belief": _policy_rows(seeds, -42.0 + hybrid_gain, 0.01),
        }
    return result


def test_development_gate_accepts_safe_ood_improvement():
    report = analyze_development(_result())
    assert report["passed"] is True
    assert report["ood_wins_at_least_0_25"] == 4


def test_confirmatory_conjunction_accepts_paired_effect():
    report = analyze_confirmatory(_result())
    assert report["confirmatory_passed"] is True
    assert all(report["criteria"].values())


def test_confirmatory_safety_failure_blocks_claim():
    result = _result()
    result["ood_combined"]["hybrid_belief"]["summary"]["trip_rate"] = 0.03
    report = analyze_confirmatory(result)
    assert report["confirmatory_passed"] is False
    assert report["criteria"]["all_hybrid_trip_rates_at_most_0_02"] is False
