from __future__ import annotations

from scripts.run_physics_residual_v12_recovery_pipeline import select_candidate


CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_shocks",
    "ood_combined",
)


def _policy(seeds, reward, trip, spec):
    return {
        "spec": spec,
        "summary": {
            "trip_rate": trip,
            "lifetime_rows": [
                {"seed": seed, "mean_reward_per_task": reward + seed * 1e-6}
                for seed in seeds
            ],
        },
    }


def _cells(*, candidate_trip=0.01, shock_gain=0.4, in_domain_gain=-0.05):
    seeds = list(range(30))
    spec = {
        "name": "candidate",
        "type": "hybrid_belief",
        "cutoff": 0.0625,
        "residual_scale": 1.0,
        "uncertainty_multiplier": 1.0,
    }
    cells = {}
    for condition in CONDITIONS:
        gain = in_domain_gain if condition == "in_domain" else 0.5
        if condition == "ood_shocks":
            gain = shock_gain
        cells[condition] = {
            "physics_belief": _policy(
                seeds, -42.0, 0.03, {"name": "physics_belief"}
            ),
            "candidate": _policy(seeds, -42.0 + gain, candidate_trip, spec),
        }
    return cells


def test_select_candidate_accepts_safe_robust_controller():
    result = select_candidate(_cells())
    assert result["passed"] is True
    assert result["selected"]["policy"] == "candidate"


def test_select_candidate_rejects_trip_rate_violation():
    result = select_candidate(_cells(candidate_trip=0.019))
    assert result["passed"] is False


def test_select_candidate_rejects_negative_shock_transfer():
    result = select_candidate(_cells(shock_gain=-0.01))
    assert result["passed"] is False
