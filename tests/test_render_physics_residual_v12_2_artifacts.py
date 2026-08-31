from __future__ import annotations

import pytest

from scripts.render_physics_residual_v12_2_artifacts import paired_effect_rows


CONDITIONS = (
    "in_domain",
    "ood_sensor_noise",
    "ood_cooling",
    "ood_combined",
    "ood_shocks",
)


def _policy(seeds, base):
    return {
        "summary": {
            "lifetime_rows": [
                {
                    "seed": seed,
                    "mean_reward_per_task": base + float(seed),
                    "trip_rate": 0.0,
                }
                for seed in seeds
            ]
        }
    }


def _cells(seeds):
    return {
        condition: {
            "physics_belief": _policy(seeds, -10.0),
            "hybrid_belief": _policy(seeds, -9.0 if condition != "in_domain" else -10.5),
        }
        for condition in CONDITIONS
    }


def test_paired_effect_rows_preserve_seed_as_analysis_unit():
    rows = paired_effect_rows(_cells([1, 2, 3]), [1, 2, 3])
    assert [row["seed"] for row in rows] == [1, 2, 3]
    assert all(row["in_domain_hybrid_minus_physics"] == -0.5 for row in rows)
    assert all(
        row["target_ood_aggregate_hybrid_minus_physics"] == 1.0 for row in rows
    )


def test_paired_effect_rows_reject_seed_mismatch():
    with pytest.raises(ValueError, match="hybrid seed mismatch"):
        paired_effect_rows(_cells([1, 2]), [1, 2, 3])
