from scripts.run_hierarchical_auto_calibration import (
    replication_summary,
    screen_passed,
)


def report(reward_difference=2.0, adaptation_gap=0.3, static=True):
    task_reward = -44.0
    return {
        "wiring_passed": True,
        "behavior_checks": {
            "static_task_prefers_high": static,
            "static_lifetime_prefers_high": static,
            "dynamic_lifetime_uses_high_more_when_cold": adaptation_gap > 0.1,
            "dynamic_lifetime_exceeds_task_reward": reward_difference > 0.0,
        },
        "rows": [
            {
                "label": "dynamic",
                "memory": "task",
                "mean_reward": task_reward,
                "cold_high_rate": 0.0,
                "hot_high_rate": 0.0,
            },
            {
                "label": "dynamic",
                "memory": "lifetime",
                "mean_reward": task_reward + reward_difference,
                "cold_high_rate": 0.5,
                "hot_high_rate": 0.5 - adaptation_gap,
            },
        ],
    }


def test_screen_requires_controls_and_adaptive_lifetime():
    assert screen_passed(report())
    assert not screen_passed(report(adaptation_gap=0.0))
    assert not screen_passed(report(static=False))


def test_replication_requires_two_seeds_and_positive_means():
    summary = replication_summary(
        [report(2.0, 0.3), report(1.0, 0.2), report(-0.5, 0.0)]
    )
    assert summary["passed"]
    assert summary["mean_reward_difference"] > 0.0
    assert summary["mean_adaptation_gap"] > 0.1


def test_replication_rejects_single_seed_success():
    summary = replication_summary(
        [report(2.0, 0.3), report(-2.0, 0.0), report(-2.0, 0.0)]
    )
    assert not summary["passed"]
