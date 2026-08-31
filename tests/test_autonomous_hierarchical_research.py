from scripts.run_autonomous_hierarchical_research import robust_gate, screen_passed


def metric(effect=2.0, gap=0.3, high_rate=0.2):
    return {
        "wiring": True,
        "static_controls": True,
        "reward_effect": effect,
        "adaptation_gap": gap,
        "lifetime_high_rate": high_rate,
        "lifetime_trip_rate": 0.0,
    }


def test_screen_requires_adaptation_effect_and_nondegenerate_policy():
    assert screen_passed(metric())
    assert not screen_passed(metric(effect=0.0))
    assert not screen_passed(metric(gap=0.1))
    assert not screen_passed(metric(high_rate=0.0))


def test_robust_gate_allows_one_failed_calibration_seed():
    rows = [metric(), metric(), metric(), metric(), metric(effect=-0.5, gap=0.0)]
    assert robust_gate(rows)["passed"]


def test_robust_gate_rejects_two_failed_calibration_seeds():
    rows = [metric(), metric(), metric(), metric(effect=-2.0, gap=0.0), metric(effect=-2.0, gap=0.0)]
    assert not robust_gate(rows)["passed"]
