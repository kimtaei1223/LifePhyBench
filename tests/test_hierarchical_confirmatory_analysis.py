from scripts.analyze_frozen_hierarchical_confirmatory import one_sample_summary


def test_positive_seed_effects_have_positive_interval_and_small_p_value():
    summary = one_sample_summary(
        [0.5, 0.7, 0.9, 1.0, 1.2, 1.3, 1.4, 1.6], bootstrap_seed=1
    )
    assert summary["bootstrap_95_ci"][0] > 0.0
    assert summary["one_sided_t_p"] < 0.05
    assert summary["positive_seeds"] == 8


def test_all_zero_control_is_serialized_with_finite_null_result():
    summary = one_sample_summary([0.0] * 8, bootstrap_seed=2)
    assert summary["bootstrap_95_ci"] == [0.0, 0.0]
    assert summary["one_sided_t_p"] == 1.0
    assert summary["one_sided_wilcoxon_p"] == 1.0
