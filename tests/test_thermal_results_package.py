from scripts.build_thermal_results_package import holm_adjust


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
