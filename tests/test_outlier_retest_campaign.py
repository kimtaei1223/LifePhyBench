import json

from scripts.run_outlier_retest_campaign import calibrated_episode_doses


def test_calibrated_episode_doses_filters_memory_mode(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "seed": 1002,
                        "memory_mode": "episode",
                        "recommended_exogenous_dose_per_step": 0.034,
                    },
                    {
                        "seed": 1002,
                        "memory_mode": "lifetime",
                        "recommended_exogenous_dose_per_step": 0.008,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    assert calibrated_episode_doses(path) == {1002: 0.034}
