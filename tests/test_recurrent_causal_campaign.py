import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_recurrent_causal_campaign import make_runs


class RecurrentCausalCampaignTests(unittest.TestCase):
    def test_orders_endogenous_before_exogenous_and_uses_separate_roots(self):
        runs = make_runs(
            endogenous_seeds=[1002],
            exogenous_seeds=[1000, 1001],
            memory_modes=["episode", "lifetime"],
            total_timesteps=1_000_000,
            endogenous_output_root=Path("outputs/endogenous"),
            exogenous_output_root=Path("outputs/exogenous"),
        )
        self.assertEqual(len(runs), 6)
        self.assertEqual(
            [(run.degradation_mode, run.memory_mode, run.seed) for run in runs],
            [
                ("endogenous_action", "episode", 1002),
                ("endogenous_action", "lifetime", 1002),
                ("exogenous_clock", "episode", 1000),
                ("exogenous_clock", "episode", 1001),
                ("exogenous_clock", "lifetime", 1000),
                ("exogenous_clock", "lifetime", 1001),
            ],
        )
        self.assertEqual(runs[0].run_directory, Path("outputs/endogenous/episode-seed1002-steps1000k"))
        self.assertEqual(runs[-1].run_directory, Path("outputs/exogenous/lifetime-seed1001-steps1000k"))


if __name__ == "__main__":
    unittest.main()
