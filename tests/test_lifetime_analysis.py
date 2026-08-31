import tempfile
import unittest
from pathlib import Path

from lifephybench.lifetime_analysis import paired_bootstrap, read_jsonl, write_jsonl


class LifetimeAnalysisTests(unittest.TestCase):
    def test_jsonl_round_trip(self):
        records = [{"seed": 1, "score": 0.5}, {"seed": 2, "score": 0.75}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl(records, path)
            self.assertEqual(read_jsonl(path), records)

    def test_paired_bootstrap_uses_only_shared_lifetimes(self):
        records = [
            {"condition": "reference", "seed": 1, "score": 1.0},
            {"condition": "treatment", "seed": 1, "score": 2.0},
            {"condition": "reference", "seed": 2, "score": 3.0},
            {"condition": "treatment", "seed": 2, "score": 5.0},
            {"condition": "treatment", "seed": 3, "score": 100.0},
        ]
        result = paired_bootstrap(
            records,
            metric="score",
            condition_key="condition",
            reference="reference",
            treatment="treatment",
            bootstrap_draws=100,
            rng_seed=1,
        )
        self.assertEqual(result.paired_lifetimes, 2)
        self.assertEqual(result.reference_mean, 2.0)
        self.assertEqual(result.treatment_mean, 3.5)
        self.assertEqual(result.mean_difference, 1.5)

    def test_paired_bootstrap_rejects_duplicate_pair(self):
        records = [
            {"condition": "reference", "seed": 1, "score": 1.0},
            {"condition": "reference", "seed": 1, "score": 2.0},
            {"condition": "treatment", "seed": 1, "score": 3.0},
            {"condition": "treatment", "seed": 2, "score": 4.0},
            {"condition": "reference", "seed": 2, "score": 2.0},
        ]
        with self.assertRaises(ValueError):
            paired_bootstrap(
                records,
                metric="score",
                condition_key="condition",
                reference="reference",
                treatment="treatment",
            )


if __name__ == "__main__":
    unittest.main()
