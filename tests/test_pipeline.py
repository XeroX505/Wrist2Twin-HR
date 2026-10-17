import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import module3_pmbnn as m3
from run_pipeline import load_record, file_hash


class PipelineTests(unittest.TestCase):
    def test_artifact_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.bin"
            path.write_bytes(b"abc")
            self.assertEqual(file_hash(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_exact_splits(self):
        df = pd.DataFrame({"activity": np.repeat(["A", "B", "C", "D"], 250)})
        for scenario, total in [("80", 800), ("50", 500), ("300", 300)]:
            train, test, train_segments, test_segments, ti, vi = m3.split_per_activity(df, scenario)
            self.assertEqual(train.sum(), total)
            self.assertEqual(test.sum(), 1000 - total)
            self.assertFalse(np.any(train & test))
            self.assertEqual(len(set(ti) & set(vi)), 0)
            self.assertEqual(sorted(np.r_[ti, vi]), list(range(1000)))
            for front, back in zip(train_segments, test_segments):
                self.assertLess(front.max(), back.min())

    def test_reject_missing_wrist_channels(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "record.csv"
            pd.DataFrame({"HR_bpm": [70, 71], "activity": ["walk", "walk"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_record(path, "dot_wrist_with_gyr")

    def test_pmbnn_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            n = 100
            df = pd.DataFrame({"t": np.arange(n), "hr": 80 + 5 * np.sin(np.arange(n) / 12),
                               "activity": np.repeat(["A", "B"], 50)})
            vo2 = 8 + np.sin(np.arange(n) / 12)
            result = m3.calibrate_and_evaluate(df, vo2, d, "synthetic", "test", "synthetic",
                                             scenarios=["80"], epochs=2)
            self.assertEqual(result.iloc[0]["n_train"], 80)
            pred = pd.read_csv(Path(d) / "part3_predictions_per_activity_80.csv")
            self.assertEqual(len(pred), 20)
            self.assertTrue(np.isfinite(pred["hr_pred"]).all())
            history = pd.read_csv(Path(d) / "part3_history_per_activity_80.csv")
            self.assertEqual(history.iloc[-1]["phys_weight"], 10000)
            self.assertTrue(all(f"lambda{i}" in history for i in range(1, 7)))


if __name__ == "__main__":
    unittest.main()
