
"""
Unit tests for ML performance history tracking.

Uses temporary files to avoid changing real monitoring records.
"""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from app.ml.monitoring import performance_history as ph


class TestPerformanceHistory(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.patches = [
            patch.object(ph, "HISTORY_DIR", self.temp_path),
            patch.object(
                ph,
                "HISTORY_FILE",
                self.temp_path / "performance_history.jsonl",
            ),
        ]

        for patcher in self.patches:
            patcher.start()

        self.metrics_one = {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.90,
            "f1_score": 0.91,
        }

        self.metrics_two = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.80,
            "f1_score": 0.81,
        }

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()

        self.temp_dir.cleanup()

    def test_record_performance(self):
        result = ph.record_model_performance(
            model_name="XGBoost",
            metrics=self.metrics_one,
            model_version="1.0",
            sample_count=100,
        )

        self.assertEqual(result["model_name"], "XGBoost")
        self.assertEqual(result["metrics"]["f1_score"], 0.91)
        self.assertTrue(ph.HISTORY_FILE.exists())

    def test_retrieve_performance_history(self):
        ph.record_model_performance(
            "XGBoost",
            self.metrics_one,
        )

        ph.record_model_performance(
            "Random Forest",
            self.metrics_two,
        )

        records = ph.get_model_performance_history()

        self.assertEqual(len(records), 2)

        xgboost_records = ph.get_model_performance_history(
            "XGBoost"
        )

        self.assertEqual(len(xgboost_records), 1)

    def test_compare_latest_performance(self):
        ph.record_model_performance(
            "XGBoost",
            self.metrics_one,
        )

        ph.record_model_performance(
            "XGBoost",
            self.metrics_two,
        )

        result = ph.compare_latest_performance("XGBoost")

        self.assertTrue(result["success"])

        self.assertAlmostEqual(
            result["metric_changes"]["f1_score"],
            -0.10,
        )

    def test_detect_performance_degradation(self):
        ph.record_model_performance(
            "XGBoost",
            self.metrics_one,
        )

        ph.record_model_performance(
            "XGBoost",
            self.metrics_two,
        )

        result = ph.detect_performance_degradation(
            "XGBoost",
            minimum_acceptable_f1=0.70,
            maximum_f1_drop=0.10,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "WARNING")

    def test_insufficient_history(self):
        ph.record_model_performance(
            "XGBoost",
            self.metrics_one,
        )

        result = ph.compare_latest_performance("XGBoost")

        self.assertFalse(result["success"])

    def test_invalid_metric_rejected(self):
        invalid_metrics = {
            "accuracy": 1.5,
            "precision": 0.9,
            "recall": 0.9,
            "f1_score": 0.9,
        }

        with self.assertRaises(ValueError):
            ph.record_model_performance(
                "XGBoost",
                invalid_metrics,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
