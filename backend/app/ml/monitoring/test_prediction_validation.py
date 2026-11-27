
"""
Unit tests for prediction validation.

Uses temporary files and controlled test records.
Does not modify actual validation data.
"""

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from app.ml.monitoring import prediction_validation as pv


class TestPredictionValidation(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()

        self.validation_dir = Path(self.temp_dir.name)

        self.predictions_file = (
            self.validation_dir / "predictions.jsonl"
        )

        self.outcomes_file = (
            self.validation_dir / "actual_outcomes.jsonl"
        )

        self.reports_dir = (
            self.validation_dir / "reports"
        )

        self.patches = [
            patch.object(
                pv, "PREDICTIONS_FILE",
                self.predictions_file
            ),
            patch.object(
                pv, "ACTUAL_OUTCOMES_FILE",
                self.outcomes_file
            ),
            patch.object(
                pv, "REPORTS_DIR",
                self.reports_dir
            ),
        ]

        for patcher in self.patches:
            patcher.start()

        self.prediction = {
            "failure_probability": 0.85,
            "failure_prediction": "FAILURE",
            "risk_level": "Critical",
            "model_metadata": {
                "model_name": "XGBoost"
            },
        }

    def tearDown(self):

        for patcher in self.patches:
            patcher.stop()

        self.temp_dir.cleanup()

    def test_record_prediction(self):

        result = pv.record_prediction(
            equipment_id="TEST-001",
            prediction=self.prediction,
            prediction_id="PRED-001",
        )

        self.assertEqual(
            result["prediction_id"],
            "PRED-001"
        )

        self.assertTrue(
            self.predictions_file.exists()
        )

        records = pv.read_jsonl(
            self.predictions_file
        )

        self.assertEqual(len(records), 1)

    def test_record_actual_outcome(self):

        result = pv.record_actual_outcome(
            prediction_id="PRED-001",
            actual_failure=True,
            source="technician_confirmed",
            notes="Controlled test record",
        )

        self.assertEqual(
            result["actual_label"],
            "FAILURE"
        )

        self.assertTrue(
            self.outcomes_file.exists()
        )

    def test_evaluate_classification(self):

        metrics = pv.evaluate_classification(
            actual_labels=[
                "NORMAL",
                "FAILURE",
                "NORMAL",
                "FAILURE",
            ],
            predicted_labels=[
                "NORMAL",
                "FAILURE",
                "FAILURE",
                "FAILURE",
            ],
        )

        self.assertEqual(
            metrics["accuracy"],
            0.75
        )

        self.assertAlmostEqual(
            metrics["precision"],
            2 / 3
        )

        self.assertEqual(
            metrics["recall"],
            1.0
        )

        self.assertAlmostEqual(
            metrics["f1_score"],
            0.8
        )

    def test_match_predictions_with_outcomes(self):

        pv.record_prediction(
            "TEST-001",
            self.prediction,
            "PRED-001",
        )

        pv.record_actual_outcome(
            prediction_id="PRED-001",
            actual_failure=True,
        )

        result = pv.evaluate_confirmed_outcomes()

        self.assertTrue(result["success"])
        self.assertEqual(result["matched_count"], 1)

        self.assertEqual(
            result["metrics"]["accuracy"],
            1.0
        )

    def test_save_validation_report(self):

        report = {
            "success": True,
            "matched_count": 1,
            "metrics": {
                "accuracy": 1.0,
                "f1_score": 1.0,
            },
        }

        report_path = Path(
            pv.save_validation_report(report)
        )

        self.assertTrue(report_path.exists())

        with report_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            saved_report = json.load(file)

        self.assertEqual(
            saved_report["metrics"]["accuracy"],
            1.0
        )

    def test_invalid_probability_rejected(self):

        invalid_prediction = {
            "failure_probability": 1.5,
            "failure_prediction": "FAILURE",
        }

        with self.assertRaises(ValueError):

            pv.record_prediction(
                "TEST-002",
                invalid_prediction,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
