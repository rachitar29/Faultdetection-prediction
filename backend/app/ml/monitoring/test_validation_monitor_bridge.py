
"""
Unit tests for the validation-monitoring bridge.

Uses temporary CSV files and mocked monitoring results.
Does not modify actual datasets or trained models.
"""

import tempfile
import unittest

import pandas as pd

from pathlib import Path
from unittest.mock import patch

from app.ml.monitoring.validation_monitor_bridge import (
    run_validation_monitoring,
)


class TestValidationMonitorBridge(unittest.TestCase):

    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        # Create actual temporary CSV files.
        self.reference_file = self.base_dir / "reference.csv"
        self.current_file = self.base_dir / "current.csv"

        sample_data = pd.DataFrame({
            "Air temperature [K]": [298.1, 300.2, 299.5],
            "Process temperature [K]": [308.6, 310.1, 309.2],
            "Rotational speed [rpm]": [1551, 1500, 1450],
            "Torque [Nm]": [42.8, 40.0, 45.0],
            "Tool wear [min]": [0, 100, 120],
            "Type": ["M", "L", "H"],
        })

        sample_data.to_csv(
            self.reference_file,
            index=False
        )

        sample_data.to_csv(
            self.current_file,
            index=False
        )

        self.confirmed_records = {
            "success": True,
            "matched_count": 2,
            "matched_records": [
                {
                    "prediction_id": "PRED-001",
                    "equipment_id": "MACHINE-001",
                    "actual_failure": "FAILURE",
                    "predicted_failure": "FAILURE",
                    "failure_probability": 0.91,
                    "outcome_source": "technician_confirmed",
                },
                {
                    "prediction_id": "PRED-002",
                    "equipment_id": "MACHINE-002",
                    "actual_failure": "NORMAL",
                    "predicted_failure": "NORMAL",
                    "failure_probability": 0.08,
                    "outcome_source": "maintenance_record",
                },
            ],
        }

        self.mock_monitoring_report = {
            "report_type": "unified_monitoring",
            "model_name": "XGBoost",
            "overall_status": "STABLE",
            "data_drift": {
                "overall_status": "STABLE"
            },
            "model_performance": {
                "accuracy": 1.0
            },
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch(
        "app.ml.monitoring.validation_monitor_bridge."
        "run_unified_monitoring"
    )
    @patch(
        "app.ml.monitoring.validation_monitor_bridge."
        "evaluate_confirmed_outcomes"
    )
    def test_confirmed_outcomes_are_passed_to_monitoring(
        self,
        mock_validation,
        mock_monitoring,
    ):

        mock_validation.return_value = self.confirmed_records

        mock_monitoring.return_value = (
            self.mock_monitoring_report.copy()
        )

        result = run_validation_monitoring(
            current_dataset=self.current_file,
            reference_dataset=self.reference_file,
            model_name="XGBoost",
        )

        mock_monitoring.assert_called_once()

        arguments = mock_monitoring.call_args.kwargs

        self.assertEqual(
            arguments["y_true"],
            ["FAILURE", "NORMAL"],
        )

        self.assertEqual(
            arguments["y_pred"],
            ["FAILURE", "NORMAL"],
        )

        self.assertEqual(
            arguments["y_probability"],
            [0.91, 0.08],
        )

        self.assertEqual(
            result["prediction_validation"][
                "matched_outcome_count"
            ],
            2,
        )

    @patch(
        "app.ml.monitoring.validation_monitor_bridge."
        "run_unified_monitoring"
    )
    @patch(
        "app.ml.monitoring.validation_monitor_bridge."
        "evaluate_confirmed_outcomes"
    )
    def test_no_confirmed_outcomes(
        self,
        mock_validation,
        mock_monitoring,
    ):

        mock_validation.return_value = {
            "success": False,
            "matched_count": 0,
            "message": "No confirmed outcomes available.",
        }

        mock_monitoring.return_value = (
            self.mock_monitoring_report.copy()
        )

        result = run_validation_monitoring(
            current_dataset=self.current_file,
            reference_dataset=self.reference_file,
        )

        arguments = mock_monitoring.call_args.kwargs

        self.assertIsNone(arguments["y_true"])
        self.assertIsNone(arguments["y_pred"])

        self.assertFalse(
            result["prediction_validation"][
                "confirmed_outcomes_available"
            ]
        )

        self.assertEqual(
            result["prediction_validation"][
                "matched_outcome_count"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
