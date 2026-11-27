
"""
Tests for the ML integration service.

Uses mocked engine output to test integration logic
without loading or modifying trained models.
"""

import unittest
from unittest.mock import patch

from app.ml.integration.ml_service import MLService
from app.ml.integration.schemas import MLPredictionRequest


class TestMLService(unittest.TestCase):

    def setUp(self):
        self.service = MLService()

        self.sensor_data = {
            "Air temperature [K]": 300.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 120,
            "Type": "M",
        }

        self.mock_engine_result = {
            "model_name": "XGBoost",
            "model_type": "supervised",
            "failure_probability": 0.08,
            "failure_probability_percent": 8.0,
            "failure_predicted": False,
            "decision_threshold": 0.5,
            "risk_level": "Low",
            "classifier_risk_level": "Low",
            "health_score": 92.0,
            "health_score_note": "Test health score",
            "maintenance_priority": "Routine",
            "maintenance_recommendation": "Continue routine monitoring.",
            "top_factors": [],
            "explanation_method": "SHAP",
            "explained_class": 0,
            "sensor_readings": self.sensor_data,
            "prediction_timestamp": "2026-11-19T10:00:00+00:00",
            "anomaly_results": {
                "Isolation Forest": {
                    "model_name": "Isolation Forest",
                    "is_anomaly": False,
                    "anomaly_score": 0.1,
                    "optimized_threshold": 0.5,
                    "threshold_source": "Saved validation threshold",
                },
                "Local Outlier Factor": {
                    "model_name": "Local Outlier Factor",
                    "is_anomaly": False,
                    "anomaly_score": 0.2,
                    "optimized_threshold": 0.5,
                    "threshold_source": "Saved validation threshold",
                },
            },
            "anomaly_assessment": {
                "detectors_flagged": 0,
                "anomaly_consensus": "Neither detector identified an anomaly.",
            },
            "risk_assessment_method": "Test heuristic",
            "rul_prediction": None,
            "rul_available": False,
            "integrated_prediction": True,
            "prediction_components": {
                "failure_classification": True,
                "anomaly_detection": True,
                "combined_risk_assessment": True,
                "shap_explanation": True,
                "remaining_useful_life": False,
            },
        }

    @patch("app.ml.integration.ml_service.predict_equipment_condition")
    def test_prediction_with_dictionary(self, mock_engine):

        mock_engine.return_value = self.mock_engine_result

        result = self.service.predict({
            "sensor_data": self.sensor_data,
            "equipment_id": "TEST-001",
        })

        self.assertTrue(result["success"])
        self.assertEqual(result["equipment_id"], "TEST-001")
        self.assertEqual(result["failure_prediction"], "NORMAL")
        self.assertEqual(result["risk_level"], "Low")
        self.assertFalse(result["anomaly_detected"])
        self.assertFalse(result["rul_available"])

        mock_engine.assert_called_once()

    @patch("app.ml.integration.ml_service.predict_equipment_condition")
    def test_prediction_with_request_object(self, mock_engine):

        mock_engine.return_value = self.mock_engine_result

        request = MLPredictionRequest(
            sensor_data=self.sensor_data,
            equipment_id="TEST-002",
        )

        result = self.service.predict(request)

        self.assertTrue(result["success"])
        self.assertEqual(result["equipment_id"], "TEST-002")

    @patch("app.ml.integration.ml_service.predict_equipment_condition")
    def test_anomaly_detection_mapping(self, mock_engine):

        engine_result = self.mock_engine_result.copy()

        engine_result["anomaly_results"] = {
            "Isolation Forest": {
                "is_anomaly": True,
                "anomaly_score": 0.8,
            },
            "Local Outlier Factor": {
                "is_anomaly": False,
                "anomaly_score": 0.2,
            },
        }

        mock_engine.return_value = engine_result

        result = self.service.predict({
            "sensor_data": self.sensor_data,
        })

        self.assertTrue(result["anomaly_detected"])

    def test_empty_sensor_data_rejected(self):

        with self.assertRaises(ValueError):
            self.service.predict({
                "sensor_data": {},
            })

    def test_invalid_request_rejected(self):

        with self.assertRaises(TypeError):
            self.service.predict("invalid request")


if __name__ == "__main__":
    unittest.main(verbosity=2)
