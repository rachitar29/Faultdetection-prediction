
from unittest.mock import patch

import pytest

from app.ml.maintenance.integrated_engine import (
    apply_rul_risk_escalation,
    predict_equipment_condition,
)


@pytest.mark.parametrize(
    "current_risk,rul_category,expected",
    [
        ("Low", "LOW", "Low"),
        ("Low", "MEDIUM", "Medium"),
        ("Low", "HIGH", "High"),
        ("Low", "CRITICAL", "Critical"),
        ("Medium", "LOW", "Medium"),
        ("High", "MEDIUM", "High"),
        ("Critical", "LOW", "Critical"),
    ],
)
def test_rul_risk_escalation(
    current_risk,
    rul_category,
    expected,
):
    result = apply_rul_risk_escalation(
        current_risk,
        {"rul_category": rul_category},
    )

    assert result["risk_level"] == expected


def test_invalid_rul_category():
    with pytest.raises(ValueError):
        apply_rul_risk_escalation(
            "Low",
            {"rul_category": "UNKNOWN"},
        )


def test_rul_escalation_does_not_reduce_risk():
    result = apply_rul_risk_escalation(
        "Critical",
        {"rul_category": "LOW"},
    )

    assert result["risk_level"] == "Critical"
    assert result["rul_escalation_applied"] is False


@patch(
    "app.ml.maintenance.integrated_engine.predict_single_machine_rul"
)
@patch(
    "app.ml.maintenance.integrated_engine.assess_equipment_risk"
)
@patch(
    "app.ml.maintenance.integrated_engine.get_anomaly_flags"
)
@patch(
    "app.ml.maintenance.integrated_engine.detect_machine_anomalies"
)
@patch(
    "app.ml.maintenance.integrated_engine.predict_equipment_health"
)
def test_critical_rul_escalates_integrated_risk(
    mock_health,
    mock_anomaly,
    mock_flags,
    mock_assessment,
    mock_rul,
):
    mock_health.return_value = {
        "failure_probability": 0.05,
        "model_name": "XGBoost",
    }

    mock_anomaly.return_value = {}
    mock_flags.return_value = {}

    mock_assessment.return_value = {
        "classifier_risk_level": "Low",
        "overall_risk_level": "Low",
        "maintenance_priority": "Routine",
        "maintenance_recommendation": "Routine monitoring.",
        "anomaly_assessment": {},
        "assessment_method": "Test risk assessment.",
    }

    mock_rul.return_value = {
        "remaining_useful_life": 8.0,
        "unit": "cycles",
        "rul_category": "CRITICAL",
        "maintenance_priority": "IMMEDIATE",
        "maintenance_recommendation": "Immediate assessment.",
    }

    result = predict_equipment_condition(
        sensor_data={"example": 1},
        rul_sensor_data={"example": 2},
    )

    assert result["classifier_risk_level"] == "Low"
    assert result["risk_level_before_rul"] == "Low"
    assert result["rul_risk_level"] == "Critical"
    assert result["risk_level"] == "Critical"
    assert result["rul_escalation_applied"] is True
    assert result["maintenance_priority"] == "Urgent"
    assert result["rul_available"] is True


@patch(
    "app.ml.maintenance.integrated_engine.assess_equipment_risk"
)
@patch(
    "app.ml.maintenance.integrated_engine.get_anomaly_flags"
)
@patch(
    "app.ml.maintenance.integrated_engine.detect_machine_anomalies"
)
@patch(
    "app.ml.maintenance.integrated_engine.predict_equipment_health"
)
def test_prediction_without_rul_preserves_existing_risk(
    mock_health,
    mock_anomaly,
    mock_flags,
    mock_assessment,
):
    mock_health.return_value = {
        "failure_probability": 0.05,
        "model_name": "XGBoost",
    }

    mock_anomaly.return_value = {}
    mock_flags.return_value = {}

    mock_assessment.return_value = {
        "classifier_risk_level": "Low",
        "overall_risk_level": "Low",
        "maintenance_priority": "Routine",
        "maintenance_recommendation": "Routine monitoring.",
        "anomaly_assessment": {},
        "assessment_method": "Test risk assessment.",
    }

    result = predict_equipment_condition(
        sensor_data={"example": 1}
    )

    assert result["risk_level"] == "Low"
    assert result["rul_available"] is False
    assert result["rul_prediction"] is None
    assert result["rul_escalation_applied"] is False
    assert result["prediction_components"]["rul_risk_escalation"] is False
