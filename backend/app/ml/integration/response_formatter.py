"""
Standardizes the existing integrated ML engine output.

Validates critical prediction fields and converts model results
into a consistent JSON-safe response.
"""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import Any
import math

from app.ml.integration.schemas import MLPredictionResponse


VALID_RISK_LEVELS = {
    "Low",
    "Medium",
    "High",
    "Critical",
}


def make_json_safe(value: Any) -> Any:
    """Convert Python and ML values into JSON-safe values."""

    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if hasattr(value, "tolist"):
        try:
            return make_json_safe(value.tolist())
        except (ValueError, TypeError):
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


def _validate_prediction(prediction: dict) -> None:
    """Validate the minimum contract required from the ML engine."""

    if not isinstance(prediction, dict):
        raise TypeError(
            "Prediction engine output must be a dictionary."
        )

    required_fields = [
        "failure_predicted",
        "failure_probability",
        "risk_level",
        "anomaly_results",
    ]

    missing = [
        field
        for field in required_fields
        if field not in prediction
    ]

    if missing:
        raise ValueError(
            "Prediction engine output is missing required fields: "
            + ", ".join(missing)
        )

    failure_predicted = prediction["failure_predicted"]

    if hasattr(failure_predicted, "item"):
        failure_predicted = failure_predicted.item()

    if not isinstance(failure_predicted, bool):
        raise ValueError(
            "failure_predicted must be a boolean."
        )

    probability = prediction["failure_probability"]

    if hasattr(probability, "item"):
        probability = probability.item()

    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
    ):
        raise ValueError(
            "failure_probability must be a numeric value."
        )

    probability = float(probability)

    if not math.isfinite(probability):
        raise ValueError(
            "failure_probability must be finite."
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "failure_probability must be between 0 and 1."
        )

    risk_level = prediction["risk_level"]

    if risk_level not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Invalid risk_level: {risk_level!r}."
        )

    anomaly_results = prediction["anomaly_results"]

    if not isinstance(anomaly_results, dict):
        raise ValueError(
            "anomaly_results must be a dictionary."
        )

    rul_available = prediction.get("rul_available", False)

    if not isinstance(rul_available, bool):
        if hasattr(rul_available, "item"):
            rul_available = rul_available.item()

    if not isinstance(rul_available, bool):
        raise ValueError(
            "rul_available must be a boolean."
        )

    escalation = prediction.get(
        "rul_escalation_applied",
        False,
    )

    if hasattr(escalation, "item"):
        escalation = escalation.item()

    if not isinstance(escalation, bool):
        raise ValueError(
            "rul_escalation_applied must be a boolean."
        )


def format_prediction_response(
    prediction: dict,
    equipment_id: str | None = None,
) -> dict:
    """
    Convert integrated engine output into a standardized response.
    """

    _validate_prediction(prediction)

    anomaly_results = prediction["anomaly_results"]
    anomaly_assessment = prediction.get(
        "anomaly_assessment",
        {},
    )

    if not isinstance(anomaly_assessment, dict):
        anomaly_assessment = {}

    anomaly_detected = any(
        result.get("is_anomaly", False)
        for result in anomaly_results.values()
        if isinstance(result, dict)
    )

    recommendations = []

    recommendation = prediction.get(
        "maintenance_recommendation"
    )

    if recommendation:
        recommendations.append(str(recommendation))

    explanations = {
        "top_factors": prediction.get("top_factors", []),
        "explanation_method": prediction.get(
            "explanation_method"
        ),
        "explained_class": prediction.get("explained_class"),
    }

    model_metadata = {
        "model_name": prediction.get("model_name"),
        "model_type": prediction.get("model_type"),
        "decision_threshold": prediction.get(
            "decision_threshold"
        ),
        "integrated_prediction": prediction.get(
            "integrated_prediction",
            False,
        ),
        "prediction_components": prediction.get(
            "prediction_components",
            {},
        ),
        "classifier_risk_level": prediction.get(
            "classifier_risk_level"
        ),
        "anomaly_assessment": anomaly_assessment,
        "risk_assessment_method": prediction.get(
            "risk_assessment_method"
        ),
        "anomaly_results": anomaly_results,
        "health_score": prediction.get("health_score"),
        "health_score_note": prediction.get(
            "health_score_note"
        ),
        "sensor_readings": prediction.get(
            "sensor_readings",
            {},
        ),
        "maintenance_priority": prediction.get(
            "maintenance_priority"
        ),
    }

    failure_prediction = (
        "FAILURE"
        if prediction["failure_predicted"]
        else "NORMAL"
    )

    risk_level = prediction["risk_level"]

    response = MLPredictionResponse(
        success=True,
        equipment_id=equipment_id,

        failure_prediction=failure_prediction,
        failure_probability=float(
            prediction["failure_probability"]
        ),

        risk_level=risk_level,

        anomaly_detected=anomaly_detected,
        anomaly_details=anomaly_results,

        rul_prediction=prediction.get("rul_prediction"),
        rul_available=prediction.get(
            "rul_available",
            False,
        ),

        risk_level_before_rul=prediction.get(
            "risk_level_before_rul",
            risk_level,
        ),
        rul_risk_level=prediction.get("rul_risk_level"),
        rul_escalation_applied=prediction.get(
            "rul_escalation_applied",
            False,
        ),

        recommendations=recommendations,
        explanations=explanations,
        model_metadata=model_metadata,

        timestamp=prediction.get(
            "prediction_timestamp"
        ) or datetime.now(timezone.utc).isoformat(),

        errors=[],
    )

    return make_json_safe(response)