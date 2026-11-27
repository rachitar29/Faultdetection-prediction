
from datetime import datetime, timezone
import logging

import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from app.ml.data.sensor_validation import validate_sensor_record
from app.ml.explainability.prediction_explainer import (
    explain_machine_prediction,
)


logger = logging.getLogger(__name__)

PRIMARY_MODEL = "XGBoost"

RISK_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
}


def load_primary_model():
    bundle = load_model_bundle(PRIMARY_MODEL)
    return bundle["model"], bundle["metadata"]


def validate_sensor_data(sensor_data, return_warnings=False):
    """
    Validate one live sensor record.

    The default return value remains a dictionary for compatibility.
    Set return_warnings=True to receive (data, warnings).
    """
    validated, warnings = validate_sensor_record(
        sensor_data,
        allow_missing_numeric=False,
    )

    if return_warnings:
        return validated, warnings

    return validated


def classify_risk(probability):
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"

    if probability < RISK_THRESHOLDS["medium"]:
        return "Medium"

    if probability < RISK_THRESHOLDS["high"]:
        return "High"

    return "Critical"


def generate_recommendation(risk_level, failure_predicted):
    if risk_level == "Critical":
        return {
            "priority": "Urgent",
            "action": (
                "Arrange an immediate equipment inspection. "
                "Follow the site's approved safety procedures "
                "before continuing operation."
            ),
        }

    if risk_level == "High":
        return {
            "priority": "High",
            "action": (
                "Schedule an equipment inspection at the "
                "earliest practical opportunity. Review "
                "operating conditions and recent sensor trends."
            ),
        }

    if risk_level == "Medium":
        return {
            "priority": "Moderate",
            "action": (
                "Increase monitoring frequency and review "
                "recent maintenance and sensor history."
            ),
        }

    if failure_predicted:
        return {
            "priority": "Review",
            "action": (
                "The model's selected decision threshold "
                "indicates a possible failure. Review the "
                "prediction and inspect the equipment."
            ),
        }

    return {
        "priority": "Routine",
        "action": (
            "Continue routine monitoring and follow the "
            "manufacturer's preventive maintenance schedule."
        ),
    }


def generate_safe_explanation(sensor_data):
    try:
        explanation = explain_machine_prediction(
            sensor_data=sensor_data,
            model_name=PRIMARY_MODEL,
            top_n=3,
        )

        return {
            "top_factors": explanation["top_factors"],
            "explanation_method": explanation["explanation_method"],
            "explained_class": explanation["explained_class"],
            "explanation_status": "available",
        }

    except Exception:
        logger.exception(
            "SHAP explanation failed for model %s. "
            "Returning prediction without feature explanations.",
            PRIMARY_MODEL,
        )

        return {
            "top_factors": [],
            "explanation_method": "Unavailable",
            "explained_class": "Machine failure",
            "explanation_status": "unavailable",
        }


def predict_equipment_health(sensor_data):
    validated_data, validation_warnings = validate_sensor_data(
        sensor_data,
        return_warnings=True,
    )

    model, metadata = load_primary_model()

    expected_features = metadata.get("feature_names", [])

    if expected_features:
        if set(expected_features) != set(validated_data.keys()):
            raise ValueError(
                "Input features do not match registered model features."
            )
    else:
        expected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    input_df = pd.DataFrame(
        [validated_data],
        columns=expected_features,
    )

    probabilities = model.predict_proba(input_df)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Registered model does not contain failure class 1."
        )

    failure_index = classes.index(1)
    failure_probability = float(probabilities[0][failure_index])

    threshold = float(metadata["threshold"])
    failure_predicted = failure_probability >= threshold

    risk_level = classify_risk(failure_probability)

    health_score = round(
        100 * (1 - failure_probability),
        2,
    )

    recommendation = generate_recommendation(
        risk_level,
        failure_predicted,
    )

    explanation = generate_safe_explanation(validated_data)

    return {
        "model_name": metadata["model_name"],
        "model_type": metadata["model_type"],
        "failure_probability": round(failure_probability, 4),
        "failure_probability_percent": round(
            failure_probability * 100,
            2,
        ),
        "failure_predicted": bool(failure_predicted),
        "decision_threshold": round(threshold, 6),
        "risk_level": risk_level,
        "health_score": health_score,
        "health_score_note": (
            "Inverse model failure probability; "
            "not a calibrated physical health measurement."
        ),
        "maintenance_priority": recommendation["priority"],
        "maintenance_recommendation": recommendation["action"],
        "top_factors": explanation["top_factors"],
        "explanation_method": explanation["explanation_method"],
        "explained_class": explanation["explained_class"],
        "explanation_status": explanation["explanation_status"],
        "sensor_readings": validated_data,
        "sensor_validation_warnings": validation_warnings,
        "prediction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }
