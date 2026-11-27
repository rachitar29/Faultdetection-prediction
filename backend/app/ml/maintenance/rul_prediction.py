
"""
NASA C-MAPSS Remaining Useful Life Prediction Service.

Loads the registered FD001 RUL regression model.
Does not retrain models.

Supports:
- Single-machine dictionary input
- Multiple-machine DataFrame input
- Model metadata validation
- RUL prediction in cycles
- RUL risk classification
- Maintenance recommendations
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_RUL_MODEL = "rul_cmapss_fd001_best"

RUL_THRESHOLDS = {
    "critical": 10,
    "high": 25,
    "medium": 50,
}


# =========================================================
# LOAD REGISTERED RUL MODEL
# =========================================================

def load_rul_model(model_name=DEFAULT_RUL_MODEL):

    bundle = load_model_bundle(model_name)

    if not bundle:
        raise FileNotFoundError(
            f"RUL model not registered: {model_name}"
        )

    model = bundle["model"]
    metadata = bundle["metadata"]

    if model is None:
        raise ValueError(
            f"No model object found for {model_name}"
        )

    if not hasattr(model, "predict"):
        raise ValueError(
            f"{model_name} does not support regression prediction."
        )

    feature_names = metadata.get("feature_names")

    if not feature_names:
        raise ValueError(
            "RUL model metadata has no feature_names."
        )

    # Accept different regression model_type labels.
    # The saved model name also identifies the RUL model.
    model_type = str(
        metadata.get("model_type", "")
    ).strip().lower()

    model_name_normalized = model_name.lower()

    is_rul_model = (
        "rul" in model_name_normalized
        or "cmapss" in model_name_normalized
        or "rul" in model_type
        or "regression" in model_type
        or "regressor" in model_type
    )

    if not is_rul_model:
        raise ValueError(
            f"{model_name} does not appear to be an RUL "
            f"regression model. Registered type: {model_type}"
        )

    return model, metadata


# =========================================================
# VALIDATE RUL INPUT
# =========================================================

def validate_rul_input(sensor_data, feature_names):

    if isinstance(sensor_data, dict):

        frame = pd.DataFrame([sensor_data])

    elif isinstance(sensor_data, pd.DataFrame):

        frame = sensor_data.copy()

    else:

        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    if frame.empty:
        raise ValueError(
            "Input data is empty."
        )

    missing = [
        column
        for column in feature_names
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing RUL features: {missing}"
        )

    # Preserve the feature order used during training.

    frame = frame[feature_names].copy()

    for column in feature_names:

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce"
        )

    values = frame.to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            "RUL input contains missing or non-finite values."
        )

    return frame


# =========================================================
# RUL CLASSIFICATION
# =========================================================

def classify_rul(value):

    if value <= RUL_THRESHOLDS["critical"]:
        return "CRITICAL"

    if value <= RUL_THRESHOLDS["high"]:
        return "HIGH"

    if value <= RUL_THRESHOLDS["medium"]:
        return "MEDIUM"

    return "LOW"


# =========================================================
# MAINTENANCE RECOMMENDATIONS
# =========================================================

def recommendation_for_rul(category):

    recommendations = {

        "CRITICAL": (
            "IMMEDIATE",
            "Arrange an immediate maintenance assessment "
            "and follow approved safety procedures."
        ),

        "HIGH": (
            "URGENT",
            "Plan maintenance promptly and inspect "
            "relevant equipment components."
        ),

        "MEDIUM": (
            "MODERATE",
            "Plan preventive maintenance and increase "
            "monitoring frequency."
        ),

        "LOW": (
            "ROUTINE",
            "Continue routine monitoring and follow "
            "the approved maintenance schedule."
        ),
    }

    if category not in recommendations:
        raise ValueError(
            f"Unknown RUL category: {category}"
        )

    priority, action = recommendations[category]

    return {
        "priority": priority,
        "action": action,
    }


# =========================================================
# PREDICT REMAINING USEFUL LIFE
# =========================================================

def predict_remaining_useful_life(
    sensor_data,
    model_name=DEFAULT_RUL_MODEL
):

    # STEP 1: Load registered model.

    model, metadata = load_rul_model(
        model_name
    )

    feature_names = metadata["feature_names"]

    # STEP 2: Validate input.

    frame = validate_rul_input(
        sensor_data,
        feature_names
    )

    # STEP 3: Generate predictions.

    values = np.asarray(
        model.predict(frame),
        dtype=float
    ).reshape(-1)

    if len(values) != len(frame):

        raise ValueError(
            "RUL model returned an unexpected number "
            "of predictions."
        )

    if not np.isfinite(values).all():

        raise ValueError(
            "RUL model returned missing or non-finite predictions."
        )

    # RUL cannot be negative.

    values = np.maximum(
        values,
        0.0
    )

    # STEP 4: Generate machine-level results.

    predictions = []

    for index, value in enumerate(values):

        rul = float(value)

        category = classify_rul(
            rul
        )

        recommendation = recommendation_for_rul(
            category
        )

        predictions.append({

            "row": index,

            "remaining_useful_life": round(
                rul,
                2
            ),

            "unit": "cycles",

            "rul_category": category,

            "maintenance_priority": (
                recommendation["priority"]
            ),

            "maintenance_recommendation": (
                recommendation["action"]
            ),

        })

    # STEP 5: Return complete response.

    return {

        "model_name": metadata.get(
            "model_name",
            model_name
        ),

        "model_type": "RUL Regression",

        "number_of_machines": len(predictions),

        "predictions": predictions,

        "prediction_timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

    }


# =========================================================
# SINGLE-MACHINE RUL PREDICTION
# =========================================================

def predict_single_machine_rul(
    sensor_data,
    model_name=DEFAULT_RUL_MODEL
):

    result = predict_remaining_useful_life(
        sensor_data,
        model_name
    )

    return result["predictions"][0]
