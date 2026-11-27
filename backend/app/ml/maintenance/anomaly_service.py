
"""
Anomaly Detection Service.

Loads registered Isolation Forest, Local Outlier Factor,
and One-Class SVM pipelines.

Uses saved validation thresholds. Does not retrain models.
Anomaly scores are not probabilities.
"""

import math

import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from app.ml.data.sensor_validation import validate_sensor_record


ANOMALY_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
]

DEFAULT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_anomaly_model(model_name):
    """Load a registered anomaly model and its threshold."""
    if model_name not in ANOMALY_MODELS:
        raise ValueError(
            f"Unsupported anomaly model: {model_name}"
        )

    bundle = load_model_bundle(model_name)

    if not isinstance(bundle, dict):
        raise ValueError(
            f"Invalid model bundle for {model_name}."
        )

    model = bundle.get("model")
    metadata = bundle.get("metadata")

    if model is None:
        raise ValueError(
            f"No model found for {model_name}."
        )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid metadata for {model_name}."
        )

    if not hasattr(model, "decision_function"):
        raise ValueError(
            f"{model_name} does not support decision_function."
        )

    threshold = metadata.get("threshold")

    if threshold is None:
        raise ValueError(
            f"No validation threshold registered for {model_name}."
        )

    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid threshold registered for {model_name}."
        ) from exc

    if not math.isfinite(threshold):
        raise ValueError(
            f"Threshold for {model_name} must be finite."
        )

    return model, metadata, threshold


def validate_feature_names(feature_names):
    """Validate the feature schema stored in model metadata."""
    if not isinstance(feature_names, (list, tuple)):
        raise ValueError(
            "Model feature_names must be a list or tuple."
        )

    if not feature_names:
        raise ValueError(
            "Model feature_names cannot be empty."
        )

    if not all(
        isinstance(feature, str) and feature.strip()
        for feature in feature_names
    ):
        raise ValueError(
            "Model feature_names contains invalid entries."
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError(
            "Model feature_names contains duplicate features."
        )

    if set(feature_names) != set(DEFAULT_FEATURES):
        raise ValueError(
            "Registered model features do not match the "
            f"expected sensor schema. Expected: {DEFAULT_FEATURES}. "
            f"Received: {list(feature_names)}"
        )

    return list(feature_names)


def prepare_sensor_input(sensor_data, feature_names):
    """
    Validate readings and arrange columns in the registered model order.
    """
    feature_names = validate_feature_names(feature_names)

    validated, _ = validate_sensor_record(
        sensor_data,
        allow_missing_numeric=False,
    )

    return pd.DataFrame(
        [[validated[feature] for feature in feature_names]],
        columns=feature_names,
    )


def predict_with_anomaly_model(model_name, sensor_data):
    """Generate an anomaly prediction using one registered model."""
    model, metadata, threshold = load_anomaly_model(model_name)

    feature_names = metadata.get(
        "feature_names",
        DEFAULT_FEATURES,
    )

    feature_names = validate_feature_names(feature_names)

    validated, validation_warnings = validate_sensor_record(
        sensor_data,
        allow_missing_numeric=False,
    )

    input_df = pd.DataFrame(
        [[validated[feature] for feature in feature_names]],
        columns=feature_names,
    )

    raw_scores = model.decision_function(input_df)
    scores = pd.Series(raw_scores).to_numpy().reshape(-1)

    if len(scores) != 1:
        raise ValueError(
            f"{model_name} returned an unexpected number "
            "of anomaly scores."
        )

    anomaly_score = float(scores[0])

    if not math.isfinite(anomaly_score):
        raise ValueError(
            f"{model_name} returned a non-finite anomaly score."
        )

    anomaly_score = -anomaly_score

    if not math.isfinite(anomaly_score):
        raise ValueError(
            f"{model_name} produced an invalid anomaly score."
        )

    is_anomaly = bool(anomaly_score >= threshold)

    return {
        "model_name": model_name,
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 6),
        "optimized_threshold": round(threshold, 6),
        "threshold_source": "Saved validation threshold",
        "sensor_validation_warnings": validation_warnings,
    }


def detect_machine_anomalies(sensor_data):
    """Run all three registered anomaly detectors."""
    # Validate once before loading/running any detector.
    validate_sensor_record(
        sensor_data,
        allow_missing_numeric=False,
    )

    results = {}

    for model_name in ANOMALY_MODELS:
        results[model_name] = predict_with_anomaly_model(
            model_name=model_name,
            sensor_data=sensor_data,
        )

    return results


def get_anomaly_flags(anomaly_results):
    """Extract validated Python boolean anomaly flags."""
    if not isinstance(anomaly_results, dict):
        raise TypeError(
            "Anomaly results must be a dictionary."
        )

    missing = [
        model_name
        for model_name in ANOMALY_MODELS
        if model_name not in anomaly_results
    ]

    if missing:
        raise ValueError(
            f"Missing anomaly detector results: {missing}"
        )

    flags = {}

    for model_name in ANOMALY_MODELS:
        result = anomaly_results[model_name]

        if not isinstance(result, dict):
            raise ValueError(
                f"Invalid result returned by {model_name}."
            )

        if "is_anomaly" not in result:
            raise ValueError(
                f"{model_name} result is missing is_anomaly."
            )

        value = result["is_anomaly"]

        if not isinstance(value, bool):
            raise ValueError(
                f"{model_name} is_anomaly must be a Python boolean."
            )

        flags[model_name] = value

    return flags
