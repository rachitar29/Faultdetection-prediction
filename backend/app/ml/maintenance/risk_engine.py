
import numpy as np
import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle


def _risk_category(probability):
    if probability >= 0.80:
        return "CRITICAL"

    if probability >= 0.60:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"


def _get_failure_probability(model, X):
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Risk scoring requires a classification model with predict_proba."
        )

    probabilities = model.predict_proba(X)

    classes = list(model.estimator.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain failure class 1.")

    return probabilities[:, classes.index(1)]


def calculate_machine_risk(model_name, sensor_data):
    """
    Calculate failure probability and risk category.

    sensor_data can be a dictionary or a pandas DataFrame.
    """

    bundle = load_model_bundle(model_name)

    if not bundle:
        raise ValueError(f"Model not found: {model_name}")

    model = bundle["model"]

    if isinstance(sensor_data, dict):
        X = pd.DataFrame([sensor_data])
    elif isinstance(sensor_data, pd.DataFrame):
        X = sensor_data.copy()
    else:
        raise TypeError(
            "sensor_data must be a dictionary or pandas DataFrame."
        )

    if X.empty:
        raise ValueError("Sensor data is empty.")

    probabilities = _get_failure_probability(model, X)

    results = []

    for index, probability in enumerate(probabilities):
        probability = float(np.clip(probability, 0, 1))

        results.append({
            "row": int(index),
            "failure_probability": round(probability, 4),
            "failure_risk_percent": round(probability * 100, 2),
            "risk_category": _risk_category(probability)
        })

    return {
        "model_name": model_name,
        "number_of_machines": len(results),
        "predictions": results
    }


def calculate_machine_health_score(model_name, sensor_data):
    """Convert failure probability into a 0-100 health score."""

    result = calculate_machine_risk(
        model_name,
        sensor_data
    )

    for prediction in result["predictions"]:
        prediction["health_score"] = round(
            100 * (1 - prediction["failure_probability"]),
            2
        )

    return result
