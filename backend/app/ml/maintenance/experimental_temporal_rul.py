"""
Experimental NASA C-MAPSS FD001 temporal RUL inference.

This module is deliberately separate from the registered production RUL
service. It loads a frozen experimental artifact and does not retrain,
register, overwrite, or modify any production model.

Input:
    A chronological DataFrame containing one engine's history:
    unit, cycle, 3 settings, and 21 sensors.

Output:
    RUL prediction for the latest available cycle.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.maintenance.rul_prediction import (
    classify_rul,
    recommendation_for_rul,
)


# ============================================================
# PATHS AND FROZEN ARTIFACT IDENTITIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

ARTIFACT_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "experimental"
    / "rul_cmapss_fd001_temporal"
)

MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

EXPECTED_MODEL_SHA256 = (
    "D6930759BDED3E386E59F0A875DF41741501011708633FEF0FC99819C9D4377E"
)

EXPECTED_METADATA_SHA256 = (
    "D05DDEB4073919907F8FD688D169298B38E3818394E2ADC9284F8BD5A42EE649"
)

EXPECTED_MODEL_NAME = "rul_cmapss_fd001_temporal_experiment"

EXPECTED_FEATURE_COUNT = 88

WINDOWS = [10, 30]

TEMPORAL_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_12",
    "sensor_14",
    "sensor_17",
]

ORIGINAL_FEATURES = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

TEMPORAL_FEATURES = []

for sensor in TEMPORAL_SENSORS:
    for window in WINDOWS:
        TEMPORAL_FEATURES.extend([
            f"{sensor}_rolling_mean_{window}",
            f"{sensor}_rolling_std_{window}",
            f"{sensor}_slope_{window}",
            f"{sensor}_deviation_{window}",
        ])

EXPECTED_FEATURES = (
    ORIGINAL_FEATURES + TEMPORAL_FEATURES
)


# ============================================================
# FILE INTEGRITY
# ============================================================

def calculate_sha256(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def verify_frozen_artifacts():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Experimental model not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Experimental metadata not found: {METADATA_PATH}"
        )

    actual_model_hash = calculate_sha256(MODEL_PATH)
    actual_metadata_hash = calculate_sha256(METADATA_PATH)

    if actual_model_hash != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "Experimental model SHA-256 verification failed. "
            "The frozen model artifact may have changed."
        )

    if actual_metadata_hash != EXPECTED_METADATA_SHA256:
        raise RuntimeError(
            "Experimental metadata SHA-256 verification failed. "
            "The frozen metadata may have changed."
        )

    return {
        "model_sha256": actual_model_hash,
        "metadata_sha256": actual_metadata_hash,
    }


# ============================================================
# LOAD EXPERIMENTAL MODEL
# ============================================================

def load_experimental_temporal_model():

    hashes = verify_frozen_artifacts()

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError(
            "Unexpected experimental model name."
        )

    if metadata.get("registered") is not False:
        raise ValueError(
            "Expected an unregistered experimental artifact."
        )

    if metadata.get("total_feature_count") != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Experimental metadata feature count is not 88."
        )

    feature_names = metadata.get("feature_names")

    if feature_names != EXPECTED_FEATURES:
        raise ValueError(
            "Metadata feature names or feature order differ "
            "from the expected FD001 temporal schema."
        )

    medians = metadata.get("imputation_medians")

    if not isinstance(medians, dict):
        raise ValueError(
            "Experimental metadata has no valid imputation medians."
        )

    if set(medians.keys()) != set(EXPECTED_FEATURES):
        raise ValueError(
            "Saved imputation median keys do not match "
            "the 88 expected model features."
        )

    median_values = np.asarray(
        [medians[name] for name in EXPECTED_FEATURES],
        dtype=float,
    )

    if not np.isfinite(median_values).all():
        raise ValueError(
            "Saved imputation medians contain non-finite values."
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Experimental artifact does not support predict()."
        )

    model_feature_count = getattr(
        model,
        "n_features_in_",
        None,
    )

    if (
        model_feature_count is not None
        and model_feature_count != EXPECTED_FEATURE_COUNT
    ):
        raise ValueError(
            f"Model expects {model_feature_count} features; "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    model_feature_names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_feature_names is not None:
        if list(model_feature_names) != EXPECTED_FEATURES:
            raise ValueError(
                "Model's internal feature names do not match "
                "the frozen feature schema."
            )

    return model, metadata, hashes


# ============================================================
# TEMPORAL SLOPE
# Must match the training implementation exactly.
# ============================================================

def calculate_rolling_slope(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    count = len(values)

    if count < 2:
        return 0.0

    x = np.arange(
        count,
        dtype=float,
    )

    x_centered = x - x.mean()
    y_centered = values - values.mean()

    denominator = np.dot(
        x_centered,
        x_centered,
    )

    if denominator == 0:
        return 0.0

    slope_per_cycle = (
        np.dot(x_centered, y_centered)
        / denominator
    )

    return float(slope_per_cycle * 10.0)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_engine_history(history):

    if not isinstance(history, pd.DataFrame):
        raise TypeError(
            "Engine history must be a pandas DataFrame. "
            "Temporal inference requires chronological history."
        )

    if history.empty:
        raise ValueError(
            "Engine history is empty."
        )

    required_columns = (
        ["unit", "cycle"] + ORIGINAL_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            f"Missing engine history columns: {missing}"
        )

    frame = history[required_columns].copy()

    if frame["unit"].isna().any():
        raise ValueError(
            "Engine ID contains missing values."
        )

    if frame["unit"].nunique() != 1:
        raise ValueError(
            "Pass history for exactly one engine per prediction."
        )

    for column in ["cycle"] + ORIGINAL_FEATURES:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    numeric_values = frame[
        ["cycle"] + ORIGINAL_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Engine history contains missing, non-numeric, "
            "or non-finite sensor values."
        )

    if (frame["cycle"] <= 0).any():
        raise ValueError(
            "Cycle values must be positive."
        )

    frame = frame.sort_values(
        "cycle",
        kind="stable",
    ).reset_index(drop=True)

    if frame["cycle"].duplicated().any():
        raise ValueError(
            "Engine history contains duplicate cycle numbers."
        )

    return frame


# ============================================================
# TEMPORAL FEATURE GENERATION
# Matches cmapss_temporal_experiment.py.
# ============================================================

def add_temporal_features(history):

    frame = history.copy()

    for sensor in TEMPORAL_SENSORS:

        values = pd.to_numeric(
            frame[sensor],
            errors="coerce",
        )

        for window in WINDOWS:

            rolling = values.rolling(
                window=window,
                min_periods=1,
            )

            rolling_mean = rolling.mean()

            rolling_std = rolling.std(
                ddof=0,
            ).fillna(0.0)

            rolling_slope = values.rolling(
                window=window,
                min_periods=2,
            ).apply(
                calculate_rolling_slope,
                raw=True,
            ).fillna(0.0)

            frame[
                f"{sensor}_rolling_mean_{window}"
            ] = rolling_mean

            frame[
                f"{sensor}_rolling_std_{window}"
            ] = rolling_std

            frame[
                f"{sensor}_slope_{window}"
            ] = rolling_slope

            frame[
                f"{sensor}_deviation_{window}"
            ] = values - rolling_mean

    missing_temporal = [
        name
        for name in TEMPORAL_FEATURES
        if name not in frame.columns
    ]

    if missing_temporal:
        raise RuntimeError(
            f"Temporal feature generation failed: {missing_temporal}"
        )

    return frame


# ============================================================
# PREPARE LATEST ENGINE SNAPSHOT
# Uses saved training medians only.
# ============================================================

def prepare_latest_snapshot(
    history,
    metadata,
):

    temporal_frame = add_temporal_features(
        history
    )

    latest = temporal_frame.tail(1).copy()

    feature_names = metadata["feature_names"]
    medians = metadata["imputation_medians"]

    features = latest[feature_names].copy()

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    saved_medians = pd.Series(
        {
            name: float(medians[name])
            for name in feature_names
        }
    )

    features = features.fillna(
        saved_medians
    ).fillna(0.0)

    values = features.to_numpy(
        dtype=float
    )

    if values.shape != (1, EXPECTED_FEATURE_COUNT):
        raise ValueError(
            f"Unexpected temporal feature matrix shape: {values.shape}"
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Prepared temporal features contain non-finite values."
        )

    return latest, features


# ============================================================
# EXPERIMENTAL TEMPORAL RUL PREDICTION
# ============================================================

def predict_experimental_temporal_rul(history):

    validated_history = validate_engine_history(
        history
    )

    model, metadata, hashes = (
        load_experimental_temporal_model()
    )

    latest, features = prepare_latest_snapshot(
        validated_history,
        metadata,
    )

    predictions = np.asarray(
        model.predict(features),
        dtype=float,
    ).reshape(-1)

    if len(predictions) != 1:
        raise ValueError(
            "Temporal model returned an unexpected number "
            "of predictions."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Temporal model returned a non-finite prediction."
        )

    rul = float(
        np.maximum(predictions[0], 0.0)
    )

    category = classify_rul(rul)

    recommendation = recommendation_for_rul(
        category
    )

    return {
        "model_name": metadata["model_name"],
        "model_type": "Experimental Temporal RUL Regression",
        "experimental": True,
        "production_approved": False,
        "engine_id": latest["unit"].iloc[0],
        "last_cycle": int(latest["cycle"].iloc[0]),
        "history_rows_used": int(len(validated_history)),
        "remaining_useful_life": round(rul, 2),
        "unit": "cycles",
        "rul_category": category,
        "maintenance_priority": recommendation["priority"],
        "maintenance_recommendation": recommendation["action"],
        "feature_count": EXPECTED_FEATURE_COUNT,
        "artifact_integrity_verified": True,
        "model_sha256": hashes["model_sha256"],
        "metadata_sha256": hashes["metadata_sha256"],
    }
