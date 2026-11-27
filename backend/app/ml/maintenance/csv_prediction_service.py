
"""
Batch CSV prediction service for the AI4I dataset.

Uses existing registered models only.
Does not train, register, overwrite, or delete models.
"""

import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


PRIMARY_MODEL = "XGBoost"

ANOMALY_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
]

REQUIRED_FEATURES = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)

MAX_ROWS = 50000
BATCH_SIZE = 2000


def validate_ai4i_dataframe(df):
    """Validate and normalize an AI4I-compatible CSV."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Uploaded data must be a CSV dataframe.")

    if df.empty:
        raise ValueError("The uploaded CSV contains no records.")

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"CSV contains {len(df)} rows. "
            f"The maximum supported size is {MAX_ROWS} rows."
        )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate CSV columns found: {duplicates}")

    missing = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "This CSV is not compatible with the AI4I model. "
            f"Missing required sensor columns: {missing}"
        )

    validated = df.copy()

    for feature in NUMERIC_FEATURES:
        validated[feature] = pd.to_numeric(
            validated[feature],
            errors="coerce",
        )

        values = validated[feature].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Column '{feature}' contains missing, "
                "non-numeric, or infinite values."
            )

    validated["Type"] = (
        validated["Type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if validated["Type"].isna().any():
        raise ValueError("Column 'Type' contains missing values.")

    invalid_types = sorted(
        set(validated["Type"].unique()) - {"L", "M", "H"}
    )

    if invalid_types:
        raise ValueError(
            "Column 'Type' contains unsupported values: "
            f"{invalid_types}. Allowed values are L, M, H."
        )

    if TARGET in validated.columns:
        labels = pd.to_numeric(
            validated[TARGET],
            errors="coerce",
        )

        if labels.isna().any():
            raise ValueError(
                f"Column '{TARGET}' contains missing or invalid labels."
            )

        if not labels.isin([0, 1]).all():
            raise ValueError(
                f"Column '{TARGET}' must contain only 0 or 1."
            )

        validated[TARGET] = labels.astype(int)

    return validated


def _load_classifier():
    """Load the existing registered XGBoost model."""

    bundle = load_model_bundle(PRIMARY_MODEL)

    model = bundle["model"]
    metadata = bundle["metadata"]

    expected_features = metadata.get(
        "feature_names",
        REQUIRED_FEATURES,
    )

    if set(expected_features) != set(REQUIRED_FEATURES):
        raise ValueError(
            "The registered XGBoost model feature schema "
            "does not match the AI4I dataset."
        )

    threshold = metadata.get("threshold")

    if threshold is None:
        raise ValueError(
            "The registered XGBoost model has no decision threshold."
        )

    return (
        model,
        metadata,
        list(expected_features),
        float(threshold),
    )


def _load_anomaly_models():
    """Load existing anomaly models and their thresholds."""

    loaded = []

    for model_name in ANOMALY_MODELS:
        bundle = load_model_bundle(model_name)

        model = bundle["model"]
        metadata = bundle["metadata"]

        if not hasattr(model, "decision_function"):
            raise ValueError(
                f"{model_name} does not support decision_function."
            )

        threshold = metadata.get("threshold")

        if threshold is None:
            raise ValueError(
                f"{model_name} has no registered threshold."
            )

        feature_names = metadata.get(
            "feature_names",
            REQUIRED_FEATURES,
        )

        if set(feature_names) != set(REQUIRED_FEATURES):
            raise ValueError(
                f"{model_name} uses a different feature schema "
                "and cannot process the AI4I dataset."
            )

        loaded.append({
            "name": model_name,
            "model": model,
            "threshold": float(threshold),
            "feature_names": list(feature_names),
        })

    return loaded


def _classify_risk(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.75:
        return "High"
    return "Critical"


def _escalate_risk(classifier_risk, anomaly_count):
    levels = ["Low", "Medium", "High", "Critical"]

    current_index = levels.index(classifier_risk)

    if anomaly_count >= 2:
        current_index = min(current_index + 1, len(levels) - 1)

    return levels[current_index]


def _maintenance_action(risk_level):
    actions = {
        "Low": (
            "Routine",
            "Continue routine monitoring and follow the "
            "manufacturer's maintenance schedule.",
        ),
        "Medium": (
            "Moderate",
            "Increase monitoring frequency and review "
            "recent sensor readings.",
        ),
        "High": (
            "High",
            "Arrange an equipment inspection and investigate "
            "the detected abnormal operating conditions.",
        ),
        "Critical": (
            "Urgent",
            "Arrange urgent equipment inspection. Follow "
            "approved site safety procedures.",
        ),
    }

    return actions[risk_level]


def _calculate_metrics(result_df):
    """Calculate metrics only when valid ground-truth labels exist."""

    if TARGET not in result_df.columns:
        return {
            "available": False,
            "reason": (
                "The uploaded CSV has no Machine failure column. "
                "Predictions are available, but evaluation metrics "
                "cannot be calculated."
            ),
        }

    y_true = result_df[TARGET].astype(int).to_numpy()
    y_pred = result_df["failure_predicted"].astype(int).to_numpy()
    y_score = result_df["failure_probability"].astype(float).to_numpy()

    metrics = {
        "available": True,
        "evaluation_rows": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, zero_division=0)), 6
        ),
        "f1_score": round(
            float(f1_score(y_true, y_pred, zero_division=0)), 6
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).tolist(),
        "confusion_matrix_labels": ["actual_0", "actual_1"],
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = round(
            float(roc_auc_score(y_true, y_score)), 6
        )
    else:
        metrics["roc_auc"] = None
        metrics["roc_auc_note"] = (
            "ROC-AUC requires both failure classes in the uploaded data."
        )

    metrics["evaluation_note"] = (
        "These metrics measure predictions on the uploaded CSV. "
        "If this CSV was used to train the model, these results "
        "are not an independent estimate of generalization."
    )

    return metrics


def predict_ai4i_csv(df):
    """
    Run batch predictions using existing registered models.

    Returns a result dataframe and a summary dictionary.
    """

    validated = validate_ai4i_dataframe(df)

    classifier, classifier_metadata, classifier_features, threshold = (
        _load_classifier()
    )

    anomaly_models = _load_anomaly_models()

    feature_frame = validated.loc[:, classifier_features].copy()

    probabilities = []

    for start in range(0, len(feature_frame), BATCH_SIZE):
        batch = feature_frame.iloc[start:start + BATCH_SIZE]

        batch_probabilities = classifier.predict_proba(batch)
        classes = list(classifier.classes_)

        if 1 not in classes:
            raise ValueError(
                "Registered XGBoost model does not contain failure class 1."
            )

        failure_index = classes.index(1)

        probabilities.extend(
            batch_probabilities[:, failure_index].astype(float).tolist()
        )

    probabilities = np.asarray(probabilities, dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The XGBoost model returned invalid probabilities."
        )

    anomaly_flags = {}
    anomaly_scores = {}

    for item in anomaly_models:
        model_name = item["name"]
        model = item["model"]
        model_features = item["feature_names"]
        threshold_value = item["threshold"]

        scores = []

        for start in range(0, len(validated), BATCH_SIZE):
            batch = validated.iloc[start:start + BATCH_SIZE]
            batch_features = batch.loc[:, model_features]

            batch_scores = -model.decision_function(batch_features)

            scores.extend(
                np.asarray(batch_scores, dtype=float).tolist()
            )

        scores = np.asarray(scores, dtype=float)

        if not np.isfinite(scores).all():
            raise ValueError(
                f"{model_name} returned invalid anomaly scores."
            )

        anomaly_scores[model_name] = scores
        anomaly_flags[model_name] = scores >= threshold_value

    result_df = df.copy().reset_index(drop=True)

    result_df.insert(
        0,
        "row_number",
        np.arange(1, len(result_df) + 1),
    )

    result_df["failure_probability"] = probabilities
    result_df["failure_probability_percent"] = probabilities * 100
    result_df["failure_predicted"] = (
        probabilities >= threshold
    ).astype(bool)

    result_df["decision_threshold"] = threshold

    for model_name in ANOMALY_MODELS:
        safe_column = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        result_df[f"{safe_column}_anomaly"] = anomaly_flags[
            model_name
        ]

        result_df[f"{safe_column}_score"] = anomaly_scores[
            model_name
        ]

    anomaly_count = np.zeros(len(result_df), dtype=int)

    for flags in anomaly_flags.values():
        anomaly_count += flags.astype(int)

    result_df["anomaly_detectors_flagged"] = anomaly_count

    result_df["anomaly_consensus_level"] = np.select(
        [
            anomaly_count == 0,
            anomaly_count == 1,
            anomaly_count == 2,
            anomaly_count == 3,
        ],
        [
            "None",
            "Single",
            "Majority",
            "Unanimous",
        ],
        default="Unknown",
    )

    classifier_risks = [
        _classify_risk(float(probability))
        for probability in probabilities
    ]

    overall_risks = [
        _escalate_risk(risk, int(count))
        for risk, count in zip(classifier_risks, anomaly_count)
    ]

    result_df["classifier_risk_level"] = classifier_risks
    result_df["risk_level"] = overall_risks
    result_df["health_score"] = (
        (1 - probabilities) * 100
    ).round(2)

    priorities = []
    recommendations = []

    for risk in overall_risks:
        priority, recommendation = _maintenance_action(risk)
        priorities.append(priority)
        recommendations.append(recommendation)

    result_df["maintenance_priority"] = priorities
    result_df["maintenance_recommendation"] = recommendations

    result_df["failure_probability"] = (
        result_df["failure_probability"].round(6)
    )
    result_df["failure_probability_percent"] = (
        result_df["failure_probability_percent"].round(4)
    )

    for model_name in ANOMALY_MODELS:
        safe_column = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        result_df[f"{safe_column}_score"] = (
            result_df[f"{safe_column}_score"].round(6)
        )

    metrics = _calculate_metrics(result_df)

    summary = {
        "status": "success",
        "dataset": "AI4I-compatible CSV",
        "rows_processed": int(len(result_df)),
        "model_name": classifier_metadata.get(
            "model_name",
            PRIMARY_MODEL,
        ),
        "model_type": classifier_metadata.get("model_type"),
        "decision_threshold": threshold,
        "failure_predictions": int(
            result_df["failure_predicted"].sum()
        ),
        "normal_predictions": int(
            (result_df["failure_predicted"] == False).sum()
        ),
        "anomaly_counts": {
            model_name: int(flags.sum())
            for model_name, flags in anomaly_flags.items()
        },
        "risk_distribution": {
            str(level): int(count)
            for level, count in result_df["risk_level"].value_counts().items()
        },
        "evaluation_metrics": metrics,
        "health_score_note": (
            "Inverse model failure probability; "
            "not a calibrated physical health measurement."
        ),
        "risk_assessment_note": (
            "Risk escalation uses the project's configurable "
            "three-detector majority-consensus heuristic, "
            "not an industrial safety standard."
        ),
    }

    return result_df, summary
