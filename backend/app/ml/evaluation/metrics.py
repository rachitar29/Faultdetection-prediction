
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def _validate_binary_inputs(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional.")

    if len(y_true) != len(scores):
        raise ValueError("Labels and scores must have equal length.")

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Binary evaluation requires both target classes.")

    return y_true, scores


def calculate_binary_metrics(
    y_true,
    scores,
    threshold=0.5,
    positive_label=1,
    beta=2.0
):
    """Evaluate binary predictions from continuous failure scores."""

    y_true, scores = _validate_binary_inputs(y_true, scores)

    if positive_label not in np.unique(y_true):
        raise ValueError("Positive label is not present.")

    if beta <= 0:
        raise ValueError("beta must be greater than zero.")

    y_binary = (y_true == positive_label).astype(int)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_binary, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_binary, predictions, zero_division=0
    )
    recall = recall_score(
        y_binary, predictions, zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_binary, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(y_binary, predictions, zero_division=0)),
        "f_beta": float(
            fbeta_score(
                y_binary, predictions, beta=beta, zero_division=0
            )
        ),
        "beta": float(beta),
        "specificity": float(specificity),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
        "false_negative_rate": float(
            fn / (fn + tp) if (fn + tp) else 0.0
        ),
        "roc_auc": float(roc_auc_score(y_binary, scores)),
        "average_precision": float(
            average_precision_score(y_binary, scores)
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true)),
        "positive_label": str(positive_label)
    }


def compare_thresholds(
    y_true,
    scores,
    thresholds,
    positive_label=1,
    beta=2.0
):
    """Evaluate a model at multiple decision thresholds."""

    return [
        calculate_binary_metrics(
            y_true,
            scores,
            threshold=float(threshold),
            positive_label=positive_label,
            beta=beta
        )
        for threshold in thresholds
    ]


def _legacy_evaluation_result(
    model_name,
    y_true,
    predictions,
    scores,
    training_time=None
):
    """Build the result structure expected by existing scripts."""

    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions).astype(int)
    scores = np.asarray(scores, dtype=float)

    if len(y_true) != len(predictions):
        raise ValueError("Labels and predictions must have equal length.")

    if len(y_true) != len(scores):
        raise ValueError("Labels and scores must have equal length.")

    if len(np.unique(y_true)) != 2:
        raise ValueError("Evaluation requires both target classes.")

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "model": model_name,
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "training_time_seconds": (
            float(training_time)
            if training_time is not None
            else None
        )
    }


def evaluate_binary_classifier(
    model_name,
    model,
    X_test,
    y_test,
    predictions,
    probabilities,
    training_time=None
):
    """
    Backward-compatible evaluator for comparison.py.

    model and X_test are retained in the signature for compatibility.
    Predictions and probabilities must be generated before calling.
    """

    return _legacy_evaluation_result(
        model_name,
        y_test,
        predictions,
        probabilities,
        training_time
    )


def evaluate_anomaly_detector(
    model_name,
    model,
    X_test,
    y_test,
    predictions,
    anomaly_scores
):
    """Backward-compatible evaluator for anomaly_comparison.py."""

    return _legacy_evaluation_result(
        model_name,
        y_test,
        predictions,
        anomaly_scores
    )
