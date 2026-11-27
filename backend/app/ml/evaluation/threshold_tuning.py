
import numpy as np

from sklearn.metrics import precision_recall_curve

from app.ml.evaluation.metrics import calculate_binary_metrics


def optimize_threshold(
    y_true,
    scores,
    positive_label=1,
    beta=2.0,
    min_precision=None
):
    """
    Optimize a threshold using validation data only.

    Higher scores must indicate a greater likelihood of failure.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if len(y_true) != len(scores):
        raise ValueError("Labels and scores must have equal length.")

    if len(y_true) == 0:
        raise ValueError("Validation data cannot be empty.")

    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Threshold optimization requires both target classes."
        )

    if positive_label not in np.unique(y_true):
        raise ValueError("Positive label is not present.")

    if beta <= 0:
        raise ValueError("beta must be greater than zero.")

    if min_precision is not None and not 0 < min_precision <= 1:
        raise ValueError("min_precision must be between 0 and 1.")

    y_binary = (y_true == positive_label).astype(int)

    precision, recall, thresholds = precision_recall_curve(
        y_binary, scores
    )

    if len(thresholds) == 0:
        raise ValueError("No threshold candidates were generated.")

    precision = precision[:-1]
    recall = recall[:-1]

    beta_squared = beta ** 2
    denominator = beta_squared * precision + recall

    f_beta_values = np.divide(
        (1 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0
    )

    valid_indices = np.arange(len(thresholds))

    if min_precision is not None:
        valid_indices = valid_indices[
            precision >= min_precision
        ]

    if len(valid_indices) == 0:
        raise ValueError(
            "No candidate threshold satisfies minimum precision."
        )

    best_index = max(
        valid_indices,
        key=lambda i: (
            f_beta_values[i],
            recall[i],
            precision[i]
        )
    )

    best_threshold = float(thresholds[best_index])

    metrics = calculate_binary_metrics(
        y_true,
        scores,
        threshold=best_threshold,
        positive_label=positive_label,
        beta=beta
    )

    return {
        "best_threshold": best_threshold,
        "optimization_metric": f"F{beta:g}",
        "beta": float(beta),
        "minimum_precision": min_precision,
        "validation_metrics": metrics,
        "candidate_thresholds": int(len(thresholds))
    }


def find_best_threshold(y_true, scores):
    """
    Legacy F1 threshold optimizer.

    Retains the return keys expected by comparison.py and
    anomaly_comparison.py.
    """

    result = optimize_threshold(
        y_true=y_true,
        scores=scores,
        positive_label=1,
        beta=1.0
    )

    metrics = result["validation_metrics"]

    return {
        "threshold": result["best_threshold"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
        "f1": metrics["f1"],
        "optimization_metric": "F1",
        "validation_metrics": metrics
    }


def evaluate_fixed_threshold(
    y_true,
    scores,
    threshold,
    positive_label=1,
    beta=2.0
):
    """Evaluate a previously selected threshold."""

    return calculate_binary_metrics(
        y_true,
        scores,
        threshold=threshold,
        positive_label=positive_label,
        beta=beta
    )
