
"""
Model Performance Monitoring

Evaluates binary machine-failure predictions and compares
current performance with an optional baseline.

This module is independent of FastAPI.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BACKEND_DIR = Path(__file__).resolve().parents[3]

DEFAULT_REPORT_DIR = (
    BACKEND_DIR / "ml_artifacts" / "monitoring"
)


def _to_binary_array(values, name):
    """Validate and convert labels or predictions to binary arrays."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")

    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty")

    if pd_is_missing(array):
        raise ValueError(f"{name} contains missing values")

    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric 0/1 values") from exc

    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains non-finite values")

    if not np.isin(numeric, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")

    return numeric.astype(int)


def pd_is_missing(array):
    """Check missing values without requiring pandas."""
    try:
        return bool(np.asarray(array != array).any())
    except (TypeError, ValueError):
        return False


def evaluate_model_performance(
    y_true,
    y_pred,
    y_probability=None,
    model_name="unknown_model",
):
    """
    Evaluate a binary failure classifier.

    Label convention:
        0 = normal
        1 = machine failure

    y_probability must contain probabilities for class 1.
    """

    actual = _to_binary_array(y_true, "y_true")
    predicted = _to_binary_array(y_pred, "y_pred")

    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have the same number of rows"
        )

    if len(np.unique(actual)) < 2:
        raise ValueError(
            "Evaluation requires both normal and failure examples"
        )

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        actual, predicted, zero_division=0
    )
    recall = recall_score(
        actual, predicted, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else 0.0
        ),
        "false_positive_rate": float(
            fp / (fp + tn) if (fp + tn) else 0.0
        ),
        "false_negative_rate": float(
            fn / (fn + tp) if (fn + tp) else 0.0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(len(actual)),
        "actual_failure_count": int(np.sum(actual == 1)),
        "predicted_failure_count": int(np.sum(predicted == 1)),
    }

    if y_probability is not None:
        probabilities = np.asarray(
            y_probability, dtype=float
        )

        if probabilities.ndim != 1:
            raise ValueError(
                "y_probability must be one-dimensional"
            )

        if len(probabilities) != len(actual):
            raise ValueError(
                "y_probability must have the same length as y_true"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "y_probability contains non-finite values"
            )

        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(
                "y_probability values must be between 0 and 1"
            )

        metrics["roc_auc"] = float(
            roc_auc_score(actual, probabilities)
        )
        metrics["average_precision"] = float(
            average_precision_score(actual, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return {
        "model_name": model_name,
        "task": "binary_machine_failure_classification",
        "positive_class": 1,
        "positive_class_meaning": "machine_failure",
        "metrics": metrics,
    }


def compare_with_baseline(
    current_report,
    baseline_metrics,
    thresholds=None,
):
    """
    Compare current metrics with a baseline metrics dictionary.

    Default warning limits:
      - Recall decrease greater than 5 percentage points
      - F1 decrease greater than 5 percentage points
      - ROC-AUC decrease greater than 3 percentage points
      - False-positive-rate increase greater than 5 percentage points
    """

    if thresholds is None:
        thresholds = {
            "recall_drop": 0.05,
            "f1_drop": 0.05,
            "roc_auc_drop": 0.03,
            "false_positive_rate_increase": 0.05,
        }

    current_metrics = current_report["metrics"]
    comparisons = {}
    warnings = []

    metric_rules = {
        "recall": ("recall_drop", "decrease"),
        "f1": ("f1_drop", "decrease"),
        "roc_auc": ("roc_auc_drop", "decrease"),
        "false_positive_rate": (
            "false_positive_rate_increase",
            "increase",
        ),
    }

    for metric_name, (threshold_name, direction) in metric_rules.items():
        baseline_value = baseline_metrics.get(metric_name)
        current_value = current_metrics.get(metric_name)

        if baseline_value is None or current_value is None:
            comparisons[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "change": None,
                "status": "NOT_AVAILABLE",
            }
            continue

        change = float(current_value - baseline_value)

        if direction == "decrease":
            triggered = change < -thresholds[threshold_name]
        else:
            triggered = change > thresholds[threshold_name]

        status = "WARNING" if triggered else "STABLE"

        comparisons[metric_name] = {
            "baseline": float(baseline_value),
            "current": float(current_value),
            "change": round(change, 6),
            "status": status,
        }

        if triggered:
            warnings.append(metric_name)

    return {
        "status": "WARNING" if warnings else "STABLE",
        "warnings": warnings,
        "thresholds": thresholds,
        "comparisons": comparisons,
    }


def save_performance_report(
    report,
    output_path=None,
):
    """Save a performance report as JSON."""
    if output_path is None:
        DEFAULT_REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path = (
            DEFAULT_REPORT_DIR
            / "model_performance_report.json"
        )
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return str(output_path.resolve())


def run_self_test():
    """Run deterministic tests without loading a trained model."""

    print("\nMODEL PERFORMANCE MONITORING SELF-TEST")
    print("=" * 55)

    actual = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    )

    predicted = np.array(
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    )

    probabilities = np.array(
        [0.05, 0.10, 0.20, 0.65, 0.90,
         0.85, 0.35, 0.95, 0.15, 0.80]
    )

    baseline = {
        "recall": 0.90,
        "f1": 0.90,
        "roc_auc": 0.95,
        "false_positive_rate": 0.05,
    }

    result = evaluate_model_performance(
        y_true=actual,
        y_pred=predicted,
        y_probability=probabilities,
        model_name="self_test_model",
    )

    comparison = compare_with_baseline(
        result,
        baseline,
    )

    report = {
        "report_type": "model_performance",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation": result,
        "baseline_comparison": comparison,
    }

    report_path = save_performance_report(
        report,
        DEFAULT_REPORT_DIR
        / "model_performance_self_test.json",
    )

    print("Accuracy:", round(result["metrics"]["accuracy"], 4))
    print("Precision:", round(result["metrics"]["precision"], 4))
    print("Recall:", round(result["metrics"]["recall"], 4))
    print("F1:", round(result["metrics"]["f1"], 4))
    print("ROC-AUC:", round(result["metrics"]["roc_auc"], 4))
    print(
        "False positive rate:",
        round(result["metrics"]["false_positive_rate"], 4),
    )
    print("Baseline comparison:", comparison["status"])
    print("Report saved:", report_path)

    # Confirm invalid input is rejected.
    try:
        evaluate_model_performance(
            [0, 1, 2],
            [0, 1, 1],
            model_name="invalid_test",
        )
    except ValueError:
        print("Invalid-label validation: PASSED")
    else:
        raise AssertionError(
            "Invalid labels were not rejected"
        )

    print("\nALL MODEL PERFORMANCE SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser(
        description="Model performance monitoring"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in tests",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
