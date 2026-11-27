
"""
Unified Monitoring Manager

Combines:
1. Dataset schema and feature drift monitoring
2. Model performance evaluation
3. Comparison against baseline metrics
4. Historical model performance tracking
5. Performance degradation detection
6. Unified monitoring status
7. JSON reporting

Existing monitoring modules are reused without modification.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.ml.monitoring.data_drift import compare_datasets

from app.ml.monitoring.model_performance import (
    evaluate_model_performance,
    compare_with_baseline,
    save_performance_report,
)

from app.ml.monitoring.performance_history import (
    record_model_performance,
    detect_performance_degradation,
)


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = BASE_DIR / "ml_artifacts" / "monitoring"


def _json_safe(value):
    """Convert NumPy and other values into JSON-compatible values."""

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    return value


def _determine_status(
    drift_report,
    performance_comparison,
    history_degradation=None,
):
    """Determine the combined monitoring status."""

    statuses = []

    if drift_report:
        statuses.append(
            str(
                drift_report.get(
                    "overall_status",
                    "UNKNOWN"
                )
            ).upper()
        )

    if performance_comparison:
        statuses.append(
            str(
                performance_comparison.get(
                    "status",
                    "UNKNOWN"
                )
            ).upper()
        )

    if history_degradation and history_degradation.get("success"):
        statuses.append(
            str(
                history_degradation.get(
                    "status",
                    "UNKNOWN"
                )
            ).upper()
        )

    if "CRITICAL" in statuses:
        return "CRITICAL"

    if "WARNING" in statuses:
        return "WARNING"

    if "STABLE" in statuses:
        return "STABLE"

    return "UNKNOWN"


def run_unified_monitoring(
    reference_dataset,
    current_dataset,
    y_true=None,
    y_pred=None,
    y_probability=None,
    model_name="unknown_model",
    baseline_metrics=None,
    ignored_columns=None,
    output_dir=None,
):
    """
    Run data drift, model performance and historical monitoring.

    Historical performance is recorded only when actual evaluation
    labels and predictions are supplied.

    Parameters
    ----------
    reference_dataset : str or Path
        Path to reference CSV dataset.

    current_dataset : str or Path
        Path to incoming/current CSV dataset.

    y_true : array-like, optional
        Actual labels for current evaluation batch.

    y_pred : array-like, optional
        Predicted labels for current evaluation batch.

    y_probability : array-like, optional
        Predicted failure probabilities.

    model_name : str
        Name of the model being monitored.

    baseline_metrics : dict, optional
        Previously saved model metrics.

    ignored_columns : list, optional
        Columns excluded from drift comparison.

    output_dir : str or Path, optional
        Directory where monitoring reports are saved.

    Returns
    -------
    dict
        Unified monitoring report.
    """

    reference_dataset = Path(reference_dataset)
    current_dataset = Path(current_dataset)

    if not reference_dataset.is_file():
        raise FileNotFoundError(
            f"Reference dataset not found: {reference_dataset}"
        )

    if not current_dataset.is_file():
        raise FileNotFoundError(
            f"Current dataset not found: {current_dataset}"
        )

    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name must be a non-empty string.")

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # --------------------------------------------------
    # STEP 1: DATA DRIFT MONITORING
    # --------------------------------------------------

    drift_path = output_dir / (
        f"unified_drift_{timestamp}.json"
    )

    drift_report = compare_datasets(
        reference_path=reference_dataset,
        current_path=current_dataset,
        ignored_columns=ignored_columns,
        output_path=drift_path,
    )

    # --------------------------------------------------
    # STEP 2: MODEL PERFORMANCE MONITORING
    # --------------------------------------------------

    performance_report = None
    performance_comparison = None

    # --------------------------------------------------
    # STEP 3: HISTORICAL PERFORMANCE TRACKING
    # --------------------------------------------------

    history_record = None
    history_degradation = None

    supplied_labels = (
        y_true is not None or y_pred is not None
    )

    if supplied_labels:

        if y_true is None or y_pred is None:
            raise ValueError(
                "Provide both y_true and y_pred."
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have equal lengths."
            )

        if len(y_true) == 0:
            raise ValueError(
                "Evaluation data cannot be empty."
            )

        if y_probability is not None:
            if len(y_probability) != len(y_true):
                raise ValueError(
                    "y_probability must match y_true length."
                )

        # Evaluate current model performance.
        performance_report = evaluate_model_performance(
            y_true=y_true,
            y_pred=y_pred,
            y_probability=y_probability,
            model_name=model_name,
        )

        # Compare with an optional baseline.
        if baseline_metrics is not None:
            performance_comparison = compare_with_baseline(
                current_report=performance_report,
                baseline_metrics=baseline_metrics,
            )

        performance_path = output_dir / (
            f"unified_performance_{timestamp}.json"
        )

        save_performance_report(
            performance_report,
            output_path=performance_path,
        )

        # Convert the existing metric naming convention:
        # model_performance uses "f1".
        # performance_history expects "f1_score".
        current_metrics = performance_report["metrics"]

        history_metrics = {
            "accuracy": current_metrics["accuracy"],
            "precision": current_metrics["precision"],
            "recall": current_metrics["recall"],
            "f1_score": current_metrics["f1"],
        }

        # Save only an actual completed evaluation.
        history_record = record_model_performance(
            model_name=model_name,
            metrics=history_metrics,
            sample_count=current_metrics["total_samples"],
        )

        # Compare the newly recorded evaluation with
        # the previous evaluation of this same model.
        history_degradation = detect_performance_degradation(
            model_name=model_name,
        )

    # --------------------------------------------------
    # STEP 4: COMBINED MONITORING STATUS
    # --------------------------------------------------

    overall_status = _determine_status(
        drift_report=drift_report,
        performance_comparison=performance_comparison,
        history_degradation=history_degradation,
    )

    # --------------------------------------------------
    # STEP 5: BUILD UNIFIED REPORT
    # --------------------------------------------------

    unified_report = {
        "report_type": "unified_monitoring",

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_name": model_name,

        "overall_status": overall_status,

        "monitoring_components": {
            "data_drift": True,

            "model_performance": (
                performance_report is not None
            ),

            "baseline_comparison": (
                performance_comparison is not None
            ),

            "performance_history": (
                history_record is not None
            ),

            "performance_degradation": (
                history_degradation is not None
                and history_degradation.get("success", False)
            ),
        },

        "data_drift": drift_report,

        "model_performance": performance_report,

        "performance_comparison": performance_comparison,

        "performance_history": {
            "recorded": history_record is not None,

            "evaluation_id": (
                history_record["evaluation_id"]
                if history_record else None
            ),

            "degradation_check": history_degradation,
        },
    }

    # --------------------------------------------------
    # STEP 6: SAVE UNIFIED REPORT
    # --------------------------------------------------

    unified_path = output_dir / (
        f"unified_monitoring_{timestamp}.json"
    )

    safe_report = _json_safe(unified_report)

    with unified_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_report,
            file,
            indent=4,
            allow_nan=False,
        )

    unified_report["report_path"] = str(unified_path)

    print("\nUNIFIED MONITORING COMPLETED")
    print("=" * 55)

    print("Model:", model_name)
    print("Overall status:", overall_status)

    print(
        "Data drift:",
        drift_report.get("overall_status", "UNKNOWN"),
    )

    print(
        "Performance evaluated:",
        performance_report is not None,
    )

    print(
        "Historical performance recorded:",
        history_record is not None,
    )

    if history_degradation and history_degradation.get("success"):
        print(
            "Historical degradation:",
            history_degradation["status"],
        )

    print("Report saved:", unified_path)

    return unified_report
