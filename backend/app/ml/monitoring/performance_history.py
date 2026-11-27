
"""
ML Model Performance History Tracker.

Stores historical evaluation reports locally.
Supports performance trend analysis and degradation alerts.
"""

import json
import math
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[3]

HISTORY_DIR = BASE_DIR / "ml_artifacts" / "monitoring"

HISTORY_FILE = HISTORY_DIR / "performance_history.jsonl"


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _read_history():

    if not HISTORY_FILE.exists():
        return []

    records = []

    with HISTORY_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid history JSON at line {line_number}: {error}"
                )

    return records


def record_model_performance(
    model_name: str,
    metrics: Dict[str, Any],
    model_version: Optional[str] = None,
    sample_count: Optional[int] = None,
):
    """Save one model evaluation result."""

    if not model_name:
        raise ValueError("model_name is required.")

    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    validated_metrics = {}

    for metric in required_metrics:

        if metric not in metrics:
            raise ValueError(
                f"Missing required metric: {metric}"
            )

        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            raise ValueError(
                f"{metric} must be a valid number."
            )

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(
                f"{metric} must be between 0 and 1."
            )

        validated_metrics[metric] = value

    if sample_count is not None:

        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 0
        ):
            raise ValueError(
                "sample_count must be a non-negative integer."
            )

    record = {
        "evaluation_id": str(uuid.uuid4()),
        "model_name": model_name,
        "model_version": model_version,
        "sample_count": sample_count,
        "metrics": validated_metrics,
        "evaluation_timestamp": _timestamp(),
    }

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with HISTORY_FILE.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(record, allow_nan=False) + "\n"
        )

    return record


def get_model_performance_history(
    model_name: Optional[str] = None,
):
    """Retrieve historical evaluations."""

    records = _read_history()

    if model_name is not None:

        records = [
            record
            for record in records
            if record["model_name"] == model_name
        ]

    return records


def compare_latest_performance(model_name: str):
    """Compare the two most recent evaluations."""

    records = get_model_performance_history(model_name)

    records.sort(
        key=lambda item: item["evaluation_timestamp"]
    )

    if len(records) < 2:

        return {
            "success": False,
            "message": "At least two evaluations are required.",
            "evaluation_count": len(records),
        }

    previous = records[-2]
    current = records[-1]

    changes = {}

    for metric in current["metrics"]:

        previous_value = previous["metrics"][metric]
        current_value = current["metrics"][metric]

        changes[metric] = round(
            current_value - previous_value,
            6
        )

    return {
        "success": True,
        "model_name": model_name,
        "previous_evaluation": previous,
        "current_evaluation": current,
        "metric_changes": changes,
        "comparison_timestamp": _timestamp(),
    }


def detect_performance_degradation(
    model_name: str,
    minimum_acceptable_f1: float = 0.70,
    maximum_f1_drop: float = 0.10,
):
    """
    Detect configurable F1 degradation.

    These are project monitoring thresholds,
    not universal industrial standards.
    """

    if not 0 <= minimum_acceptable_f1 <= 1:
        raise ValueError(
            "minimum_acceptable_f1 must be between 0 and 1."
        )

    if not 0 <= maximum_f1_drop <= 1:
        raise ValueError(
            "maximum_f1_drop must be between 0 and 1."
        )

    comparison = compare_latest_performance(model_name)

    if not comparison["success"]:
        return comparison

    previous_f1 = comparison[
        "previous_evaluation"
    ]["metrics"]["f1_score"]

    current_f1 = comparison[
        "current_evaluation"
    ]["metrics"]["f1_score"]

    f1_drop = previous_f1 - current_f1

    threshold_reached = (
        f1_drop >= maximum_f1_drop
        or math.isclose(
            f1_drop,
            maximum_f1_drop,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    )

    if current_f1 < minimum_acceptable_f1:
        status = "CRITICAL"

    elif threshold_reached:
        status = "WARNING"

    else:
        status = "STABLE"

    return {
        "success": True,
        "model_name": model_name,
        "status": status,
        "previous_f1": previous_f1,
        "current_f1": current_f1,
        "f1_drop": round(f1_drop, 6),
        "minimum_acceptable_f1": minimum_acceptable_f1,
        "maximum_f1_drop": maximum_f1_drop,
        "checked_at": _timestamp(),
    }
