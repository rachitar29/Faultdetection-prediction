
"""
Prediction Validation and Actual Outcome Tracking.

Responsibilities:
1. Record ML predictions.
2. Record technician-confirmed outcomes.
3. Match predictions with actual outcomes.
4. Calculate classification performance.
5. Save validation reports.

This module uses local JSONL files.
It does not implement database operations.
"""

import json
import math
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ==========================================
# CONFIGURATION
# ==========================================

BACKEND_DIR = Path(__file__).resolve().parents[3]

VALIDATION_DIR = (
    BACKEND_DIR
    / "ml_artifacts"
    / "validation"
)

PREDICTIONS_FILE = VALIDATION_DIR / "predictions.jsonl"

ACTUAL_OUTCOMES_FILE = (
    VALIDATION_DIR / "actual_outcomes.jsonl"
)

REPORTS_DIR = VALIDATION_DIR / "reports"


# ==========================================
# UTILITIES
# ==========================================

def current_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


def make_json_safe(value: Any):

    if isinstance(value, dict):

        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            make_json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):

        try:
            return make_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    if isinstance(value, float):

        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:

        return value

    return str(value)


def append_jsonl(file_path: Path, record: Dict[str, Any]):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                make_json_safe(record),
                allow_nan=False
            ) + "\n"
        )


def read_jsonl(file_path: Path):

    if not file_path.exists():
        return []

    records = []

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON in {file_path}, "
                    f"line {line_number}: {error}"
                )

    return records


# ==========================================
# RECORD PREDICTION
# ==========================================

def record_prediction(
    equipment_id: str,
    prediction: Dict[str, Any],
    prediction_id: Optional[str] = None,
):
    """
    Save a model prediction for future validation.

    The prediction must contain failure_probability
    and failure_prediction.
    """

    if not equipment_id:
        raise ValueError("equipment_id is required.")

    if not isinstance(prediction, dict):
        raise TypeError("prediction must be a dictionary.")

    probability = prediction.get("failure_probability")

    if probability is None:
        raise ValueError(
            "failure_probability is required."
        )

    probability = float(probability)

    if not 0 <= probability <= 1:
        raise ValueError(
            "failure_probability must be between 0 and 1."
        )

    predicted_label = prediction.get("failure_prediction")

    if predicted_label not in ["NORMAL", "FAILURE"]:
        raise ValueError(
            "failure_prediction must be NORMAL or FAILURE."
        )

    record = {
        "prediction_id": prediction_id or str(uuid.uuid4()),
        "equipment_id": equipment_id,
        "failure_probability": probability,
        "failure_prediction": predicted_label,
        "risk_level": prediction.get("risk_level"),
        "model_name": prediction.get(
            "model_metadata", {}
        ).get("model_name"),
        "prediction_timestamp": prediction.get(
            "timestamp"
        ) or current_timestamp(),
        "recorded_timestamp": current_timestamp(),
    }

    append_jsonl(
        PREDICTIONS_FILE,
        record
    )

    return record


# ==========================================
# RECORD ACTUAL OUTCOME
# ==========================================

def record_actual_outcome(
    prediction_id: str,
    actual_failure: bool,
    source: str = "technician_confirmed",
    notes: Optional[str] = None,
):
    """
    Record the observed outcome associated with a prediction.

    actual_failure:
        True  = confirmed failure
        False = confirmed normal outcome

    source:
        technician_confirmed, maintenance_record,
        or verified_sensor_outcome.
    """

    if not prediction_id:
        raise ValueError("prediction_id is required.")

    if not isinstance(actual_failure, bool):
        raise TypeError(
            "actual_failure must be True or False."
        )

    allowed_sources = {
        "technician_confirmed",
        "maintenance_record",
        "verified_sensor_outcome",
    }

    if source not in allowed_sources:
        raise ValueError(
            f"source must be one of {sorted(allowed_sources)}."
        )

    record = {
        "outcome_id": str(uuid.uuid4()),
        "prediction_id": prediction_id,
        "actual_failure": actual_failure,
        "actual_label": (
            "FAILURE" if actual_failure else "NORMAL"
        ),
        "source": source,
        "notes": notes,
        "outcome_timestamp": current_timestamp(),
    }

    append_jsonl(
        ACTUAL_OUTCOMES_FILE,
        record
    )

    return record


# ==========================================
# CLASSIFICATION EVALUATION
# ==========================================

def evaluate_classification(
    actual_labels,
    predicted_labels,
):
    """Calculate binary classification metrics."""

    if len(actual_labels) != len(predicted_labels):
        raise ValueError(
            "Actual and predicted label counts must match."
        )

    if not actual_labels:
        raise ValueError(
            "At least one confirmed outcome is required."
        )

    allowed = {"NORMAL", "FAILURE"}

    if not set(actual_labels).issubset(allowed):
        raise ValueError("Invalid actual labels.")

    if not set(predicted_labels).issubset(allowed):
        raise ValueError("Invalid predicted labels.")

    actual = [
        1 if label == "FAILURE" else 0
        for label in actual_labels
    ]

    predicted = [
        1 if label == "FAILURE" else 0
        for label in predicted_labels
    ]

    return {
        "sample_count": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1_score": f1_score(
            actual, predicted, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).tolist(),
        "class_labels": ["NORMAL", "FAILURE"],
    }


# ==========================================
# EVALUATE CONFIRMED OUTCOMES
# ==========================================

def evaluate_confirmed_outcomes():
    """
    Match stored predictions with confirmed outcomes
    and evaluate the matched records.

    Unconfirmed predictions are excluded.
    """

    predictions = read_jsonl(PREDICTIONS_FILE)
    outcomes = read_jsonl(ACTUAL_OUTCOMES_FILE)

    prediction_map = {
        record["prediction_id"]: record
        for record in predictions
    }

    matched = []
    unmatched = []

    seen_outcomes = set()

    for outcome in outcomes:

        prediction_id = outcome["prediction_id"]

        if prediction_id not in prediction_map:
            unmatched.append(prediction_id)
            continue

        if prediction_id in seen_outcomes:
            raise ValueError(
                "Multiple actual outcomes found for "
                f"prediction ID: {prediction_id}"
            )

        seen_outcomes.add(prediction_id)

        matched.append({
            "prediction_id": prediction_id,
            "equipment_id": prediction_map[
                prediction_id
            ]["equipment_id"],
            "actual_failure": outcome["actual_label"],
            "predicted_failure": prediction_map[
                prediction_id
            ]["failure_prediction"],
            "failure_probability": prediction_map[
                prediction_id
            ]["failure_probability"],
            "outcome_source": outcome["source"],
        })

    if not matched:

        return {
            "success": False,
            "message": "No matched confirmed outcomes available.",
            "matched_count": 0,
            "unmatched_outcomes": unmatched,
        }

    metrics = evaluate_classification(
        actual_labels=[
            item["actual_failure"]
            for item in matched
        ],
        predicted_labels=[
            item["predicted_failure"]
            for item in matched
        ],
    )

    return {
        "success": True,
        "matched_count": len(matched),
        "unmatched_outcomes": unmatched,
        "metrics": metrics,
        "matched_records": matched,
        "evaluation_timestamp": current_timestamp(),
    }


# ==========================================
# SAVE VALIDATION REPORT
# ==========================================

def save_validation_report(report):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        "validation_"
        + datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        + ".json"
    )

    report_path = REPORTS_DIR / filename

    with report_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            make_json_safe(report),
            file,
            indent=4,
            allow_nan=False
        )

    return str(report_path)
