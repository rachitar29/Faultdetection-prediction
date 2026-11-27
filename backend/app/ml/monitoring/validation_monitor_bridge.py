
"""
Bridge between prediction validation and unified monitoring.

Connects:
1. Technician-confirmed actual outcomes.
2. Previously recorded ML predictions.
3. Data drift monitoring.
4. Model performance evaluation.
5. Baseline comparison.

Existing monitoring modules remain unchanged.
"""

from pathlib import Path
from typing import Optional, Dict, Any

from app.ml.monitoring.prediction_validation import (
    evaluate_confirmed_outcomes,
)

from app.ml.monitoring.unified_monitor import (
    run_unified_monitoring,
)


BACKEND_DIR = Path(__file__).resolve().parents[3]

DEFAULT_REFERENCE_DATASET = (
    BACKEND_DIR
    / "ml_data"
    / "raw"
    / "ai4i2020.csv"
)


def run_validation_monitoring(
    current_dataset,
    reference_dataset=None,
    model_name="XGBoost",
    baseline_metrics: Optional[Dict[str, Any]] = None,
    ignored_columns=None,
    output_dir=None,
):
    """
    Run unified monitoring using confirmed prediction outcomes.

    Parameters
    ----------
    current_dataset:
        CSV containing the current incoming sensor data.

    reference_dataset:
        Reference training dataset. Defaults to AI4I.

    model_name:
        Name of the monitored model.

    baseline_metrics:
        Previously saved baseline model metrics.

    ignored_columns:
        Columns excluded from drift monitoring.

    output_dir:
        Optional monitoring report directory.
    """

    current_dataset = Path(current_dataset)

    if not current_dataset.is_file():
        raise FileNotFoundError(
            f"Current dataset not found: {current_dataset}"
        )

    if reference_dataset is None:
        reference_dataset = DEFAULT_REFERENCE_DATASET

    reference_dataset = Path(reference_dataset)

    if not reference_dataset.is_file():
        raise FileNotFoundError(
            f"Reference dataset not found: {reference_dataset}"
        )

    # STEP 1: Retrieve technician-confirmed outcomes.
    validation_result = evaluate_confirmed_outcomes()

    y_true = None
    y_pred = None
    y_probability = None

    matched_count = 0

    if validation_result.get("success"):

        matched_records = validation_result[
            "matched_records"
        ]

        # STEP 2: Use only matched confirmed outcomes.
        model_records = [
            record
            for record in matched_records
            if (
                record.get("model_name") in (None, model_name)
            )
        ]

        if model_records:

            y_true = [
                record["actual_failure"]
                for record in model_records
            ]

            y_pred = [
                record["predicted_failure"]
                for record in model_records
            ]

            y_probability = [
                record["failure_probability"]
                for record in model_records
            ]

            matched_count = len(model_records)

    # STEP 3: Run existing unified monitoring.
    report = run_unified_monitoring(
        reference_dataset=reference_dataset,
        current_dataset=current_dataset,
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
        model_name=model_name,
        baseline_metrics=baseline_metrics,
        ignored_columns=ignored_columns,
        output_dir=output_dir,
    )

    # STEP 4: Add validation information.
    report["prediction_validation"] = {
        "confirmed_outcomes_available": matched_count > 0,
        "matched_outcome_count": matched_count,
        "performance_metrics_source": (
            "Technician-confirmed or otherwise verified outcomes"
            if matched_count
            else "No matching confirmed outcomes available"
        ),
    }

    return report
