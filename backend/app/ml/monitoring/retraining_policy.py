
"""Intelligent model retraining decision policy.

Evaluates unified monitoring reports and recommends whether
a model should continue running, be reviewed, or be retrained.

This module does not train models or overwrite model registry entries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_status_values(data: Any) -> list[str]:
    """Recursively collect status values from a report."""

    statuses = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower().endswith("status") and isinstance(value, str):
                statuses.append(value.upper())

            statuses.extend(_find_status_values(value))

    elif isinstance(data, list):
        for item in data:
            statuses.extend(_find_status_values(item))

    return statuses


def evaluate_retraining_need(report: dict) -> dict:
    """Evaluate a unified monitoring report.

    Decision rules:
    - CRITICAL drift or performance status: RETRAIN_RECOMMENDED
    - WARNING drift or performance status: REVIEW_REQUIRED
    - Otherwise: NO_RETRAINING_REQUIRED
    """

    if not isinstance(report, dict):
        raise TypeError("Monitoring report must be a dictionary.")

    drift_report = report.get("data_drift", {})
    performance_report = report.get("performance_comparison", {})

    if not isinstance(drift_report, dict):
        drift_report = {}

    if not isinstance(performance_report, dict):
        performance_report = {}

    drift_statuses = _find_status_values(drift_report)
    performance_statuses = _find_status_values(performance_report)

    # Also consider the top-level monitoring status.
    overall_status = str(
        report.get("overall_status", "")
    ).upper()

    reasons = []

    if "CRITICAL" in drift_statuses:
        reasons.append(
            "Critical data drift detected."
        )

    if "CRITICAL" in performance_statuses:
        reasons.append(
            "Critical model performance degradation detected."
        )

    if overall_status == "CRITICAL" and not reasons:
        reasons.append(
            "Unified monitoring reports a critical condition."
        )

    if reasons:
        decision = "RETRAIN_RECOMMENDED"
        priority = "HIGH"

    elif (
        "WARNING" in drift_statuses
        or "WARNING" in performance_statuses
        or overall_status == "WARNING"
    ):
        decision = "REVIEW_REQUIRED"
        priority = "MEDIUM"

        if "WARNING" in drift_statuses:
            reasons.append(
                "Data drift warning detected."
            )

        if "WARNING" in performance_statuses:
            reasons.append(
                "Model performance warning detected."
            )

        if overall_status == "WARNING":
            reasons.append(
                "Unified monitoring reports a warning."
            )

    else:
        decision = "NO_RETRAINING_REQUIRED"
        priority = "LOW"
        reasons.append(
            "No critical drift or performance warning was found."
        )

    return {
        "decision_type": "MODEL_RETRAINING_ASSESSMENT",
        "model_name": report.get(
            "model_name",
            "unknown_model",
        ),
        "decision": decision,
        "priority": priority,
        "overall_monitoring_status": overall_status or "UNKNOWN",
        "drift_statuses": sorted(set(drift_statuses)),
        "performance_statuses": sorted(set(performance_statuses)),
        "reasons": reasons,
        "automatic_retraining": False,
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def evaluate_report_file(report_path: str | Path) -> dict:
    """Load a unified monitoring JSON report and evaluate it."""

    report_path = Path(report_path)

    if not report_path.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {report_path}"
        )

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    return evaluate_retraining_need(report)


def save_retraining_decision(
    decision_report: dict,
    output_path: str | Path,
) -> Path:
    """Save the retraining decision as a JSON file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            decision_report,
            file,
            indent=4,
        )

    return output_path


def run_self_test() -> None:
    """Test stable, warning, and critical monitoring scenarios."""

    print("\nRETRAINING POLICY SELF-TEST")
    print("=" * 60)

    stable_report = {
        "model_name": "AI4I XGBoost",
        "overall_status": "STABLE",
        "data_drift": {
            "overall_status": "STABLE",
        },
        "performance_comparison": {
            "status": "STABLE",
        },
    }

    warning_report = {
        "model_name": "AI4I XGBoost",
        "overall_status": "WARNING",
        "data_drift": {
            "overall_status": "WARNING",
        },
        "performance_comparison": {
            "status": "STABLE",
        },
    }

    critical_report = {
        "model_name": "AI4I XGBoost",
        "overall_status": "CRITICAL",
        "data_drift": {
            "overall_status": "CRITICAL",
        },
        "performance_comparison": {
            "status": "CRITICAL",
        },
    }

    stable_result = evaluate_retraining_need(stable_report)
    warning_result = evaluate_retraining_need(warning_report)
    critical_result = evaluate_retraining_need(critical_report)

    assert stable_result["decision"] == "NO_RETRAINING_REQUIRED"
    assert warning_result["decision"] == "REVIEW_REQUIRED"
    assert critical_result["decision"] == "RETRAIN_RECOMMENDED"

    assert stable_result["automatic_retraining"] is False
    assert warning_result["automatic_retraining"] is False
    assert critical_result["automatic_retraining"] is False

    print(
        "Stable scenario:",
        stable_result["decision"],
    )

    print(
        "Warning scenario:",
        warning_result["decision"],
    )

    print(
        "Critical scenario:",
        critical_result["decision"],
    )

    print("\nALL RETRAINING POLICY TESTS PASSED")


if __name__ == "__main__":
    run_self_test()
