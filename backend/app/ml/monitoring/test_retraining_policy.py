
"""Integration test connecting unified monitoring to retraining policy."""

from datetime import datetime, timezone
from pathlib import Path

from app.ml.monitoring.retraining_policy import (
    evaluate_report_file,
    save_retraining_decision,
)


BACKEND_DIR = Path(__file__).resolve().parents[3]

MONITORING_DIR = (
    BACKEND_DIR / "ml_artifacts" / "monitoring"
)


def find_latest_monitoring_report() -> Path:
    """Find the newest unified monitoring JSON report."""

    reports = list(
        MONITORING_DIR.glob("unified_monitoring_*.json")
    )

    if not reports:
        raise FileNotFoundError(
            "No unified monitoring reports found in: "
            f"{MONITORING_DIR}"
        )

    return max(
        reports,
        key=lambda path: path.stat().st_mtime,
    )


def main():
    print("\nRETRAINING POLICY INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Locate the latest unified monitoring report.
    monitoring_report_path = find_latest_monitoring_report()

    print("\nMonitoring report selected:")
    print(monitoring_report_path)

    # Step 2: Evaluate the monitoring report.
    decision_report = evaluate_report_file(
        monitoring_report_path
    )

    # Step 3: Generate a unique decision report filename.
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    decision_path = (
        MONITORING_DIR
        / f"retraining_decision_{timestamp}.json"
    )

    # Step 4: Save the decision.
    saved_path = save_retraining_decision(
        decision_report,
        decision_path,
    )

    # Step 5: Display the results.
    print("\nRETRAINING DECISION")
    print("=" * 60)

    print(
        "Model:",
        decision_report.get("model_name"),
    )

    print(
        "Monitoring status:",
        decision_report.get(
            "overall_monitoring_status"
        ),
    )

    print(
        "Decision:",
        decision_report.get("decision"),
    )

    print(
        "Priority:",
        decision_report.get("priority"),
    )

    print("\nReasons:")

    for reason in decision_report.get("reasons", []):
        print("-", reason)

    print(
        "\nAutomatic retraining:",
        decision_report.get("automatic_retraining"),
    )

    print("\nDecision report saved:")
    print(saved_path)

    # Step 6: Basic validation.
    if not saved_path.exists():
        raise AssertionError(
            "Retraining decision report was not saved."
        )

    if decision_report.get("decision") not in {
        "NO_RETRAINING_REQUIRED",
        "REVIEW_REQUIRED",
        "RETRAIN_RECOMMENDED",
    }:
        raise AssertionError(
            "Unexpected retraining decision."
        )

    print("\nRETRAINING POLICY INTEGRATION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
