
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

from app.ml.algorithms.model_registry import load_model_bundle


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_NAME = "XGBoost"
SAFE_MODEL_DIR = "xgboost"

METADATA_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "model_registry"
    / SAFE_MODEL_DIR
    / "metadata.json"
)

MODEL_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "model_registry"
    / SAFE_MODEL_DIR
    / "model.joblib"
)

REPORT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "xgboost_threshold_evaluation.json"
)

EXPECTED_ORIGINAL_THRESHOLD = 0.884595513343811
MINIMUM_PRECISION = 0.80


def load_evaluation_report():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {REPORT_PATH}"
        )

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if report.get("model_name") != MODEL_NAME:
        raise ValueError("Evaluation report model mismatch.")

    candidate = report["thresholds"][
        "optimized_candidate_threshold"
    ]

    test_metrics = report["independent_test"][
        "optimized_threshold"
    ]

    precision = float(test_metrics["precision"])

    if precision < MINIMUM_PRECISION:
        raise ValueError(
            "Independent test precision does not meet "
            "the minimum requirement."
        )

    if not report["constraint_verification"][
        "cross_validation_precision_at_least_0_80"
    ]:
        raise ValueError(
            "Cross-validation precision constraint failed."
        )

    return float(candidate), precision, report


def write_metadata_atomically(metadata):
    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=METADATA_PATH.parent,
        suffix=".tmp",
        delete=False
    ) as temporary_file:

        json.dump(metadata, temporary_file, indent=4)
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(METADATA_PATH)


def main():
    parser = argparse.ArgumentParser(
        description="Safely promote the validated XGBoost threshold."
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the threshold after explicit confirmation."
    )

    args = parser.parse_args()

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    candidate, precision, report = load_evaluation_report()

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    current_threshold = float(metadata["threshold"])

    if abs(
        current_threshold - EXPECTED_ORIGINAL_THRESHOLD
    ) > 1e-10:
        raise ValueError(
            "Registered threshold differs from the "
            "expected original threshold. Stopping safely."
        )

    print("=" * 65)
    print("XGBOOST THRESHOLD PROMOTION")
    print("=" * 65)

    print("Current registered threshold:", current_threshold)
    print("Validated candidate threshold:", candidate)
    print("Independent test precision:", round(precision, 4))
    print("Production model replacement: NO")

    if not args.apply:
        print("\nDRY RUN ONLY")
        print("No files were modified.")
        print("To apply, run the command with --apply.")
        return

    confirmation = input(
        "\nType PROMOTE XGBOOST THRESHOLD to continue: "
    )

    if confirmation != "PROMOTE XGBOOST THRESHOLD":
        print("\nPromotion cancelled. No files modified.")
        return

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    backup_path = METADATA_PATH.with_name(
        f"metadata.json.backup_{timestamp}"
    )

    shutil.copy2(METADATA_PATH, backup_path)

    updated_metadata = dict(metadata)

    updated_metadata["threshold"] = candidate

    updated_metrics = dict(
        updated_metadata.get("metrics", {})
    )

    updated_metrics["threshold"] = candidate
    updated_metrics["threshold_type"] = (
        "Validated balanced threshold"
    )

    updated_metadata["metrics"] = updated_metrics

    updated_metadata["threshold_promotion"] = {
        "previous_threshold": current_threshold,
        "new_threshold": candidate,
        "evaluation_report": str(REPORT_PATH),
        "independent_test_precision": precision,
        "promoted_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    try:
        write_metadata_atomically(updated_metadata)

        verified_bundle = load_model_bundle(MODEL_NAME)

        verified_threshold = float(
            verified_bundle["metadata"]["threshold"]
        )

        if abs(verified_threshold - candidate) > 1e-10:
            raise RuntimeError(
                "Reload verification failed."
            )

        print("\nPROMOTION SUCCESSFUL")
        print("Verified threshold:", verified_threshold)
        print("Backup:", backup_path)
        print("Model artifact unchanged:", MODEL_PATH.exists())

    except Exception:
        shutil.copy2(backup_path, METADATA_PATH)
        print("\nPromotion failed. Original metadata restored.")
        raise


if __name__ == "__main__":
    main()
