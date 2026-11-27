
import json
from pathlib import Path

from app.ml.datasets.dataset_validator import (
    validate_sensor_dataset_file
)


def main():

    backend_dir = Path(__file__).resolve().parents[3]

    test_file = (
        backend_dir
        / "ml_data"
        / "validity_test"
        / "sensor_validity_test.csv"
    )

    clean_file = (
        backend_dir
        / "ml_data"
        / "validity_test"
        / "sensor_validity_clean.csv"
    )

    report_dir = (
        backend_dir
        / "ml_artifacts"
        / "reports"
    )

    report_dir.mkdir(parents=True, exist_ok=True)

    if not test_file.exists():
        raise FileNotFoundError(
            f"Dirty test dataset not found: {test_file}"
        )

    if not clean_file.exists():
        raise FileNotFoundError(
            f"Clean test dataset not found: {clean_file}"
        )

    print("=" * 65)
    print("INDUSTRIAL SENSOR VALIDITY TEST")
    print("=" * 65)

    # Validate dirty dataset.
    print("\n1. VALIDATING DIRTY DATASET")

    dirty_result = validate_sensor_dataset_file(
        test_file,
        target_column="machine_failure",
        min_rows=10
    )

    print("\nDirty dataset valid:", dirty_result["valid"])

    print("\nErrors:")
    for error in dirty_result["errors"]:
        print("-", error)

    print("\nWarnings:")
    for warning in dirty_result["warnings"]:
        print("-", warning)

    # Validate clean dataset.
    print("\n2. VALIDATING CLEAN DATASET")

    clean_result = validate_sensor_dataset_file(
        clean_file,
        target_column="machine_failure",
        min_rows=10
    )

    print("\nClean dataset valid:", clean_result["valid"])

    print("\nErrors:")
    for error in clean_result["errors"]:
        print("-", error)

    print("\nWarnings:")
    for warning in clean_result["warnings"]:
        print("-", warning)

    # Save both reports.
    report = {
        "dirty_dataset": dirty_result,
        "clean_dataset": clean_result
    }

    report_path = (
        report_dir / "sensor_validity_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("\n3. TEST SUMMARY")

    print(
        "Dirty dataset correctly rejected:",
        not dirty_result["valid"]
    )

    print(
        "Clean dataset passed:",
        clean_result["valid"]
    )

    print("\nReport saved:", report_path)

    print("\nSENSOR VALIDITY TEST COMPLETED!")


if __name__ == "__main__":
    main()
