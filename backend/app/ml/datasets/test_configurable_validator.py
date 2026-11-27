
from pathlib import Path

from app.ml.datasets.configurable_validator import (
    validate_configured_dataset,
    save_validation_report,
)


def print_result(title, result):
    print(f"\n{title}")
    print("Valid:", result["valid"])
    print("Rows:", result.get("rows"))
    print("Columns:", result.get("columns"))

    errors = result.get("errors", [])
    warnings = result.get("warnings", [])

    print("Errors:", len(errors))
    for error in errors:
        print("[ERROR]", error)

    print("Warnings:", len(warnings))
    for warning in warnings:
        print("[WARNING]", warning)


def main():
    backend_dir = Path(__file__).resolve().parents[3]

    data_dir = backend_dir / "ml_data" / "validity_test"

    config_path = (
        backend_dir
        / "app"
        / "ml"
        / "configurations"
        / "dataset_configs"
        / "industrial_sensor_config.json"
    )

    dirty_file = data_dir / "sensor_validity_test.csv"
    clean_file = data_dir / "sensor_validity_clean.csv"

    report_dir = backend_dir / "ml_artifacts" / "reports"

    for path in [config_path, dirty_file, clean_file]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("=" * 65)
    print("CONFIGURABLE INDUSTRIAL VALIDATION TEST")
    print("=" * 65)

    dirty_result = validate_configured_dataset(
        file_path=dirty_file,
        config_path=config_path,
    )

    clean_result = validate_configured_dataset(
        file_path=clean_file,
        config_path=config_path,
    )

    print_result("1. DIRTY DATASET", dirty_result)
    print_result("2. CLEAN DATASET", clean_result)

    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "configurable_validation_report.json"

    save_validation_report(
        {
            "dirty_dataset": dirty_result,
            "clean_dataset": clean_result,
        },
        report_path,
    )

    print("\n3. TEST SUMMARY")
    print("Dirty dataset rejected:", not dirty_result["valid"])
    print("Clean dataset accepted:", clean_result["valid"])
    print("Report saved:", report_path)

    if dirty_result["valid"]:
        raise AssertionError("Dirty dataset should be rejected.")

    if not clean_result["valid"]:
        raise AssertionError("Clean dataset should be accepted.")

    print("\nCONFIGURABLE VALIDATION TEST PASSED!")


if __name__ == "__main__":
    main()
