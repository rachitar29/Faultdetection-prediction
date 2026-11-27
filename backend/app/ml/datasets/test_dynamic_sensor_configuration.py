
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.datasets.configurable_validator import (
    validate_configured_dataset
)


def create_test_datasets(data_dir):

    data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    rows = 1000

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="min"
    )

    clean_df = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "machine_id": [
            f"MACHINE_{i % 10 + 1}"
            for i in range(rows)
        ],
        "vibration": np.random.uniform(1, 30, rows),
        "pressure": np.random.uniform(50, 300, rows),
        "humidity": np.random.uniform(20, 80, rows),
        "voltage": np.random.uniform(200, 450, rows),
        "current": np.random.uniform(10, 150, rows),
        "machine_failure": np.random.choice(
            [0, 1],
            size=rows,
            p=[0.95, 0.05]
        )
    })

    dirty_df = clean_df.copy()

    # Inject sensor range errors
    dirty_df.loc[10, "vibration"] = 80
    dirty_df.loc[20, "pressure"] = -10
    dirty_df.loc[30, "humidity"] = 150
    dirty_df.loc[40, "voltage"] = 600
    dirty_df.loc[50, "current"] = -5

    # Inject a non-numeric sensor reading
    dirty_df["vibration"] = dirty_df["vibration"].astype(object)
    dirty_df.loc[60, "vibration"] = "sensor_error"

    # Inject missing sensor data
    dirty_df.loc[70, "humidity"] = np.nan

    # Inject invalid target label
    dirty_df.loc[80, "machine_failure"] = 7

    # Inject invalid timestamp
    dirty_df.loc[90, "timestamp"] = "not-a-valid-date"

    clean_path = data_dir / "alternative_sensor_clean.csv"
    dirty_path = data_dir / "alternative_sensor_dirty.csv"

    clean_df.to_csv(clean_path, index=False)
    dirty_df.to_csv(dirty_path, index=False)

    return clean_path, dirty_path


def save_report(report, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )


def main():

    backend_dir = Path(__file__).resolve().parents[3]

    data_dir = (
        backend_dir
        / "ml_data"
        / "validity_test"
        / "alternative_sensors"
    )

    config_path = (
        backend_dir
        / "app"
        / "ml"
        / "configurations"
        / "dataset_configs"
        / "alternative_sensor_config.json"
    )

    report_path = (
        backend_dir
        / "ml_artifacts"
        / "reports"
        / "alternative_sensor_validation_report.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    print("=" * 65)
    print("DYNAMIC SENSOR CONFIGURATION TEST")
    print("=" * 65)

    clean_path, dirty_path = create_test_datasets(data_dir)

    print("\nTest datasets created successfully.")

    print("\n1. VALIDATING CLEAN DATASET")

    clean_result = validate_configured_dataset(
        file_path=clean_path,
        config_path=config_path
    )

    print("Valid:", clean_result.get("valid"))
    print("Rows:", clean_result.get(
        "rows",
        clean_result.get("row_count", "Not reported")
    ))

    print("Errors:", clean_result.get("errors", []))
    print("Warnings:", clean_result.get("warnings", []))

    print("\n2. VALIDATING DIRTY DATASET")

    dirty_result = validate_configured_dataset(
        file_path=dirty_path,
        config_path=config_path
    )

    print("Valid:", dirty_result.get("valid"))
    print("Rows:", dirty_result.get(
        "rows",
        dirty_result.get("row_count", "Not reported")
    ))

    print("Errors:", dirty_result.get("errors", []))
    print("Warnings:", dirty_result.get("warnings", []))

    report = {
        "configuration": config_path.name,
        "clean_dataset": clean_result,
        "dirty_dataset": dirty_result
    }

    save_report(report, report_path)

    print("\n3. TEST SUMMARY")

    print(
        "Clean dataset accepted:",
        clean_result.get("valid") is True
    )

    print(
        "Dirty dataset rejected:",
        dirty_result.get("valid") is False
    )

    print("\nReport saved:", report_path)

    if clean_result.get("valid") is not True:
        raise AssertionError(
            "Clean alternative sensor dataset was not accepted."
        )

    if dirty_result.get("valid") is not False:
        raise AssertionError(
            "Dirty alternative sensor dataset was not rejected."
        )

    print("\nDYNAMIC SENSOR CONFIGURATION TEST PASSED!")


if __name__ == "__main__":
    main()
