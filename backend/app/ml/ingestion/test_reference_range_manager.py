
import json
import tempfile
from pathlib import Path

import pandas as pd

from app.ml.ingestion.reference_range_manager import (
    calculate_healthy_reference_ranges,
    save_reference_ranges,
    load_reference_ranges,
    evaluate_sensor_readings,
    DEFAULT_CONFIG_PATH,
    BACKEND_ROOT,
)


def main():

    dataset_path = (
        BACKEND_ROOT
        / "ml_data"
        / "raw"
        / "ai4i2020.csv"
    )

    print("\nTEST 1: Load AI4I dataset")

    assert dataset_path.exists(), (
        f"Dataset not found: {dataset_path}"
    )

    df = pd.read_csv(dataset_path)

    assert len(df) == 10000

    print("Dataset rows:", len(df))
    print("PASS")

    print("\nTEST 2: Calculate healthy reference ranges")

    reference = calculate_healthy_reference_ranges(df)

    assert reference["healthy_record_count"] == 9661
    assert len(reference["sensors"]) == 5

    assert reference["reference_type"] == (
        "healthy_data_statistical_reference"
    )

    assert reference["engineering_limits_available"] is False

    print("Healthy records:", reference["healthy_record_count"])
    print("Sensors:", len(reference["sensors"]))
    print("PASS")

    print("\nTEST 3: Verify sensor units and boundaries")

    expected_units = {
        "Air temperature [K]": "Kelvin",
        "Process temperature [K]": "Kelvin",
        "Rotational speed [rpm]": "RPM",
        "Torque [Nm]": "Nm",
        "Tool wear [min]": "minutes",
    }

    for sensor, expected_unit in expected_units.items():

        details = reference["sensors"][sensor]

        assert details["unit"] == expected_unit

        assert (
            details["statistical_lower_bound"]
            <= details["statistical_upper_bound"]
        )

        print(
            sensor,
            "->",
            details["unit"],
            "| Range:",
            round(details["statistical_lower_bound"], 3),
            "-",
            round(details["statistical_upper_bound"], 3),
        )

    print("PASS")

    print("\nTEST 4: Save and reload reference ranges")

    with tempfile.TemporaryDirectory() as temp_dir:

        output_path = (
            Path(temp_dir) / "reference_ranges.json"
        )

        saved_path = save_reference_ranges(
            reference,
            output_path
        )

        assert saved_path.exists()

        loaded_reference = load_reference_ranges(
            saved_path
        )

        assert (
            loaded_reference["healthy_record_count"]
            == 9661
        )

        assert (
            loaded_reference["sensors"]
            == reference["sensors"]
        )

        print("JSON saved and reloaded successfully.")
        print("PASS")

    print("\nTEST 5: Evaluate normal sensor readings")

    normal_data = df[
        df["Machine failure"] == 0
    ].head(10).copy()

    evaluation = evaluate_sensor_readings(
        normal_data,
        reference
    )

    assert evaluation["total_rows"] == 10

    assert len(evaluation["sensor_results"]) == 5

    print("Rows evaluated:", evaluation["total_rows"])
    print("Sensors evaluated:", len(evaluation["sensor_results"]))
    print("PASS")

    print("\nTEST 6: Detect unusual sensor readings")

    unusual_data = normal_data.head(2).copy()

    unusual_data.loc[
        unusual_data.index[0],
        "Torque [Nm]"
    ] = 9999

    evaluation = evaluate_sensor_readings(
        unusual_data,
        reference
    )

    torque_result = evaluation["sensor_results"][
        "Torque [Nm]"
    ]

    assert torque_result["outside_reference_count"] >= 1

    print(
        "Unusual torque readings:",
        torque_result["outside_reference_count"]
    )

    print("PASS")

    print("\nTEST 7: Verify input data is not modified")

    original = normal_data.copy(deep=True)

    calculate_healthy_reference_ranges(original)

    pd.testing.assert_frame_equal(
        original,
        normal_data
    )

    print("Original dataframe unchanged.")
    print("PASS")

    print("\n" + "=" * 50)
    print("ALL REFERENCE RANGE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
