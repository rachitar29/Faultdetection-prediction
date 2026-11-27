
from pathlib import Path
import pandas as pd

from app.ml.datasets.dataset_manager import DatasetManager


def create_sample_dataset(file_path):
    """Generate a sample industrial sensor dataset."""

    df = pd.DataFrame({
        "Machine ID": [f"M{i:03d}" for i in range(1, 31)],

        "Temperature [C]": [
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
            75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            85, 86, 87, 88, 89, 90, 91, 92, 93, 94
        ],

        "Vibration [mm/s]": [
            1.1, 1.2, 1.1, 1.3, 1.4, 1.2, 1.5, 1.6,
            1.4, 1.3, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2,
            2.5, 2.8, 3.0, 3.1, 3.4, 3.7, 4.0, 4.5,
            4.8, 5.0, 5.2, 5.5, 5.8, 6.0
        ],

        "Pressure [bar]": [
            5.0, 5.1, 5.2, 5.0, 5.1, 5.3, 5.2, 5.1,
            5.0, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            6.8, 6.9, 7.0, 7.1, 7.2, 7.3
        ],

        "Humidity [%]": [
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69
        ],

        "Voltage [V]": [
            220, 221, 219, 222, 220, 221, 223, 220,
            219, 222, 221, 224, 223, 222, 225, 224,
            226, 225, 227, 228, 226, 229, 230, 228,
            231, 232, 230, 233, 234, 235
        ],

        "Current [A]": [
            10, 10.1, 10.2, 10.1, 10.3, 10.2, 10.4,
            10.5, 10.3, 10.2, 10.5, 10.6, 10.7, 10.8,
            10.9, 11, 11.1, 11.2, 11.3, 11.4, 11.5,
            11.6, 11.7, 11.8, 11.9, 12, 12.1, 12.2,
            12.3, 12.4
        ],

        "Machine failure": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
    })

    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)

    return file_path


def main():

    backend_dir = Path(__file__).resolve().parents[3]

    dataset_path = (
        backend_dir / "ml_data" / "industrial" / "machine.csv"
    )

    # Generate sample data if the file doesn't exist.
    if not dataset_path.exists():
        create_sample_dataset(dataset_path)
        print("Sample industrial dataset created.")

    print("\nSource dataset:", dataset_path)

    manager = DatasetManager()

    result = manager.add_dataset(
        file_path=str(dataset_path),
        name="Industrial Machine",
        dataset_type="industrial",
        task_type="classification",
        target_column="Machine failure",
        description="Sample industrial predictive maintenance dataset",
        source="Local generated test dataset",

        exclude_columns=["Machine ID"],

        vibration_sensor=True,
        acoustic_sensor=True,
        humidity_sensor=True,
        pressure_sensor=True,
        temperature_sensor=True,
        voltage_sensor=True,
        current_sensor=True,

        custom_fault_codes=["F01", "F02", "F03"],

        fault_types=[
            "Bearing failure",
            "Overheating",
            "Excessive vibration",
            "Pressure abnormality"
        ]
    )

    if result["duplicate"]:
        print("\nDataset already registered.")
        print("Existing ID:", result["existing_dataset"]["dataset_id"])
        return

    if not result["success"]:
        raise RuntimeError("Dataset import failed.")

    print("\nDATASET IMPORT SUCCESSFUL")
    print("=" * 45)

    print("Dataset ID:", result["dataset"]["dataset_id"])
    print("Stored file:", result["stored_file"])
    print("Profile report:", result["profile_report"])

    print("\nDETECTED NUMERIC FEATURES")

    for feature in result["schema"]["numeric_features"]:
        print("-", feature)

    print("\nREGISTERED CUSTOM PARAMETERS")

    for key, value in result["dataset"]["custom_parameters"].items():
        print(f"{key}: {value}")

    expected_features = [
        "Temperature [C]",
        "Vibration [mm/s]",
        "Pressure [bar]",
        "Humidity [%]",
        "Voltage [V]",
        "Current [A]"
    ]

    for feature in expected_features:
        assert feature in result["schema"]["numeric_features"], (
            f"Missing feature: {feature}"
        )

    assert result["success"] is True

    loaded_df = manager.get_dataset_dataframe(
        result["dataset"]["dataset_id"]
    )

    assert len(loaded_df) == 30

    print("\nCUSTOM INDUSTRIAL IMPORT TEST PASSED!")


if __name__ == "__main__":
    main()
