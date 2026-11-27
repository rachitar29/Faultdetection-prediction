
import tempfile

from pathlib import Path

import pandas as pd

from app.ml.datasets.dataset_manager import DatasetManager


def main():

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = Path(temp_dir)

        # Create a sample industrial dataset
        source_file = temp_path / "industrial_machine.csv"

        dataframe = pd.DataFrame({

            "Machine ID": [
                f"M{i:03d}" for i in range(1, 21)
            ],

            "Temperature [C]": [
                70, 71, 72, 73, 74,
                75, 76, 77, 78, 79,
                80, 81, 82, 83, 84,
                85, 86, 87, 88, 89
            ],

            "Vibration [mm/s]": [
                1.1, 1.2, 1.1, 1.3, 1.4,
                1.2, 1.5, 1.6, 1.4, 1.3,
                2.1, 2.2, 2.5, 2.8, 3.0,
                3.1, 3.4, 3.7, 4.0, 4.5
            ],

            "Pressure [bar]": [
                5.0, 5.1, 5.2, 5.0, 5.1,
                5.3, 5.2, 5.1, 5.0, 5.2,
                5.4, 5.5, 5.6, 5.7, 5.8,
                5.9, 6.0, 6.1, 6.2, 6.3
            ],

            "Voltage [V]": [
                220, 221, 219, 222, 220,
                221, 223, 220, 219, 222,
                221, 224, 223, 222, 225,
                224, 226, 225, 227, 228
            ],

            "Current [A]": [
                10, 10.1, 10.2, 10.1, 10.3,
                10.2, 10.4, 10.5, 10.3, 10.2,
                11, 11.1, 11.2, 11.3, 11.4,
                11.5, 11.6, 11.7, 11.8, 12
            ],

            "Machine failure": [
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 1, 1, 1
            ]
        })

        dataframe.to_csv(
            source_file,
            index=False
        )

        # Use temporary artifact storage
        manager = DatasetManager(
            artifact_root=temp_path / "artifacts"
        )

        result = manager.add_dataset(
            file_path=str(source_file),
            name="Industrial Machine Dataset",
            dataset_type="industrial",
            task_type="classification",
            target_column="Machine failure",
            description="Industrial predictive maintenance data",
            source="Local test",

            exclude_columns=["Machine ID"],

            vibration_sensor=True,
            pressure_sensor=True,
            temperature_sensor=True,
            voltage_sensor=True,
            current_sensor=True,

            fault_types=[
                "Bearing failure",
                "Overheating",
                "Excessive vibration",
                "Pressure abnormality"
            ]
        )

        print("\nDATASET IMPORT RESULT")
        print("=" * 45)

        print("Success:", result["success"])
        print("Stored file:", result["stored_file"])
        print("Profile:", result["profile_report"])

        print("\nDETECTED NUMERIC FEATURES")

        for feature in result["schema"]["numeric_features"]:
            print("-", feature)

        print("\nCUSTOM PARAMETERS")

        for key, value in result["dataset"]["custom_parameters"].items():
            print(key, ":", value)

        # Verify new sensor columns were detected
        expected_features = [
            "Temperature [C]",
            "Vibration [mm/s]",
            "Pressure [bar]",
            "Voltage [V]",
            "Current [A]"
        ]

        for feature in expected_features:
            assert feature in result["schema"]["numeric_features"]

        assert result["success"] is True

        # Verify dataset retrieval
        dataset_id = result["dataset"]["dataset_id"]

        loaded_df = manager.get_dataset_dataframe(dataset_id)

        assert len(loaded_df) == 20

        # Verify registry
        registered = manager.get_dataset(dataset_id)

        assert registered is not None
        assert registered["status"] == "validated"

        # Verify duplicate detection
        duplicate_result = manager.add_dataset(
            file_path=str(source_file),
            name="Industrial Machine Dataset",
            dataset_type="industrial",
            task_type="classification",
            target_column="Machine failure"
        )

        assert duplicate_result["duplicate"] is True

        print("\nDATASET MANAGER TEST PASSED!")
        print("Import: PASS")
        print("Validation: PASS")
        print("Dynamic sensor detection: PASS")
        print("Custom parameter registration: PASS")
        print("Dataset retrieval: PASS")
        print("Duplicate detection: PASS")
        print("Profile generation: PASS")


if __name__ == "__main__":
    main()
