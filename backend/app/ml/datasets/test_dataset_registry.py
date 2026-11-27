
import tempfile
from pathlib import Path

from app.ml.datasets.dataset_registry import DatasetRegistry


def main():

    with tempfile.TemporaryDirectory() as temp_dir:

        registry_file = (
            Path(temp_dir) / "dataset_registry.json"
        )

        registry = DatasetRegistry(
            registry_path=registry_file
        )

        # Register AI4I
        ai4i = registry.register_dataset(
            name="AI4I 2020",
            file_path="datasets/ai4i/ai4i2020.csv",
            dataset_type="ai4i",
            task_type="classification",
            target_column="Machine failure",
            source="UCI Repository"
        )

        assert ai4i["name"] == "AI4I 2020"

        # Register an industrial dataset with additional parameters
        industrial = registry.register_dataset(
            name="Industrial Machine Dataset",
            file_path="datasets/industrial/machine.csv",
            dataset_type="industrial",
            task_type="classification",
            target_column="Machine failure",

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
            ],

            sensor_units={
                "vibration": "mm/s",
                "pressure": "bar",
                "temperature": "C",
                "voltage": "V",
                "current": "A"
            }
        )

        print("\nINDUSTRIAL DATASET CUSTOM PARAMETERS")

        for key, value in industrial["custom_parameters"].items():
            print(f"{key}: {value}")

        # Verify custom parameters
        parameters = industrial["custom_parameters"]

        assert parameters["vibration_sensor"] is True
        assert parameters["pressure_sensor"] is True
        assert parameters["voltage_sensor"] is True

        assert "Bearing failure" in parameters["fault_types"]

        # Update custom parameters
        registry.update_custom_parameters(
            industrial["dataset_id"],
            {
                "humidity_sensor": True,
                "acoustic_sensor": True
            }
        )

        updated = registry.get_dataset(
            industrial["dataset_id"]
        )

        assert updated["custom_parameters"]["humidity_sensor"]
        assert updated["custom_parameters"]["acoustic_sensor"]

        # Verify dataset filtering
        assert len(registry.list_datasets()) == 2

        assert len(
            registry.list_datasets(
                dataset_type="industrial"
            )
        ) == 1

        # Verify retrieval and counting
        assert registry.get_dataset(
            industrial["dataset_id"]
        ) is not None

        assert registry.count() == 2

        print("\nCUSTOM DATASET REGISTRY TEST PASSED!")
        print("AI4I registration: PASS")
        print("Industrial registration: PASS")
        print("Custom sensor parameters: PASS")
        print("Custom fault types: PASS")
        print("Custom parameter updates: PASS")
        print("Dataset filtering: PASS")


if __name__ == "__main__":
    main()
