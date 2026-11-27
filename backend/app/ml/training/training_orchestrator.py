
import json
from pathlib import Path

from app.ml.training.training_pipeline import train_dataset


def train_multiple_datasets(
    dataset_specs,
    output_path=None,
    test_size=0.20,
    random_state=42,
    tasks=("classification", "anomaly_detection"),
    save_models=True
):
    """
    Train configured datasets independently.

    Each dataset specification must contain:
        name
        dataset_path
        config_path
    """

    if not dataset_specs:
        raise ValueError(
            "At least one dataset specification is required."
        )

    results = []

    for specification in dataset_specs:

        required = [
            "name",
            "dataset_path",
            "config_path"
        ]

        missing = [
            key for key in required
            if key not in specification
        ]

        if missing:
            raise ValueError(
                f"Dataset specification missing: {missing}"
            )

        print("\n" + "=" * 65)
        print(
            "STARTING DATASET:",
            specification["name"]
        )
        print("=" * 65)

        try:

            result = train_dataset(
                dataset_path=specification["dataset_path"],
                config_path=specification["config_path"],
                dataset_name=specification["name"],
                test_size=test_size,
                random_state=random_state,
                tasks=tasks,
                save_models=save_models
            )

            result["status"] = (
                "completed"
                if result["successful_models"] > 0
                else "failed"
            )

            results.append(result)

        except Exception as exc:

            results.append({
                "dataset_name": specification["name"],
                "status": "failed",
                "error": str(exc),
                "successful_models": 0
            })

    total_models = sum(
        result.get("successful_models", 0)
        for result in results
    )

    failed_datasets = sum(
        result["status"] == "failed"
        for result in results
    )

    report = {
        "datasets_processed": len(results),
        "total_models_trained": total_models,
        "failed_datasets": failed_datasets,
        "results": results
    }

    if output_path is None:

        backend_dir = Path(__file__).resolve().parents[3]

        output_path = (
            backend_dir
            / "ml_artifacts"
            / "reports"
            / "multi_dataset_training_report.json"
        )

    output_path = Path(output_path)

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

    print("\n" + "=" * 65)
    print("MULTI-DATASET TRAINING SUMMARY")
    print("=" * 65)

    print("Datasets processed:", len(results))
    print("Models trained:", total_models)
    print("Failed datasets:", failed_datasets)
    print("Report:", output_path)

    return report


if __name__ == "__main__":

    backend_dir = Path(__file__).resolve().parents[3]

    config_dir = (
        backend_dir
        / "app"
        / "ml"
        / "configurations"
        / "dataset_configs"
    )

    data_dir = backend_dir / "ml_data"

    dataset_specs = [
        {
            "name": "AI4I",
            "dataset_path": (
                data_dir / "raw" / "ai4i2020.csv"
            ),
            "config_path": (
                config_dir / "ai4i_config.json"
            )
        },
        {
            "name": "Alternative Industrial Sensors",
            "dataset_path": (
                data_dir
                / "validity_test"
                / "alternative_sensors"
                / "alternative_sensor_clean.csv"
            ),
            "config_path": (
                config_dir / "alternative_sensor_config.json"
            )
        }
    ]

    train_multiple_datasets(
        dataset_specs=dataset_specs,
        test_size=0.20,
        random_state=42,
        tasks=("classification", "anomaly_detection"),
        save_models=True
    )
