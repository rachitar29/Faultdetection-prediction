
from pathlib import Path

import pandas as pd

from app.ml.data.dynamic_preprocessor import (
    DynamicDatasetPreprocessor
)


def run_preprocessing_test(
    dataset_name,
    dataset_path,
    config_path,
    artifact_path
):

    print("\n" + "=" * 65)
    print(f"TESTING: {dataset_name}")
    print("=" * 65)

    df = pd.read_csv(dataset_path)

    print("Original shape:", df.shape)

    # Split dataset before fitting transformations.
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    # Initialize preprocessor.
    processor = DynamicDatasetPreprocessor(
        config_path
    )

    # Fit on training data only.
    X_train, y_train = processor.fit_transform(
        train_df
    )

    # Transform test data using fitted pipeline.
    X_test = processor.transform(
        test_df
    )

    print("\nDetected features:")

    information = processor.get_feature_information()

    for key, value in information.items():
        print(f"{key}: {value}")

    print("\nTraining features:", X_train.shape)
    print("Testing features:", X_test.shape)

    print(
        "Training target:",
        None if y_train is None else y_train.shape
    )

    # Check for invalid numeric output.
    if X_train.isna().any().any():
        raise AssertionError(
            "Training data contains missing values after preprocessing."
        )

    if X_test.isna().any().any():
        raise AssertionError(
            "Testing data contains missing values after preprocessing."
        )

    # Save fitted preprocessing artifact.
    processor.save(artifact_path)

    # Reload artifact.
    loaded_processor = DynamicDatasetPreprocessor.load(
        artifact_path
    )

    # Verify identical test transformation.
    reloaded_X_test = loaded_processor.transform(
        test_df
    )

    pd.testing.assert_frame_equal(
        X_test,
        reloaded_X_test
    )

    print("\nArtifact saved:", artifact_path)
    print("Artifact reload test: PASSED")
    print("Preprocessing test: PASSED")

    return True


def main():

    backend_dir = Path(__file__).resolve().parents[3]

    config_dir = (
        backend_dir
        / "app"
        / "ml"
        / "configurations"
        / "dataset_configs"
    )

    data_dir = (
        backend_dir
        / "ml_data"
    )

    artifact_dir = (
        backend_dir
        / "ml_artifacts"
        / "preprocessors"
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    datasets = [
        {
            "name": "AI4I 2020",
            "data": data_dir / "raw" / "ai4i2020.csv",
            "config": config_dir / "ai4i_config.json",
            "artifact": artifact_dir / "ai4i_preprocessor.joblib"
        },
        {
            "name": "Alternative Industrial Sensors",
            "data": (
                data_dir
                / "validity_test"
                / "alternative_sensors"
                / "alternative_sensor_clean.csv"
            ),
            "config": (
                config_dir / "alternative_sensor_config.json"
            ),
            "artifact": (
                artifact_dir
                / "alternative_sensor_preprocessor.joblib"
            )
        }
    ]

    print("=" * 65)
    print("MULTI-DATASET DYNAMIC PREPROCESSING")
    print("=" * 65)

    results = []

    for dataset in datasets:

        for key in ["data", "config"]:

            if not dataset[key].exists():
                raise FileNotFoundError(
                    f"Missing {key} file: {dataset[key]}"
                )

        result = run_preprocessing_test(
            dataset["name"],
            dataset["data"],
            dataset["config"],
            dataset["artifact"]
        )

        results.append(result)

    print("\n" + "=" * 65)
    print("FINAL TEST SUMMARY")
    print("=" * 65)

    print("Datasets tested:", len(results))
    print("Successful:", sum(results))
    print("Failed:", len(results) - sum(results))

    if all(results):
        print("\nALL PREPROCESSING TESTS PASSED!")


if __name__ == "__main__":
    main()
