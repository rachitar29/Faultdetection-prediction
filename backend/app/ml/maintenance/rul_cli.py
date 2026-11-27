"""
NASA C-MAPSS FD001 RUL Prediction CLI.

Loads the registered RUL regression model.
Does not retrain any model.

Features:
- Reads NASA FD001 test data.
- Selects the latest cycle for every engine.
- Predicts remaining useful life.
- Generates maintenance categories and recommendations.
- Exports predictions to CSV and JSON.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from app.ml.maintenance.rul_prediction import (
    load_rul_model,
    predict_remaining_useful_life,
)


BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = (
    BASE_DIR
    / "ml_data"
    / "raw"
    / "CMAPSSData"
    / "test_FD001.txt"
)

DEFAULT_OUTPUT = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
)

MODEL_NAME = "rul_cmapss_fd001_best"

NASA_FEATURES = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

NASA_COLUMNS = ["engine_id", "cycle"] + NASA_FEATURES


def load_nasa_test_data(file_path):
    """Load and validate NASA C-MAPSS test data."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"NASA dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
    )

    if dataframe.shape[1] != 26:
        raise ValueError(
            "Invalid NASA dataset format. "
            f"Expected 26 columns, got {dataframe.shape[1]}."
        )

    dataframe.columns = NASA_COLUMNS

    if dataframe.empty:
        raise ValueError("NASA dataset is empty.")

    if dataframe.isnull().any().any():
        raise ValueError(
            "NASA dataset contains missing values."
        )

    dataframe["engine_id"] = pd.to_numeric(
        dataframe["engine_id"],
        errors="raise"
    )

    dataframe["cycle"] = pd.to_numeric(
        dataframe["cycle"],
        errors="raise"
    )

    return dataframe


def prepare_latest_engine_readings(dataframe, feature_names):
    """Select the latest sensor reading for every engine."""

    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    latest_readings = (
        dataframe
        .sort_values(["engine_id", "cycle"])
        .drop_duplicates(
            subset=["engine_id"],
            keep="last"
        )
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    sensor_dataframe = latest_readings[
        feature_names
    ].copy()

    return latest_readings, sensor_dataframe


def run_prediction(input_file, output_dir):
    """Execute NASA RUL prediction and export reports."""

    print("\nNASA C-MAPSS RUL PREDICTION")
    print("=" * 65)

    print("\nInput file:", input_file)

    print("\nLoading registered RUL model...")

    _, metadata = load_rul_model(MODEL_NAME)

    feature_names = metadata["feature_names"]

    print("Model:", MODEL_NAME)
    print("Expected features:", len(feature_names))

    print("\nLoading NASA dataset...")

    dataframe = load_nasa_test_data(input_file)

    print("Total readings:", len(dataframe))
    print("Total engines:", dataframe["engine_id"].nunique())

    latest_readings, sensor_dataframe = (
        prepare_latest_engine_readings(
            dataframe,
            feature_names
        )
    )

    print(
        "\nSelected latest readings:",
        len(latest_readings)
    )

    print("\nRunning RUL prediction...")

    result = predict_remaining_useful_life(
        sensor_dataframe,
        model_name=MODEL_NAME,
    )

    predictions = pd.DataFrame(
        result["predictions"]
    )

    predictions.insert(
        0,
        "engine_id",
        latest_readings["engine_id"].astype(int).to_numpy()
    )

    predictions.insert(
        1,
        "latest_cycle",
        latest_readings["cycle"].astype(int).to_numpy()
    )

    predictions = predictions.rename(
        columns={
            "remaining_useful_life": "predicted_rul_cycles",
            "rul_category": "risk_category",
        }
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir / "nasa_fd001_rul_predictions.csv"
    )

    json_path = (
        output_dir / "nasa_fd001_rul_report.json"
    )

    predictions.to_csv(
        csv_path,
        index=False
    )

    summary = {
        "model_name": result["model_name"],
        "model_type": result["model_type"],
        "dataset": "NASA C-MAPSS FD001",
        "input_file": str(input_file),
        "total_engines": len(predictions),
        "unit": "cycles",
        "risk_distribution": (
            predictions["risk_category"]
            .value_counts()
            .to_dict()
        ),
        "prediction_timestamp": result[
            "prediction_timestamp"
        ],
        "predictions": predictions.to_dict(
            orient="records"
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as report_file:

        json.dump(
            summary,
            report_file,
            indent=4,
            default=str
        )

    print("\nNASA RUL PREDICTION REPORT")
    print("=" * 65)

    print("Total engines:", len(predictions))

    print("\nRUL CATEGORY DISTRIBUTION")

    for category, count in summary["risk_distribution"].items():
        print(f"{category}: {count}")

    print("\nENGINE-WISE PREDICTIONS")
    print("-" * 65)

    display_columns = [
        "engine_id",
        "latest_cycle",
        "predicted_rul_cycles",
        "risk_category",
        "maintenance_priority",
    ]

    print(
        predictions[display_columns].to_string(
            index=False
        )
    )

    print("\nFILES SAVED")
    print("CSV:", csv_path)
    print("JSON:", json_path)

    print("\nNASA RUL PREDICTION COMPLETED")

    return summary


def main():

    parser = argparse.ArgumentParser(
        description="NASA C-MAPSS RUL prediction CLI."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to NASA FD001 test dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for prediction reports."
    )

    args = parser.parse_args()

    run_prediction(
        input_file=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
