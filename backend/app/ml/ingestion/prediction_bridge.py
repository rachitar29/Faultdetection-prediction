
"""Connect validated uploaded sensor readings to the unified prediction engine."""

from pathlib import Path

import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.ingestion.ingestion_pipeline import ingest_and_validate
from app.ml.maintenance.integrated_engine import (
    predict_equipment_condition,
)


MODEL_NAME = "XGBoost"


def load_failure_model():
    """Verify the registered XGBoost model and its saved metadata."""

    bundle = load_model_bundle(MODEL_NAME)

    if not isinstance(bundle, dict):
        raise TypeError("Unexpected model bundle format.")

    model = bundle.get("model")
    metadata = bundle.get("metadata", {})

    if model is None:
        raise ValueError("Registered model bundle has no model.")

    feature_names = metadata.get("feature_names")

    if not feature_names:
        raise ValueError(
            "Model metadata does not contain feature_names."
        )

    threshold = float(metadata.get("threshold", 0.5))

    if not 0 < threshold < 1:
        raise ValueError(
            f"Invalid saved decision threshold: {threshold}"
        )

    return model, metadata, feature_names, threshold


def _prepare_sensor_reading(row, feature_names):
    """Extract only the registered model's sensor features."""

    return {
        feature: row[feature]
        for feature in feature_names
    }


def predict_validated_dataframe(dataframe):
    """
    Run the unified prediction engine on every validated reading.

    Includes:
    - XGBoost failure prediction
    - Isolation Forest anomaly detection
    - Local Outlier Factor anomaly detection
    - Combined risk assessment
    - SHAP explanations
    - Maintenance recommendations

    NASA RUL is not calculated here because it requires
    separate NASA C-MAPSS-compatible input.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("No sensor readings were provided.")

    _, metadata, feature_names, threshold = load_failure_model()

    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model input features: {missing_features}"
        )

    prediction_records = []
    integrated_results = []

    # Process each reading through the existing unified engine.
    for row_index, (_, row) in enumerate(dataframe.iterrows()):

        sensor_data = _prepare_sensor_reading(
            row,
            feature_names
        )

        try:
            integrated_result = predict_equipment_condition(
                sensor_data=sensor_data
            )

        except Exception as exc:
            raise RuntimeError(
                f"Prediction failed for input row {row_index}: {exc}"
            ) from exc

        failure_probability = float(
            integrated_result["failure_probability"]
        )

        failure_predicted = bool(
            integrated_result["failure_predicted"]
        )

        anomaly_results = integrated_result["anomaly_results"]

        anomaly_detected = any(
            result["is_anomaly"]
            for result in anomaly_results.values()
        )

        record = {
            "failure_probability": failure_probability,
            "failure_probability_percent": (
                integrated_result["failure_probability_percent"]
            ),
            "failure_prediction": int(failure_predicted),
            "condition": (
                "FAILURE_PREDICTED"
                if failure_predicted
                else "NORMAL"
            ),
            "decision_threshold": (
                integrated_result["decision_threshold"]
            ),
            "classifier_risk_level": (
                integrated_result["classifier_risk_level"]
            ),
            "risk_level": integrated_result["risk_level"],
            "health_score": integrated_result["health_score"],
            "maintenance_priority": (
                integrated_result["maintenance_priority"]
            ),
            "maintenance_recommendation": (
                integrated_result["maintenance_recommendation"]
            ),
            "anomaly_detected": anomaly_detected,
            "anomaly_results": anomaly_results,
            "anomaly_assessment": (
                integrated_result["anomaly_assessment"]
            ),
            "top_factors": integrated_result["top_factors"],
            "explanation_method": (
                integrated_result["explanation_method"]
            ),
            "explained_class": (
                integrated_result["explained_class"]
            ),
            "risk_assessment_method": (
                integrated_result["risk_assessment_method"]
            ),
            "rul_prediction": None,
            "rul_available": False,
            "prediction_timestamp": (
                integrated_result["prediction_timestamp"]
            ),
        }

        prediction_records.append(record)
        integrated_results.append(integrated_result)

    predictions_dataframe = dataframe.copy()

    prediction_columns = pd.DataFrame(
        prediction_records,
        index=dataframe.index
    )

    predictions_dataframe = pd.concat(
        [predictions_dataframe, prediction_columns],
        axis=1
    )

    predicted_failures = int(
        predictions_dataframe["failure_prediction"].sum()
    )

    anomaly_count = int(
        predictions_dataframe["anomaly_detected"].sum()
    )

    return {
        "model_name": MODEL_NAME,
        "model_description": metadata.get("description"),
        "threshold": threshold,
        "total_readings": len(dataframe),
        "predicted_failures": predicted_failures,
        "predicted_normal": len(dataframe) - predicted_failures,
        "readings_with_anomalies": anomaly_count,
        "risk_distribution": (
            predictions_dataframe["risk_level"]
            .value_counts()
            .to_dict()
        ),
        "integrated_prediction": True,
        "prediction_components": {
            "failure_classification": True,
            "isolation_forest": True,
            "local_outlier_factor": True,
            "combined_risk_assessment": True,
            "shap_explanation": True,
            "maintenance_recommendations": True,
            "remaining_useful_life": False,
        },
        "predictions": predictions_dataframe,
        "integrated_results": integrated_results,
    }


def predict_uploaded_file(file_path):
    """
    Complete workflow:
    uploaded file -> ingestion -> validation -> compatibility
    -> reference ranges -> integrated equipment prediction.
    """

    ingestion_result = ingest_and_validate(file_path)

    if not ingestion_result["is_valid"]:
        raise ValueError(
            "Uploaded data failed validation. "
            f"Missing features: {ingestion_result['missing_features']}. "
            f"Validation errors: "
            f"{ingestion_result['validation_report']['errors']}"
        )

    prediction_result = predict_validated_dataframe(
        ingestion_result["dataframe"]
    )

    prediction_result["file_name"] = Path(file_path).name
    prediction_result["ingestion_status"] = (
        ingestion_result["status"]
    )
    prediction_result["column_mapping"] = (
        ingestion_result["column_mapping"]
    )
    prediction_result["validation_report"] = (
        ingestion_result["validation_report"]
    )
    prediction_result["compatibility_report"] = (
        ingestion_result["compatibility_report"]
    )
    prediction_result["reference_range_report"] = (
        ingestion_result["reference_range_report"]
    )
    prediction_result["ingestion_warnings"] = (
        ingestion_result["warnings"]
    )

    return prediction_result


def print_prediction_report(result):
    """Display the integrated equipment prediction report."""

    print("\n" + "=" * 75)
    print("AI PREDICTIVE MAINTENANCE - INTEGRATED EQUIPMENT REPORT")
    print("=" * 75)

    print("File:", result.get("file_name", "DataFrame input"))
    print("Model:", result["model_name"])
    print("XGBoost threshold:", result["threshold"])
    print("Total readings:", result["total_readings"])
    print("Predicted failures:", result["predicted_failures"])
    print("Predicted normal:", result["predicted_normal"])
    print("Readings with anomalies:", result["readings_with_anomalies"])

    print("\nRISK DISTRIBUTION")
    print("-" * 75)

    for risk, count in result["risk_distribution"].items():
        print(f"{risk}: {count}")

    print("\nROW-WISE EQUIPMENT ASSESSMENT")
    print("-" * 75)

    display_columns = [
        "failure_probability_percent",
        "condition",
        "anomaly_detected",
        "classifier_risk_level",
        "risk_level",
        "maintenance_priority",
    ]

    print(
        result["predictions"][display_columns].to_string(
            index=True
        )
    )

    print("\nPREDICTION COMPONENTS")
    print("-" * 75)

    for component, enabled in result["prediction_components"].items():
        print(f"{component}: {enabled}")

    print("\nNASA RUL: Not calculated; separate C-MAPSS input required.")

    print("=" * 75)


if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run integrated predictive maintenance on a sensor CSV."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ml_artifacts/monitoring/ingestion_test.csv"),
        help="Path to the input sensor CSV file."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml_artifacts/reports"),
        help="Directory for prediction CSV and JSON report."
    )

    args = parser.parse_args()

    input_file = args.input

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input CSV file not found: {input_file}"
        )

    print("\\nINPUT FILE:", input_file)

    result = predict_uploaded_file(input_file)

    print_prediction_report(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        args.output_dir / f"{input_file.stem}_predictions.csv"
    )

    json_path = (
        args.output_dir / f"{input_file.stem}_report.json"
    )

    result["predictions"].to_csv(
        csv_path,
        index=False
    )

    summary = {
        key: value
        for key, value in result.items()
        if key not in ("predictions", "integrated_results")
    }

    with json_path.open("w", encoding="utf-8") as report_file:
        json.dump(
            summary,
            report_file,
            indent=4,
            default=str
        )

    print("\\nPREDICTION FILES SAVED")
    print("CSV:", csv_path)
    print("JSON:", json_path)

    print("\\nINTEGRATED PREDICTION COMPLETED")
