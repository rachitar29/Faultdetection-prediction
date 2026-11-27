
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

MODEL_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "model_registry"
    / "xgboost"
    / "model.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "model_registry"
    / "xgboost"
    / "metadata.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "twf_prediction_analysis.json"
)

TARGET = "Machine failure"
TWF = "TWF"

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def main():

    print("\nTWF PREDICTION PROBABILITY ANALYSIS")
    print("=" * 65)

    # Load dataset
    dataframe = pd.read_csv(DATA_PATH)

    # Reproduce the existing independent split
    train_data, test_data = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=dataframe[TARGET],
    )

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET]

    # Select actual TWF failures from the test set
    twf_records = test_data[
        (test_data[TARGET] == 1)
        & (test_data[TWF] == 1)
    ].copy()

    if twf_records.empty:
        raise ValueError("No TWF records found in the test set.")

    # Load registered pipeline and threshold
    registered_model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["threshold"])

    # Train a temporary clone on the reference training split
    evaluation_model = clone(registered_model)
    evaluation_model.fit(X_train, y_train)

    # Calculate failure probabilities for TWF records
    X_twf = twf_records[FEATURE_COLUMNS]

    probabilities = evaluation_model.predict_proba(X_twf)[:, 1]

    results = []

    print(f"\nProduction threshold: {threshold:.6f}")
    print(f"TWF records in test set: {len(twf_records)}")

    print("\nINDIVIDUAL TWF PREDICTIONS")
    print("-" * 90)

    for (index, record), probability in zip(
        twf_records.iterrows(),
        probabilities,
    ):

        predicted = bool(probability >= threshold)

        status = "DETECTED" if predicted else "MISSED"

        distance = threshold - float(probability)

        result = {
            "dataset_index": int(index),
            "UDI": int(record["UDI"]),
            "tool_wear": float(record["Tool wear [min]"]),
            "torque": float(record["Torque [Nm]"]),
            "rotational_speed": float(record["Rotational speed [rpm]"]),
            "machine_type": str(record["Type"]),
            "failure_probability": float(probability),
            "threshold": threshold,
            "prediction": status,
            "distance_below_threshold": (
                float(distance) if not predicted else 0.0
            ),
        }

        results.append(result)

        print(
            f"UDI={result['UDI']:5d} | "
            f"Tool wear={result['tool_wear']:6.1f} | "
            f"Probability={probability:.6f} | "
            f"{status}"
        )

    # Summary
    detected_count = sum(
        item["prediction"] == "DETECTED"
        for item in results
    )

    missed_count = len(results) - detected_count

    results.sort(key=lambda item: item["failure_probability"])

    report = {
        "threshold": threshold,
        "total_twf_records": len(results),
        "detected": detected_count,
        "missed": missed_count,
        "records_sorted_by_probability": results,
        "production_model_modified": False,
        "production_threshold_modified": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nSUMMARY")
    print("-" * 65)
    print("TWF records:", len(results))
    print("Detected:", detected_count)
    print("Missed:", missed_count)

    print("\nREPORT SAVED:")
    print(OUTPUT_PATH)

    print("\nPRODUCTION MODEL MODIFIED: NO")
    print("PRODUCTION THRESHOLD MODIFIED: NO")


if __name__ == "__main__":
    main()
