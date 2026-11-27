
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.evaluation.metrics import calculate_binary_metrics


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "evaluations"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "XGBoost"
TARGET_COLUMN = "Machine failure"

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():

    print("\nAI4I FAILURE-CASE EVALUATION")
    print("=" * 65)

    # 1. Load registered model and promoted threshold

    bundle = load_model_bundle(MODEL_NAME)

    registered_model = bundle["model"]
    metadata = bundle["metadata"]

    threshold = float(metadata["threshold"])
    feature_names = metadata["feature_names"]

    print("Model:", MODEL_NAME)
    print("Promoted threshold:", threshold)

    # 2. Load dataset

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    dataframe = pd.read_csv(DATA_PATH)

    X = dataframe.drop(
        columns=[TARGET_COLUMN] + EXCLUDED_COLUMNS,
        errors="ignore",
    )

    y = dataframe[TARGET_COLUMN].astype(int)

    X = X[feature_names]

    # 3. Recreate independent split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDATASET INFORMATION")
    print("-" * 65)
    print("Reference samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Actual failure cases:", int(y_test.sum()))
    print("Actual normal cases:", int((y_test == 0).sum()))

    # 4. Train evaluation clone only

    evaluation_model = clone(registered_model)

    evaluation_model.fit(X_train, y_train)

    # 5. Predict probabilities

    classes = list(evaluation_model.classes_)

    if 1 not in classes:
        raise ValueError("Failure class 1 not found.")

    positive_index = classes.index(1)

    probabilities = evaluation_model.predict_proba(
        X_test
    )[:, positive_index]

    # 6. Evaluate promoted threshold

    results = calculate_binary_metrics(
        y_true=y_test,
        scores=probabilities,
        threshold=threshold,
        positive_label=1,
        beta=2.0,
    )

    # 7. Identify missed failure records

    predictions = (probabilities >= threshold).astype(int)

    failure_indices = y_test.index[y_test == 1]

    missed_failure_indices = y_test.index[
        (y_test == 1) & (predictions == 0)
    ]

    detected_failure_indices = y_test.index[
        (y_test == 1) & (predictions == 1)
    ]

    missed_failures = [
        int(index) for index in missed_failure_indices
    ]

    detected_failures = [
        int(index) for index in detected_failure_indices
    ]

    report = {
        "report_type": "failure_case_evaluation",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_name": MODEL_NAME,
        "decision_threshold": threshold,

        "dataset": {
            "name": "AI4I 2020",
            "reference_samples": len(X_train),
            "test_samples": len(X_test),
            "actual_failures": int(y_test.sum()),
            "actual_normal": int((y_test == 0).sum()),
        },

        "evaluation_method": {
            "method": "Stratified 80/20 holdout",
            "random_state": RANDOM_STATE,
            "evaluation_model": "Cloned registered pipeline",
            "production_model_modified": False,
        },

        "metrics": results,

        "failure_case_analysis": {
            "actual_failure_count": len(failure_indices),
            "detected_failure_count": len(detected_failures),
            "missed_failure_count": len(missed_failures),
            "detected_failure_indices": detected_failures,
            "missed_failure_indices": missed_failures,
        },
    }

    # 8. Save report

    report_path = (
        REPORT_DIR / "xgboost_failure_case_evaluation.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # 9. Display results

    print("\nFAILURE DETECTION RESULTS")
    print("-" * 65)

    for key, value in results.items():
        print(f"{key}: {value}")

    print("\nFAILURE CASE ANALYSIS")
    print("-" * 65)

    print("Actual failures:", len(failure_indices))
    print("Detected failures:", len(detected_failures))
    print("Missed failures:", len(missed_failures))

    print("\nMISSED FAILURE RECORD INDICES")
    print(missed_failures)

    print("\nREPORT SAVED:")
    print(report_path)

    print("\nPRODUCTION MODEL MODIFIED: NO")
    print("\nFAILURE-CASE EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
