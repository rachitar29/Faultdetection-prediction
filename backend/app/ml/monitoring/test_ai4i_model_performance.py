
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from app.ml.algorithms.model_registry import load_model_bundle


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "monitoring"
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


def calculate_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def run_ai4i_model_performance_test():

    print("\nAI4I INDEPENDENT MODEL PERFORMANCE EVALUATION")
    print("=" * 65)

    # 1. Load registered model and metadata

    bundle = load_model_bundle(MODEL_NAME)

    registered_model = bundle["model"]
    metadata = bundle["metadata"]

    current_threshold = float(metadata["threshold"])

    promotion_info = metadata.get(
        "threshold_promotion", {}
    )

    original_threshold = float(
        promotion_info.get(
            "previous_threshold",
            current_threshold
        )
    )

    feature_names = metadata.get("feature_names", [])

    print("Registered model:", MODEL_NAME)
    print("Original threshold:", original_threshold)
    print("Promoted threshold:", current_threshold)

    # 2. Load dataset

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=[TARGET_COLUMN] + EXCLUDED_COLUMNS,
        errors="ignore",
    )

    y = df[TARGET_COLUMN].astype(int)

    if not feature_names:
        raise ValueError(
            "Model metadata has no feature names."
        )

    missing = [
        feature for feature in feature_names
        if feature not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Required features missing: {missing}"
        )

    X = X[feature_names]

    # 3. Create independent holdout split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDATASET SPLIT")
    print("-" * 65)
    print("Total records:", len(df))
    print("Reference records:", len(X_train))
    print("Independent test records:", len(X_test))
    print("Test failure records:", int(y_test.sum()))
    print("Test normal records:", int((y_test == 0).sum()))

    # 4. Clone the registered pipeline.
    # The registered production model is NOT modified.

    evaluation_model = clone(registered_model)

    print("\nTraining evaluation clone on reference data...")

    evaluation_model.fit(X_train, y_train)

    # 5. Generate independent test probabilities

    classes = list(evaluation_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Evaluation model does not contain failure class 1."
        )

    positive_index = classes.index(1)

    probabilities = evaluation_model.predict_proba(
        X_test
    )[:, positive_index]

    # 6. Evaluate both thresholds on the same untouched test set

    original_results = calculate_metrics(
        y_test,
        probabilities,
        original_threshold,
    )

    promoted_results = calculate_metrics(
        y_test,
        probabilities,
        current_threshold,
    )

    # 7. Build report

    report = {
        "report_type": "independent_model_performance",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_name": MODEL_NAME,

        "production_model_modified": False,

        "dataset": {
            "name": "AI4I 2020",
            "total_records": len(df),
            "reference_records": len(X_train),
            "independent_test_records": len(X_test),
            "test_failure_records": int(y_test.sum()),
            "test_normal_records": int((y_test == 0).sum()),
        },

        "evaluation_method": {
            "method": "Stratified 80/20 holdout",
            "random_state": RANDOM_STATE,
            "evaluation_model": "Clone of registered pipeline",
            "evaluation_model_trained_only_on_reference_data": True,
            "production_model_retrained": False,
        },

        "threshold_comparison": {
            "original": original_results,
            "promoted": promoted_results,
        },

        "constraint_verification": {
            "promoted_precision_at_least_0_80": (
                promoted_results["precision"] >= 0.80
            ),
        },
    }

    report_path = (
        REPORT_DIR
        / "ai4i_xgboost_independent_performance_report.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # 8. Display results

    print("\nORIGINAL THRESHOLD RESULTS")
    print("-" * 65)

    for key, value in original_results.items():
        print(f"{key}: {value}")

    print("\nPROMOTED THRESHOLD RESULTS")
    print("-" * 65)

    for key, value in promoted_results.items():
        print(f"{key}: {value}")

    print("\nCONSTRAINT VERIFICATION")
    print("-" * 65)

    print(
        "Independent test precision >= 0.80:",
        report["constraint_verification"][
            "promoted_precision_at_least_0_80"
        ],
    )

    print("\nREPORT SAVED:")
    print(report_path)

    print("\nPRODUCTION MODEL MODIFIED: NO")
    print("\nINDEPENDENT PERFORMANCE EVALUATION COMPLETED")

    return report


if __name__ == "__main__":
    run_ai4i_model_performance_test()
