
"""
Balanced XGBoost threshold optimization evaluation.

Strategy:
- Minimum precision: 0.80
- Optimize F1-score
- No hard minimum recall constraint
- Stratified 5-fold cross-validation
- Independent untouched test evaluation

The registered production model is never modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from app.ml.algorithms.model_registry import load_model_bundle

from app.ml.monitoring.threshold_optimizer import (
    optimize_threshold_cv,
    calculate_threshold_metrics,
)


BACKEND_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BACKEND_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

MODEL_NAME = "XGBoost"

TARGET_COLUMN = "Machine failure"

IGNORED_COLUMNS = [
    "UDI",
    "Product ID",
    TARGET_COLUMN,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

MINIMUM_PRECISION = 0.80
MINIMUM_RECALL = None

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5


def main():

    print("\nBALANCED XGBOOST THRESHOLD OPTIMIZATION")
    print("=" * 65)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    reference_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=dataframe[TARGET_COLUMN],
    )

    X_reference = reference_df.drop(
        columns=IGNORED_COLUMNS,
        errors="ignore",
    )

    y_reference = (
        reference_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    X_test = test_df.drop(
        columns=IGNORED_COLUMNS,
        errors="ignore",
    )

    y_test = (
        test_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    bundle = load_model_bundle(MODEL_NAME)

    original_model = bundle["model"]
    metadata = bundle["metadata"]

    original_threshold = float(
        metadata["threshold"]
    )

    print("Model:", MODEL_NAME)
    print("Reference samples:", len(X_reference))
    print("Test samples:", len(X_test))
    print("Original threshold:", original_threshold)

    print("\nOPTIMIZATION SETTINGS")
    print("=" * 65)
    print("Minimum precision:", MINIMUM_PRECISION)
    print("Minimum recall:", MINIMUM_RECALL)
    print("Objective: Maximum F1-score")
    print("Cross-validation folds:", CV_FOLDS)

    # --------------------------------------------------
    # STEP 1: CROSS-VALIDATED THRESHOLD OPTIMIZATION
    # --------------------------------------------------

    optimization = optimize_threshold_cv(
        estimator=original_model,
        X=X_reference,
        y=y_reference,
        minimum_precision=MINIMUM_PRECISION,
        minimum_recall=MINIMUM_RECALL,
        n_splits=CV_FOLDS,
        random_state=RANDOM_STATE,
    )

    optimized_threshold = float(
        optimization["threshold"]
    )

    print("\nTHRESHOLD OPTIMIZATION RESULTS")
    print("=" * 65)

    print("Original threshold:", original_threshold)
    print("Optimized threshold:", optimized_threshold)

    print(
        "Precision constraint satisfied:",
        optimization["constraints_met"],
    )

    print("Cross-validation metrics:")
    for metric, value in optimization["metrics"].items():
        print(f"{metric}: {value}")

    # --------------------------------------------------
    # STEP 2: REFIT A FRESH MODEL ON REFERENCE DATA
    # --------------------------------------------------

    evaluation_model = clone(original_model)

    evaluation_model.fit(
        X_reference,
        y_reference,
    )

    # --------------------------------------------------
    # STEP 3: PREDICT UNTOUCHED TEST DATA
    # --------------------------------------------------

    probabilities = np.asarray(
        evaluation_model.predict_proba(X_test)
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected two-dimensional class probabilities."
        )

    classes = list(evaluation_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "Failure class 1 was not found."
        )

    positive_index = classes.index(1)

    failure_probabilities = probabilities[
        :, positive_index
    ]

    # --------------------------------------------------
    # STEP 4: EVALUATE BOTH THRESHOLDS
    # --------------------------------------------------

    original_metrics = calculate_threshold_metrics(
        y_true=y_test,
        probabilities=failure_probabilities,
        threshold=original_threshold,
    )

    optimized_metrics = calculate_threshold_metrics(
        y_true=y_test,
        probabilities=failure_probabilities,
        threshold=optimized_threshold,
    )

    print("\nINDEPENDENT TEST RESULTS")
    print("=" * 65)

    print("\nOriginal threshold:")
    for metric, value in original_metrics.items():
        print(f"{metric}: {value}")

    print("\nOptimized threshold:")
    for metric, value in optimized_metrics.items():
        print(f"{metric}: {value}")

    # --------------------------------------------------
    # STEP 5: CONFUSION-RELATED COUNTS
    # --------------------------------------------------

    original_predictions = (
        failure_probabilities >= original_threshold
    ).astype(int)

    optimized_predictions = (
        failure_probabilities >= optimized_threshold
    ).astype(int)

    original_false_negatives = int(
        np.sum(
            (y_test == 1)
            & (original_predictions == 0)
        )
    )

    optimized_false_negatives = int(
        np.sum(
            (y_test == 1)
            & (optimized_predictions == 0)
        )
    )

    original_false_positives = int(
        np.sum(
            (y_test == 0)
            & (original_predictions == 1)
        )
    )

    optimized_false_positives = int(
        np.sum(
            (y_test == 0)
            & (optimized_predictions == 1)
        )
    )

    print("\nFAILURE DETECTION COMPARISON")
    print("=" * 65)

    print(
        "Original false negatives:",
        original_false_negatives,
    )

    print(
        "Optimized false negatives:",
        optimized_false_negatives,
    )

    print(
        "Original false positives:",
        original_false_positives,
    )

    print(
        "Optimized false positives:",
        optimized_false_positives,
    )

    print("\nCONSTRAINT VERIFICATION")
    print("=" * 65)

    print(
        "Cross-validation precision constraint:",
        optimization["constraints_met"],
    )

    print(
        "Independent test precision >= 80%:",
        optimized_metrics["precision"] >= MINIMUM_PRECISION,
    )

    print(
        "Independent test recall:",
        optimized_metrics["recall"],
    )

    print(
        "Independent test F1:",
        optimized_metrics["f1"],
    )

    print("\nEVALUATION COMPLETED")
    print(
        "Production model and registered threshold were not modified."
    )


if __name__ == "__main__":
    main()
