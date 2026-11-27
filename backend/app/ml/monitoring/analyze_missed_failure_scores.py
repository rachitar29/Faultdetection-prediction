
import json
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split

from app.ml.algorithms.model_registry import load_model_bundle


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

REPORT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "xgboost_failure_case_evaluation.json"
)

TARGET = "Machine failure"

EXCLUDED = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():

    print("\nMISSED FAILURE PROBABILITY ANALYSIS")
    print("=" * 65)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    bundle = load_model_bundle("XGBoost")

    model = clone(bundle["model"])
    metadata = bundle["metadata"]

    threshold = float(metadata["threshold"])
    features = metadata["feature_names"]

    dataframe = pd.read_csv(DATA_PATH)

    X = dataframe.drop(
        columns=[TARGET] + EXCLUDED,
        errors="ignore",
    )[features]

    y = dataframe[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)

    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        X_test
    )[:, positive_index]

    test_results = pd.DataFrame({
        "original_index": X_test.index,
        "actual_failure": y_test.values,
        "failure_probability": probabilities,
    })

    test_results["predicted_failure"] = (
        test_results["failure_probability"] >= threshold
    )

    missed_indices = report[
        "failure_case_analysis"
    ]["missed_failure_indices"]

    missed = test_results[
        test_results["original_index"].isin(missed_indices)
    ].copy()

    missed = missed.sort_values("failure_probability")

    missed["distance_below_threshold"] = (
        threshold - missed["failure_probability"]
    )

    print("Promoted threshold:", threshold)
    print("Missed failures:", len(missed))

    print("\nMISSED FAILURE PROBABILITIES")
    print("-" * 65)

    print(
        missed[
            [
                "original_index",
                "failure_probability",
                "distance_below_threshold",
            ]
        ].to_string(index=False)
    )

    print("\nPROBABILITY DISTRIBUTION")
    print("-" * 65)

    print(
        missed["failure_probability"].describe().round(5)
    )

    print("\nMISSED FAILURES BY PROBABILITY RANGE")
    print("-" * 65)

    ranges = [
        (0.0, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, threshold),
    ]

    for lower, upper in ranges:

        count = (
            (missed["failure_probability"] >= lower)
            & (missed["failure_probability"] < upper)
        ).sum()

        print(
            f"{lower:.2f} - {upper:.3f}: {count}"
        )

    print("\nANALYSIS COMPLETED")
    print("Production model modified: NO")


if __name__ == "__main__":
    main()
