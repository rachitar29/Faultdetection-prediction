
"""Test data drift detection using the real AI4I 2020 dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.monitoring.data_drift import DataDriftDetector


BACKEND_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BACKEND_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

REPORT_PATH = (
    BACKEND_DIR
    / "ml_artifacts"
    / "monitoring"
    / "ai4i_drift_baseline_report.json"
)

IGNORED_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def main():
    print("\nAI4I REAL DATA DRIFT TEST")
    print("=" * 55)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    print("Dataset:", DATASET_PATH.name)
    print("Total rows:", len(dataframe))
    print("Total columns:", len(dataframe.columns))

    target_column = "Machine failure"

    if target_column in dataframe.columns:
        stratify_values = dataframe[target_column]
    else:
        stratify_values = None

    reference_df, current_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=stratify_values,
    )

    print("\nREFERENCE DATA")
    print("Rows:", len(reference_df))

    print("\nINCOMING TEST BATCH")
    print("Rows:", len(current_df))

    detector = DataDriftDetector(
        ignored_columns=IGNORED_COLUMNS,
        psi_warning=0.10,
        psi_critical=0.25,
    )

    report = detector.compare(
        reference_df,
        current_df,
    )

    report["dataset_name"] = "AI4I 2020"
    report["test_type"] = (
        "Random holdout baseline comparison"
    )

    saved_path = detector.save_report(
        report,
        REPORT_PATH,
    )

    print("\nDRIFT ANALYSIS RESULTS")
    print("=" * 55)

    print("Overall status:", report["overall_status"])

    print(
        "Features compared:",
        report["summary"]["features_compared"],
    )

    print(
        "Features with drift:",
        report["summary"]["features_with_drift"],
    )

    print(
        "Drifted features:",
        report["drifted_features"],
    )

    print("\nINDIVIDUAL FEATURE RESULTS")
    print("-" * 55)

    for feature, result in report["feature_reports"].items():
        print(
            f"{feature}: "
            f"PSI={result.get('psi')} | "
            f"Status={result['status']}"
        )

    print("\nREPORT SAVED:")
    print(saved_path)

    print("\nAI4I REAL DATA DRIFT TEST COMPLETED")


if __name__ == "__main__":
    main()
