from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.monitoring.unified_monitor import run_unified_monitoring


BACKEND_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BACKEND_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

OUTPUT_DIR = (
    BACKEND_DIR / "ml_artifacts" / "monitoring"
)

TEST_DATA_DIR = OUTPUT_DIR / "unified_test_data"

MODEL_NAME = "XGBoost"

TARGET_COLUMN = "Machine failure"

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

    print("\nAI4I UNIFIED MONITORING TEST")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    print("Dataset:", DATASET_PATH.name)
    print("Total rows:", len(dataframe))

    reference_df, current_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=dataframe[TARGET_COLUMN],
    )

    print("Reference rows:", len(reference_df))
    print("Current rows:", len(current_df))

    TEST_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    reference_path = (
        TEST_DATA_DIR / "ai4i_reference.csv"
    )

    current_path = (
        TEST_DATA_DIR / "ai4i_current.csv"
    )

    reference_df.to_csv(
        reference_path,
        index=False
    )

    current_df.to_csv(
        current_path,
        index=False
    )

    bundle = load_model_bundle(MODEL_NAME)

    model = bundle["model"]
    metadata = bundle["metadata"]

    print("\nMODEL LOADED:", MODEL_NAME)

    X_current = current_df.drop(
        columns=IGNORED_COLUMNS,
        errors="ignore"
    )

    y_true = (
        current_df[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Model does not support predict_proba."
        )

    probabilities = np.asarray(
        model.predict_proba(X_current)
    )

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected binary classification probabilities."
        )

    probabilities = probabilities[:, 1]

    threshold = float(
        metadata.get("threshold", 0.5)
    )

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    print("Threshold:", threshold)
    print("Actual failures:", int(y_true.sum()))
    print("Predicted failures:", int(y_pred.sum()))

    report = run_unified_monitoring(
        reference_dataset=reference_path,
        current_dataset=current_path,
        y_true=y_true,
        y_pred=y_pred,
        y_probability=probabilities,
        model_name=MODEL_NAME,
        baseline_metrics=metadata.get("metrics"),
        ignored_columns=IGNORED_COLUMNS,
        output_dir=OUTPUT_DIR,
    )

    print("\nUNIFIED MONITORING RESULTS")
    print("=" * 60)

    print(
        "Overall status:",
        report.get("overall_status")
    )

    print(
        "Data drift status:",
        report.get("data_drift", {}).get(
            "overall_status",
            "Not available"
        )
    )

    print(
        "Performance evaluated:",
        report.get("model_performance") is not None
    )

    print(
        "Performance comparison:",
        report.get("performance_comparison")
    )

    print("\nGENERATED REPORTS")
    print("=" * 60)

    print("Output directory:", OUTPUT_DIR)

    for file in OUTPUT_DIR.glob("*.json"):
        print(file.name)

    print("\nUNIFIED MONITORING TEST COMPLETED")


if __name__ == "__main__":
    main()
