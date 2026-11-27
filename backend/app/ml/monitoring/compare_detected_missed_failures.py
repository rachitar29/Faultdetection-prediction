
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

REPORT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "xgboost_failure_case_evaluation.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "detected_vs_missed_failure_analysis.json"
)

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def main():

    print("\nDETECTED VS MISSED FAILURE ANALYSIS")
    print("=" * 65)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    dataframe = pd.read_csv(DATA_PATH)

    analysis = report["failure_case_analysis"]

    detected_indices = analysis["detected_failure_indices"]
    missed_indices = analysis["missed_failure_indices"]

    detected = dataframe.loc[detected_indices].copy()
    missed = dataframe.loc[missed_indices].copy()

    print("Detected failures:", len(detected))
    print("Missed failures:", len(missed))

    # Sensor comparison

    print("\nSENSOR COMPARISON")
    print("-" * 65)

    comparison = {}

    for feature in SENSOR_COLUMNS:

        detected_mean = float(detected[feature].mean())
        missed_mean = float(missed[feature].mean())

        detected_median = float(detected[feature].median())
        missed_median = float(missed[feature].median())

        comparison[feature] = {
            "detected_mean": detected_mean,
            "missed_mean": missed_mean,
            "detected_median": detected_median,
            "missed_median": missed_median,
            "mean_difference": missed_mean - detected_mean,
        }

        print(f"\n{feature}")
        print(f"  Detected mean: {detected_mean:.3f}")
        print(f"  Missed mean:   {missed_mean:.3f}")
        print(f"  Mean difference: {missed_mean - detected_mean:.3f}")

    # Machine type comparison

    print("\nMACHINE TYPE COMPARISON")
    print("-" * 65)

    detected_types = detected["Type"].value_counts().to_dict()
    missed_types = missed["Type"].value_counts().to_dict()

    print("Detected:", detected_types)
    print("Missed:", missed_types)

    # Failure mode comparison

    print("\nFAILURE MODE COMPARISON")
    print("-" * 65)

    failure_mode_comparison = {}

    for mode in FAILURE_MODES:

        detected_count = int(detected[mode].sum())
        missed_count = int(missed[mode].sum())

        failure_mode_comparison[mode] = {
            "detected_count": detected_count,
            "missed_count": missed_count,
        }

        print(
            f"{mode}: detected={detected_count}, "
            f"missed={missed_count}"
        )

    # Save report

    result = {
        "detected_failure_count": len(detected),
        "missed_failure_count": len(missed),
        "sensor_comparison": comparison,
        "machine_type_comparison": {
            "detected": detected_types,
            "missed": missed_types,
        },
        "failure_mode_comparison": failure_mode_comparison,
        "production_model_modified": False,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nANALYSIS REPORT SAVED:")
    print(OUTPUT_PATH)

    print("\nPRODUCTION MODEL MODIFIED: NO")
    print("\nCOMPARISON COMPLETED")


if __name__ == "__main__":
    main()
