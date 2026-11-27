from pathlib import Path
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


def main():

    print("\nMISSED FAILURE PATTERN ANALYSIS")
    print("=" * 65)

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    missed_indices = report[
        "failure_case_analysis"
    ]["missed_failure_indices"]

    dataframe = pd.read_csv(DATA_PATH)

    missed = dataframe.loc[missed_indices].copy()

    if missed.empty:
        print("No missed failure records found.")
        return

    sensor_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    print("Missed failure records:", len(missed))

    print("\nMISSED FAILURE SENSOR STATISTICS")
    print("-" * 65)

    print(
        missed[sensor_columns].describe().round(3).to_string()
    )

    print("\nMISSED FAILURE MACHINE TYPE DISTRIBUTION")
    print("-" * 65)

    print(
        missed["Type"].value_counts().to_string()
    )

    print("\nFAILURE MODE DISTRIBUTION")
    print("-" * 65)

    failure_modes = [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    for column in failure_modes:
        if column in missed.columns:
            print(
                f"{column}: {int(missed[column].sum())}"
            )

    print("\nMISSED FAILURE RECORDS")
    print("-" * 65)

    print(
        missed[
            ["UDI"] + sensor_columns + ["Type"]
        ].to_string(index=False)
    )

    print("\nANALYSIS COMPLETED")
    print(
        "No model or production metadata was modified."
    )


if __name__ == "__main__":
    main()
