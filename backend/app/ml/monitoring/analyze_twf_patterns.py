
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH = BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"

OUTPUT_PATH = (
    BASE_DIR
    / "ml_artifacts"
    / "evaluations"
    / "twf_pattern_analysis.json"
)

SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_COLUMN = "Machine failure"
TWF_COLUMN = "TWF"


def summarize_group(data):
    summary = {
        "record_count": int(len(data)),
        "machine_types": {
            str(key): int(value)
            for key, value in data["Type"].value_counts().items()
        },
        "sensors": {},
    }

    for column in SENSOR_COLUMNS:
        values = data[column]

        summary["sensors"][column] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "max": float(values.max()),
        }

    return summary


def main():
    print("\nTWF FAILURE PATTERN ANALYSIS")
    print("=" * 65)

    dataframe = pd.read_csv(DATA_PATH)

    # Reproduce the existing independent test split.
    _, test_data = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=42,
        stratify=dataframe[FAILURE_COLUMN],
    )

    # Three mutually exclusive groups.
    twf = test_data[
        (test_data[FAILURE_COLUMN] == 1)
        & (test_data[TWF_COLUMN] == 1)
    ]

    other_failures = test_data[
        (test_data[FAILURE_COLUMN] == 1)
        & (test_data[TWF_COLUMN] == 0)
    ]

    normal = test_data[
        test_data[FAILURE_COLUMN] == 0
    ]

    groups = {
        "twf_failures": twf,
        "other_failures_without_twf": other_failures,
        "normal_machines": normal,
    }

    results = {}

    for group_name, group_data in groups.items():
        print(f"\n{group_name.upper()}")
        print("-" * 65)

        results[group_name] = summarize_group(group_data)

        print("Records:", len(group_data))
        print("Machine types:", group_data["Type"].value_counts().to_dict())

        for column in SENSOR_COLUMNS:
            values = group_data[column]

            print(
                f"{column}: "
                f"mean={values.mean():.2f}, "
                f"median={values.median():.2f}, "
                f"range={values.min():.2f}-{values.max():.2f}"
            )

    result = {
        "test_split_size": int(len(test_data)),
        "split_random_state": 42,
        "groups": results,
        "production_model_modified": False,
        "threshold_modified": False,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nREPORT SAVED:")
    print(OUTPUT_PATH)
    print("\nPRODUCTION MODEL MODIFIED: NO")
    print("THRESHOLD MODIFIED: NO")
    print("\nANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
