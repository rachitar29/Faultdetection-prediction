
import pandas as pd

from app.ml.datasets.compatibility_checker import (
    check_dataset_compatibility,
)


def create_valid_dataset():
    return pd.DataFrame({
        "Air temperature [K]": [300.0, 301.0, 299.0],
        "Process temperature [K]": [310.0, 311.0, 309.0],
        "Rotational speed [rpm]": [1500, 1600, 1400],
        "Torque [Nm]": [40.0, 42.0, 38.0],
        "Tool wear [min]": [100, 120, 80],
        "Type": ["L", "M", "H"],
    })


def main():

    print("\nTEST 1: Valid AI4I dataset")

    valid_df = create_valid_dataset()
    result = check_dataset_compatibility(valid_df)

    print("Compatible:", result["compatible"])
    print("Errors:", result["errors"])

    assert result["compatible"] is True
    assert len(result["errors"]) == 0

    print("PASS")

    print("\nTEST 2: Missing required feature")

    missing_df = valid_df.drop(
        columns=["Torque [Nm]"]
    )

    result = check_dataset_compatibility(missing_df)

    assert result["compatible"] is False
    assert "Torque [Nm]" in result["missing_features"]

    print("PASS")

    print("\nTEST 3: Invalid numeric value")

    invalid_df = valid_df.copy()

    # Convert column to object before inserting invalid text
    invalid_df["Rotational speed [rpm]"] = (
        invalid_df["Rotational speed [rpm]"].astype(object)
    )

    invalid_df.loc[0, "Rotational speed [rpm]"] = "invalid"

    result = check_dataset_compatibility(invalid_df)

    assert result["compatible"] is False
    assert any(
        "Rotational speed [rpm]" in error
        for error in result["errors"]
    )

    print("PASS")

    print("\nTEST 4: Unknown machine category")

    category_df = valid_df.copy()
    category_df.loc[0, "Type"] = "X"

    result = check_dataset_compatibility(category_df)

    assert result["compatible"] is False
    assert "X" in result["categorical_checks"]["Type"][
        "unknown_categories"
    ]

    print("PASS")

    print("\nTEST 5: Failure-mode leakage columns")

    leakage_df = valid_df.copy()
    leakage_df["TWF"] = [0, 1, 0]
    leakage_df["HDF"] = [0, 0, 0]

    result = check_dataset_compatibility(leakage_df)

    assert result["compatible"] is True
    assert "TWF" in result["leakage_columns"]
    assert "HDF" in result["leakage_columns"]

    print("PASS")

    print("\nTEST 6: Missing values")

    missing_value_df = valid_df.copy()
    missing_value_df.loc[0, "Torque [Nm]"] = None
    missing_value_df.loc[1, "Type"] = None

    result = check_dataset_compatibility(missing_value_df)

    assert result["compatible"] is True
    assert len(result["warnings"]) > 0

    print("PASS")

    print("\n" + "=" * 45)
    print("ALL COMPATIBILITY CHECKER TESTS PASSED")
    print("=" * 45)


if __name__ == "__main__":
    main()
