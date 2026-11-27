
import pandas as pd

REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

BINARY_COLUMNS = [
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


def validate_dataset(df: pd.DataFrame):
    """
    Validate AI4I dataset structure and data quality.
    """

    print("\nDATASET VALIDATION")
    print("=" * 55)

    errors = []

    # 1. Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    else:
        print("1. Required columns: PASSED")

    # 2. Check missing values
    missing_values = df.isnull().sum()
    total_missing = int(missing_values.sum())

    print(f"\n2. Missing values: {total_missing}")

    if total_missing > 0:
        print(missing_values[missing_values > 0])
        errors.append("Dataset contains missing values.")

    # 3. Check duplicate records
    duplicates = int(df.duplicated().sum())

    print(f"\n3. Duplicate records: {duplicates}")

    # 4. Check binary columns
    print("\n4. Binary column validation")

    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue

        values = set(df[col].dropna().unique())

        if not values.issubset({0, 1}):
            errors.append(
                f"{col} contains unexpected values: {values}"
            )
            print(f"{col}: FAILED")
        else:
            print(f"{col}: PASSED")

    # 5. Check machine type
    if "Type" in df.columns:
        valid_types = {"L", "M", "H"}
        actual_types = set(df["Type"].dropna().unique())

        if not actual_types.issubset(valid_types):
            errors.append("Unexpected machine type found.")
            print("\n5. Machine type: FAILED")
        else:
            print("\n5. Machine type: PASSED")

    # 6. Check target distribution
    if "Machine failure" in df.columns:
        print("\n6. Target distribution")
        print(df["Machine failure"].value_counts())

    # 7. Check unique identifiers
    if "UDI" in df.columns:
        print(
            "\n7. Unique UDI values:",
            df["UDI"].nunique()
        )

    print("\n" + "=" * 55)

    if errors:
        print("VALIDATION RESULT: FAILED")

        for error in errors:
            print("-", error)

        raise ValueError(
            "Please resolve dataset validation errors."
        )

    print("VALIDATION RESULT: PASSED")

    return True


if __name__ == "__main__":

    from app.ml.data.data_loader import load_dataset

    df = load_dataset()

    validate_dataset(df)
