
import re

import pandas as pd


# Patterns used to identify identifier columns
ID_PATTERNS = [
    r"^id$",
    r"^id[_\s-]",
    r"[_\s-]id$",
    r"^udi$",
    r"^index$",
    r"^serial[_\s-]?number$",
    r"^product[_\s-]?id$",
    r"^machine[_\s-]?id$",
    r"^equipment[_\s-]?id$",
    r"^asset[_\s-]?id$",
]


# Patterns used to identify date and time columns
DATETIME_PATTERNS = [
    r"timestamp",
    r"datetime",
    r"date",
    r"time",
]


# Patterns used to identify possible target columns
TARGET_PATTERNS = [
    r"target",
    r"label",
    r"class",
    r"failure",
    r"fault",
    r"status",
    r"rul",
    r"remaining[_\s-]?useful[_\s-]?life",
    r"anomaly",
]


# Patterns used to identify potential target leakage
LEAKAGE_PATTERNS = [
    r"failure[_\s-]?mode",
    r"failure[_\s-]?type",
    r"failure[_\s-]?reason",
]


def _matches_pattern(column_name, patterns):
    """Check whether a column name matches any supplied regex pattern."""

    name = str(column_name).strip().lower()

    return any(
        re.search(pattern, name, flags=re.IGNORECASE)
        for pattern in patterns
    )


def detect_schema(
    df,
    target_column=None,
    exclude_columns=None,
    high_cardinality_threshold=0.90,
):
    """
    Detect dataset schema and identify:

    - Numeric features
    - Categorical features
    - Datetime columns
    - Possible target columns
    - Identifier columns
    - High-cardinality columns
    - Constant columns
    - Excluded columns
    - Potential target leakage warnings

    This function does not modify the original DataFrame.
    """

    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not 0 < high_cardinality_threshold <= 1:
        raise ValueError(
            "high_cardinality_threshold must be greater than 0 "
            "and less than or equal to 1."
        )

    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    # User-configured exclusions
    user_exclusions = set(exclude_columns or [])

    warnings = []

    # Check unknown exclusions
    unknown_exclusions = user_exclusions - set(df.columns)

    for column in unknown_exclusions:
        warnings.append(
            f"Configured exclusion '{column}' was not found in the dataset."
        )

    # Initialize result categories
    numeric_features = []
    categorical_features = []
    datetime_columns = []
    possible_target_columns = []
    id_columns = []
    high_cardinality_columns = []
    constant_columns = []

    excluded = {
        column
        for column in user_exclusions
        if column in df.columns
    }

    total_rows = len(df)

    # Inspect each column
    for column in df.columns:

        name = str(column)
        series = df[column]

        # 1. Explicitly configured exclusions
        if column in user_exclusions:
            continue

        # 2. Constant or entirely empty columns
        unique_count = series.nunique(dropna=True)

        if unique_count <= 1:
            constant_columns.append(name)
            excluded.add(column)
            continue

        # 3. Explicit target column
        if column == target_column:
            possible_target_columns.append(name)
            continue

        # 4. Identifier columns
        if _matches_pattern(name, ID_PATTERNS):
            id_columns.append(name)
            excluded.add(column)
            continue

        # 5. Datetime columns
        if (
            _matches_pattern(name, DATETIME_PATTERNS)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            datetime_columns.append(name)
            excluded.add(column)
            continue

        # 6. Potential target leakage columns
        # IMPORTANT: Check before broad target patterns such as "failure".
        if _matches_pattern(name, LEAKAGE_PATTERNS):
            excluded.add(column)

            warnings.append(
                f"'{name}' may contain failure information. "
                "Review for target leakage."
            )

            continue

        # 7. Possible target columns
        if _matches_pattern(name, TARGET_PATTERNS):
            possible_target_columns.append(name)
            excluded.add(column)
            continue

        # 8. Numeric sensor features
        if pd.api.types.is_numeric_dtype(series):

            if pd.api.types.is_bool_dtype(series):
                categorical_features.append(name)
            else:
                numeric_features.append(name)

            continue

        # 9. Boolean columns
        if pd.api.types.is_bool_dtype(series):
            categorical_features.append(name)
            continue

        # 10. High-cardinality non-numeric columns
        cardinality_ratio = unique_count / total_rows

        if cardinality_ratio >= high_cardinality_threshold:

            high_cardinality_columns.append(name)
            excluded.add(column)

            warnings.append(
                f"'{name}' has high cardinality "
                f"({cardinality_ratio:.2%} unique values). "
                "Review whether it is an identifier."
            )

            continue

        # 11. Remaining non-numeric columns are categorical
        categorical_features.append(name)

    # Prepare schema result
    result = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),

        "numeric_features": numeric_features,
        "categorical_features": categorical_features,

        "datetime_columns": datetime_columns,
        "possible_target_columns": possible_target_columns,

        "id_columns": id_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "constant_columns": constant_columns,

        "excluded_columns": sorted(
            str(column) for column in excluded
        ),

        "warnings": warnings,
    }

    return result


def print_schema_report(schema):
    """Print a readable summary of the detected dataset schema."""

    print("=" * 65)
    print("DATASET SCHEMA DETECTION REPORT")
    print("=" * 65)

    print("\nTotal rows:", schema.get("total_rows"))
    print("Total columns:", schema.get("total_columns"))

    sections = [
        ("Numeric features", "numeric_features"),
        ("Categorical features", "categorical_features"),
        ("Datetime columns", "datetime_columns"),
        ("Possible targets", "possible_target_columns"),
        ("ID columns", "id_columns"),
        ("High-cardinality columns", "high_cardinality_columns"),
        ("Constant columns", "constant_columns"),
        ("Excluded columns", "excluded_columns"),
    ]

    for title, key in sections:

        print(f"\n{title}:")

        values = schema.get(key, [])

        if values:
            for value in values:
                print("  -", value)
        else:
            print("  None")

    print("\nWarnings:")

    warnings = schema.get("warnings", [])

    if warnings:
        for warning in warnings:
            print("  -", warning)
    else:
        print("  None")

    print("\n" + "=" * 65)
