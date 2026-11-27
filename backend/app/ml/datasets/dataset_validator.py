
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# Default ranges for the synthetic industrial sensor dataset.
# These are demonstration limits, not universal industrial safety limits.
DEFAULT_SENSOR_RANGES = {
    "air_temperature": (0, 500),
    "process_temperature": (0, 500),
    "rotational_speed": (0, 10000),
    "torque": (0, 1000),
    "tool_wear": (0, 10000),
}

DEFAULT_CATEGORICAL_RULES = {
    "machine_type": ["L", "M", "H"],
}

DEFAULT_TIMESTAMP_COLUMN = "timestamp"
DEFAULT_STATUS_COLUMN = "sensor_status"
DEFAULT_TARGET_COLUMN = "machine_failure"


def load_dataset(file_path):
    """Load a CSV or Excel dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {SUPPORTED_EXTENSIONS}"
        )

    if extension == ".csv":
        return pd.read_csv(path)

    if extension in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError("Unable to load dataset.")


def validate_dataframe(
    df,
    target_column=None,
    required_columns=None,
    min_rows=10,
    max_missing_percentage=80.0
):
    """
    General dataset validation.

    Checks:
    - Empty datasets
    - Minimum rows
    - Duplicate and empty column names
    - Missing values
    - Constant columns
    - Target and required columns
    - Infinite numeric values
    - Duplicate rows
    """

    errors = []
    warnings = []

    if not isinstance(df, pd.DataFrame):
        return {
            "valid": False,
            "errors": ["Input must be a pandas DataFrame."],
            "warnings": [],
            "statistics": {}
        }

    if df.empty:
        return {
            "valid": False,
            "errors": ["Dataset is empty."],
            "warnings": [],
            "statistics": {"rows": 0, "columns": len(df.columns)}
        }

    row_count, column_count = df.shape

    if row_count < min_rows:
        errors.append(
            f"Dataset has {row_count} rows. "
            f"At least {min_rows} rows are required."
        )

    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        errors.append(
            f"Duplicate column names found: {duplicate_columns}"
        )

    empty_names = [
        col for col in df.columns
        if str(col).strip() == ""
    ]

    if empty_names:
        errors.append("Dataset contains empty column names.")

    if required_columns:
        missing_required = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_required:
            errors.append(
                f"Required columns missing: {missing_required}"
            )

    if target_column is not None:
        if target_column not in df.columns:
            errors.append(
                f"Target column '{target_column}' was not found."
            )
        elif df[target_column].isna().all():
            errors.append(
                f"Target column '{target_column}' contains only missing values."
            )
        elif df[target_column].nunique(dropna=True) < 2:
            warnings.append(
                f"Target column '{target_column}' has fewer than "
                "two distinct values."
            )

    missing_counts = df.isna().sum()
    missing_percentages = (
        missing_counts / row_count * 100
    ).round(2)

    high_missing_columns = missing_percentages[
        missing_percentages > max_missing_percentage
    ].to_dict()

    if high_missing_columns:
        warnings.append(
            f"Columns with more than {max_missing_percentage}% "
            f"missing values: {high_missing_columns}"
        )

    columns_with_missing_values = missing_counts[
        missing_counts > 0
    ].to_dict()

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        warnings.append(
            f"Constant or entirely empty columns found: {constant_columns}"
        )

    numeric_df = df.select_dtypes(include="number")

    infinite_columns = []

    if not numeric_df.empty:
        infinite_columns = numeric_df.columns[
            numeric_df.isin([float("inf"), float("-inf")]).any()
        ].tolist()

    if infinite_columns:
        warnings.append(
            f"Infinite numeric values found: {infinite_columns}"
        )

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows:
        warnings.append(
            f"Found {duplicate_rows} duplicate rows."
        )

    column_types = {
        str(col): str(dtype)
        for col, dtype in df.dtypes.items()
    }

    statistics = {
        "rows": int(row_count),
        "columns": int(column_count),
        "numeric_columns": int(
            len(df.select_dtypes(include="number").columns)
        ),
        "categorical_columns": int(
            len(df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns)
        ),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows,
        "missing_values_by_column": columns_with_missing_values,
        "missing_percentage_by_column": missing_percentages.to_dict(),
        "constant_columns": constant_columns,
        "column_types": column_types
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "statistics": statistics
    }


def validate_sensor_dataframe(
    df,
    sensor_ranges=None,
    categorical_rules=None,
    timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
    status_column=DEFAULT_STATUS_COLUMN,
    target_column=DEFAULT_TARGET_COLUMN,
    min_rows=10
):
    """
    Validate industrial sensor readings.

    Configurable sensor ranges and category rules allow this
    function to support different machine types and datasets.
    """

    result = validate_dataframe(
        df=df,
        target_column=target_column if target_column in df.columns else None,
        min_rows=min_rows
    )

    errors = result["errors"]
    warnings = result["warnings"]

    sensor_ranges = (
        sensor_ranges
        if sensor_ranges is not None
        else DEFAULT_SENSOR_RANGES
    )

    categorical_rules = (
        categorical_rules
        if categorical_rules is not None
        else DEFAULT_CATEGORICAL_RULES
    )

    sensor_report = {}

    if not isinstance(df, pd.DataFrame) or df.empty:
        result["sensor_checks"] = sensor_report
        return result

    # 1. Numeric sensor validation and operating ranges.
    for column, limits in sensor_ranges.items():

        if column not in df.columns:
            warnings.append(
                f"Configured sensor column '{column}' is missing."
            )
            continue

        minimum, maximum = limits

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_text = (
            df[column].notna()
            & converted.isna()
        )

        if invalid_text.any():
            errors.append(
                f"{column}: {int(invalid_text.sum())} "
                "non-numeric sensor readings found."
            )

        infinite_values = converted.isin(
            [float("inf"), float("-inf")]
        )

        if infinite_values.any():
            errors.append(
                f"{column}: infinite sensor readings found."
            )

        outside_range = (
            converted.notna()
            & (
                (converted < minimum)
                | (converted > maximum)
            )
        )

        if outside_range.any():
            errors.append(
                f"{column}: {int(outside_range.sum())} readings "
                f"outside configured range [{minimum}, {maximum}]."
            )

        sensor_report[column] = {
            "missing": int(df[column].isna().sum()),
            "non_numeric": int(invalid_text.sum()),
            "outside_range": int(outside_range.sum()),
            "minimum_allowed": minimum,
            "maximum_allowed": maximum
        }

    # 2. Categorical validation.
    for column, allowed_values in categorical_rules.items():

        if column not in df.columns:
            warnings.append(
                f"Configured categorical column '{column}' is missing."
            )
            continue

        normalized = df[column].astype("string").str.strip()

        invalid_categories = (
            df[column].notna()
            & ~normalized.isin(allowed_values)
        )

        if invalid_categories.any():
            found = normalized[invalid_categories].value_counts().to_dict()

            errors.append(
                f"{column}: unexpected category values found: {found}"
            )

        sensor_report[column] = {
            "missing": int(df[column].isna().sum()),
            "invalid_categories": int(invalid_categories.sum()),
            "allowed_values": allowed_values
        }

    # 3. Timestamp validation.
    if timestamp_column in df.columns:

        parsed_timestamps = pd.to_datetime(
            df[timestamp_column],
            errors="coerce"
        )

        invalid_timestamps = (
            df[timestamp_column].notna()
            & parsed_timestamps.isna()
        )

        if df[timestamp_column].isna().any():
            errors.append(
                f"{timestamp_column}: missing timestamps found."
            )

        if invalid_timestamps.any():
            errors.append(
                f"{timestamp_column}: "
                f"{int(invalid_timestamps.sum())} invalid timestamps found."
            )

        valid_timestamps = parsed_timestamps.dropna()

        out_of_order = (
            len(valid_timestamps) > 1
            and not valid_timestamps.is_monotonic_increasing
        )

        if out_of_order:
            warnings.append(
                f"{timestamp_column}: timestamps are not in ascending order."
            )

        sensor_report[timestamp_column] = {
            "missing": int(df[timestamp_column].isna().sum()),
            "invalid": int(invalid_timestamps.sum()),
            "out_of_order": bool(out_of_order)
        }

    # 4. Sensor status validation.
    if status_column in df.columns:

        status_values = (
            df[status_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        error_status_count = int(
            (status_values == "ERROR").sum()
        )

        empty_status_count = int(
            (status_values == "").sum()
        )

        if error_status_count:
            warnings.append(
                f"{status_column}: {error_status_count} readings "
                "reported sensor ERROR status."
            )

        if empty_status_count:
            errors.append(
                f"{status_column}: {empty_status_count} missing or empty statuses."
            )

        sensor_report[status_column] = {
            "error_status_count": error_status_count,
            "empty_status_count": empty_status_count
        }

    # 5. Target validation.
    if target_column in df.columns:

        target_values = pd.to_numeric(
            df[target_column],
            errors="coerce"
        )

        invalid_target = (
            target_values.isna()
            | ~target_values.isin([0, 1])
        )

        if invalid_target.any():
            errors.append(
                f"{target_column}: {int(invalid_target.sum())} "
                "invalid labels found. Expected 0 or 1."
            )

        sensor_report[target_column] = {
            "invalid_labels": int(invalid_target.sum()),
            "allowed_labels": [0, 1]
        }

    # 6. High-cardinality identifier warning.
    identifier_candidates = []

    for column in df.select_dtypes(
        include=["object", "string"]
    ).columns:

        if column in {
            timestamp_column,
            status_column
        }:
            continue

        unique_count = df[column].nunique(dropna=True)
        unique_ratio = unique_count / max(len(df), 1)

        if unique_ratio >= 0.90 and unique_count > 20:
            identifier_candidates.append(column)

    if identifier_candidates:
        warnings.append(
            "Possible identifier/high-cardinality columns: "
            f"{identifier_candidates}. Review before using as ML features."
        )

    result["valid"] = len(errors) == 0
    result["errors"] = errors
    result["warnings"] = warnings
    result["sensor_checks"] = sensor_report

    return result


def validate_dataset_file(
    file_path,
    target_column=None,
    required_columns=None,
    min_rows=10
):
    """Load and perform general dataset validation."""

    try:
        df = load_dataset(file_path)

        result = validate_dataframe(
            df=df,
            target_column=target_column,
            required_columns=required_columns,
            min_rows=min_rows
        )

        result["file_name"] = Path(file_path).name
        result["file_format"] = Path(file_path).suffix.lower()

        return result

    except Exception as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "statistics": {}
        }


def validate_sensor_dataset_file(
    file_path,
    sensor_ranges=None,
    categorical_rules=None,
    timestamp_column=DEFAULT_TIMESTAMP_COLUMN,
    status_column=DEFAULT_STATUS_COLUMN,
    target_column=DEFAULT_TARGET_COLUMN,
    min_rows=10
):
    """Load and perform industrial sensor validation."""

    try:
        df = load_dataset(file_path)

        result = validate_sensor_dataframe(
            df=df,
            sensor_ranges=sensor_ranges,
            categorical_rules=categorical_rules,
            timestamp_column=timestamp_column,
            status_column=status_column,
            target_column=target_column,
            min_rows=min_rows
        )

        result["file_name"] = Path(file_path).name
        result["file_format"] = Path(file_path).suffix.lower()

        return result

    except Exception as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "sensor_checks": {},
            "statistics": {}
        }
