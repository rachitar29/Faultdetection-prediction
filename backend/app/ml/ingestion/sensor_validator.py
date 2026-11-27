
"""Validate normalized sensor data before ML prediction."""

import numpy as np
import pandas as pd

from app.ml.data.sensor_validation import AI4I_OBSERVED_RANGES


REQUIRED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

VALID_MACHINE_TYPES = {"L", "M", "H"}

TEMPERATURE_FEATURES = {
    "Air temperature [K]",
    "Process temperature [K]",
}

NON_NEGATIVE_FEATURES = {
    "Rotational speed [rpm]",
    "Tool wear [min]",
}


def validate_sensor_data(dataframe):
    """
    Validate a normalized sensor DataFrame.

    Does not silently remove rows or modify the original data.
    Returns a validation report and a cleaned copy.

    Missing values remain validation errors here. Batch prediction
    has a separate policy that permits missing numeric values for
    the registered fitted model imputer.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    report = {
        "is_valid": False,
        "status": "FAILED",
        "total_rows": len(dataframe),
        "total_columns": len(dataframe.columns),
        "missing_features": [],
        "missing_values": {},
        "invalid_numeric_values": {},
        "infinite_values": {},
        "invalid_machine_types": [],
        "negative_values": {},
        "out_of_distribution_values": {},
        "physical_validation": {
            "temperatures_at_or_below_zero_kelvin": {},
            "negative_rotational_speed": 0,
            "negative_tool_wear": 0,
            "negative_torque": 0,
        },
        "duplicate_rows": 0,
        "warnings": [],
        "errors": [],
    }

    if dataframe.empty:
        report["errors"].append("Dataset is empty.")

        return {
            "dataframe": dataframe.copy(),
            "report": report,
        }

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in dataframe.columns
    ]

    report["missing_features"] = missing_features

    if missing_features:
        report["errors"].append(
            "Required sensor features are missing."
        )

    working_df = dataframe.copy()

    # Validate numeric columns and convert valid numeric values.
    for column in NUMERIC_FEATURES:

        if column not in working_df.columns:
            continue

        original = working_df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        invalid_mask = original.notna() & converted.isna()

        invalid_count = int(invalid_mask.sum())

        if invalid_count:
            report["invalid_numeric_values"][column] = (
                invalid_count
            )

        working_df[column] = converted

        missing_count = int(
            working_df[column].isna().sum()
        )

        if missing_count:
            report["missing_values"][column] = missing_count

        numeric_values = working_df[column].dropna().to_numpy(
            dtype=float
        )

        infinite_count = int(
            np.isinf(numeric_values).sum()
        )

        if infinite_count:
            report["infinite_values"][column] = (
                infinite_count
            )

        finite_mask = np.isfinite(
            working_df[column].to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )

        finite_values = working_df.loc[
            finite_mask,
            column,
        ]

        negative_count = int(
            (finite_values < 0).sum()
        )

        if negative_count:
            report["negative_values"][column] = (
                negative_count
            )

        # Physical temperature validation.
        if column in TEMPERATURE_FEATURES:
            invalid_temperature_count = int(
                (finite_values <= 0).sum()
            )

            if invalid_temperature_count:
                report[
                    "physical_validation"
                ][
                    "temperatures_at_or_below_zero_kelvin"
                ][column] = invalid_temperature_count

                report["errors"].append(
                    f"{column}: {invalid_temperature_count} "
                    "reading(s) at or below absolute zero "
                    "(0 K) are physically invalid."
                )

        # Negative rotational speed and tool wear are rejected.
        if column in NON_NEGATIVE_FEATURES:
            if negative_count:
                report["physical_validation"][
                    "negative_rotational_speed"
                    if column == "Rotational speed [rpm]"
                    else "negative_tool_wear"
                ] = negative_count

                report["errors"].append(
                    f"{column}: {negative_count} negative "
                    "reading(s) detected. Negative values "
                    "are not accepted."
                )

        # Negative torque is accepted but requires review.
        if column == "Torque [Nm]" and negative_count:
            report["physical_validation"][
                "negative_torque"
            ] = negative_count

            report["warnings"].append(
                f"Torque [Nm]: {negative_count} negative "
                "reading(s) detected. Review sensor direction "
                "and sign convention."
            )

        # Compare finite readings with the observed AI4I dataset
        # distribution. These are not certified safety limits.
        observed_range = AI4I_OBSERVED_RANGES.get(column)

        if observed_range is not None:
            lower, upper = observed_range

            below_count = int(
                (finite_values < lower).sum()
            )

            above_count = int(
                (finite_values > upper).sum()
            )

            outside_count = below_count + above_count

            if outside_count:
                report["out_of_distribution_values"][column] = {
                    "below_observed_minimum": below_count,
                    "above_observed_maximum": above_count,
                    "total_outside_observed_range": outside_count,
                    "observed_minimum": lower,
                    "observed_maximum": upper,
                }

                report["warnings"].append(
                    f"{column}: {outside_count} reading(s) "
                    f"outside the AI4I observed training range "
                    f"[{lower:g}, {upper:g}]. These are "
                    "out-of-distribution warnings, not "
                    "certified safety limits."
                )

    # Validate and normalize categorical machine types.
    if "Type" in working_df.columns:

        machine_type_series = (
            working_df["Type"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        missing_type_mask = (
            working_df["Type"].isna()
            | machine_type_series.eq("").fillna(False)
        )

        missing_type_count = int(
            missing_type_mask.sum()
        )

        if missing_type_count:
            report["missing_values"]["Type"] = (
                missing_type_count
            )

        invalid_type_mask = (
            ~missing_type_mask
            & ~machine_type_series.isin(VALID_MACHINE_TYPES)
        )

        invalid_types = sorted(
            set(
                machine_type_series.loc[
                    invalid_type_mask
                ].dropna().tolist()
            )
        )

        report["invalid_machine_types"] = invalid_types

        if invalid_types:
            invalid_count = int(invalid_type_mask.sum())

            report["errors"].append(
                f"Invalid machine type values detected "
                f"({invalid_count} row(s)): {invalid_types}. "
                "Expected L, M, or H."
            )

        # Normalize only valid machine types in the returned copy.
        valid_type_mask = (
            ~missing_type_mask
            & machine_type_series.isin(VALID_MACHINE_TYPES)
        )

        working_df.loc[valid_type_mask, "Type"] = (
            machine_type_series.loc[valid_type_mask]
        )

    # Check duplicates without removing any rows.
    report["duplicate_rows"] = int(
        working_df.duplicated().sum()
    )

    if report["duplicate_rows"]:
        report["warnings"].append(
            "Duplicate rows detected. They have not been removed."
        )

    # Preserve the generic negative-reading diagnostic.
    if report["negative_values"]:
        report["warnings"].append(
            "Negative sensor readings detected. "
            "Review the feature-specific validation results."
        )

    # Detect completely empty columns.
    empty_columns = [
        column
        for column in working_df.columns
        if working_df[column].isna().all()
    ]

    if empty_columns:
        report["errors"].append(
            f"Completely empty columns: {empty_columns}"
        )

    # Critical numeric problems.
    if report["invalid_numeric_values"]:
        report["errors"].append(
            "Some numeric columns contain non-numeric values."
        )

    if report["infinite_values"]:
        report["errors"].append(
            "Infinite sensor values detected."
        )

    if report["missing_values"]:
        report["errors"].append(
            "Missing sensor values detected."
        )

    report["is_valid"] = len(report["errors"]) == 0

    report["status"] = (
        "VALID" if report["is_valid"] else "FAILED"
    )

    return {
        "dataframe": working_df,
        "report": report,
    }


def print_validation_report(report):
    """Print a readable sensor validation report."""

    print("\nSENSOR DATA VALIDATION")
    print("=" * 55)

    print("Status:", report["status"])
    print("Rows:", report["total_rows"])
    print("Columns:", report["total_columns"])
    print("Duplicate rows:", report["duplicate_rows"])

    print("\nMissing features:")
    print(report["missing_features"])

    print("\nMissing values:")
    print(report["missing_values"])

    print("\nInvalid numeric values:")
    print(report["invalid_numeric_values"])

    print("\nInfinite values:")
    print(report["infinite_values"])

    print("\nInvalid machine types:")
    print(report["invalid_machine_types"])

    print("\nNegative values:")
    print(report["negative_values"])

    print("\nOut-of-distribution values:")
    print(report["out_of_distribution_values"])

    print("\nPhysical validation:")
    print(report["physical_validation"])

    print("\nWarnings:")
    for warning in report["warnings"]:
        print("-", warning)

    print("\nErrors:")
    for error in report["errors"]:
        print("-", error)

    print("=" * 55)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "Air temperature [K]": [298.1, 299.2, 300.1],
        "Process temperature [K]": [308.5, 309.1, 310.2],
        "Rotational speed [rpm]": [1500, 1450, 1400],
        "Torque [Nm]": [40.2, 42.1, 45.0],
        "Tool wear [min]": [10, 20, 30],
        "Type": ["M", "L", "H"],
    })

    result = validate_sensor_data(sample)

    print_validation_report(result["report"])

    if result["report"]["is_valid"]:
        print("\nSELF-TEST PASSED")
    else:
        print("\nSELF-TEST FAILED")
