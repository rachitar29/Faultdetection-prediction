
import json
from pathlib import Path

import pandas as pd

from app.ml.datasets.dataset_validator import (
    load_dataset,
    validate_sensor_dataframe
)


def load_validation_config(config_path):
    """Load and validate a dataset's JSON validation configuration."""

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Validation configuration not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    if "dataset_type" not in config:
        raise ValueError("Configuration requires dataset_type.")

    config.setdefault("required_columns", [])
    config.setdefault("sensors", {})
    config.setdefault("categorical_rules", {})
    config.setdefault("missing_value_policy", {})
    config.setdefault("timestamp", {})
    config.setdefault("sensor_status", {})
    config.setdefault("target", {})

    for name, sensor in config["sensors"].items():
        if not isinstance(sensor, dict):
            raise ValueError(
                f"Sensor configuration for '{name}' must be an object."
            )

        if not sensor.get("enabled", True):
            continue

        if "min" not in sensor or "max" not in sensor:
            raise ValueError(
                f"Enabled sensor '{name}' requires min and max."
            )

        if sensor["min"] > sensor["max"]:
            raise ValueError(
                f"Sensor '{name}' has min greater than max."
            )

    return config


def validate_configured_dataset(
    file_path,
    config_path
):
    """
    Validate a dataset using an external JSON configuration.

    The source dataset is only read. It is never modified.
    """

    try:
        config = load_validation_config(config_path)
        dataframe = load_dataset(file_path)

        errors = []
        warnings = []

        # -------------------------------------------------
        # 1. Required columns
        # -------------------------------------------------

        required_columns = config.get(
            "required_columns", []
        )

        missing_required = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_required:
            errors.append(
                f"Required columns missing: {missing_required}"
            )

        # -------------------------------------------------
        # 2. Build enabled sensor ranges
        # -------------------------------------------------

        sensor_ranges = {}

        for name, sensor in config["sensors"].items():

            if sensor.get("enabled", True):
                sensor_ranges[name] = (
                    sensor["min"],
                    sensor["max"]
                )

        # -------------------------------------------------
        # 3. Existing sensor validation engine
        # -------------------------------------------------

        timestamp_config = config.get("timestamp", {})
        status_config = config.get("sensor_status", {})
        target_config = config.get("target", {})

        timestamp_column = timestamp_config.get(
            "column"
        )

        status_column = status_config.get(
            "column"
        )

        # Target validation is performed below using the
        # allowed labels from this configuration.
        base_result = validate_sensor_dataframe(
            df=dataframe,
            sensor_ranges=sensor_ranges,
            categorical_rules=config.get(
                "categorical_rules", {}
            ),
            timestamp_column=timestamp_column,
            status_column=status_column,
            target_column=None,
            min_rows=config.get("min_rows", 10)
        )

        errors.extend(base_result["errors"])
        warnings.extend(base_result["warnings"])

        # -------------------------------------------------
        # 4. Missing-value policies
        # -------------------------------------------------

        missing_policy = config.get(
            "missing_value_policy", {}
        )

        missing_report = {}

        for column, policy in missing_policy.items():

            if column not in dataframe.columns:
                warnings.append(
                    f"Missing-value policy references unknown "
                    f"column '{column}'."
                )
                continue

            policy = str(policy).lower().strip()

            if policy not in {"reject", "warn", "allow"}:
                errors.append(
                    f"Invalid missing-value policy '{policy}' "
                    f"for column '{column}'."
                )
                continue

            missing_count = int(
                dataframe[column].isna().sum()
            )

            missing_report[column] = {
                "missing_count": missing_count,
                "policy": policy
            }

            if missing_count == 0:
                continue

            message = (
                f"{column}: {missing_count} missing values "
                f"(policy: {policy})."
            )

            if policy == "reject":
                errors.append(message)

            elif policy == "warn":
                warnings.append(message)

        # -------------------------------------------------
        # 5. Timestamp policy
        # -------------------------------------------------

        if timestamp_column:
            if timestamp_column not in dataframe.columns:
                if timestamp_config.get("required", True):
                    errors.append(
                        f"Required timestamp column "
                        f"'{timestamp_column}' is missing."
                    )

            else:
                timestamps = pd.to_datetime(
                    dataframe[timestamp_column],
                    errors="coerce"
                )

                invalid_count = int(
                    (
                        dataframe[timestamp_column].notna()
                        & timestamps.isna()
                    ).sum()
                )

                missing_count = int(
                    dataframe[timestamp_column].isna().sum()
                )

                if invalid_count:
                    errors.append(
                        f"{timestamp_column}: "
                        f"{invalid_count} invalid timestamps."
                    )

                if missing_count and timestamp_config.get(
                    "required", True
                ):
                    errors.append(
                        f"{timestamp_column}: "
                        f"{missing_count} missing timestamps."
                    )

        # -------------------------------------------------
        # 6. Sensor status policy
        # -------------------------------------------------

        if status_column and status_column in dataframe.columns:

            allowed_statuses = status_config.get(
                "allowed_values", ["OK", "WARNING", "ERROR"]
            )

            status_values = (
                dataframe[status_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            invalid_status = (
                status_values != ""
                ) & ~status_values.isin(allowed_statuses)

            if invalid_status.any():
                errors.append(
                    f"{status_column}: unexpected status values "
                    f"{status_values[invalid_status].unique().tolist()}."
                )

            error_statuses = status_config.get(
                "error_values", ["ERROR"]
            )

            error_count = int(
                status_values.isin(error_statuses).sum()
            )

            if error_count:
                warnings.append(
                    f"{status_column}: {error_count} readings "
                    "have an error status."
                )

        # -------------------------------------------------
        # 7. Configurable target validation
        # -------------------------------------------------

        target_column = target_config.get("column")

        if target_column:

            if target_column not in dataframe.columns:
                errors.append(
                    f"Target column '{target_column}' is missing."
                )

            else:
                allowed_labels = target_config.get(
                    "allowed_values"
                )

                target_values = dataframe[target_column]

                if target_values.isna().any():
                    errors.append(
                        f"Target column '{target_column}' "
                        "contains missing labels."
                    )

                if allowed_labels is not None:
                    invalid_target = (
                        target_values.notna()
                        & ~target_values.isin(allowed_labels)
                    )

                    invalid_count = int(
                        invalid_target.sum()
                    )

                    if invalid_count:
                        errors.append(
                            f"Target column '{target_column}': "
                            f"{invalid_count} invalid labels. "
                            f"Allowed values: {allowed_labels}."
                        )

        # -------------------------------------------------
        # 8. Final report
        # -------------------------------------------------

        result = {
            "valid": len(errors) == 0,
            "dataset_type": config["dataset_type"],
            "file_name": Path(file_path).name,
            "config_file": Path(config_path).name,
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "errors": errors,
            "warnings": warnings,
            "missing_value_checks": missing_report,
            "sensor_checks": base_result.get(
                "sensor_checks", {}
            ),
            "statistics": base_result.get(
                "statistics", {}
            )
        }

        return result

    except Exception as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "missing_value_checks": {},
            "sensor_checks": {},
            "statistics": {}
        }


def save_validation_report(result, output_path):
    """Save a validation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=4,
            default=str
        )

    return path
