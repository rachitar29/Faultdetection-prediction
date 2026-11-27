
"""
Reference Range Manager for Industrial Sensor Data.

Calculates statistical reference ranges from healthy observations.
These ranges are for statistical monitoring, NOT certified
equipment operating limits or safety thresholds.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = (
    BACKEND_ROOT
    / "app"
    / "ml"
    / "configurations"
    / "dataset_configs"
    / "ai4i_config.json"
)

DEFAULT_OUTPUT_PATH = (
    BACKEND_ROOT
    / "ml_artifacts"
    / "reference_ranges"
    / "ai4i_2020_reference_ranges.json"
)


def calculate_healthy_reference_ranges(
    dataframe,
    config_path=DEFAULT_CONFIG_PATH,
    target_column=None,
    healthy_value=None,
    lower_percentile=1,
    upper_percentile=99,
):
    """
    Calculate statistical reference ranges from healthy records.

    The input dataframe is never modified.

    Percentile boundaries identify unusual observations.
    They must not be interpreted as engineering safety limits.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Input dataset is empty.")

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Invalid percentile boundaries.")

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    target_column = (
        target_column or config.get("target_column")
    )

    if not target_column:
        raise ValueError("Target column is not configured.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column missing: {target_column}"
        )

    if healthy_value is None:
        healthy_value = config.get("negative_class", 0)

    numeric_features = config.get("numeric_features", [])

    if not numeric_features:
        raise ValueError(
            "No numeric features defined in dataset configuration."
        )

    missing_features = [
        feature
        for feature in numeric_features
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing sensor features: {missing_features}"
        )

    healthy_df = dataframe[
        dataframe[target_column] == healthy_value
    ].copy()

    if healthy_df.empty:
        raise ValueError(
            "No healthy observations found for the requested class."
        )

    sensor_metadata = config.get("sensor_metadata", {})
    reference_ranges = {}

    for feature in numeric_features:

        values = pd.to_numeric(
            healthy_df[feature],
            errors="coerce"
        )

        invalid_mask = (
            healthy_df[feature].notna()
            & values.isna()
        )

        if invalid_mask.any():
            raise ValueError(
                f"{feature} contains non-numeric values."
            )

        values = values.dropna()

        if values.empty:
            raise ValueError(
                f"No valid healthy observations for {feature}."
            )

        if not values.map(lambda value: abs(value) != float("inf")).all():
            raise ValueError(
                f"Infinite values detected in {feature}."
            )

        metadata = sensor_metadata.get(feature, {})

        reference_ranges[feature] = {
            "sensor_type": metadata.get("sensor_type"),
            "unit": metadata.get("unit"),
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "observed_min": float(values.min()),
            "observed_max": float(values.max()),
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
            "statistical_lower_bound": float(
                values.quantile(lower_percentile / 100)
            ),
            "statistical_upper_bound": float(
                values.quantile(upper_percentile / 100)
            ),
        }

    return {
        "dataset_id": config.get("dataset_id"),
        "dataset_name": config.get("dataset_name"),
        "dataset_source": config.get("source"),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reference_type": "healthy_data_statistical_reference",
        "target_column": target_column,
        "healthy_class": healthy_value,
        "healthy_record_count": int(len(healthy_df)),
        "percentile_method": "pandas quantile",
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile,
        "engineering_limits_available": False,
        "disclaimer": (
            "These are statistical ranges derived from healthy "
            "dataset observations. They are not manufacturer "
            "operating limits or certified safety thresholds."
        ),
        "sensors": reference_ranges,
    }


def save_reference_ranges(
    reference_data,
    output_path=DEFAULT_OUTPUT_PATH,
):
    """Save generated reference ranges as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            reference_data,
            file,
            indent=4
        )

    return output_path


def load_reference_ranges(reference_path=DEFAULT_OUTPUT_PATH):
    """Load previously generated reference ranges."""

    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference range file not found: {reference_path}"
        )

    with reference_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def evaluate_sensor_readings(
    dataframe,
    reference_data,
):
    """
    Identify readings outside statistical reference boundaries.

    This function flags unusual readings for review.
    It does not declare a machine faulty.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Input dataset is empty.")

    sensors = reference_data.get("sensors", {})

    report = {
        "total_rows": len(dataframe),
        "sensor_results": {},
        "warnings": [],
        "status": "COMPLETED",
    }

    for feature, limits in sensors.items():

        if feature not in dataframe.columns:
            report["warnings"].append(
                f"Missing sensor column: {feature}"
            )
            continue

        values = pd.to_numeric(
            dataframe[feature],
            errors="coerce"
        )

        invalid_count = int(
            (
                dataframe[feature].notna()
                & values.isna()
            ).sum()
        )

        infinite_mask = (
            values.notna()
            & ~values.map(lambda value: abs(value) != float("inf"))
        )

        infinite_count = int(infinite_mask.sum())

        lower = limits["statistical_lower_bound"]
        upper = limits["statistical_upper_bound"]

        outside_mask = (
            values.notna()
            & ~infinite_mask
            & ((values < lower) | (values > upper))
        )

        report["sensor_results"][feature] = {
            "unit": limits.get("unit"),
            "lower_reference_bound": lower,
            "upper_reference_bound": upper,
            "outside_reference_count": int(
                outside_mask.sum()
            ),
            "invalid_value_count": invalid_count,
            "infinite_value_count": infinite_count,
            "outside_reference_row_indices": [
                str(index)
                for index in dataframe.index[outside_mask]
            ],
        }

    return report


if __name__ == "__main__":

    dataset_path = (
        BACKEND_ROOT
        / "ml_data"
        / "raw"
        / "ai4i2020.csv"
    )

    df = pd.read_csv(dataset_path)

    reference_data = calculate_healthy_reference_ranges(df)

    saved_path = save_reference_ranges(reference_data)

    print("\nREFERENCE RANGE GENERATION")
    print("=" * 60)
    print("Dataset:", reference_data["dataset_name"])
    print("Healthy records:", reference_data["healthy_record_count"])
    print("Sensors:", len(reference_data["sensors"]))
    print("Saved:", saved_path)

    for sensor, details in reference_data["sensors"].items():
        print(f"\n{sensor} ({details['unit']})")
        print(
            f"Statistical range: "
            f"{details['statistical_lower_bound']:.3f} - "
            f"{details['statistical_upper_bound']:.3f}"
        )

    print("\nReference ranges generated successfully.")
