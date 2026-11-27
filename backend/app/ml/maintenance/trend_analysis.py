
"""
Sensor Trend Analysis

Analyses timestamped historical machine readings.

Features:
- Sensor trend direction
- Rate of change per hour
- Percentage change
- Configurable operating-limit checks
- Maintenance monitoring summary

This module analyses trends; it does not predict
Remaining Useful Life (RUL).
"""

import numpy as np
import pandas as pd

from app.ml.data.preprocessing import NUMERIC_FEATURES


# ==========================================
# CONFIGURATION
# ==========================================

DEFAULT_TREND_CHANGE_PERCENT = 2.0

TIMESTAMP_COLUMN = "timestamp"


# ==========================================
# VALIDATE HISTORICAL READINGS
# ==========================================

def validate_sensor_history(readings):

    if not isinstance(readings, list):
        raise TypeError(
            "Sensor history must be a list of dictionaries."
        )

    if len(readings) < 2:
        raise ValueError(
            "At least two historical readings are required."
        )

    df = pd.DataFrame(readings)

    required_columns = (
        [TIMESTAMP_COLUMN] + NUMERIC_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing historical columns: {missing}"
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
        utc=True
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "All timestamps must be valid."
        )

    for feature in NUMERIC_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        if not np.isfinite(
            df[feature].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{feature} contains invalid or non-finite values."
            )

    df = df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)

    if df[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate timestamps are not allowed."
        )

    return df


# ==========================================
# CALCULATE TREND DIRECTION
# ==========================================

def classify_trend(
    percentage_change,
    change_threshold=DEFAULT_TREND_CHANGE_PERCENT
):

    if percentage_change > change_threshold:
        return "Increasing"

    if percentage_change < -change_threshold:
        return "Decreasing"

    return "Stable"


# ==========================================
# ANALYSE ONE SENSOR
# ==========================================

def analyse_sensor_trend(
    df,
    feature,
    change_threshold=DEFAULT_TREND_CHANGE_PERCENT
):

    values = df[feature].to_numpy(dtype=float)

    timestamps = df[TIMESTAMP_COLUMN]

    first_value = float(values[0])
    last_value = float(values[-1])

    absolute_change = last_value - first_value

    if abs(first_value) > 1e-12:
        percentage_change = (
            absolute_change / abs(first_value)
        ) * 100
    else:
        percentage_change = None

    elapsed_hours = (
        timestamps.iloc[-1] - timestamps.iloc[0]
    ).total_seconds() / 3600

    if elapsed_hours <= 0:
        raise ValueError(
            "Historical readings must span a positive time interval."
        )

    # Calculate linear trend slope per hour.

    elapsed = (
        timestamps - timestamps.iloc[0]
    ).dt.total_seconds().to_numpy() / 3600

    slope_per_hour = float(
        np.polyfit(elapsed, values, 1)[0]
    )

    if percentage_change is None:
        direction = (
            "Increasing"
            if absolute_change > 0
            else "Decreasing"
            if absolute_change < 0
            else "Stable"
        )
    else:
        direction = classify_trend(
            percentage_change,
            change_threshold
        )

    return {
        "feature": feature,
        "first_value": round(first_value, 4),
        "latest_value": round(last_value, 4),
        "absolute_change": round(absolute_change, 4),
        "percentage_change": (
            round(percentage_change, 2)
            if percentage_change is not None
            else None
        ),
        "trend_direction": direction,
        "slope_per_hour": round(slope_per_hour, 6),
        "time_window_hours": round(elapsed_hours, 4),
        "reading_count": len(values)
    }


# ==========================================
# CHECK CONFIGURABLE OPERATING LIMITS
# ==========================================

def check_operating_limits(latest_reading, operating_limits=None):

    if operating_limits is None:
        operating_limits = {}

    if not isinstance(operating_limits, dict):
        raise TypeError(
            "Operating limits must be a dictionary."
        )

    alerts = []

    for feature, limits in operating_limits.items():

        if feature not in NUMERIC_FEATURES:
            raise ValueError(
                f"Unsupported sensor feature: {feature}"
            )

        if not isinstance(limits, dict):
            raise ValueError(
                f"Limits for {feature} must be a dictionary."
            )

        value = latest_reading[feature]

        minimum = limits.get("min")
        maximum = limits.get("max")

        if minimum is not None and value < float(minimum):
            alerts.append({
                "feature": feature,
                "reading": value,
                "limit": float(minimum),
                "condition": "Below minimum"
            })

        if maximum is not None and value > float(maximum):
            alerts.append({
                "feature": feature,
                "reading": value,
                "limit": float(maximum),
                "condition": "Above maximum"
            })

        if (
            minimum is not None
            and maximum is not None
            and float(minimum) > float(maximum)
        ):
            raise ValueError(
                f"Minimum limit exceeds maximum for {feature}."
            )

    return alerts


# ==========================================
# COMPLETE SENSOR TREND ANALYSIS
# ==========================================

def analyse_sensor_history(
    readings,
    operating_limits=None,
    change_threshold=DEFAULT_TREND_CHANGE_PERCENT
):

    if change_threshold < 0:
        raise ValueError(
            "Change threshold cannot be negative."
        )

    df = validate_sensor_history(readings)

    # Analyse every numeric sensor feature.

    trends = {}

    for feature in NUMERIC_FEATURES:

        trends[feature] = analyse_sensor_trend(
            df,
            feature,
            change_threshold
        )

    # Get the latest sensor readings.

    latest_row = df.iloc[-1]

    latest_reading = {
        feature: float(latest_row[feature])
        for feature in NUMERIC_FEATURES
    }

    # Check optional engineering limits.

    limit_alerts = check_operating_limits(
        latest_reading,
        operating_limits
    )

    increasing_features = [
        feature
        for feature, result in trends.items()
        if result["trend_direction"] == "Increasing"
    ]

    decreasing_features = [
        feature
        for feature, result in trends.items()
        if result["trend_direction"] == "Decreasing"
    ]

    stable_features = [
        feature
        for feature, result in trends.items()
        if result["trend_direction"] == "Stable"
    ]

    if limit_alerts:
        summary = (
            "One or more configured operating limits "
            "have been exceeded. Review the affected sensors."
        )

    elif increasing_features:
        summary = (
            "Increasing sensor trends detected. "
            "Review the affected sensor history."
        )

    else:
        summary = (
            "No configured operating-limit violations "
            "or increasing trends were detected."
        )

    return {
        "trend_analysis": trends,
        "latest_sensor_readings": latest_reading,
        "increasing_features": increasing_features,
        "decreasing_features": decreasing_features,
        "stable_features": stable_features,
        "operating_limit_alerts": limit_alerts,
        "summary": summary,
        "reading_count": len(df),
        "analysis_timestamp": pd.Timestamp.now(
            tz="UTC"
        ).isoformat()
    }
