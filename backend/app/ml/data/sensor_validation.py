
import math


REQUIRED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

NUMERIC_FEATURES = REQUIRED_FEATURES[:-1]
VALID_MACHINE_TYPES = {"L", "M", "H"}

# Observed AI4I 2020 training-data ranges.
# These are distribution references, NOT certified safety limits.
AI4I_OBSERVED_RANGES = {
    "Air temperature [K]": (295.3, 304.5),
    "Process temperature [K]": (305.7, 313.8),
    "Rotational speed [rpm]": (1168.0, 2886.0),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0.0, 253.0),
}


def validate_sensor_record(sensor_data, allow_missing_numeric=False):
    """
    Validate one normalized AI4I sensor record.

    Returns:
        (validated_data, warnings)

    Missing numeric values are permitted only when
    allow_missing_numeric=True, for the existing batch imputer.
    """

    if not isinstance(sensor_data, dict):
        raise TypeError("Sensor data must be a dictionary.")

    missing_features = [
        feature for feature in REQUIRED_FEATURES
        if feature not in sensor_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing sensor features: {missing_features}"
        )

    validated = {}
    warnings = []

    for feature in NUMERIC_FEATURES:
        value = sensor_data[feature]

        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            if allow_missing_numeric:
                validated[feature] = None
                warnings.append(
                    f"{feature}: missing; fitted model imputer will be used"
                )
                continue

            raise ValueError(f"{feature} cannot be missing.")

        if isinstance(value, bool):
            raise ValueError(f"{feature} must be numeric.")

        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc

        if not math.isfinite(number):
            raise ValueError(f"{feature} must be finite.")

        if feature in (
            "Air temperature [K]",
            "Process temperature [K]",
        ) and number <= 0:
            raise ValueError(
                f"{feature} must be above absolute zero (0 K)."
            )

        if feature in (
            "Rotational speed [rpm]",
            "Tool wear [min]",
        ) and number < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

        if feature == "Torque [Nm]" and number < 0:
            warnings.append(
                "Torque [Nm]: negative value detected; "
                "review sensor direction and sign convention."
            )

        observed_range = AI4I_OBSERVED_RANGES.get(feature)

        if observed_range is not None:
            lower, upper = observed_range

            if number < lower or number > upper:
                warnings.append(
                    f"{feature}: value {number:g} is outside the "
                    f"AI4I observed training range "
                    f"[{lower:g}, {upper:g}]. This is an "
                    "out-of-distribution warning, not a safety limit."
                )

        validated[feature] = number

    machine_type = sensor_data["Type"]

    if not isinstance(machine_type, str):
        raise ValueError("Machine Type must be a string.")

    machine_type = machine_type.strip().upper()

    if machine_type not in VALID_MACHINE_TYPES:
        raise ValueError("Machine Type must be L, M, or H.")

    validated["Type"] = machine_type

    return validated, warnings
