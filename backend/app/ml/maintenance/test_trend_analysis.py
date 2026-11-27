
from pprint import pprint

from app.ml.maintenance.trend_analysis import (
    analyse_sensor_history
)


def main():

    print("\nSENSOR TREND ANALYSIS TEST")
    print("=" * 60)

    # Synthetic readings for testing only.
    # These are not real machine measurements.

    readings = [
        {
            "timestamp": "2026-11-16T09:00:00Z",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 100
        },
        {
            "timestamp": "2026-11-16T09:15:00Z",
            "Air temperature [K]": 301.0,
            "Process temperature [K]": 310.5,
            "Rotational speed [rpm]": 1490,
            "Torque [Nm]": 42,
            "Tool wear [min]": 105
        },
        {
            "timestamp": "2026-11-16T09:30:00Z",
            "Air temperature [K]": 302.0,
            "Process temperature [K]": 311.0,
            "Rotational speed [rpm]": 1480,
            "Torque [Nm]": 44,
            "Tool wear [min]": 110
        },
        {
            "timestamp": "2026-11-16T09:45:00Z",
            "Air temperature [K]": 303.0,
            "Process temperature [K]": 312.0,
            "Rotational speed [rpm]": 1470,
            "Torque [Nm]": 46,
            "Tool wear [min]": 115
        }
    ]

    # Example limits are illustrative test settings,
    # not manufacturer-approved operating limits.

    operating_limits = {
        "Torque [Nm]": {
            "min": 0,
            "max": 45
        },
        "Air temperature [K]": {
            "min": 290,
            "max": 305
        }
    }

    result = analyse_sensor_history(
        readings=readings,
        operating_limits=operating_limits
    )

    print("\nTREND SUMMARY")
    print(result["summary"])

    print("\nINCREASING FEATURES")
    pprint(result["increasing_features"])

    print("\nDECREASING FEATURES")
    pprint(result["decreasing_features"])

    print("\nSTABLE FEATURES")
    pprint(result["stable_features"])

    print("\nOPERATING LIMIT ALERTS")
    pprint(result["operating_limit_alerts"])

    print("\nDETAILED SENSOR TRENDS")
    pprint(result["trend_analysis"])

    print("\nSENSOR TREND ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
