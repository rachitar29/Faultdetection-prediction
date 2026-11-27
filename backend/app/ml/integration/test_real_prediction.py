
"""
Real ML integration smoke test.

Loads one actual AI4I dataset row and runs it through
the registered failure and anomaly models.
"""

from pathlib import Path
import json

import pandas as pd

from app.ml.integration.ml_service import MLService


BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

REQUIRED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]


def run_real_prediction():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    missing = [
        column
        for column in REQUIRED_FEATURES
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required features: {missing}"
        )

    # Select one actual sensor record.
    sensor_data = (
        dataset.iloc[0][REQUIRED_FEATURES].to_dict()
    )

    # Initialize integration service.
    service = MLService()

    # Run actual registered ML models.
    result = service.predict({
        "equipment_id": "AI4I-SMOKE-TEST-001",
        "sensor_data": sensor_data,
    })

    print("\n========== REAL ML PREDICTION ==========")

    print(
        json.dumps(
            result,
            indent=4,
            allow_nan=False
        )
    )

    assert result["success"] is True
    assert result["failure_prediction"] in [
        "NORMAL",
        "FAILURE",
    ]

    assert result["risk_level"] in [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    assert isinstance(
        result["anomaly_detected"],
        bool
    )

    assert result["rul_available"] is False

    print("\nREAL ML PREDICTION TEST PASSED")


if __name__ == "__main__":
    run_real_prediction()
