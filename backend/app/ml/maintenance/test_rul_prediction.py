
"""
Integrated Failure Prediction + NASA RUL Test

AI4I:
- Failure classification
- Anomaly detection
- Risk assessment
- SHAP explanation

NASA C-MAPSS:
- Remaining Useful Life prediction

No models are retrained.
"""

from pathlib import Path

import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.maintenance.integrated_engine import (
    predict_equipment_condition,
)


BASE_DIR = Path(__file__).resolve().parents[3]

AI4I_PATH = (
    BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

NASA_PATH = (
    BASE_DIR
    / "ml_data"
    / "raw"
    / "CMAPSSData"
    / "test_FD001.txt"
)

RUL_MODEL_NAME = "rul_cmapss_fd001_best"

AI4I_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

NASA_FEATURES = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

NASA_COLUMNS = ["engine_id", "cycle"] + NASA_FEATURES


def load_ai4i_input():
    if not AI4I_PATH.exists():
        raise FileNotFoundError(
            f"AI4I dataset missing: {AI4I_PATH}"
        )

    df = pd.read_csv(AI4I_PATH)

    missing = [
        f for f in AI4I_FEATURES
        if f not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing AI4I features: {missing}"
        )

    row = df.iloc[0][AI4I_FEATURES].to_dict()

    for feature in AI4I_FEATURES[:-1]:
        row[feature] = float(row[feature])

    row["Type"] = str(row["Type"]).strip().upper()

    return row


def load_nasa_input():
    if not NASA_PATH.exists():
        raise FileNotFoundError(
            f"NASA dataset missing: {NASA_PATH}"
        )

    bundle = load_model_bundle(RUL_MODEL_NAME)

    if not bundle:
        raise FileNotFoundError(
            f"RUL model not found: {RUL_MODEL_NAME}"
        )

    expected = bundle["metadata"].get(
        "feature_names", []
    )

    if len(expected) != 24:
        raise ValueError(
            f"Expected 24 model features, got {len(expected)}"
        )

    df = pd.read_csv(
        NASA_PATH,
        sep=r"\s+",
        header=None,
    )

    if df.shape[1] != 26:
        raise ValueError(
            f"Expected 26 NASA columns, got {df.shape[1]}"
        )

    df.columns = NASA_COLUMNS

    engine_id = int(df["engine_id"].iloc[0])

    engine = df[df["engine_id"] == engine_id]

    latest = engine.sort_values("cycle").iloc[-1]

    rul_input = {
        feature: float(latest[feature])
        for feature in expected
    }

    return engine_id, int(latest["cycle"]), rul_input


def main():
    print("\nINTEGRATED FAILURE + RUL TEST")
    print("=" * 55)

    print("\nLoading AI4I input...")
    ai4i_input = load_ai4i_input()

    print("Loading NASA input...")
    engine_id, cycle, rul_input = load_nasa_input()

    print("NASA engine:", engine_id)
    print("NASA cycle:", cycle)
    print("NASA feature count:", len(rul_input))

    print("\nRunning unified prediction...")

    result = predict_equipment_condition(
        sensor_data=ai4i_input,
        rul_sensor_data=rul_input,
    )

    required = [
        "failure_probability",
        "risk_level",
        "anomaly_results",
        "anomaly_assessment",
        "top_factors",
        "rul_prediction",
        "rul_available",
        "prediction_components",
    ]

    missing = [
        key for key in required
        if key not in result
    ]

    if missing:
        raise AssertionError(
            f"Missing output fields: {missing}"
        )

    if result["rul_available"] is not True:
        raise AssertionError("RUL branch did not execute")

    print("\nFAILURE PREDICTION")
    print("Probability:", result["failure_probability"])
    print("Risk:", result["risk_level"])

    print("\nANOMALY RESULTS")

    for name, value in result["anomaly_results"].items():
        print(name, value)

    print("\nSHAP TOP FACTORS")
    print(result["top_factors"])

    print("\nNASA RUL RESULT")
    print(result["rul_prediction"])

    print("\nCOMPONENT STATUS")

    for name, enabled in result["prediction_components"].items():
        print(f"{name}: {enabled}")

    print("\n" + "=" * 55)
    print("INTEGRATED FAILURE + RUL TEST PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
