
import json
import pandas as pd

from app.ml.explainability.feature_analysis import (
    run_feature_analysis
)

from app.ml.maintenance.risk_engine import (
    calculate_machine_risk,
    calculate_machine_health_score
)

from app.ml.maintenance.recommendations import (
    generate_maintenance_recommendations
)


MODEL_NAME = "ai4i_random_forest_optimized_60_20_20"

SAMPLE_SENSOR_DATA = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 45.0,
    "Tool wear [min]": 100,
    "Type": "M"
}


def main():
    print("\n1. TESTING RISK SCORING")

    risk = calculate_machine_risk(
        MODEL_NAME,
        SAMPLE_SENSOR_DATA
    )

    print(json.dumps(risk, indent=4))

    print("\n2. TESTING HEALTH SCORE")

    health = calculate_machine_health_score(
        MODEL_NAME,
        SAMPLE_SENSOR_DATA
    )

    print(json.dumps(health, indent=4))

    print("\n3. TESTING MAINTENANCE RECOMMENDATIONS")

    recommendations = generate_maintenance_recommendations(
        MODEL_NAME,
        SAMPLE_SENSOR_DATA
    )

    print(json.dumps(recommendations, indent=4))

    print("\n4. TESTING FEATURE IMPORTANCE AND SHAP")

    dataset_path = "ml_data/raw/ai4i2020.csv"

    df = pd.read_csv(dataset_path)

    X = df.drop(
        columns=[
            "Machine failure",
            "UDI",
            "Product ID",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF"
        ],
        errors="ignore"
    ).head(100)

    analysis = run_feature_analysis(
        MODEL_NAME,
        X,
        run_shap=True
    )

    print(json.dumps(analysis, indent=4, default=str))

    print("\nALL TESTS FINISHED")


if __name__ == "__main__":
    main()
