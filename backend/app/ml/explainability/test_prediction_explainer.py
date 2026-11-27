
from app.ml.explainability.prediction_explainer import (
    explain_machine_prediction
)


def main():

    sample_machine = {
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 120,
        "Type": "L"
    }

    print("\nSHAP PREDICTION EXPLANATION")
    print("=" * 60)

    explanation = explain_machine_prediction(
        sensor_data=sample_machine,
        model_name="XGBoost",
        top_n=3
    )

    print("\nTOP CONTRIBUTING FACTORS")

    for index, factor in enumerate(
        explanation["top_factors"],
        start=1
    ):
        print(f"\n{index}. {factor['feature']}")
        print("Impact:", factor["impact"])
        print("SHAP value:", factor["shap_value"])

    print("\nEXPLANATION DETAILS")

    for key, value in explanation.items():
        if key != "top_factors":
            print(f"{key}: {value}")

    print("\nSHAP PREDICTION EXPLANATION COMPLETED")


if __name__ == "__main__":
    main()
