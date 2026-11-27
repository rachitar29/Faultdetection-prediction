
from pprint import pprint

from app.ml.maintenance.integrated_engine import (
    predict_equipment_condition
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

    print("\nINTEGRATED EQUIPMENT CONDITION TEST")
    print("=" * 60)

    result = predict_equipment_condition(
        sensor_data=sample_machine
    )

    print("\nFAILURE PREDICTION")
    print("-" * 60)

    print("Model:", result["model_name"])
    print("Failure probability:", result["failure_probability"])
    print("Decision threshold:", result["decision_threshold"])
    print("Failure predicted:", result["failure_predicted"])

    print("\nANOMALY DETECTION")
    print("-" * 60)
    pprint(result["anomaly_results"])

    print("\nRISK ASSESSMENT")
    print("-" * 60)
    print("Classifier risk:", result["classifier_risk_level"])
    print("Overall risk:", result["risk_level"])
    print("Maintenance priority:", result["maintenance_priority"])
    print("Recommendation:", result["maintenance_recommendation"])

    print("\nRUL STATUS")
    print("-" * 60)
    print("RUL available:", result["rul_available"])
    print("RUL prediction:", result["rul_prediction"])

    print("\nINTEGRATION COMPONENTS")
    print("-" * 60)
    pprint(result["prediction_components"])

    assert result["decision_threshold"] == round(
        0.824862539768219, 6
    )

    assert result["integrated_prediction"] is True
    assert result["rul_available"] is False

    print("\nINTEGRATED PREDICTION TEST PASSED")


if __name__ == "__main__":
    main()
