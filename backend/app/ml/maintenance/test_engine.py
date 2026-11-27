
from app.ml.maintenance.engine import (
    predict_equipment_health
)


def main():

    # Example sensor reading using AI4I feature names

    sample_machine = {
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 120,
        "Type": "L"
    }

    print("\nEQUIPMENT HEALTH PREDICTION")
    print("=" * 55)

    result = predict_equipment_health(sample_machine)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nPREDICTION TEST COMPLETED")


if __name__ == "__main__":
    main()
