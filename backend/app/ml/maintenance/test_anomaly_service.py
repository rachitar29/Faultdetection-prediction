
from app.ml.maintenance.anomaly_service import (
    detect_machine_anomalies,
    get_anomaly_flags
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

    print("\nSAVED ANOMALY MODEL TEST")
    print("=" * 60)

    results = detect_machine_anomalies(
        sample_machine
    )

    for model_name, result in results.items():

        print("\nMODEL:", model_name)
        print("Anomaly detected:", result["is_anomaly"])
        print("Anomaly score:", result["anomaly_score"])
        print("Optimized threshold:", result["optimized_threshold"])

    flags = get_anomaly_flags(results)

    print("\nANOMALY FLAGS FOR RISK ASSESSMENT")
    print(flags)

    print("\nANOMALY SERVICE TEST COMPLETED")


if __name__ == "__main__":
    main()
