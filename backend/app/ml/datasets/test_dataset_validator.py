
import pandas as pd

from app.ml.datasets.dataset_validator import validate_dataframe


def main():
    df = pd.DataFrame({
        "Air temperature [K]": [298.1, 298.2, 298.3, 298.4, 298.5,
                                298.6, 298.7, 298.8, 298.9, 299.0],
        "Rotational speed [rpm]": [1500, 1510, 1520, 1530, 1540,
                                    1550, 1560, 1570, 1580, 1590],
        "Torque [Nm]": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        "Machine failure": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    })

    result = validate_dataframe(
        df,
        target_column="Machine failure",
        min_rows=10
    )

    print("\nDATASET VALIDATION RESULT")
    print("=" * 40)

    print("Valid:", result["valid"])
    print("Errors:", result["errors"])
    print("Warnings:", result["warnings"])

    print("\nSTATISTICS")
    for key, value in result["statistics"].items():
        print(f"{key}: {value}")

    assert result["valid"] is True
    print("\nTEST PASSED!")


if __name__ == "__main__":
    main()
