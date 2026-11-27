
import pandas as pd

from app.ml.datasets.schema_detector import (
    detect_schema,
    print_schema_report
)


def main():

    df = pd.DataFrame({
        "UDI": range(1, 11),

        "Air temperature [K]": [
            298.1, 298.2, 298.3, 298.4, 298.5,
            298.6, 298.7, 298.8, 298.9, 299.0
        ],

        "Rotational speed [rpm]": [
            1500, 1510, 1520, 1530, 1540,
            1550, 1560, 1570, 1580, 1590
        ],

        "Torque [Nm]": [
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49
        ],

        "Type": [
            "L", "L", "M", "L", "H",
            "M", "L", "H", "M", "L"
        ],

        "Machine failure": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        ],

        "TWF": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        ],

        "Constant column": [1] * 10
    })

    schema = detect_schema(
        df,
        target_column="Machine failure",
        exclude_columns=["TWF"]
    )

    print_schema_report(schema)

    assert "Air temperature [K]" in schema["numeric_features"]
    assert "Torque [Nm]" in schema["numeric_features"]
    assert "Type" in schema["categorical_features"]

    assert "UDI" in schema["id_columns"]
    assert "Machine failure" in schema["possible_target_columns"]

    assert "TWF" in schema["excluded_columns"]
    assert "Constant column" in schema["constant_columns"]

    print("\nSCHEMA DETECTION TEST PASSED!")


if __name__ == "__main__":
    main()
