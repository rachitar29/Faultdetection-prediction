
import pandas as pd

from app.ml.datasets.dataset_profiler import (
    profile_dataframe,
    save_profile
)


def main():
    df = pd.DataFrame({
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
        "Machine failure": [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        ],
        "Machine type": [
            "L", "L", "M", "L", "H",
            "M", "L", "H", "M", "L"
        ]
    })

    profile = profile_dataframe(
        df,
        target_column="Machine failure"
    )

    print("\nDATASET PROFILE")
    print("=" * 45)

    print("\nOverview:")
    print(profile["dataset_overview"])

    print("\nNumeric columns:")
    for name, details in profile["numeric_summary"].items():
        print(name, details)

    print("\nCategorical columns:")
    for name, details in profile["categorical_summary"].items():
        print(name, details)

    print("\nTarget distribution:")
    print(profile["target_summary"])

    save_profile(
        profile,
        "ml_artifacts/reports/test_dataset_profile.json"
    )

    assert profile["dataset_overview"]["rows"] == 10
    assert "Torque [Nm]" in profile["numeric_summary"]
    assert profile["target_summary"]["target_column"] == "Machine failure"

    print("\nPROFILE TEST PASSED!")
    print("Report saved successfully.")


if __name__ == "__main__":
    main()
