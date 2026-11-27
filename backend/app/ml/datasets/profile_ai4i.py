
import json
from pathlib import Path
from pprint import pprint

from app.ml.datasets.dataset_validator import load_dataset
from app.ml.datasets.dataset_profiler import (
    profile_dataframe,
    save_profile
)


def main():

    backend_dir = Path(__file__).resolve().parents[3]

    dataset_path = (
        backend_dir / "ml_data" / "raw" / "ai4i2020.csv"
    )

    report_path = (
        backend_dir
        / "ml_artifacts"
        / "reports"
        / "ai4i_full_profile.json"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"AI4I dataset not found: {dataset_path}"
        )

    print("=" * 60)
    print("AI4I 2020 FULL DATASET PROFILING")
    print("=" * 60)

    # Load original CSV without modifying it.
    dataframe = load_dataset(dataset_path)

    print("\nDATASET DIMENSIONS")
    print("Rows:", len(dataframe))
    print("Columns:", len(dataframe.columns))

    print("\nCOLUMN NAMES")
    for column in dataframe.columns:
        print("-", column)

    # Generate complete profile.
    profile = profile_dataframe(
        dataframe,
        target_column="Machine failure"
    )

    # Save complete report.
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_profile(profile, report_path)

    print("\nPROFILE CONTENT")
    pprint(profile, sort_dicts=False)

    print("\nREPORT SAVED:")
    print(report_path)

    print("\nAI4I FULL DATASET PROFILING COMPLETED!")


if __name__ == "__main__":
    main()
