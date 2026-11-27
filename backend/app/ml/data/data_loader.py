
from pathlib import Path
import pandas as pd


# Project directory
BASE_DIR = Path(__file__).resolve().parents[3]

# Dataset directory
DATA_DIR = BASE_DIR / "ml_data" / "raw"


def load_dataset(file_path=None):
    """
    Load the AI4I predictive maintenance CSV dataset.
    """

    if file_path is None:
        csv_files = list(DATA_DIR.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV file found inside {DATA_DIR}"
            )

        if len(csv_files) > 1:
            raise ValueError(
                "Multiple CSV files found. Specify the dataset path."
            )

        file_path = csv_files[0]

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("\nDATASET LOADED SUCCESSFULLY")
    print("=" * 55)
    print(f"File: {file_path.name}")

    return df


def inspect_dataset(df):
    """
    Inspect dataset structure and quality.
    """

    print("\n1. DATASET DIMENSIONS")
    print("=" * 55)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n2. COLUMN NAMES")
    print("=" * 55)

    for column in df.columns:
        print(column)

    print("\n3. DATA TYPES")
    print("=" * 55)
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print("=" * 55)
    print(df.isnull().sum())

    print("\n5. DUPLICATE RECORDS")
    print("=" * 55)
    print(df.duplicated().sum())

    print("\n6. DATASET PREVIEW")
    print("=" * 55)
    print(df.head())

    print("\n7. NUMERICAL STATISTICS")
    print("=" * 55)
    print(df.describe().T)

    if "Machine failure" in df.columns:
        print("\n8. MACHINE FAILURE DISTRIBUTION")
        print("=" * 55)
        print(df["Machine failure"].value_counts())

        print("\nFailure percentage:")
        print(
            df["Machine failure"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

    failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    available_columns = [
        col for col in failure_columns if col in df.columns
    ]

    if available_columns:
        print("\n9. FAILURE TYPE DISTRIBUTION")
        print("=" * 55)
        print(df[available_columns].sum())

    print("\n10. CORRELATION MATRIX")
    print("=" * 55)

    numeric_df = df.select_dtypes(include="number")
    print(numeric_df.corr().round(2))

    print("\nDATASET INSPECTION COMPLETED")


if __name__ == "__main__":

    dataset = load_dataset()

    inspect_dataset(dataset)
