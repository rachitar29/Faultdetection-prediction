
from pathlib import Path
import json

import numpy as np
import pandas as pd


def profile_dataframe(df, target_column=None, top_categories=10):
    """
    Generate a detailed profile of a tabular dataset.

    Includes:
    - Dataset dimensions
    - Column data types
    - Missing values
    - Numeric statistics
    - Categorical statistics
    - Outlier estimates using the IQR method
    - Numeric correlations
    - Target distribution
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    if top_categories < 1:
        raise ValueError("top_categories must be at least 1.")

    rows, columns = df.shape

    # Basic information
    profile = {
        "dataset_overview": {
            "rows": int(rows),
            "columns": int(columns),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / (1024 ** 2), 3
            )
        },
        "columns": {},
        "numeric_summary": {},
        "categorical_summary": {},
        "correlations": {},
        "target_summary": None
    }

    # Analyze every column
    for column in df.columns:
        series = df[column]
        missing_count = int(series.isna().sum())

        column_info = {
            "data_type": str(series.dtype),
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_count / rows * 100, 2
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "unique_percentage": round(
                series.nunique(dropna=True) / rows * 100, 2
            ),
            "is_constant": bool(series.nunique(dropna=True) <= 1)
        }

        profile["columns"][str(column)] = column_info

        # Numeric column analysis
        if pd.api.types.is_numeric_dtype(series):
            clean = series.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            if not clean.empty:
                q1 = float(clean.quantile(0.25))
                q3 = float(clean.quantile(0.75))
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outlier_count = int(
                    ((clean < lower_bound) |
                     (clean > upper_bound)).sum()
                )

                profile["numeric_summary"][str(column)] = {
                    "mean": float(clean.mean()),
                    "median": float(clean.median()),
                    "standard_deviation": float(clean.std()),
                    "minimum": float(clean.min()),
                    "maximum": float(clean.max()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(iqr),
                    "estimated_outlier_count": outlier_count,
                    "estimated_outlier_percentage": round(
                        outlier_count / len(clean) * 100, 2
                    )
                }
            else:
                profile["numeric_summary"][str(column)] = {
                    "message": "No valid numeric values."
                }

        # Categorical column analysis
        else:
            counts = series.value_counts(dropna=True)

            profile["categorical_summary"][str(column)] = {
                "unique_values": int(series.nunique(dropna=True)),
                "most_common_values": {
                    str(key): int(value)
                    for key, value in counts.head(
                        top_categories
                    ).items()
                }
            }

    # Correlation matrix for numeric columns
    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df.replace(
        [np.inf, -np.inf], np.nan
    )

    if numeric_df.shape[1] >= 2:
        correlation = numeric_df.corr()

        profile["correlations"] = {
            str(column): {
                str(other): (
                    None if pd.isna(value) else round(float(value), 4)
                )
                for other, value in correlation[column].items()
            }
            for column in correlation.columns
        }

    # Target column analysis
    if target_column is not None:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' does not exist."
            )

        target = df[target_column]
        counts = target.value_counts(dropna=False)

        profile["target_summary"] = {
            "target_column": target_column,
            "unique_values": int(target.nunique(dropna=True)),
            "missing_values": int(target.isna().sum()),
            "class_distribution": {
                str(key): int(value)
                for key, value in counts.items()
            }
        }

    return profile


def save_profile(profile, output_path):
    """Save the profile report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert_numpy(value):
        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating,)):
            return float(value)

        if isinstance(value, np.ndarray):
            return value.tolist()

        return str(value)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            profile,
            file,
            indent=4,
            ensure_ascii=False,
            default=convert_numpy,
            allow_nan=False
        )

    return str(path)


def profile_dataset(
    file_path,
    target_column=None,
    output_path=None
):
    """
    Load a CSV or Excel file, profile it,
    and optionally save the report.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    extension = path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(path)
    elif extension in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            "Only CSV, XLSX and XLS files are supported."
        )

    profile = profile_dataframe(
        df,
        target_column=target_column
    )

    if output_path:
        profile["dataset_overview"]["source_file"] = path.name
        profile["dataset_overview"]["profile_report"] = save_profile(
            profile,
            output_path
        )

    return profile
