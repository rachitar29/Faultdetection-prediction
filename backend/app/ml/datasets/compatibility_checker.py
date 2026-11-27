
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Correct project root: backend/
BACKEND_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = (
    BACKEND_ROOT
    / "ml_artifacts"
    / "model_registry"
    / "xgboost"
    / "model.joblib"
)

DEFAULT_METADATA_PATH = (
    BACKEND_ROOT
    / "ml_artifacts"
    / "model_registry"
    / "xgboost"
    / "metadata.json"
)

LEAKAGE_COLUMNS = {
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def check_dataset_compatibility(
    dataframe,
    model_path=DEFAULT_MODEL_PATH,
    metadata_path=DEFAULT_METADATA_PATH,
):
    """
    Check whether an incoming dataset is compatible
    with the registered XGBoost model.

    Does not modify the model, threshold, metadata,
    or input dataframe.
    """

    result = {
        "compatible": False,
        "errors": [],
        "warnings": [],
        "required_features": [],
        "missing_features": [],
        "extra_columns": [],
        "leakage_columns": [],
        "numeric_checks": {},
        "categorical_checks": {},
        "missing_value_counts": {},
        "model_name": None,
    }

    # 1. Validate input
    if not isinstance(dataframe, pd.DataFrame):
        result["errors"].append(
            "Input must be a pandas DataFrame."
        )
        return result

    if dataframe.empty:
        result["errors"].append(
            "Input dataset is empty."
        )
        return result

    # 2. Check artifact paths
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        result["errors"].append(
            f"Model file not found: {model_path}"
        )
        return result

    if not metadata_path.exists():
        result["errors"].append(
            f"Metadata file not found: {metadata_path}"
        )
        return result

    # 3. Load model and metadata
    try:
        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

        model = joblib.load(model_path)

    except Exception as exc:
        result["errors"].append(
            f"Could not load model or metadata: {exc}"
        )
        return result

    result["model_name"] = metadata.get("model_name")

    # 4. Verify trained feature schema
    metadata_features = metadata.get("feature_names", [])

    pipeline_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if pipeline_features is None:
        result["errors"].append(
            "Model does not expose fitted feature names."
        )
        return result

    pipeline_features = list(pipeline_features)

    if not metadata_features:
        result["errors"].append(
            "Model metadata has no feature_names."
        )
        return result

    if metadata_features != pipeline_features:
        result["errors"].append(
            "Model metadata features do not match "
            "the fitted pipeline features."
        )
        return result

    required_features = pipeline_features
    result["required_features"] = required_features

    # 5. Required columns
    missing_features = [
        column
        for column in required_features
        if column not in dataframe.columns
    ]

    result["missing_features"] = missing_features

    if missing_features:
        result["errors"].append(
            f"Missing required features: {missing_features}"
        )

    # 6. Extra and leakage columns
    result["extra_columns"] = [
        column
        for column in dataframe.columns
        if column not in required_features
    ]

    result["leakage_columns"] = [
        column
        for column in dataframe.columns
        if column in LEAKAGE_COLUMNS
    ]

    if result["extra_columns"]:
        result["warnings"].append(
            "Extra columns will not be used for model inference."
        )

    if result["leakage_columns"]:
        result["warnings"].append(
            "Identifier or failure-mode columns detected. "
            "They must not be passed as model features."
        )

    if missing_features:
        return result

    # 7. Numeric feature validation
    for column in NUMERIC_FEATURES:
        original = dataframe[column]

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        invalid_mask = (
            original.notna()
            & converted.isna()
        )

        infinite_mask = (
            converted.notna()
            & ~np.isfinite(converted)
        )

        invalid_count = int(
            invalid_mask.sum() + infinite_mask.sum()
        )

        missing_count = int(
            converted.isna().sum()
        )

        result["numeric_checks"][column] = {
            "valid": invalid_count == 0,
            "invalid_values": invalid_count,
            "missing_values": missing_count,
        }

        result["missing_value_counts"][column] = (
            missing_count
        )

        if invalid_count:
            result["errors"].append(
                f"{column} contains "
                f"{invalid_count} invalid or infinite values."
            )

        if missing_count:
            result["warnings"].append(
                f"{column} contains {missing_count} missing "
                "values. The fitted imputer can handle these."
            )

    # 8. Categorical feature validation
    try:
        preprocessor = model.named_steps["preprocessor"]

        encoder = (
            preprocessor
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )

        accepted_categories = set(
            encoder.categories_[0].tolist()
        )

    except Exception as exc:
        result["errors"].append(
            f"Could not inspect fitted categorical encoder: {exc}"
        )
        return result

    category_series = dataframe["Type"]

    observed_categories = set(
        category_series.dropna().unique().tolist()
    )

    unknown_categories = sorted(
        observed_categories - accepted_categories
    )

    result["categorical_checks"]["Type"] = {
        "accepted_categories": sorted(accepted_categories),
        "observed_categories": sorted(observed_categories),
        "unknown_categories": unknown_categories,
        "missing_values": int(category_series.isna().sum()),
    }

    if unknown_categories:
        result["errors"].append(
            f"Unknown Type categories: {unknown_categories}. "
            f"Expected: {sorted(accepted_categories)}"
        )

    if category_series.isna().any():
        result["warnings"].append(
            "Type contains missing values. "
            "The fitted categorical imputer can handle these."
        )

    # 9. Final decision
    result["compatible"] = len(result["errors"]) == 0

    return result
