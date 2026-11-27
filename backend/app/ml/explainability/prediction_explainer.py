
"""Cached SHAP explanations for registered equipment prediction models."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from app.ml.algorithms.model_registry import load_model_bundle
from app.ml.algorithms.model_registry import REGISTRY_DIR, _safe_model_name

from app.ml.data.data_loader import load_dataset
from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


BACKGROUND_SIZE = 100
TOP_FEATURES = 3


def clean_feature_names(feature_names):
    """Remove preprocessing prefixes from transformed feature names."""

    cleaned = []

    for name in feature_names:
        name = str(name)

        for prefix in (
            "numeric__",
            "categorical__",
            "num__",
            "cat__",
        ):
            name = name.replace(prefix, "")

        cleaned.append(name)

    return cleaned


def to_dense_array(data):
    """Convert sparse transformed data into a NumPy array."""

    if hasattr(data, "toarray"):
        data = data.toarray()

    return np.asarray(data)


def select_failure_shap_values(values):
    """Extract SHAP values for the failure class."""

    if isinstance(values, list):
        if len(values) < 2:
            raise ValueError(
                "SHAP output does not contain failure class values."
            )

        return np.asarray(values[1])

    values = np.asarray(values)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        if values.shape[1] == 2:
            return values[:, 1, :]

    if values.ndim == 2:
        return values

    raise ValueError(
        f"Unexpected SHAP output shape: {values.shape}"
    )


# ==========================================
# MODEL ARTIFACT VERSION
# ==========================================

def get_model_artifact_version(model_name):
    """
    Return model artifact identity and version information.

    The timestamp and file size detect ordinary artifact
    replacements, including replacements outside the registry.

    Returns:
        tuple: (model_path, modified_ns, file_size)
    """

    safe_name = _safe_model_name(model_name)

    model_path = (
        REGISTRY_DIR
        / safe_name
        / "model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    model_path = model_path.resolve()
    model_stat = model_path.stat()

    return (
        str(model_path),
        model_stat.st_mtime_ns,
        model_stat.st_size,
    )


# ==========================================
# CACHED EXPLANATION CONTEXT
# ==========================================

@lru_cache(maxsize=4)
def _load_explanation_context(
    model_name,
    model_path,
    modified_ns,
    file_size,
):
    """
    Load and cache the model, background data and SHAP explainer.

    Artifact version information is part of the cache key.
    """

    # 1. Load registered model.
    bundle = load_model_bundle(model_name)

    model = bundle["model"]
    metadata = bundle["metadata"]

    # 2. Validate model pipeline.
    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Registered model must be a fitted sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise ValueError(
            "Model pipeline has no preprocessor step."
        )

    if "classifier" not in model.named_steps:
        raise ValueError(
            "Model pipeline has no classifier step."
        )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # 3. Identify original model features.
    feature_names = metadata.get("feature_names")

    if not feature_names:
        feature_names = (
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        )

    # 4. Load background dataset.
    dataset = load_dataset()

    missing_features = [
        feature
        for feature in feature_names
        if feature not in dataset.columns
    ]

    if missing_features:
        raise ValueError(
            "Background dataset is missing model features: "
            f"{missing_features}"
        )

    background_size = min(
        BACKGROUND_SIZE,
        len(dataset)
    )

    background_df = dataset.sample(
        n=background_size,
        random_state=42,
    ).loc[:, feature_names].copy()

    # 5. Transform background data.
    background_transformed = preprocessor.transform(
        background_df
    )

    background_transformed = to_dense_array(
        background_transformed
    )

    # 6. Get transformed feature names.
    transformed_feature_names = clean_feature_names(
        preprocessor.get_feature_names_out()
    )

    # 7. Create SHAP explainer.
    explainer = shap.TreeExplainer(
        classifier,
        data=background_transformed,
        feature_perturbation="interventional",
        model_output="probability",
    )

    return {
        "model": model,
        "metadata": metadata,
        "preprocessor": preprocessor,
        "explainer": explainer,
        "feature_names": feature_names,
        "transformed_feature_names": transformed_feature_names,
        "background_size": background_size,
        "artifact_version": {
            "model_path": model_path,
            "modified_ns": modified_ns,
            "file_size": file_size,
        },
    }


# ==========================================
# EXPLAIN MACHINE PREDICTION
# ==========================================

def explain_machine_prediction(
    sensor_data,
    model_name="XGBoost",
    top_n=TOP_FEATURES,
):
    """
    Explain one machine prediction using cached SHAP components.

    Positive SHAP value:
        Pushes the prediction towards failure.

    Negative SHAP value:
        Pushes the prediction away from failure.
    """

    # 1. Get the current model artifact version.
    model_path, modified_ns, file_size = (
        get_model_artifact_version(model_name)
    )

    # 2. Retrieve the context for this exact artifact version.
    context = _load_explanation_context(
        model_name,
        model_path,
        modified_ns,
        file_size,
    )

    feature_names = context["feature_names"]

    # 3. Validate input features.
    missing_features = [
        feature
        for feature in feature_names
        if feature not in sensor_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing explanation input features: {missing_features}"
        )

    input_df = pd.DataFrame(
        [sensor_data],
        columns=feature_names,
    )

    # 4. Transform the new reading.
    input_transformed = context["preprocessor"].transform(
        input_df
    )

    input_transformed = to_dense_array(
        input_transformed
    )

    # 5. Calculate SHAP values.
    raw_values = context["explainer"].shap_values(
        input_transformed
    )

    failure_values = select_failure_shap_values(
        raw_values
    )[0]

    transformed_feature_names = (
        context["transformed_feature_names"]
    )

    if len(failure_values) != len(transformed_feature_names):
        raise ValueError(
            "SHAP values and transformed feature names do not match."
        )

    # 6. Rank feature contributions.
    contributions = []

    for index, shap_value in enumerate(failure_values):

        shap_value = float(shap_value)

        contributions.append({
            "feature": transformed_feature_names[index],
            "shap_value": round(shap_value, 6),
            "impact": (
                "increases failure risk"
                if shap_value > 0
                else "decreases failure risk"
                if shap_value < 0
                else "no meaningful contribution"
            ),
            "absolute_impact": abs(shap_value),
        })

    contributions.sort(
        key=lambda item: item["absolute_impact"],
        reverse=True,
    )

    top_factors = contributions[:top_n]

    for factor in top_factors:
        factor.pop("absolute_impact")

    # 7. Return explanation.
    return {
        "model_name": model_name,
        "explained_class": "Machine failure",
        "explanation_method": "SHAP TreeExplainer",
        "top_factors": top_factors,
        "feature_count": len(transformed_feature_names),
        "background_sample_count": context["background_size"],
    }


# ==========================================
# CACHE MANAGEMENT
# ==========================================

def clear_explanation_cache():
    """Clear all cached SHAP explanation contexts."""

    _load_explanation_context.cache_clear()


def get_explanation_cache_info():
    """Return SHAP cache statistics for diagnostics."""

    return _load_explanation_context.cache_info()
