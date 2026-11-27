
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.ml.algorithms.model_registry import load_model_bundle


BASE_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BASE_DIR / "ml_artifacts" / "feature_analysis"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(model_name):
    bundle = load_model_bundle(model_name)

    if not bundle:
        raise ValueError(f"Model not found: {model_name}")

    return bundle["model"], bundle["metadata"]


def _get_transformed_data(model_bundle, X):
    estimator = model_bundle.estimator
    raw_X = model_bundle.input_preprocessor._prepare_input(X)

    if not hasattr(estimator, "steps") or len(estimator.steps) < 2:
        raise ValueError("Expected a preprocessing and estimator pipeline.")

    preprocessing_pipeline = estimator[:-1]
    transformed = preprocessing_pipeline.transform(raw_X)

    final_estimator = estimator.steps[-1][1]

    try:
        feature_names = preprocessing_pipeline.get_feature_names_out()
    except Exception:
        feature_names = [
            f"feature_{i}" for i in range(transformed.shape[1])
        ]

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed), list(feature_names), final_estimator


def calculate_feature_importance(model_name, X):
    """Calculate tree-based feature importance."""

    model_bundle, metadata = _load_model(model_name)

    transformed, feature_names, final_estimator = (
        _get_transformed_data(model_bundle, X)
    )

    if not hasattr(final_estimator, "feature_importances_"):
        raise ValueError(
            "This model does not expose tree-based feature importance."
        )

    importances = final_estimator.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError("Feature importance dimensions do not match.")

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    output_path = REPORT_DIR / f"{model_name}_feature_importance.csv"
    result.to_csv(output_path, index=False)

    return {
        "model_name": model_name,
        "features": result.to_dict(orient="records"),
        "report_path": str(output_path)
    }


def plot_feature_importance(model_name, X, top_n=15):
    result = calculate_feature_importance(model_name, X)

    importance_df = pd.DataFrame(result["features"]).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()

    output_path = REPORT_DIR / f"{model_name}_feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    result["plot_path"] = str(output_path)

    return result


def calculate_shap_values(model_name, X, max_samples=100):
    """Calculate SHAP values for tree-based classification models."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install it with: pip install shap"
        ) from exc

    model_bundle, metadata = _load_model(model_name)

    transformed, feature_names, final_estimator = (
        _get_transformed_data(model_bundle, X)
    )

    if len(transformed) == 0:
        raise ValueError("No input rows supplied.")

    sample = transformed[:max_samples]

    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(sample)

    # Handle different SHAP output formats.
    if isinstance(shap_values, list):
        values = shap_values[-1]
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[:, :, -1]

    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    mean_abs = np.abs(values).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": mean_abs
    }).sort_values("mean_absolute_shap", ascending=False)

    output_path = REPORT_DIR / f"{model_name}_shap_importance.csv"
    importance.to_csv(output_path, index=False)

    plt.figure()
    shap.summary_plot(
        values,
        sample,
        feature_names=feature_names,
        show=False
    )

    plot_path = REPORT_DIR / f"{model_name}_shap_summary.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "model_name": model_name,
        "sample_count": len(sample),
        "features": importance.to_dict(orient="records"),
        "csv_path": str(output_path),
        "plot_path": str(plot_path)
    }


def run_feature_analysis(model_name, X, run_shap=True):
    """Run feature importance and optional SHAP analysis."""

    result = {
        "model_name": model_name,
        "feature_importance": None,
        "shap": None,
        "errors": []
    }

    try:
        result["feature_importance"] = plot_feature_importance(
            model_name,
            X
        )
    except Exception as exc:
        result["errors"].append({
            "component": "feature_importance",
            "error": str(exc)
        })

    if run_shap:
        try:
            result["shap"] = calculate_shap_values(
                model_name,
                X
            )
        except Exception as exc:
            result["errors"].append({
                "component": "shap",
                "error": str(exc)
            })

    report_path = REPORT_DIR / f"{model_name}_analysis.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, default=str)

    result["report_path"] = str(report_path)

    return result
