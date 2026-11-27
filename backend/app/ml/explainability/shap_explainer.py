
from pathlib import Path

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from app.ml.data.data_loader import load_dataset
from app.ml.data.preprocessing import prepare_data


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "trained_models"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "shap_explanations"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FEATURE NAME CLEANING
# ==========================================

def clean_feature_names(feature_names):

    cleaned = []

    for name in feature_names:

        name = str(name)

        name = name.replace("numeric__", "")
        name = name.replace("categorical__", "")
        name = name.replace("num__", "")
        name = name.replace("cat__", "")

        cleaned.append(name)

    return cleaned


# ==========================================
# SELECT FAILURE CLASS SHAP VALUES
# ==========================================

def get_failure_shap_values(shap_values):

    # Older SHAP versions may return a list:
    # [normal_class_values, failure_class_values]

    if isinstance(shap_values, list):

        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)

    # Newer SHAP versions may return:
    # (samples, features, classes)

    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        # Alternative output layout:
        # (samples, classes, features)

        if values.shape[1] == 2:
            return values[:, 1, :]

    if values.ndim != 2:

        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    return values


# ==========================================
# SELECT FAILURE CLASS BASE VALUE
# ==========================================

def get_failure_base_value(base_values, sample_count):

    values = np.asarray(base_values)

    if values.ndim == 0:
        return np.full(sample_count, float(values))

    if values.ndim == 1:

        if len(values) == 2:
            return np.full(sample_count, float(values[1]))

        if len(values) == sample_count:
            return values

    if values.ndim == 2:

        if values.shape[1] == 2:
            return values[:, 1]

        if values.shape[0] == 2:
            return values[1, :]

    raise ValueError(
        f"Unexpected SHAP base value shape: {values.shape}"
    )


# ==========================================
# EXPLAIN ONE CLASSIFICATION MODEL
# ==========================================

def explain_model(
    model_name,
    model_path,
    X_background,
    X_test,
    y_test
):

    print("\n" + "=" * 65)
    print("SHAP EXPLAINABILITY:", model_name)
    print("=" * 65)

    # STEP 1: Load trained pipeline

    if not model_path.exists():

        print("MODEL NOT FOUND:", model_path)
        return None

    model = joblib.load(model_path)

    # Handle model packages if present

    if isinstance(model, dict) and "model" in model:
        model = model["model"]

    if not hasattr(model, "named_steps"):

        raise ValueError(
            f"{model_name} is not a fitted sklearn Pipeline."
        )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # STEP 2: Transform sensor data using
    # the fitted model's own preprocessor

    background_transformed = preprocessor.transform(
        X_background
    )

    test_transformed = preprocessor.transform(
        X_test
    )

    if hasattr(background_transformed, "toarray"):
        background_transformed = background_transformed.toarray()

    if hasattr(test_transformed, "toarray"):
        test_transformed = test_transformed.toarray()

    background_transformed = np.asarray(
        background_transformed
    )

    test_transformed = np.asarray(
        test_transformed
    )

    # STEP 3: Extract feature names

    feature_names = preprocessor.get_feature_names_out()

    feature_names = clean_feature_names(feature_names)

    print("\nTRANSFORMED FEATURES:")
    print(feature_names)

    # STEP 4: Build SHAP TreeExplainer
    # Explain positive-class (failure) probabilities.

    explainer = shap.TreeExplainer(
        classifier,
        data=background_transformed,
        feature_perturbation="interventional",
        model_output="probability"
    )

    # STEP 5: Calculate SHAP values

    print("\nCALCULATING SHAP VALUES...")

    raw_shap_values = explainer.shap_values(
        test_transformed
    )

    shap_values = get_failure_shap_values(
        raw_shap_values
    )

    base_values = get_failure_base_value(
        explainer.expected_value,
        len(test_transformed)
    )

    if shap_values.shape[1] != len(feature_names):

        raise ValueError(
            "SHAP feature count does not match feature names."
        )

    # STEP 6: Create SHAP Explanation object

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=test_transformed,
        feature_names=feature_names
    )

    # STEP 7: Calculate global feature importance

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.mean(
            np.abs(shap_values),
            axis=0
        )
    })

    importance_df = importance_df.sort_values(
        by="mean_absolute_shap",
        ascending=False
    )

    print("\nSHAP FEATURE IMPORTANCE")
    print("-" * 50)

    print(importance_df.to_string(index=False))

    # STEP 8: Save importance CSV

    csv_path = (
        REPORT_DIR /
        f"{model_name.lower().replace(' ', '_')}_shap_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # STEP 9: SHAP beeswarm summary plot

    plt.figure()

    shap.plots.beeswarm(
        explanation,
        max_display=len(feature_names),
        show=False
    )

    plt.title(
        f"{model_name} - SHAP Summary"
    )

    plt.tight_layout()

    summary_path = (
        REPORT_DIR /
        f"{model_name.lower().replace(' ', '_')}_shap_summary.png"
    )

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # STEP 10: SHAP global bar plot

    shap.plots.bar(
        explanation,
        max_display=len(feature_names),
        show=False
    )

    plt.title(
        f"{model_name} - SHAP Feature Importance"
    )

    plt.tight_layout()

    bar_path = (
        REPORT_DIR /
        f"{model_name.lower().replace(' ', '_')}_shap_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # STEP 11: Explain an individual machine

    # Prefer a real failure example in the test sample.
    # Otherwise explain the highest predicted failure probability.

    probabilities = classifier.predict_proba(
        test_transformed
    )[:, 1]

    failure_positions = np.flatnonzero(
        np.asarray(y_test) == 1
    )

    if len(failure_positions) > 0:

        selected_index = int(failure_positions[0])

    else:

        selected_index = int(np.argmax(probabilities))

    individual_explanation = explanation[selected_index]

    # STEP 12: Save waterfall plot

    shap.plots.waterfall(
        individual_explanation,
        max_display=len(feature_names),
        show=False
    )

    plt.title(
        f"{model_name} - Individual Failure Explanation"
    )

    plt.tight_layout()

    waterfall_path = (
        REPORT_DIR /
        f"{model_name.lower().replace(' ', '_')}_waterfall.png"
    )

    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # STEP 13: Display individual prediction

    actual_label = int(
        np.asarray(y_test)[selected_index]
    )

    predicted_label = int(
        probabilities[selected_index] >= 0.5
    )

    print("\nINDIVIDUAL MACHINE EXPLANATION")
    print("-" * 50)

    print("Actual failure label:", actual_label)

    print("Predicted failure:", predicted_label)

    print(
        "Predicted failure probability:",
        round(float(probabilities[selected_index]), 4)
    )

    print("\nSAVED FILES")

    print("Importance CSV:", csv_path)
    print("Summary plot:", summary_path)
    print("Bar plot:", bar_path)
    print("Waterfall plot:", waterfall_path)

    return importance_df


# ==========================================
# RUN SHAP FOR BOTH MODELS
# ==========================================

def run_shap_analysis():

    print("\nLOADING AI4I DATASET")

    df = load_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df)

    # Use a reproducible sample to reduce memory usage.

    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    # Explain up to 200 test observations.

    sample_size = min(200, len(X_test))

    X_explain = X_test.sample(
        n=sample_size,
        random_state=42
    )

    y_explain = y_test.loc[X_explain.index]

    models = {
        "Random Forest": MODEL_DIR / "random_forest.joblib",
        "XGBoost": MODEL_DIR / "xgboost.joblib"
    }

    all_results = []

    for model_name, model_path in models.items():

        importance_df = explain_model(
            model_name,
            model_path,
            background,
            X_explain,
            y_explain
        )

        if importance_df is not None:

            importance_df["model"] = model_name

            all_results.append(importance_df)

    # Save combined comparison

    if all_results:

        comparison_df = pd.concat(
            all_results,
            ignore_index=True
        )

        comparison_path = (
            REPORT_DIR / "shap_model_comparison.csv"
        )

        comparison_df.to_csv(
            comparison_path,
            index=False
        )

        print("\n" + "=" * 65)
        print("SHAP ANALYSIS COMPLETED")
        print("=" * 65)

        print(
            "\nCOMPARISON SAVED:",
            comparison_path
        )

    else:

        raise RuntimeError(
            "No trained classification models were found."
        )


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    run_shap_analysis()
