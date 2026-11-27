
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "trained_models"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "feature_importance"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FEATURE NAME CLEANING
# ==========================================

def clean_feature_names(feature_names):

    cleaned = []

    for name in feature_names:

        name = str(name)

        name = name.replace("num__", "")
        name = name.replace("cat__", "")
        name = name.replace("numeric__", "")
        name = name.replace("categorical__", "")

        cleaned.append(name)

    return cleaned


# ==========================================
# EXTRACT MODEL AND PREPROCESSOR
# ==========================================

def extract_feature_importance(model_path):

    saved_model = joblib.load(model_path)

    # Support both raw pipelines and model packages
    if isinstance(saved_model, dict) and "model" in saved_model:
        model = saved_model["model"]
    else:
        model = saved_model

    estimator = None
    feature_names = None

    # Check pipeline steps
    if hasattr(model, "named_steps"):

        for step_name, step in model.named_steps.items():

            # Find the trained classifier
            if hasattr(step, "feature_importances_"):
                estimator = step

            # Find transformed feature names
            if hasattr(step, "get_feature_names_out"):

                try:
                    feature_names = step.get_feature_names_out()
                except Exception:
                    pass

    # Support standalone estimators
    elif hasattr(model, "feature_importances_"):

        estimator = model

    if estimator is None:

        raise ValueError(
            f"No feature importance found in {model_path.name}"
        )

    importances = np.asarray(
        estimator.feature_importances_
    )

    # Generate fallback names if necessary
    if feature_names is None:

        feature_names = [
            f"feature_{i}"
            for i in range(len(importances))
        ]

    feature_names = clean_feature_names(feature_names)

    # Verify feature counts
    if len(feature_names) != len(importances):

        raise ValueError(
            f"Feature mismatch in {model_path.name}: "
            f"{len(feature_names)} names but "
            f"{len(importances)} importance values."
        )

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(
        by="importance",
        ascending=False
    )


# ==========================================
# ANALYSE RANDOM FOREST AND XGBOOST
# ==========================================

def run_feature_importance():

    model_files = {
        "Random Forest": MODEL_DIR / "random_forest.joblib",
        "XGBoost": MODEL_DIR / "xgboost.joblib"
    }

    all_results = []

    for model_name, model_path in model_files.items():

        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE:", model_name)
        print("=" * 60)

        if not model_path.exists():

            print("MODEL NOT FOUND:", model_path)
            continue

        # Extract importance values
        importance_df = extract_feature_importance(
            model_path
        )

        importance_df["model"] = model_name

        print(
            importance_df.to_string(index=False)
        )

        all_results.append(importance_df)

        # Save individual CSV
        csv_path = (
            REPORT_DIR /
            f"{model_name.lower().replace(' ', '_')}_importance.csv"
        )

        importance_df.to_csv(
            csv_path,
            index=False
        )

        # ==================================
        # CREATE FEATURE IMPORTANCE CHART
        # ==================================

        plot_df = importance_df.sort_values(
            by="importance",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_df["feature"],
            plot_df["importance"]
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Sensor Parameters")

        plt.title(
            f"{model_name} - Feature Importance"
        )

        plt.tight_layout()

        image_path = (
            REPORT_DIR /
            f"{model_name.lower().replace(' ', '_')}_importance.png"
        )

        plt.savefig(
            image_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("\nCSV SAVED:", csv_path)
        print("CHART SAVED:", image_path)

    # ======================================
    # COMPARE BOTH MODELS
    # ======================================

    if not all_results:

        raise RuntimeError(
            "No models were available for analysis."
        )

    combined_df = pd.concat(
        all_results,
        ignore_index=True
    )

    comparison_path = (
        REPORT_DIR / "feature_importance_comparison.csv"
    )

    combined_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
    print("=" * 60)

    print("\nCOMBINED RESULTS")

    print(
        combined_df.to_string(index=False)
    )

    print("\nCOMPARISON SAVED:", comparison_path)


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    run_feature_importance()
