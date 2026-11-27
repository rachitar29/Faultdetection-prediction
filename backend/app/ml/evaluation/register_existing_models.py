
from pathlib import Path
import joblib
import pandas as pd

from app.ml.algorithms.model_registry import save_model_bundle

from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


BASE_DIR = Path(__file__).resolve().parents[3]

TRAINED_DIR = BASE_DIR / "ml_artifacts" / "trained_models"
TUNED_DIR = BASE_DIR / "ml_artifacts" / "tuned_models"
RESULT_DIR = BASE_DIR / "ml_artifacts" / "hyperparameter_tuning"

FEATURE_NAMES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_saved_artifact(path):
    if not path.exists():
        print("SKIPPED - file not found:", path)
        return None

    artifact = joblib.load(path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        print("SKIPPED - unexpected artifact format:", path)
        return None

    return artifact


def load_tuned_metrics(model_name):
    path = RESULT_DIR / "tuned_model_comparison.csv"

    if not path.exists():
        return {}

    try:
        df = pd.read_csv(path)

        selected = df[
            (df["model_name"] == model_name)
            & (df["threshold_type"] == "Optimized")
        ]

        if selected.empty:
            return {}

        return selected.iloc[0].to_dict()

    except Exception as error:
        print("Could not read tuned metrics:", error)
        return {}


def register_one(
    model_name,
    artifact_path,
    model_type,
    metrics=None
):
    artifact = load_saved_artifact(artifact_path)

    if artifact is None:
        return

    model = artifact["model"]
    threshold = artifact.get("threshold")

    if threshold is None:
        raise ValueError(
            f"No saved threshold found for {model_name}"
        )

    artifact_features = artifact.get(
        "features",
        artifact.get("feature_names", FEATURE_NAMES)
    )

    if list(artifact_features) != FEATURE_NAMES:
        raise ValueError(
            f"Feature mismatch for {model_name}.\n"
            f"Artifact features: {artifact_features}\n"
            f"Expected features: {FEATURE_NAMES}"
        )

    best_parameters = artifact.get(
        "best_parameters", {}
    )

    save_model_bundle(
        model_name=model_name,
        model=model,
        threshold=threshold,
        model_type=model_type,
        feature_names=artifact_features,
        metrics=metrics or artifact.get(
            "validation_metrics", {}
        ),
        best_parameters=best_parameters,
        description=(
            "AI4I 2020 predictive maintenance model. "
            "Complete fitted preprocessing and estimator pipeline. "
            "Threshold selected using validation data."
        )
    )


def main():

    print("\nREGISTERING EXISTING MODELS")
    print("=" * 60)

    # Tuned supervised models

    register_one(
        "Random Forest",
        TUNED_DIR / "random_forest_tuned.joblib",
        "Supervised",
        load_tuned_metrics("Random Forest")
    )

    register_one(
        "XGBoost",
        TUNED_DIR / "xgboost_tuned.joblib",
        "Supervised",
        load_tuned_metrics("XGBoost")
    )

    # Unsupervised anomaly detection models

    register_one(
        "Isolation Forest",
        TRAINED_DIR / "isolation_forest.joblib",
        "Unsupervised"
    )

    register_one(
        "Local Outlier Factor",
        TRAINED_DIR / "local_outlier_factor.joblib",
        "Unsupervised"
    )

    register_one(
        "One-Class SVM",
        TRAINED_DIR / "one-class_svm.joblib",
        "Unsupervised"
    )

    print("\nMODEL REGISTRATION FINISHED")


if __name__ == "__main__":
    main()
