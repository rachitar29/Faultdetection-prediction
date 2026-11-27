
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from app.ml.data.data_loader import load_dataset
from app.ml.data.validation import validate_dataset
from app.ml.data.preprocessing import prepare_data

from app.ml.algorithms.classification import (
    build_classification_models
)

from app.ml.algorithms.anomaly_detection import (
    build_anomaly_models,
    get_anomaly_predictions,
    get_anomaly_scores
)

from app.ml.evaluation.metrics import (
    evaluate_binary_classifier,
    evaluate_anomaly_detector
)

from app.ml.evaluation.threshold_tuning import (
    find_best_threshold
)


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "trained_models"
RESULT_DIR = BASE_DIR / "ml_artifacts" / "unified_comparison"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# HELPER: SAVE RESULTS
# ==========================================

def clean_metrics(metrics):
    """Convert metric values into CSV-friendly values."""

    cleaned = {}

    for key, value in metrics.items():

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (list, dict)):
            value = str(value)

        cleaned[key] = value

    return cleaned


# ==========================================
# MAIN COMPARISON
# ==========================================

def run_unified_comparison():

    print("\n" + "=" * 70)
    print("UNIFIED EQUIPMENT FAILURE MODEL COMPARISON")
    print("=" * 70)

    # STEP 1: Load and validate dataset

    print("\nSTEP 1: LOADING DATASET")

    df = load_dataset()

    print("\nSTEP 2: VALIDATING DATASET")

    validate_dataset(df)

    # STEP 2: Prepare data

    print("\nSTEP 3: PREPARING DATA")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df)

    # Split training data into fitting and validation data.
    # Validation labels are used for threshold selection only.

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train
    )

    print("\nDATA DISTRIBUTION")
    print("-" * 55)
    print("Training samples:", len(X_fit))
    print("Validation samples:", len(X_val))
    print("Test samples:", len(X_test))

    # STEP 3: Build all four models

    classification_models = build_classification_models(
        preprocessor,
        y_fit
    )

    anomaly_models = build_anomaly_models(
        preprocessor
    )

    all_models = {}

    all_models.update(classification_models)
    all_models.update(anomaly_models)

    results = []

    # ==========================================
    # TRAIN AND EVALUATE EACH MODEL
    # ==========================================

    for name, model in all_models.items():

        print("\n" + "#" * 65)
        print("MODEL:", name)
        print("#" * 65)

        is_supervised = name in classification_models

        # STEP 4: Train model

        print("\nTRAINING MODEL")

        start_time = time.perf_counter()

        if is_supervised:
            model.fit(X_fit, y_fit)
        else:
            # Anomaly detectors do not use failure labels.
            model.fit(X_fit)

        training_time = time.perf_counter() - start_time

        print(f"Training time: {training_time:.2f} seconds")

        # ==========================================
        # VALIDATION SCORES
        # ==========================================

        if is_supervised:

            validation_scores = model.predict_proba(
                X_val
            )[:, 1]

        else:

            validation_scores = get_anomaly_scores(
                model,
                X_val
            )

        # STEP 5: Optimize threshold using validation data

        best_result = find_best_threshold(
            y_val,
            validation_scores
        )

        best_threshold = best_result["threshold"]

        print("\nOPTIMIZED VALIDATION THRESHOLD")
        print("-" * 55)
        print("Threshold:", round(best_threshold, 6))
        print("Precision:", round(best_result["precision"], 4))
        print("Recall:", round(best_result["recall"], 4))
        print("F1-score:", round(best_result["f1_score"], 4))

        # ==========================================
        # FINAL TEST SCORES
        # ==========================================

        if is_supervised:

            test_scores = model.predict_proba(
                X_test
            )[:, 1]

            default_predictions = (
                test_scores >= 0.5
            ).astype(int)

        else:

            test_scores = get_anomaly_scores(
                model,
                X_test
            )

            default_predictions = get_anomaly_predictions(
                model,
                X_test
            )

        optimized_predictions = (
            test_scores >= best_threshold
        ).astype(int)

        # ==========================================
        # EVALUATE DEFAULT AND OPTIMIZED THRESHOLDS
        # ==========================================

        threshold_results = {
            "Default": default_predictions,
            "Optimized": optimized_predictions
        }

        for threshold_type, predictions in threshold_results.items():

            print("\n" + "-" * 55)
            print(name, "-", threshold_type)
            print("-" * 55)

            if is_supervised:

                metrics = evaluate_binary_classifier(
                    name + " - " + threshold_type,
                    model,
                    X_test,
                    y_test,
                    predictions=predictions,
                    probabilities=test_scores,
                    training_time=training_time
                )

            else:

                metrics = evaluate_anomaly_detector(
                    name + " - " + threshold_type,
                    model,
                    X_test,
                    y_test,
                    predictions,
                    test_scores
                )

            metrics = clean_metrics(metrics)

            metrics["model_name"] = name
            metrics["model_type"] = (
                "Supervised" if is_supervised
                else "Unsupervised"
            )
            metrics["threshold_type"] = threshold_type
            metrics["threshold"] = (
                0.5 if threshold_type == "Default"
                and is_supervised
                else best_threshold if threshold_type == "Optimized"
                else "Detector default"
            )
            metrics["validation_f1"] = best_result["f1_score"]
            metrics["training_time_seconds"] = round(
                training_time, 3
            )

            results.append(metrics)

        # ==========================================
        # SAVE MODEL
        # ==========================================

        filename = name.lower().replace(" ", "_") + ".joblib"

        model_path = MODEL_DIR / filename

        joblib.dump(
            {
                "model": model,
                "threshold": best_threshold,
                "model_name": name,
                "model_type": (
                    "Supervised" if is_supervised
                    else "Unsupervised"
                )
            },
            model_path
        )

        print("\nMODEL SAVED:", model_path)

    # ==========================================
    # FINAL UNIFIED COMPARISON
    # ==========================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("UNIFIED MODEL COMPARISON")
    print("=" * 100)

    display_columns = [
        "model_name",
        "model_type",
        "threshold_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "training_time_seconds"
    ]

    available_columns = [
        col for col in display_columns
        if col in results_df.columns
    ]

    print(
        results_df[available_columns].to_string(
            index=False
        )
    )

    # STEP 6: Save complete results

    results_path = RESULT_DIR / "unified_model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nRESULTS SAVED:", results_path)

    # STEP 7: Save optimized threshold summary

    threshold_df = results_df[
        results_df["threshold_type"] == "Optimized"
    ][
        ["model_name", "model_type", "threshold"]
    ]

    threshold_path = RESULT_DIR / "unified_thresholds.csv"

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    print("\nTHRESHOLDS SAVED:", threshold_path)

    print("\n" + "=" * 70)
    print("UNIFIED MODEL COMPARISON COMPLETED")
    print("=" * 70)


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    run_unified_comparison()
