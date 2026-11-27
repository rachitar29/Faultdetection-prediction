
from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from app.ml.data.data_loader import load_dataset
from app.ml.data.validation import validate_dataset
from app.ml.data.preprocessing import prepare_data

from app.ml.algorithms.anomaly_detection import (
    build_anomaly_models,
    get_anomaly_predictions,
    get_anomaly_scores
)

from app.ml.evaluation.metrics import (
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
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = BASE_DIR / "ml_artifacts" / "anomaly_comparison"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# MAIN COMPARISON FUNCTION
# ==========================================

def run_anomaly_comparison():

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION MODEL COMPARISON")
    print("=" * 70)

    # STEP 1: Load dataset
    print("\nSTEP 1: LOADING DATASET")

    df = load_dataset()

    # STEP 2: Validate dataset
    print("\nSTEP 2: VALIDATING DATASET")

    validate_dataset(df)

    # STEP 3: Prepare data
    print("\nSTEP 3: PREPARING DATA")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data(df)

    # STEP 4: Create validation split
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

    # STEP 5: Build models
    models = build_anomaly_models(preprocessor)

    all_results = []
    threshold_summary = []

    # ==========================================
    # TRAIN AND EVALUATE MODELS
    # ==========================================

    for name, model in models.items():

        print("\n" + "#" * 70)
        print("MODEL:", name)
        print("#" * 70)

        # STEP 6: Train without failure labels
        print("\nTRAINING MODEL")

        start_time = time.perf_counter()

        model.fit(X_fit)

        training_time = time.perf_counter() - start_time

        print(f"Training time: {training_time:.2f} seconds")

        # STEP 7: Validation anomaly scores
        validation_scores = get_anomaly_scores(
            model,
            X_val
        )

        # STEP 8: Optimize threshold using validation labels
        best_result = find_best_threshold(
            y_val,
            validation_scores
        )

        best_threshold = best_result["threshold"]

        print("\nOPTIMIZED THRESHOLD")
        print("-" * 55)
        print("Threshold:", round(best_threshold, 6))
        print("Validation precision:", round(best_result["precision"], 4))
        print("Validation recall:", round(best_result["recall"], 4))
        print("Validation F1:", round(best_result["f1_score"], 4))

        # STEP 9: Get final test scores
        test_scores = get_anomaly_scores(
            model,
            X_test
        )

        # ==========================================
        # DEFAULT THRESHOLD EVALUATION
        # ==========================================

        print("\nDEFAULT THRESHOLD EVALUATION")

        default_predictions = get_anomaly_predictions(
            model,
            X_test
        )

        default_metrics = evaluate_anomaly_detector(
            name + " - Default",
            model,
            X_test,
            y_test,
            default_predictions,
            test_scores
        )

        default_metrics["threshold_type"] = "Default"
        default_metrics["optimal_threshold"] = best_threshold
        default_metrics["validation_f1"] = best_result["f1_score"]
        default_metrics["training_time_seconds"] = round(training_time, 2)

        all_results.append(default_metrics)

        # ==========================================
        # OPTIMIZED THRESHOLD EVALUATION
        # ==========================================

        print("\nOPTIMIZED THRESHOLD EVALUATION")

        optimized_predictions = (
            test_scores >= best_threshold
        ).astype(int)

        optimized_metrics = evaluate_anomaly_detector(
            name + " - Optimized",
            model,
            X_test,
            y_test,
            optimized_predictions,
            test_scores
        )

        optimized_metrics["threshold_type"] = "Optimized"
        optimized_metrics["optimal_threshold"] = best_threshold
        optimized_metrics["validation_f1"] = best_result["f1_score"]
        optimized_metrics["training_time_seconds"] = round(training_time, 2)

        all_results.append(optimized_metrics)

        # ==========================================
        # SAVE MODEL AND THRESHOLD
        # ==========================================

        filename = name.lower().replace(" ", "_") + ".joblib"

        model_path = MODEL_DIR / filename

        joblib.dump(
            {
                "model": model,
                "threshold": best_threshold,
                "model_name": name,
                "features": list(X_fit.columns),
                "threshold_source": "validation",
                "validation_metrics": best_result
            },
            model_path
        )

        print("\nMODEL SAVED:", model_path)

        threshold_summary.append({
            "model": name,
            "threshold": best_threshold,
            "threshold_source": "validation"
        })

    # ==========================================
    # FINAL COMPARISON
    # ==========================================

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False
    )

    print("\n" + "=" * 90)
    print("FINAL ANOMALY DETECTION COMPARISON")
    print("=" * 90)

    display_columns = [
        col for col in [
            "model_name",
            "threshold_type",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "optimal_threshold",
            "training_time_seconds"
        ]
        if col in results_df.columns
    ]

    print(results_df[display_columns].to_string(index=False))

    # STEP 10: Save comparison CSV
    results_path = RESULT_DIR / "anomaly_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nCOMPARISON SAVED:", results_path)

    # STEP 11: Save threshold summary
    threshold_df = pd.DataFrame(threshold_summary)

    threshold_path = RESULT_DIR / "anomaly_thresholds.csv"

    threshold_df.to_csv(
        threshold_path,
        index=False
    )

    print("\nTHRESHOLDS SAVED:", threshold_path)

    print("\nANOMALY DETECTION COMPARISON COMPLETED")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    run_anomaly_comparison()
