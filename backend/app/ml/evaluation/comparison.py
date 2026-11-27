
from pathlib import Path
import time
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from app.ml.data.data_loader import load_dataset
from app.ml.data.validation import validate_dataset
from app.ml.data.preprocessing import prepare_data

from app.ml.algorithms.classification import (
    build_classification_models
)

from app.ml.evaluation.metrics import (
    evaluate_binary_classifier
)

from app.ml.evaluation.threshold_tuning import (
    find_best_threshold
)

from app.ml.evaluation.calibration import (
    evaluate_calibration,
    plot_reliability_diagram
)

from app.ml.evaluation.visualization import (
    plot_roc_curve,
    plot_precision_recall,
    plot_confusion_matrix
)


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "trained_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# MAIN MODEL COMPARISON
# ==========================================================

def run_model_comparison():

    print("\n" + "=" * 65)
    print("AI EQUIPMENT FAULT DETECTION")
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 65)

    # ======================================================
    # 1. LOAD AND VALIDATE DATASET
    # ======================================================

    print("\nSTEP 1: LOADING DATASET")

    df = load_dataset()

    validate_dataset(df)

    # ======================================================
    # 2. PREPARE DATA
    # ======================================================

    print("\nSTEP 2: PREPARING DATA")

    X_train, X_test, y_train, y_test, preprocessor = (
        prepare_data(df)
    )

    # Separate training data into fitting and validation.
    # Test data remains untouched until final evaluation.

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    print("\nDATASET DISTRIBUTION")
    print("-" * 50)

    print("Model training:", len(X_fit))
    print("Validation:", len(X_val))
    print("Final testing:", len(X_test))

    # ======================================================
    # 3. BUILD MODELS
    # ======================================================

    print("\nSTEP 3: BUILDING CLASSIFICATION MODELS")

    models = build_classification_models(
        preprocessor,
        y_fit
    )

    results = []
    thresholds = {}
    calibration_results = []

    # ======================================================
    # 4. TRAIN EACH MODEL
    # ======================================================

    for name, model in models.items():

        print("\n" + "#" * 65)
        print("TRAINING MODEL:", name)
        print("#" * 65)

        start = time.perf_counter()

        model.fit(X_fit, y_fit)

        training_time = time.perf_counter() - start

        print(
            f"Training time: {training_time:.4f} seconds"
        )

        # ==================================================
        # 5. VALIDATION PREDICTIONS
        # ==================================================

        print("\nSTEP 5: VALIDATION PREDICTIONS")

        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "Failure class 1 not found in model classes."
            )

        failure_index = classes.index(1)

        val_probabilities = model.predict_proba(X_val)

        val_scores = val_probabilities[:, failure_index]

        # ==================================================
        # 6. OPTIMIZE CLASSIFICATION THRESHOLD
        # ==================================================

        print("\nSTEP 6: THRESHOLD OPTIMIZATION")

        optimal = find_best_threshold(
            y_val,
            val_scores
        )

        optimal_threshold = optimal["threshold"]

        thresholds[name] = optimal_threshold

        print("\nOPTIMAL THRESHOLD RESULTS")

        print(
            "Threshold:",
            round(optimal_threshold, 5)
        )

        print(
            "Validation F1:",
            round(optimal["f1_score"], 4)
        )

        print(
            "Validation Precision:",
            round(optimal["precision"], 4)
        )

        print(
            "Validation Recall:",
            round(optimal["recall"], 4)
        )

        # ==================================================
        # 7. FINAL TEST PREDICTIONS
        # ==================================================

        print("\nSTEP 7: FINAL TEST PREDICTIONS")

        test_probabilities = model.predict_proba(X_test)

        failure_scores = test_probabilities[:, failure_index]

        default_predictions = (
            failure_scores >= 0.50
        ).astype(int)

        optimized_predictions = (
            failure_scores >= optimal_threshold
        ).astype(int)

        # ==================================================
        # 8. DEFAULT THRESHOLD EVALUATION
        # ==================================================

        print("\nSTEP 8: DEFAULT THRESHOLD EVALUATION")

        default_metrics = evaluate_binary_classifier(
            model_name=name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            predictions=default_predictions,
            probabilities=failure_scores,
            training_time=training_time
        )

        default_metrics["threshold"] = 0.50
        default_metrics["threshold_type"] = "Default"

        results.append(default_metrics)

        # ==================================================
        # 9. OPTIMIZED THRESHOLD EVALUATION
        # ==================================================

        print("\nSTEP 9: OPTIMIZED THRESHOLD EVALUATION")

        optimized_metrics = evaluate_binary_classifier(
            model_name=name,
            model=model,
            X_test=X_test,
            y_test=y_test,
            predictions=optimized_predictions,
            probabilities=failure_scores,
            training_time=training_time
        )

        optimized_metrics["threshold"] = optimal_threshold

        optimized_metrics["threshold_type"] = "Optimized"

        results.append(optimized_metrics)

        # ==================================================
        # 10. PROBABILITY CALIBRATION
        # ==================================================

        print("\nSTEP 10: PROBABILITY CALIBRATION")

        calibration_data = evaluate_calibration(
            model_name=name,
            y_test=y_test,
            probabilities=failure_scores,
            n_bins=10
        )

        calibration_metrics = calibration_data["results"]

        calibration_results.append(
            calibration_metrics
        )

        plot_reliability_diagram(
            model_name=name,
            calibration_data=calibration_data
        )

        # Add calibration metrics to both
        # threshold evaluation records.

        for record in results[-2:]:
            record.update(calibration_metrics)

        # ==================================================
        # 11. ROC CURVE
        # ==================================================

        print("\nSTEP 11: ROC CURVE")

        plot_roc_curve(
            name,
            y_test,
            failure_scores
        )

        # ==================================================
        # 12. PRECISION RECALL CURVE
        # ==================================================

        print("\nSTEP 12: PRECISION RECALL CURVE")

        plot_precision_recall(
            name,
            y_test,
            failure_scores
        )

        # ==================================================
        # 13. CONFUSION MATRICES
        # ==================================================

        print("\nSTEP 13: CONFUSION MATRICES")

        plot_confusion_matrix(
            name + "_default",
            y_test,
            default_predictions
        )

        plot_confusion_matrix(
            name + "_optimized",
            y_test,
            optimized_predictions
        )

        # ==================================================
        # 14. SAVE TRAINED MODEL
        # ==================================================

        filename = (
            name.lower().replace(" ", "_") + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(
            model,
            model_path
        )

        print("\nMODEL SAVED:", model_path)

    # ======================================================
    # 15. SAVE OPTIMAL THRESHOLDS
    # ======================================================

    print("\nSTEP 15: SAVING OPTIMAL THRESHOLDS")

    threshold_path = MODEL_DIR / "optimal_thresholds.json"

    with open(
        threshold_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            thresholds,
            file,
            indent=4
        )

    print("THRESHOLDS SAVED:", threshold_path)

    # ======================================================
    # 16. SAVE CALIBRATION RESULTS
    # ======================================================

    calibration_df = pd.DataFrame(
        calibration_results
    )

    calibration_path = (
        MODEL_DIR / "calibration_comparison.csv"
    )

    calibration_df.to_csv(
        calibration_path,
        index=False
    )

    print(
        "CALIBRATION RESULTS SAVED:",
        calibration_path
    )

    # ======================================================
    # 17. FINAL MODEL COMPARISON
    # ======================================================

    print("\nSTEP 17: FINAL MODEL COMPARISON")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="pr_auc",
        ascending=False
    )

    print("\nFINAL RESULTS")

    print(
        results_df.to_string(index=False)
    )

    # ======================================================
    # 18. EXPORT COMPARISON CSV
    # ======================================================

    comparison_path = (
        MODEL_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print(
        "\nMODEL COMPARISON SAVED:",
        comparison_path
    )

    # ======================================================
    # 19. FINAL SUMMARY
    # ======================================================

    print("\n" + "=" * 65)
    print("ALL MODEL ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nTRAINED MODELS:")

    for name in models:
        print("-", name)

    print("\nSAVED THRESHOLDS:")

    for name, threshold in thresholds.items():
        print(name, ":", round(threshold, 5))

    print("\nOUTPUT DIRECTORY:")
    print(MODEL_DIR)


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    run_model_comparison()
