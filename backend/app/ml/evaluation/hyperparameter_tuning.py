
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from app.ml.data.data_loader import load_dataset
from app.ml.data.validation import validate_dataset
from app.ml.data.preprocessing import prepare_data

from app.ml.algorithms.classification import (
    build_classification_models
)

from app.ml.evaluation.threshold_tuning import (
    find_best_threshold
)

from app.ml.evaluation.metrics import (
    evaluate_binary_classifier
)


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "tuned_models"
RESULT_DIR = BASE_DIR / "ml_artifacts" / "hyperparameter_tuning"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# HYPERPARAMETER SEARCH SPACE
# ==========================================

PARAMETER_SEARCH = {

    "Random Forest": {
        "classifier__n_estimators": [200, 300, 400, 500],
        "classifier__max_depth": [8, 12, 16, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.8],
        "classifier__class_weight": [
            "balanced",
            "balanced_subsample"
        ]
    },

    "XGBoost": {
        "classifier__n_estimators": [200, 300, 400, 500],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [
            0.01, 0.03, 0.05, 0.1
        ],
        "classifier__subsample": [
            0.7, 0.8, 0.9, 1.0
        ],
        "classifier__colsample_bytree": [
            0.7, 0.8, 0.9, 1.0
        ],
        "classifier__min_child_weight": [
            1, 3, 5, 7
        ],
        "classifier__reg_lambda": [
            1, 2, 5, 10
        ],
        "classifier__scale_pos_weight": [
            10, 20, 28, 35
        ]
    }
}


# ==========================================
# MAIN TUNING FUNCTION
# ==========================================

def run_hyperparameter_tuning():

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    # STEP 1: Load dataset

    print("\nSTEP 1: LOADING DATASET")

    df = load_dataset()

    # STEP 2: Validate dataset

    print("\nSTEP 2: VALIDATING DATASET")

    validate_dataset(df)

    # STEP 3: Prepare dataset

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

    # STEP 5: Build classification models

    models = build_classification_models(
        preprocessor,
        y_fit
    )

    # Five-fold stratified cross-validation

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    all_results = []

    # ==========================================
    # TUNE EACH MODEL
    # ==========================================

    for name, model in models.items():

        print("\n" + "#" * 70)
        print("TUNING:", name)
        print("#" * 70)

        start_time = time.perf_counter()

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=PARAMETER_SEARCH[name],
            n_iter=12,
            scoring="average_precision",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=2,
            refit=True,
            return_train_score=True,
            error_score="raise"
        )

        # Cross-validation uses training data only

        search.fit(X_fit, y_fit)

        tuning_time = time.perf_counter() - start_time

        best_model = search.best_estimator_

        print("\nBEST PARAMETERS")
        print("-" * 55)

        for parameter, value in search.best_params_.items():
            print(parameter, ":", value)

        print(
            "\nBest cross-validation PR-AUC:",
            round(search.best_score_, 5)
        )

        print(
            "Tuning time:",
            round(tuning_time, 2),
            "seconds"
        )

        # ==========================================
        # SAVE CROSS-VALIDATION RESULTS
        # ==========================================

        cv_results = pd.DataFrame(search.cv_results_)

        cv_path = RESULT_DIR / (
            name.lower().replace(" ", "_")
            + "_cv_results.csv"
        )

        cv_results.to_csv(cv_path, index=False)

        print("\nCV RESULTS SAVED:", cv_path)

        # ==========================================
        # VALIDATION THRESHOLD OPTIMIZATION
        # ==========================================

        validation_probabilities = best_model.predict_proba(
            X_val
        )[:, 1]

        best_threshold_result = find_best_threshold(
            y_val,
            validation_probabilities
        )

        best_threshold = best_threshold_result["threshold"]

        print("\nOPTIMIZED THRESHOLD")
        print("-" * 55)
        print("Threshold:", round(best_threshold, 6))
        print("Validation precision:",
              round(best_threshold_result["precision"], 4))
        print("Validation recall:",
              round(best_threshold_result["recall"], 4))
        print("Validation F1:",
              round(best_threshold_result["f1_score"], 4))

        # ==========================================
        # FINAL TEST EVALUATION
        # ==========================================

        test_probabilities = best_model.predict_proba(
            X_test
        )[:, 1]

        default_predictions = (
            test_probabilities >= 0.5
        ).astype(int)

        optimized_predictions = (
            test_probabilities >= best_threshold
        ).astype(int)

        for threshold_name, predictions in [
            ("Default", default_predictions),
            ("Optimized", optimized_predictions)
        ]:

            print("\n" + "-" * 55)
            print(name, "-", threshold_name)
            print("-" * 55)

            metrics = evaluate_binary_classifier(
                name + " - " + threshold_name,
                best_model,
                X_test,
                y_test,
                predictions=predictions,
                probabilities=test_probabilities,
                training_time=tuning_time
            )

            metrics["model_name"] = name
            metrics["threshold_type"] = threshold_name
            metrics["threshold"] = (
                0.5 if threshold_name == "Default"
                else best_threshold
            )
            metrics["cv_best_pr_auc"] = search.best_score_
            metrics["validation_f1"] = (
                best_threshold_result["f1_score"]
            )
            metrics["tuning_time_seconds"] = round(
                tuning_time, 2
            )

            all_results.append(metrics)

        # ==========================================
        # SAVE BEST MODEL
        # ==========================================

        model_path = MODEL_DIR / (
            name.lower().replace(" ", "_")
            + "_tuned.joblib"
        )

        joblib.dump(
            {
                "model": best_model,
                "threshold": best_threshold,
                "best_parameters": search.best_params_,
                "cv_best_pr_auc": search.best_score_,
                "model_name": name
            },
            model_path
        )

        print("\nBEST MODEL SAVED:", model_path)

        # Save parameters separately

        parameter_path = RESULT_DIR / (
            name.lower().replace(" ", "_")
            + "_best_parameters.json"
        )

        with open(parameter_path, "w", encoding="utf-8") as file:
            json.dump(
                search.best_params_,
                file,
                indent=4,
                default=str
            )

        print("BEST PARAMETERS SAVED:", parameter_path)

    # ==========================================
    # FINAL COMPARISON
    # ==========================================

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 100)
    print("TUNED MODEL COMPARISON")
    print("=" * 100)

    columns = [
        "model_name",
        "threshold_type",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "cv_best_pr_auc",
        "tuning_time_seconds"
    ]

    columns = [
        col for col in columns
        if col in results_df.columns
    ]

    print(results_df[columns].to_string(index=False))

    # Save final comparison

    results_path = RESULT_DIR / "tuned_model_comparison.csv"

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nCOMPARISON SAVED:", results_path)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("=" * 70)


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    run_hyperparameter_tuning()
