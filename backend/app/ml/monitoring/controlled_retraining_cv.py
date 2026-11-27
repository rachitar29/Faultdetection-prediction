
"""
Controlled retraining with cross-validated threshold optimization.

Uses:
- Existing AI4I dataset configuration
- Existing preprocessing
- Existing XGBoost model selector
- Five-fold out-of-fold threshold optimization
- Existing model registry
- Protected baseline evaluation

The original baseline is never overwritten.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd

from app.ml.data.data_loader import load_dataset
from app.ml.data.dynamic_preprocessor import DynamicDatasetPreprocessor

from app.ml.training.model_selector import select_models

from app.ml.algorithms.model_registry import (
    load_model_bundle,
    save_model_bundle,
)

from app.ml.monitoring.threshold_optimizer import (
    optimize_threshold_cv,
)

from app.ml.monitoring.controlled_retraining import (
    DatasetModelBundle,
    split_dataset,
    get_positive_probability,
    calculate_metrics,
    compare_models,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BASE_DIR / "ml_data" / "raw" / "ai4i2020.csv"
)

CONFIG_PATH = (
    BASE_DIR
    / "app"
    / "ml"
    / "configurations"
    / "dataset_configs"
    / "ai4i_config.json"
)

OUTPUT_DIR = (
    BASE_DIR / "ml_artifacts" / "monitoring"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_MODEL_NAME = "XGBoost"

CANDIDATE_ALGORITHM = "XGBoost"

N_SPLITS = 5
RANDOM_STATE = 42

PRECISION_TOLERANCE = 0.01
MAX_METRIC_DROP = 0.01


# ==========================================================
# REPORT SAVING
# ==========================================================

def save_report(report):
    """Save the cross-validated retraining report."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    report_path = (
        OUTPUT_DIR
        / f"controlled_retraining_cv_{timestamp}.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    return report_path


# ==========================================================
# MAIN RETRAINING WORKFLOW
# ==========================================================

def run_cross_validated_retraining():

    print("\nCROSS-VALIDATED CONTROLLED RETRAINING")
    print("=" * 65)

    # ------------------------------------------------------
    # 1. Check files
    # ------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration not found: {CONFIG_PATH}"
        )

    # ------------------------------------------------------
    # 2. Load protected baseline
    # ------------------------------------------------------

    baseline_bundle = load_model_bundle(
        BASELINE_MODEL_NAME
    )

    baseline_model = baseline_bundle["model"]
    baseline_metadata = baseline_bundle["metadata"]

    baseline_threshold = float(
        baseline_metadata.get("threshold", 0.5)
    )

    baseline_features = baseline_metadata.get(
        "feature_names",
        [],
    )

    if not baseline_features:
        raise ValueError(
            "Baseline model metadata has no feature_names."
        )

    print("\nProtected baseline:", BASELINE_MODEL_NAME)

    # ------------------------------------------------------
    # 3. Load dataset
    # ------------------------------------------------------

    dataframe = load_dataset(DATASET_PATH)

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    processor = DynamicDatasetPreprocessor(
        CONFIG_PATH
    )

    target_column = processor.target_column

    print("Dataset rows:", len(dataframe))
    print("Target column:", target_column)

    # ------------------------------------------------------
    # 4. Split dataset
    # ------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(
        dataframe,
        target_column,
    )

    print("\nDATA SPLIT")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Testing:", len(X_test))

    # ------------------------------------------------------
    # 5. Fit candidate preprocessing on training data
    # ------------------------------------------------------

    processor.fit(
        pd.concat(
            [X_train, y_train],
            axis=1,
        )
    )

    raw_X_train = processor._prepare_input(
        X_train
    )

    raw_X_test = processor._prepare_input(
        X_test
    )

    # ------------------------------------------------------
    # 6. Check baseline feature compatibility
    # ------------------------------------------------------

    missing_features = [
        feature
        for feature in baseline_features
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing baseline features: {missing_features}"
        )

    baseline_X_val = X_val[baseline_features]
    baseline_X_test = X_test[baseline_features]

    # ------------------------------------------------------
    # 7. Calculate baseline validation precision
    # ------------------------------------------------------

    baseline_val_probabilities = (
        get_positive_probability(
            baseline_model,
            baseline_X_val,
        )
    )

    baseline_val_metrics = calculate_metrics(
        y_val,
        baseline_val_probabilities,
        baseline_threshold,
    )

    minimum_precision = max(
        0.0,
        baseline_val_metrics["precision"]
        - PRECISION_TOLERANCE,
    )

    print("\nBASELINE VALIDATION")
    print(
        "Baseline precision:",
        round(baseline_val_metrics["precision"], 4),
    )

    print(
        "Minimum candidate precision:",
        round(minimum_precision, 4),
    )

    # ------------------------------------------------------
    # 8. Build candidate model
    # ------------------------------------------------------

    available_models = select_models(
        task="classification",
        preprocessor=processor.preprocessor,
        y_train=y_train,
    )

    if CANDIDATE_ALGORITHM not in available_models:
        raise KeyError(
            f"Candidate algorithm not found: "
            f"{CANDIDATE_ALGORITHM}"
        )

    candidate_estimator = available_models[
        CANDIDATE_ALGORITHM
    ]

    # ------------------------------------------------------
    # 9. Cross-validated threshold optimization
    # ------------------------------------------------------

    print("\nSTARTING FIVE-FOLD THRESHOLD OPTIMIZATION")

    threshold_result = optimize_threshold_cv(
        estimator=candidate_estimator,
        X=raw_X_train,
        y=y_train,
        minimum_precision=minimum_precision,
        minimum_recall=None,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    candidate_threshold = threshold_result["threshold"]

    print("\nSELECTED THRESHOLD")
    print(candidate_threshold)

    # ------------------------------------------------------
    # 10. Fit final candidate on all training data
    # ------------------------------------------------------

    print("\nFITTING FINAL CANDIDATE")

    candidate_estimator.fit(
        raw_X_train,
        y_train,
    )

    # ------------------------------------------------------
    # 11. Evaluate baseline and candidate on test data
    # ------------------------------------------------------

    baseline_test_probabilities = (
        get_positive_probability(
            baseline_model,
            baseline_X_test,
        )
    )

    candidate_test_probabilities = (
        get_positive_probability(
            candidate_estimator,
            raw_X_test,
        )
    )

    baseline_test_metrics = calculate_metrics(
        y_test,
        baseline_test_probabilities,
        baseline_threshold,
    )

    candidate_test_metrics = calculate_metrics(
        y_test,
        candidate_test_probabilities,
        candidate_threshold,
    )

    # ------------------------------------------------------
    # 12. Compare candidate against baseline
    # ------------------------------------------------------

    approval = compare_models(
        baseline_test_metrics,
        candidate_test_metrics,
        max_metric_drop=MAX_METRIC_DROP,
    )

    if not threshold_result["constraints_met"]:
        approval["approved"] = False

        if "cross_validation_precision_constraint" not in (
            approval["failed_metrics"]
        ):
            approval["failed_metrics"].append(
                "cross_validation_precision_constraint"
            )

    # ------------------------------------------------------
    # 13. Create unique registry name
    # ------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    candidate_name = (
        f"ai4i_xgboost_cv_candidate_{timestamp}"
    )

    registration_result = None

    # ------------------------------------------------------
    # 14. Register approved candidate only
    # ------------------------------------------------------

    if approval["approved"]:

        candidate_bundle = DatasetModelBundle(
            processor,
            candidate_estimator,
        )

        registration_result = save_model_bundle(
            model_name=candidate_name,
            model=candidate_bundle,
            threshold=candidate_threshold,
            model_type="classification",
            feature_names=processor.feature_columns,
            metrics=candidate_test_metrics,
            description=(
                "AI4I XGBoost candidate trained with "
                "cross-validated threshold optimization."
            ),
        )

        decision = "APPROVED_AND_REGISTERED"

    else:
        decision = "REJECTED"

    # ------------------------------------------------------
    # 15. Save report
    # ------------------------------------------------------

    report = {
        "report_type": "CROSS_VALIDATED_CONTROLLED_RETRAINING",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "baseline_model": BASELINE_MODEL_NAME,
        "candidate_model": candidate_name,
        "dataset": str(DATASET_PATH),
        "decision": decision,
        "cross_validation": {
            "folds": N_SPLITS,
            "method": "StratifiedKFold",
            "prediction_type": "out_of_fold",
        },
        "baseline_validation_metrics": baseline_val_metrics,
        "threshold_optimization": {
            key: value
            for key, value in threshold_result.items()
            if key != "oof_probabilities"
        },
        "baseline_test_metrics": baseline_test_metrics,
        "candidate_test_metrics": candidate_test_metrics,
        "approval": approval,
        "registered": registration_result is not None,
        "existing_baseline_modified": False,
        "automatic_production_promotion": False,
    }

    report_path = save_report(report)

    # ------------------------------------------------------
    # 16. Display results
    # ------------------------------------------------------

    print("\nFINAL MODEL COMPARISON")
    print("=" * 65)

    print("\nBASELINE")
    print(json.dumps(baseline_test_metrics, indent=4))

    print("\nCANDIDATE")
    print(json.dumps(candidate_test_metrics, indent=4))

    print("\nFINAL DECISION")
    print("=" * 65)

    print("Decision:", decision)
    print("Failed metrics:", approval["failed_metrics"])
    print("Candidate registered:", report["registered"])
    print("Existing baseline modified: NO")
    print("Production promotion: NO")

    print("\nReport saved:")
    print(report_path)

    print("\nCROSS-VALIDATED RETRAINING COMPLETED")

    return report


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    result = run_cross_validated_retraining()

    print("\nFINAL RESULT")
    print("=" * 65)
    print("Decision:", result["decision"])
