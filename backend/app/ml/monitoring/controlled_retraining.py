
"""Controlled offline retraining with precision-constrained thresholds.

Features:
1. Loads the existing registered baseline.
2. Trains a new XGBoost candidate locally.
3. Uses validation data for threshold optimization.
4. Constrains candidate precision relative to baseline precision.
5. Evaluates both models on the same test split.
6. Registers approved candidates under unique names.
7. Never overwrites the existing baseline.

All operations run locally.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

from app.ml.data.data_loader import load_dataset
from app.ml.data.dynamic_preprocessor import DynamicDatasetPreprocessor
from app.ml.training.model_selector import select_models

from app.ml.algorithms.model_registry import (
    load_model_bundle,
    save_model_bundle,
)


# ==========================================================
# PATH CONFIGURATION
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


# ==========================================================
# RETRAINING CONFIGURATION
# ==========================================================

BASELINE_MODEL_NAME = "XGBoost"

CANDIDATE_ALGORITHM = "XGBoost"

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

# Candidate precision must be within 1 percentage point
# of baseline validation precision.
PRECISION_TOLERANCE = 0.01

# Final test-set approval tolerance.
MAX_METRIC_DROP = 0.01


# ==========================================================
# MODEL WRAPPER
# ==========================================================

class DatasetModelBundle:
    """Wrap a fitted estimator and its input preprocessor."""

    def __init__(self, input_preprocessor, estimator):
        self.input_preprocessor = input_preprocessor
        self.estimator = estimator

    def _prepare(self, X):
        return self.input_preprocessor._prepare_input(X)

    def predict(self, X):
        return self.estimator.predict(self._prepare(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._prepare(X))

    def decision_function(self, X):
        return self.estimator.decision_function(self._prepare(X))


# ==========================================================
# DATASET SPLITTING
# ==========================================================

def split_dataset(
    dataframe,
    target_column,
    test_size=TEST_SIZE,
    validation_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
):
    """Create stratified train, validation and test splits."""

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    y = dataframe[target_column]

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Only binary target labels 0 and 1 are supported."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Dataset must contain both target classes."
        )

    if test_size <= 0 or validation_size <= 0:
        raise ValueError(
            "Split proportions must be positive."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "Test and validation proportions must total less than 1."
        )

    X = dataframe.drop(columns=[target_column])

    stratify_first = (
        y if y.value_counts().min() >= 3 else None
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_first,
    )

    validation_fraction = validation_size / (1 - test_size)

    stratify_second = (
        y_train_val
        if y_train_val.value_counts().min() >= 2
        else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=stratify_second,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ==========================================================
# PROBABILITY HELPERS
# ==========================================================

def get_positive_probability(model, X):
    """Return probabilities for positive class 1."""

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model does not support predict_proba."
        )

    probabilities = np.asarray(
        model.predict_proba(X)
    )

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Expected binary classification probabilities."
        )

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "estimator"):
        classes = getattr(model.estimator, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)

    if classes is None:
        raise ValueError(
            "Cannot determine model class ordering."
        )

    classes = list(classes)

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 is missing."
        )

    return probabilities[:, classes.index(1)]


# ==========================================================
# METRIC CALCULATION
# ==========================================================

def calculate_metrics(y_true, probabilities, threshold):
    """Calculate binary classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "threshold": float(threshold),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
    else:
        metrics["roc_auc"] = None

    return metrics


# ==========================================================
# THRESHOLD OPTIMIZATION
# ==========================================================

def optimize_threshold_with_precision_constraint(
    y_true,
    probabilities,
    minimum_precision,
):
    """Maximize validation F1 while satisfying minimum precision."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
        pos_label=1,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "Unable to generate candidate thresholds."
        )

    precision_values = precision[:-1]
    recall_values = recall[:-1]

    f1_values = (
        2 * precision_values * recall_values
        / (
            precision_values
            + recall_values
            + 1e-12
        )
    )

    eligible_indices = np.where(
        precision_values >= minimum_precision
    )[0]

    if len(eligible_indices) == 0:
        # No threshold satisfies the precision constraint.
        # Choose the threshold with the highest precision.
        # The candidate will still undergo final approval.
        best_index = max(
            range(len(thresholds)),
            key=lambda index: (
                precision_values[index],
                f1_values[index],
                thresholds[index],
            ),
        )

        constraint_met = False

    else:
        # Highest F1 among thresholds satisfying precision.
        best_index = max(
            eligible_indices,
            key=lambda index: (
                f1_values[index],
                precision_values[index],
                thresholds[index],
            ),
        )

        constraint_met = True

    return {
        "threshold": float(thresholds[best_index]),
        "validation_precision": float(
            precision_values[best_index]
        ),
        "validation_recall": float(
            recall_values[best_index]
        ),
        "validation_f1": float(
            f1_values[best_index]
        ),
        "minimum_required_precision": float(
            minimum_precision
        ),
        "precision_constraint_met": constraint_met,
    }


# ==========================================================
# BASELINE METRIC NORMALIZATION
# ==========================================================

def normalize_baseline_metrics(metrics):
    """Normalize historical metric naming conventions."""

    metrics = metrics or {}

    return {
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get(
            "f1",
            metrics.get("f1_score"),
        ),
        "roc_auc": metrics.get("roc_auc"),
    }


# ==========================================================
# FINAL MODEL COMPARISON
# ==========================================================

def compare_models(
    baseline_metrics,
    candidate_metrics,
    max_metric_drop=MAX_METRIC_DROP,
):
    """Compare baseline and candidate on the same test data."""

    required_metrics = [
        "recall",
        "precision",
        "f1",
        "roc_auc",
    ]

    comparison = {}
    failed_metrics = []

    for metric in required_metrics:

        baseline_value = baseline_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)

        if baseline_value is None or candidate_value is None:
            comparison[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "status": "NOT_COMPARABLE",
            }

            failed_metrics.append(metric)
            continue

        difference = candidate_value - baseline_value

        passed = difference >= -max_metric_drop

        comparison[metric] = {
            "baseline": float(baseline_value),
            "candidate": float(candidate_value),
            "difference": float(difference),
            "status": "PASSED" if passed else "FAILED",
        }

        if not passed:
            failed_metrics.append(metric)

    return {
        "approved": len(failed_metrics) == 0,
        "failed_metrics": failed_metrics,
        "comparison": comparison,
        "maximum_allowed_metric_drop": max_metric_drop,
    }


# ==========================================================
# REPORT SAVING
# ==========================================================

def save_retraining_report(report):
    """Save the retraining report locally."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    report_path = (
        OUTPUT_DIR
        / f"controlled_retraining_{timestamp}.json"
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
# CONTROLLED RETRAINING
# ==========================================================

def run_controlled_retraining(
    dataset_path=DATASET_PATH,
    config_path=CONFIG_PATH,
    dataset_name="AI4I",
    baseline_model_name=BASELINE_MODEL_NAME,
    precision_tolerance=PRECISION_TOLERANCE,
    max_metric_drop=MAX_METRIC_DROP,
):
    """Train and evaluate a candidate without overwriting baseline."""

    print("\nCONTROLLED OFFLINE RETRAINING")
    print("=" * 65)

    dataset_path = Path(dataset_path)
    config_path = Path(config_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    # ------------------------------------------------------
    # 1. Load protected baseline
    # ------------------------------------------------------

    baseline_bundle = load_model_bundle(
        baseline_model_name
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
            "Baseline metadata has no feature_names."
        )

    print("Protected baseline:", baseline_model_name)

    # ------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------

    dataframe = load_dataset(dataset_path)

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    processor = DynamicDatasetPreprocessor(config_path)

    target_column = processor.target_column

    print("Dataset:", dataset_name)
    print("Rows:", len(dataframe))
    print("Target:", target_column)

    # ------------------------------------------------------
    # 3. Split dataset
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
    # 4. Fit candidate preprocessing on training data only
    # ------------------------------------------------------

    processor.fit(
        pd.concat(
            [X_train, y_train],
            axis=1,
        )
    )

    raw_X_train = processor._prepare_input(X_train)
    raw_X_val = processor._prepare_input(X_val)
    raw_X_test = processor._prepare_input(X_test)

    # ------------------------------------------------------
    # 5. Validate baseline input features
    # ------------------------------------------------------

    missing_features = [
        feature
        for feature in baseline_features
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Baseline features missing from dataset: {missing_features}"
        )

    baseline_X_val = X_val[baseline_features]
    baseline_X_test = X_test[baseline_features]

    # ------------------------------------------------------
    # 6. Calculate baseline validation precision
    # ------------------------------------------------------

    baseline_val_probabilities = get_positive_probability(
        baseline_model,
        baseline_X_val,
    )

    baseline_val_metrics = calculate_metrics(
        y_val,
        baseline_val_probabilities,
        baseline_threshold,
    )

    minimum_precision = max(
        0.0,
        baseline_val_metrics["precision"] - precision_tolerance,
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
    # 7. Build and train candidate
    # ------------------------------------------------------

    available_models = select_models(
        task="classification",
        preprocessor=processor.preprocessor,
        y_train=y_train,
    )

    if CANDIDATE_ALGORITHM not in available_models:
        raise KeyError(
            f"{CANDIDATE_ALGORITHM} not found in model selector."
        )

    candidate_estimator = available_models[
        CANDIDATE_ALGORITHM
    ]

    print("\nTraining candidate:", CANDIDATE_ALGORITHM)

    candidate_estimator.fit(
        raw_X_train,
        y_train,
    )

    # ------------------------------------------------------
    # 8. Optimize candidate threshold using validation data
    # ------------------------------------------------------

    candidate_val_probabilities = get_positive_probability(
        candidate_estimator,
        raw_X_val,
    )

    threshold_result = (
        optimize_threshold_with_precision_constraint(
            y_true=y_val,
            probabilities=candidate_val_probabilities,
            minimum_precision=minimum_precision,
        )
    )

    candidate_threshold = threshold_result["threshold"]

    print("\nTHRESHOLD OPTIMIZATION")
    print(
        "Selected threshold:",
        candidate_threshold,
    )

    print(
        "Validation precision:",
        round(threshold_result["validation_precision"], 4),
    )

    print(
        "Validation F1:",
        round(threshold_result["validation_f1"], 4),
    )

    print(
        "Precision constraint met:",
        threshold_result["precision_constraint_met"],
    )

    # ------------------------------------------------------
    # 9. Evaluate baseline and candidate on the same test set
    # ------------------------------------------------------

    baseline_test_probabilities = get_positive_probability(
        baseline_model,
        baseline_X_test,
    )

    candidate_test_probabilities = get_positive_probability(
        candidate_estimator,
        raw_X_test,
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
    # 10. Compare candidate and baseline
    # ------------------------------------------------------

    approval = compare_models(
        baseline_test_metrics,
        candidate_test_metrics,
        max_metric_drop=max_metric_drop,
    )

    # Precision constraint is also required.
    if not threshold_result["precision_constraint_met"]:
        approval["approved"] = False

        if "precision_constraint" not in approval["failed_metrics"]:
            approval["failed_metrics"].append(
                "precision_constraint"
            )

    # ------------------------------------------------------
    # 11. Create unique candidate name
    # ------------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    candidate_registry_name = (
        f"{dataset_name.lower()}_xgboost_candidate_{timestamp}"
    )

    registration_result = None

    # ------------------------------------------------------
    # 12. Register candidate only if approved
    # ------------------------------------------------------

    if approval["approved"]:

        candidate_bundle = DatasetModelBundle(
            processor,
            candidate_estimator,
        )

        registration_result = save_model_bundle(
            model_name=candidate_registry_name,
            model=candidate_bundle,
            threshold=candidate_threshold,
            model_type="classification",
            feature_names=processor.feature_columns,
            metrics=candidate_test_metrics,
            description=(
                f"Precision-constrained controlled retraining "
                f"candidate for {dataset_name}, compared against "
                f"{baseline_model_name}."
            ),
        )

        decision = "APPROVED_AND_REGISTERED"

    else:
        decision = "REJECTED"

    # ------------------------------------------------------
    # 13. Save report
    # ------------------------------------------------------

    report = {
        "report_type": "CONTROLLED_MODEL_RETRAINING",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "baseline_model_name": baseline_model_name,
        "candidate_algorithm": CANDIDATE_ALGORITHM,
        "candidate_registry_name": candidate_registry_name,
        "decision": decision,
        "baseline_validation_metrics": baseline_val_metrics,
        "threshold_optimization": threshold_result,
        "baseline_test_metrics": baseline_test_metrics,
        "candidate_test_metrics": candidate_test_metrics,
        "approval": approval,
        "registered": registration_result is not None,
        "existing_baseline_modified": False,
        "automatic_production_promotion": False,
    }

    report_path = save_retraining_report(report)

    # ------------------------------------------------------
    # 14. Display results
    # ------------------------------------------------------

    print("\nFINAL TEST METRICS")
    print("=" * 65)

    print("\nBaseline:")
    print(json.dumps(baseline_test_metrics, indent=4))

    print("\nCandidate:")
    print(json.dumps(candidate_test_metrics, indent=4))

    print("\nAPPROVAL RESULT")
    print("=" * 65)

    print("Decision:", decision)
    print("Failed metrics:", approval["failed_metrics"])
    print("Candidate registered:", report["registered"])
    print("Existing baseline modified: NO")
    print("Production promotion: NO")

    print("\nReport saved:")
    print(report_path)

    print("\nCONTROLLED RETRAINING COMPLETED")

    return report


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    result = run_controlled_retraining()

    print("\nFINAL RESULT")
    print("=" * 65)
    print("Decision:", result["decision"])
