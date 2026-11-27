
"""
Cross-validated threshold optimization for binary classification.

Features:
- Stratified K-fold cross-validation.
- Out-of-fold probability predictions.
- Precision-constrained threshold selection.
- Optional recall constraint.
- F1-score optimization.
- No network or cloud dependencies.

The returned threshold must be evaluated on a separate,
untouched test dataset before model approval.
"""

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)


# ==========================================================
# VALIDATION
# ==========================================================

def validate_inputs(
    estimator,
    X,
    y,
    n_splits,
    minimum_precision,
    minimum_recall,
):
    """Validate input data and optimization settings."""

    if estimator is None:
        raise ValueError("Estimator cannot be None.")

    if X is None or y is None:
        raise ValueError("X and y are required.")

    if len(X) != len(y):
        raise ValueError("X and y have different lengths.")

    if len(X) == 0:
        raise ValueError("Training data is empty.")

    y_values = np.asarray(y)

    if not set(np.unique(y_values)).issubset({0, 1}):
        raise ValueError(
            "Only binary target labels 0 and 1 are supported."
        )

    if len(np.unique(y_values)) != 2:
        raise ValueError(
            "Both target classes must be present."
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    class_counts = np.bincount(
        y_values.astype(int),
        minlength=2,
    )

    if class_counts.min() < n_splits:
        raise ValueError(
            "Each class must contain at least n_splits samples. "
            f"Class counts: {class_counts.tolist()}"
        )

    if not 0 <= minimum_precision <= 1:
        raise ValueError(
            "minimum_precision must be between 0 and 1."
        )

    if minimum_recall is not None:
        if not 0 <= minimum_recall <= 1:
            raise ValueError(
                "minimum_recall must be between 0 and 1."
            )

    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            "Estimator must support predict_proba."
        )


# ==========================================================
# POSITIVE CLASS PROBABILITIES
# ==========================================================

def extract_positive_probabilities(
    estimator,
    X,
):
    """Extract probabilities belonging to class 1."""

    probabilities = np.asarray(
        estimator.predict_proba(X)
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional probability array."
        )

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Expected probabilities for exactly two classes."
        )

    classes = getattr(
        estimator,
        "classes_",
        None,
    )

    if classes is None:
        raise ValueError(
            "Estimator does not expose classes_."
        )

    classes = list(classes)

    if 1 not in classes:
        raise ValueError(
            "Positive class 1 was not found."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


# ==========================================================
# THRESHOLD METRICS
# ==========================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate classification metrics for a threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
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
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None

    return metrics


# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

def select_best_threshold(
    y_true,
    probabilities,
    minimum_precision=0.0,
    minimum_recall=None,
):
    """
    Select the threshold with the highest F1-score
    while satisfying the supplied constraints.

    If no threshold satisfies all constraints, the
    highest-F1 threshold is returned with constraints_met=False.
    """

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Labels and probabilities have different lengths."
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
        pos_label=1,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No classification thresholds could be generated."
        )

    # precision and recall have one more element than thresholds.
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

    precision_mask = (
        precision_values >= minimum_precision
    )

    if minimum_recall is None:
        recall_mask = np.ones(
            len(thresholds),
            dtype=bool,
        )
    else:
        recall_mask = (
            recall_values >= minimum_recall
        )

    eligible_indices = np.where(
        precision_mask & recall_mask
    )[0]

    constraints_met = len(eligible_indices) > 0

    if constraints_met:
        candidate_indices = eligible_indices
    else:
        # Do not silently claim constraints were satisfied.
        # Return the best F1 threshold for diagnosis.
        candidate_indices = np.arange(
            len(thresholds)
        )

    best_index = max(
        candidate_indices,
        key=lambda index: (
            f1_values[index],
            precision_values[index],
            recall_values[index],
            thresholds[index],
        ),
    )

    selected_threshold = float(
        thresholds[best_index]
    )

    selected_metrics = calculate_threshold_metrics(
        y_true,
        probabilities,
        selected_threshold,
    )

    return {
        "threshold": selected_threshold,
        "metrics": selected_metrics,
        "constraints_met": bool(constraints_met),
        "minimum_precision": float(minimum_precision),
        "minimum_recall": (
            None
            if minimum_recall is None
            else float(minimum_recall)
        ),
        "eligible_threshold_count": int(
            len(eligible_indices)
        ),
        "total_threshold_count": int(
            len(thresholds)
        ),
    }


# ==========================================================
# CROSS-VALIDATED THRESHOLD OPTIMIZATION
# ==========================================================

def optimize_threshold_cv(
    estimator,
    X,
    y,
    minimum_precision,
    minimum_recall=None,
    n_splits=5,
    random_state=42,
):
    """
    Generate out-of-fold probabilities and optimize threshold.

    Important:
    - The estimator is cloned for cross-validation.
    - The original estimator is not fitted or modified.
    - Threshold selection uses out-of-fold predictions only.
    - Final evaluation must use separate untouched test data.
    """

    validate_inputs(
        estimator=estimator,
        X=X,
        y=y,
        n_splits=n_splits,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
    )

    y_values = np.asarray(y).astype(int)

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    print("\nCROSS-VALIDATED THRESHOLD OPTIMIZATION")
    print("=" * 60)

    print("Folds:", n_splits)
    print("Training samples:", len(y_values))
    print("Minimum precision:", minimum_precision)
    print("Minimum recall:", minimum_recall)

    # Clone ensures that the supplied estimator remains untouched.
    estimator_copy = clone(estimator)

    # Each row receives a prediction from a model that did not
    # train on that row.
    oof_probabilities = cross_val_predict(
        estimator_copy,
        X,
        y_values,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=1,
    )

    classes = np.asarray([0, 1])

    if oof_probabilities.ndim != 2:
        raise ValueError(
            "Cross-validation returned invalid probability dimensions."
        )

    if oof_probabilities.shape[1] != 2:
        raise ValueError(
            "Cross-validation did not return two class probabilities."
        )

    positive_probabilities = oof_probabilities[
        :,
        list(classes).index(1),
    ]

    result = select_best_threshold(
        y_true=y_values,
        probabilities=positive_probabilities,
        minimum_precision=minimum_precision,
        minimum_recall=minimum_recall,
    )

    result["cross_validation"] = {
        "method": "StratifiedKFold",
        "folds": int(n_splits),
        "random_state": int(random_state),
        "prediction_type": "out_of_fold",
        "sample_count": int(len(y_values)),
    }

    result["oof_probabilities"] = (
        positive_probabilities
    )

    print("\nOPTIMIZATION RESULTS")
    print("=" * 60)

    print("Selected threshold:", result["threshold"])
    print(
        "OOF precision:",
        round(result["metrics"]["precision"], 4),
    )
    print(
        "OOF recall:",
        round(result["metrics"]["recall"], 4),
    )
    print(
        "OOF F1:",
        round(result["metrics"]["f1"], 4),
    )
    print(
        "Constraints satisfied:",
        result["constraints_met"],
    )

    return result


# ==========================================================
# SELF TEST
# ==========================================================

def run_self_test():
    """Run a local test without loading external datasets."""

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    print("\nTHRESHOLD OPTIMIZER SELF-TEST")
    print("=" * 60)

    X, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        weights=[0.90, 0.10],
        random_state=42,
    )

    estimator = RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    result = optimize_threshold_cv(
        estimator=estimator,
        X=X,
        y=y,
        minimum_precision=0.50,
        minimum_recall=0.30,
        n_splits=5,
        random_state=42,
    )

    assert 0 <= result["threshold"] <= 1

    assert len(
        result["oof_probabilities"]
    ) == len(y)

    assert result["metrics"]["f1"] >= 0

    print("\nSELF-TEST PASSED")

    return result


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    run_self_test()
