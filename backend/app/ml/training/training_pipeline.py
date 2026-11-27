
from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

from app.ml.data.data_loader import load_dataset
from app.ml.data.dynamic_preprocessor import DynamicDatasetPreprocessor
from app.ml.training.model_selector import select_models
from app.ml.algorithms.anomaly_detection import get_anomaly_scores
from app.ml.algorithms.model_registry import save_model_bundle


class DatasetModelBundle:
    """Prediction wrapper accepting raw DataFrames."""

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


def _safe_name(value):
    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(value).strip().lower()
    ).strip("_")

    if not name:
        raise ValueError("Dataset name cannot be empty.")

    return name


def _get_target(config):
    target = config.get("target")

    if isinstance(target, dict):
        return target.get("column")

    if isinstance(target, str):
        return target

    return config.get("target_column")


def _split_dataset(
    df,
    target_column,
    test_size=0.20,
    validation_size=0.20,
    random_state=42
):
    """Create stratified 60/20/20 train/validation/test splits."""

    if not target_column or target_column not in df.columns:
        raise ValueError(
            f"Valid target column required: {target_column}"
        )

    if test_size <= 0 or validation_size <= 0:
        raise ValueError("Split proportions must be greater than zero.")

    if test_size + validation_size >= 1:
        raise ValueError(
            "Test and validation proportions must total less than 1."
        )

    y = df[target_column]

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    X = df.drop(columns=[target_column])

    stratify_first = (
        y if y.value_counts().min() >= 3 else None
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_first
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
        stratify=stratify_second
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _get_positive_probability(model, X):
    probabilities = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


def _select_threshold(y_true, scores):
    """Select an F1-optimal threshold using validation data only."""

    y_array = np.asarray(y_true)

    if len(np.unique(y_array)) < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(
        y_array,
        scores,
        pos_label=1
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(np.argmax(f1_values))

    return float(thresholds[best_index])


def _calculate_metrics(y_true, predictions, scores):
    """Calculate classification evaluation metrics."""

    metrics: dict[str, float | None] = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        )
    }

    if len(np.unique(y_true)) == 2 and scores is not None:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )
    else:
        metrics["roc_auc"] = None

    return metrics


def _classification_metrics(model, X, y, threshold):
    probabilities = _get_positive_probability(model, X)

    predictions = (probabilities >= threshold).astype(int)

    return _calculate_metrics(
        y,
        predictions,
        probabilities
    )


def _anomaly_metrics(model, X, y, threshold):
    scores = get_anomaly_scores(model, X)

    predictions = (scores >= threshold).astype(int)

    return _calculate_metrics(
        y,
        predictions,
        scores
    )


def train_dataset(
    dataset_path,
    config_path,
    dataset_name,
    test_size=0.20,
    random_state=42,
    tasks=("classification", "anomaly_detection"),
    save_models=True,
    validation_size=0.20
):
    """
    Train and evaluate models using 60/20/20 splitting.

    Training data:
        Used for fitting models.

    Validation data:
        Used for threshold optimization.

    Testing data:
        Used only for final evaluation.

    Anomaly models do not use target labels during fitting.
    """

    dataset_path = Path(dataset_path)
    config_path = Path(config_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {config_path}"
        )

    df = load_dataset(dataset_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    processor = DynamicDatasetPreprocessor(config_path)

    target_column = processor.target_column

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = _split_dataset(
        df,
        target_column,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state
    )

    # Fit preprocessing only on training data.
    processor.fit(
        pd.concat([X_train, y_train], axis=1)
    )

    raw_X_train = processor._prepare_input(X_train)
    raw_X_val = processor._prepare_input(X_val)
    raw_X_test = processor._prepare_input(X_test)

    dataset_slug = _safe_name(dataset_name)

    results = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "target_column": target_column,
        "rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "testing_rows": int(len(X_test)),
        "split": {
            "training": 0.60,
            "validation": 0.20,
            "testing": 0.20
        },
        "feature_columns": processor.feature_columns,
        "numeric_features": processor.numeric_columns,
        "categorical_features": processor.categorical_columns,
        "models": {},
        "errors": {}
    }

    # ---------------- CLASSIFICATION ----------------

    if "classification" in tasks:

        if not set(y_train.unique()).issubset({0, 1}):
            results["errors"]["classification"] = (
                "Only binary labels 0 and 1 are supported."
            )

        elif y_train.nunique() < 2:
            results["errors"]["classification"] = (
                "Training split contains fewer than two classes."
            )

        else:
            try:
                classification_models = select_models(
                    task="classification",
                    preprocessor=processor.preprocessor,
                    y_train=y_train
                )

                for model_name, estimator in classification_models.items():
                    try:
                        print(
                            f"\nTraining {model_name} for {dataset_name}"
                        )

                        estimator.fit(raw_X_train, y_train)

                        # Select threshold using validation data.
                        val_scores = _get_positive_probability(
                            estimator,
                            raw_X_val
                        )

                        threshold = _select_threshold(
                            y_val,
                            val_scores
                        )

                        # Final evaluation on untouched test data.
                        metrics = _classification_metrics(
                            estimator,
                            raw_X_test,
                            y_test,
                            threshold
                        )

                        default_metrics = _classification_metrics(
                            estimator,
                            raw_X_test,
                            y_test,
                            0.5
                        )

                        bundle = DatasetModelBundle(
                            processor,
                            estimator
                        )

                        registry_name = (
                            f"{dataset_slug}_"
                            f"{_safe_name(model_name)}_60_20_20"
                        )

                        if save_models:
                            save_model_bundle(
                                model_name=registry_name,
                                model=bundle,
                                threshold=threshold,
                                model_type="classification",
                                feature_names=processor.feature_columns,
                                metrics=metrics,
                                description=(
                                    f"{model_name} trained for "
                                    f"{dataset_name} using 60/20/20 split."
                                )
                            )

                        results["models"][registry_name] = {
                            "algorithm": model_name,
                            "task": "classification",
                            "threshold": threshold,
                            "metrics": metrics,
                            "default_threshold_metrics": default_metrics,
                            "saved": save_models
                        }

                    except Exception as exc:
                        results["errors"][
                            f"classification_{model_name}"
                        ] = str(exc)

            except Exception as exc:
                results["errors"]["classification"] = str(exc)

    # ---------------- ANOMALY DETECTION ----------------

    if "anomaly_detection" in tasks:

        try:
            anomaly_models = select_models(
                task="anomaly_detection",
                preprocessor=processor.preprocessor
            )

            for model_name, estimator in anomaly_models.items():
                try:
                    print(
                        f"\nTraining {model_name} for {dataset_name}"
                    )

                    # Labels are NOT used during model fitting.
                    estimator.fit(raw_X_train)

                    # Calibrate anomaly threshold on validation labels.
                    val_scores = get_anomaly_scores(
                        estimator,
                        raw_X_val
                    )

                    threshold = _select_threshold(
                        y_val,
                        val_scores
                    )

                    metrics = _anomaly_metrics(
                        estimator,
                        raw_X_test,
                        y_test,
                        threshold
                    )

                    bundle = DatasetModelBundle(
                        processor,
                        estimator
                    )

                    registry_name = (
                        f"{dataset_slug}_"
                        f"{_safe_name(model_name)}_60_20_20"
                    )

                    if save_models:
                        save_model_bundle(
                            model_name=registry_name,
                            model=bundle,
                            threshold=threshold,
                            model_type="anomaly_detection",
                            feature_names=processor.feature_columns,
                            metrics=metrics,
                            description=(
                                f"{model_name} trained for "
                                f"{dataset_name} using 60/20/20 split. "
                                "Labels were not used during fitting."
                            )
                        )

                    results["models"][registry_name] = {
                        "algorithm": model_name,
                        "task": "anomaly_detection",
                        "threshold": threshold,
                        "metrics": metrics,
                        "saved": save_models
                    }

                except Exception as exc:
                    results["errors"][
                        f"anomaly_{model_name}"
                    ] = str(exc)

        except Exception as exc:
            results["errors"]["anomaly_detection"] = str(exc)

    # Save a separate preprocessor artifact.
    if save_models:
        from app.ml.algorithms.model_registry import REGISTRY_DIR

        preprocessor_path = (
            REGISTRY_DIR
            / f"{dataset_slug}_preprocessor_60_20_20.joblib"
        )

        processor.save(preprocessor_path)

        results["preprocessor_path"] = str(
            preprocessor_path
        )

    results["successful_models"] = len(results["models"])
    results["failed_components"] = len(results["errors"])

    return results
