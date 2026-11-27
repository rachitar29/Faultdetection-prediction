
from pathlib import Path
import json
import copy
import traceback

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from sklearn.metrics import f1_score, average_precision_score

from app.ml.data.data_loader import load_dataset
from app.ml.data.dynamic_preprocessor import DynamicDatasetPreprocessor
from app.ml.training.model_selector import select_models
from app.ml.training.training_pipeline import (
    _split_dataset,
    _select_threshold,
    _calculate_metrics,
    DatasetModelBundle
)
from app.ml.algorithms.anomaly_detection import get_anomaly_scores
from app.ml.algorithms.model_registry import (
    save_model_bundle,
    REGISTRY_DIR
)


REPORT_DIR = Path("ml_artifacts/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _last_step_name(pipeline):
    if not hasattr(pipeline, "steps") or not pipeline.steps:
        raise ValueError("Expected an sklearn Pipeline.")

    return pipeline.steps[-1][0]


def _classification_parameter_grid(model_name):
    if "random forest" in model_name.lower():
        return {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [None, 8, 12, 16, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.8],
            "class_weight": ["balanced", "balanced_subsample"]
        }

    if "xgboost" in model_name.lower():
        return {
            "n_estimators": [200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0]
        }

    raise ValueError(f"No classification search space for {model_name}")


def _anomaly_parameter_candidates(model_name, training_rows):
    if "isolation" in model_name.lower():
        return [
            {
                "n_estimators": trees,
                "contamination": contamination
            }
            for trees in [150, 250, 350]
            for contamination in [0.01, 0.03, 0.05, 0.08, 0.10]
        ]

    if "local" in model_name.lower():
        return [
            {
                "n_neighbors": neighbors,
                "contamination": contamination
            }
            for neighbors in [10, 20, 30, 40]
            if neighbors < training_rows
            for contamination in [0.01, 0.03, 0.05, 0.08, 0.10]
        ]

    raise ValueError(f"No anomaly search space for {model_name}")


def _optimize_classification(
    model_name,
    estimator,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    step_name = _last_step_name(estimator)
    parameter_grid = _classification_parameter_grid(model_name)

    minimum_class_count = int(y_train.value_counts().min())

    if minimum_class_count < 2:
        raise ValueError(
            "Not enough samples in every class for cross-validation."
        )

    cv_splits = min(3, minimum_class_count)

    prefixed_grid = {
        f"{step_name}__{key}": values
        for key, values in parameter_grid.items()
    }

    total_combinations = len(list(ParameterGrid(prefixed_grid)))

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=prefixed_grid,
        n_iter=min(10, total_combinations),
        scoring="average_precision",
        cv=cv_splits,
        random_state=42,
        n_jobs=1,
        refit=True,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    val_probabilities = best_model.predict_proba(X_val)[:, 1]

    threshold = _select_threshold(
        y_val,
        val_probabilities
    )

    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = _calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities
    )

    validation_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    validation_f1 = float(
        f1_score(y_val, validation_predictions, zero_division=0)
    )

    return {
        "model": best_model,
        "threshold": threshold,
        "metrics": metrics,
        "validation_f1": validation_f1,
        "best_parameters": search.best_params_,
        "best_cv_average_precision": float(search.best_score_)
    }


def _optimize_anomaly(
    model_name,
    estimator,
    X_train,
    y_val,
    X_val,
    X_test,
    y_test
):
    step_name = _last_step_name(estimator)

    candidates = _anomaly_parameter_candidates(
        model_name,
        len(X_train)
    )

    if not candidates:
        raise ValueError("No valid anomaly parameter combinations.")

    best_result = None

    for parameters in candidates:
        candidate = clone(estimator)

        candidate.set_params(**{
            f"{step_name}__{key}": value
            for key, value in parameters.items()
        })

        candidate.fit(X_train)

        validation_scores = get_anomaly_scores(
            candidate,
            X_val
        )

        threshold = _select_threshold(
            y_val,
            validation_scores
        )

        validation_predictions = (
            validation_scores >= threshold
        ).astype(int)

        validation_f1 = float(
            f1_score(
                y_val,
                validation_predictions,
                zero_division=0
            )
        )

        if (
            best_result is None
            or validation_f1 > best_result["validation_f1"]
        ):
            best_result = {
                "model": candidate,
                "threshold": threshold,
                "validation_f1": validation_f1,
                "best_parameters": parameters
            }

    if best_result is None:
        raise RuntimeError("Anomaly optimization failed.")

    best_model = best_result["model"]

    test_scores = get_anomaly_scores(
        best_model,
        X_test
    )

    test_predictions = (
        test_scores >= best_result["threshold"]
    ).astype(int)

    metrics = _calculate_metrics(
        y_test,
        test_predictions,
        test_scores
    )

    best_result["metrics"] = metrics

    return best_result


def optimize_dataset(
    dataset_path,
    config_path,
    dataset_name,
    random_state=42,
    save_models=True
):
    """Optimize all four algorithms for one dataset."""

    dataset_path = Path(dataset_path)
    config_path = Path(config_path)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    df = load_dataset(dataset_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    processor = DynamicDatasetPreprocessor(config_path)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = _split_dataset(
        df,
        processor.target_column,
        test_size=0.20,
        validation_size=0.20,
        random_state=random_state
    )

    processor.fit(
        pd.concat([X_train, y_train], axis=1)
    )

    X_train = processor._prepare_input(X_train)
    X_val = processor._prepare_input(X_val)
    X_test = processor._prepare_input(X_test)

    results = {
        "dataset_name": dataset_name,
        "dataset_rows": len(df),
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "testing_rows": len(X_test),
        "models": {},
        "errors": {}
    }

    selected_models = select_models(
        task="all",
        preprocessor=processor.preprocessor,
        y_train=y_train
    )

    for model_name, estimator in selected_models.items():
        print(f"\nOptimizing {model_name} for {dataset_name}")

        try:
            if model_name in ("Random Forest", "XGBoost"):
                optimized = _optimize_classification(
                    model_name,
                    estimator,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test
                )
                model_type = "classification"

            else:
                optimized = _optimize_anomaly(
                    model_name,
                    estimator,
                    X_train,
                    y_val,
                    X_val,
                    X_test,
                    y_test
                )
                model_type = "anomaly_detection"

            registry_name = (
                f"{dataset_name.lower().replace(' ', '_')}_"
                f"{model_name.lower().replace(' ', '_')}_optimized_60_20_20"
            )

            bundle = DatasetModelBundle(
                processor,
                optimized["model"]
            )

            if save_models:
                save_model_bundle(
                    model_name=registry_name,
                    model=bundle,
                    threshold=optimized["threshold"],
                    model_type=model_type,
                    feature_names=processor.feature_columns,
                    metrics=optimized["metrics"],
                    best_parameters=optimized["best_parameters"],
                    description=(
                        f"Optimized {model_name} for {dataset_name}. "
                        "Hyperparameters selected without using test data."
                    )
                )

            results["models"][registry_name] = {
                "algorithm": model_name,
                "task": model_type,
                "threshold": float(optimized["threshold"]),
                "validation_f1": float(optimized["validation_f1"]),
                "metrics": optimized["metrics"],
                "best_parameters": optimized["best_parameters"],
                "saved": save_models
            }

            if "best_cv_average_precision" in optimized:
                results["models"][registry_name][
                    "best_cv_average_precision"
                ] = optimized["best_cv_average_precision"]

        except Exception as exc:
            results["errors"][model_name] = {
                "error": str(exc),
                "traceback": traceback.format_exc()
            }

    results["successful_models"] = len(results["models"])
    results["failed_models"] = len(results["errors"])

    return results


def optimize_multiple_datasets(dataset_specs):
    """Run optimization for multiple configured datasets."""

    report = {
        "datasets_processed": 0,
        "models_optimized": 0,
        "failed_datasets": 0,
        "results": []
    }

    for spec in dataset_specs:
        try:
            result = optimize_dataset(
                dataset_path=spec["dataset_path"],
                config_path=spec["config_path"],
                dataset_name=spec["name"]
            )

            report["results"].append(result)
            report["datasets_processed"] += 1
            report["models_optimized"] += result["successful_models"]

        except Exception as exc:
            report["failed_datasets"] += 1
            report["results"].append({
                "dataset_name": spec["name"],
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc()
            })

    report_path = REPORT_DIR / "model_optimization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("\n" + "=" * 55)
    print("MODEL OPTIMIZATION SUMMARY")
    print("=" * 55)
    print("Datasets processed:", report["datasets_processed"])
    print("Models optimized:", report["models_optimized"])
    print("Failed datasets:", report["failed_datasets"])
    print("Report:", report_path)

    return report
