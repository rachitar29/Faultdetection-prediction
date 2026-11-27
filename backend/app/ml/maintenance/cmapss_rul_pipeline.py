
"""
NASA C-MAPSS FD001 Remaining Useful Life (RUL) Pipeline

Complete workflow:
1. Load NASA FD001 data
2. Generate capped RUL labels
3. Split training engines to prevent engine-level leakage
4. Train three regression models
5. Compare validation performance
6. Evaluate on NASA FD001 test engine endpoints
7. Save the best model in the existing model registry
8. Save reports and test predictions

This pipeline does not overwrite classification or anomaly models.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.ml.algorithms.model_registry import save_model_bundle


warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"

ARTIFACT_DIR = BASE_DIR / "ml_artifacts"

REPORT_DIR = ARTIFACT_DIR / "reports"

PREPROCESSOR_DIR = ARTIFACT_DIR / "preprocessors"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

MODEL_NAME = "rul_cmapss_fd001_best"

RUL_CAP = 125
RANDOM_STATE = 42

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = COLUMN_NAMES[2:]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================

def load_cmapss_file(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required NASA dataset file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    # NASA files contain 26 actual columns.
    # Some versions may contain trailing empty columns.

    dataframe = dataframe.dropna(axis=1, how="all")

    if dataframe.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected number of columns in {file_path.name}. "
            f"Expected {len(COLUMN_NAMES)}, "
            f"found {dataframe.shape[1]}."
        )

    dataframe.columns = COLUMN_NAMES

    return dataframe


def load_all_data():

    print("\nLoading NASA C-MAPSS FD001 dataset...")

    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)

    rul_df = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    )

    rul_values = rul_df.iloc[:, 0].astype(float).to_numpy()

    if len(rul_values) != test_df["unit"].nunique():
        raise ValueError(
            "RUL label count does not match the number "
            "of test engines."
        )

    print(f"Training rows: {len(train_df)}")
    print(f"Training engines: {train_df['unit'].nunique()}")
    print(f"Test rows: {len(test_df)}")
    print(f"Test engines: {test_df['unit'].nunique()}")

    return train_df, test_df, rul_values


# ============================================================
# RUL LABEL GENERATION
# ============================================================

def add_rul_labels(dataframe):

    result = dataframe.copy()

    maximum_cycles = (
        result.groupby("unit")["cycle"]
        .transform("max")
    )

    result["RUL"] = maximum_cycles - result["cycle"]

    # Common capped-RUL target used in C-MAPSS studies.

    result["RUL"] = result["RUL"].clip(
        upper=RUL_CAP
    )

    return result


# ============================================================
# ENGINE-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def split_by_engine(dataframe):

    engine_ids = dataframe["unit"].unique()

    rng = np.random.RandomState(RANDOM_STATE)
    shuffled_ids = rng.permutation(engine_ids)

    validation_count = max(
        1,
        int(round(len(shuffled_ids) * 0.20))
    )

    validation_ids = shuffled_ids[:validation_count]

    training_ids = shuffled_ids[validation_count:]

    train_part = dataframe[
        dataframe["unit"].isin(training_ids)
    ].copy()

    validation_part = dataframe[
        dataframe["unit"].isin(validation_ids)
    ].copy()

    if set(training_ids).intersection(set(validation_ids)):
        raise RuntimeError(
            "Engine leakage detected between training and validation."
        )

    print(f"\nTraining engines: {len(training_ids)}")
    print(f"Validation engines: {len(validation_ids)}")

    return train_part, validation_part


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(dataframe):

    X = dataframe[FEATURE_COLUMNS].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(X.median(numeric_only=True))

    return X


# ============================================================
# REGRESSION MODELS
# ============================================================

def build_models():

    return {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=180,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE
        )
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    predicted = np.clip(
        predicted,
        0,
        None
    )

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "r2": float(
            r2_score(actual, predicted)
        )
    }


# ============================================================
# NASA RUL SCORING FUNCTION
# ============================================================

def nasa_score(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - actual

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1,
        np.exp(errors / 10.0) - 1
    )

    return float(np.sum(scores))


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

def train_and_compare(train_df, validation_df):

    X_train = prepare_features(train_df)
    y_train = train_df["RUL"].to_numpy()

    X_validation = prepare_features(validation_df)
    y_validation = validation_df["RUL"].to_numpy()

    models = build_models()

    results = {}
    trained_models = {}

    print("\nTraining regression models...")

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_validation)

        predictions = np.clip(
            predictions,
            0,
            None
        )

        metrics = calculate_metrics(
            y_validation,
            predictions
        )

        results[model_name] = metrics
        trained_models[model_name] = model

        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R2: {metrics['r2']:.4f}")

    # Lowest validation MAE is the selection criterion.

    best_name = min(
        results,
        key=lambda name: results[name]["mae"]
    )

    best_model = trained_models[best_name]

    print("\nBest validation model:", best_name)

    return (
        best_name,
        best_model,
        results
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_test_engines(
    model,
    test_df,
    actual_rul
):

    print("\nEvaluating final cycle of test engines...")

    final_rows = (
        test_df.sort_values("cycle")
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
    )

    X_test = prepare_features(final_rows)

    predicted_rul = model.predict(X_test)

    predicted_rul = np.clip(
        predicted_rul,
        0,
        None
    )

    actual_rul = np.asarray(
        actual_rul,
        dtype=float
    )

    metrics = calculate_metrics(
        actual_rul,
        predicted_rul
    )

    metrics["nasa_score"] = nasa_score(
        actual_rul,
        predicted_rul
    )

    predictions_df = pd.DataFrame({
        "engine_id": final_rows["unit"].to_numpy(),
        "actual_rul": actual_rul,
        "predicted_rul": np.round(predicted_rul, 2),
        "absolute_error": np.round(
            np.abs(actual_rul - predicted_rul),
            2
        )
    })

    prediction_path = (
        REPORT_DIR / "cmapss_fd001_rul_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    print("\nTest metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"NASA score: {metrics['nasa_score']:.4f}")

    return metrics, predictions_df


# ============================================================
# MODEL REGISTRATION
# ============================================================

def register_best_model(
    model,
    model_name,
    validation_metrics,
    test_metrics
):

    print("\nRegistering RUL model...")

    # Regression has no binary decision threshold.
    # 0.0 is stored only to satisfy the existing registry API.

    metadata_metrics = {
        "validation": validation_metrics,
        "test": test_metrics
    }

    save_model_bundle(
        model_name=MODEL_NAME,
        model=model,
        threshold=0.0,
        model_type="Regression",
        feature_names=FEATURE_COLUMNS,
        metrics=metadata_metrics,
        best_parameters=model.get_params(),
        description=(
            "NASA C-MAPSS FD001 Remaining Useful Life "
            "regression model. Selected using validation MAE. "
            "Predicts capped RUL in operating cycles."
        )
    )

    print(f"Registered model: {MODEL_NAME}")


# ============================================================
# REPORT GENERATION
# ============================================================

def save_report(
    best_name,
    validation_results,
    test_metrics,
    predictions_df
):

    report = {
        "dataset": "NASA C-MAPSS FD001",
        "task": "Remaining Useful Life Regression",
        "rul_cap": RUL_CAP,
        "selection_metric": "Validation MAE",
        "best_validation_model": best_name,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "test_engine_count": int(len(predictions_df)),
        "model_registry_name": MODEL_NAME,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    report_path = (
        REPORT_DIR / "cmapss_fd001_rul_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(f"\nReport saved: {report_path}")


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_pipeline():

    print("=" * 65)
    print("NASA C-MAPSS FD001 RUL TRAINING PIPELINE")
    print("=" * 65)

    create_directories()

    train_df, test_df, actual_rul = load_all_data()

    train_df = add_rul_labels(train_df)

    training_part, validation_part = split_by_engine(
        train_df
    )

    best_name, best_model, validation_results = (
        train_and_compare(
            training_part,
            validation_part
        )
    )

    # Retrain the selected algorithm on all available training engines.
    # The validation split is used only for model selection.
    print("\\nRefitting selected model using all training engines...")

    final_model = build_models()[best_name]

    X_full_train = prepare_features(train_df)
    y_full_train = train_df["RUL"].to_numpy()

    final_model.fit(X_full_train, y_full_train)

    print(f"Final training rows: {len(X_full_train)}")
    print(f"Final training engines: {train_df['unit'].nunique()}")

    # Evaluate the refitted model on the untouched NASA test engines.
    test_metrics, predictions_df = evaluate_test_engines(
        final_model,
        test_df,
        actual_rul
    )

    best_validation_metrics = validation_results[best_name]

    register_best_model(
        final_model,
        best_name,
        best_validation_metrics,
        test_metrics
    )

    save_report(
        best_name,
        validation_results,
        test_metrics,
        predictions_df
    )

    print("\n" + "=" * 65)
    print("NASA C-MAPSS FD001 RUL PIPELINE COMPLETED")
    print("=" * 65)

    print(f"Selected model: {best_name}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test R2: {test_metrics['r2']:.4f}")

    print("\nSaved model:", MODEL_NAME)

    return {
        "model_name": MODEL_NAME,
        "best_algorithm": best_name,
        "validation_results": validation_results,
        "test_metrics": test_metrics
    }


if __name__ == "__main__":
    run_pipeline()
