"""
NASA C-MAPSS FD001 Temporal RUL Experiment

Compares:
1. Existing registered HistGradientBoosting baseline
2. Experimental HistGradientBoosting with temporal sensor features

IMPORTANT:
- Does not modify the existing registered RUL model.
- Does not call save_model_bundle().
- Uses only current and historical sensor observations.
- Splits training and validation by engine ID.
- Saves experimental artifacts separately.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from app.ml.algorithms.model_registry import load_model_bundle


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"

ARTIFACT_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "experimental"
    / "rul_cmapss_fd001_temporal"
)

REPORT_DIR = BASE_DIR / "ml_artifacts" / "reports"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

BASELINE_MODEL_NAME = "rul_cmapss_fd001_best"

RUL_CAP = 125
RANDOM_STATE = 42

WINDOWS = [10, 30]

TEMPORAL_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_12",
    "sensor_14",
    "sensor_17"
]

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = COLUMN_NAMES[2:]

TEMPORAL_FEATURE_COLUMNS = []

for sensor in TEMPORAL_SENSORS:
    for window in WINDOWS:
        TEMPORAL_FEATURE_COLUMNS.extend([
            f"{sensor}_rolling_mean_{window}",
            f"{sensor}_rolling_std_{window}",
            f"{sensor}_slope_{window}",
            f"{sensor}_deviation_{window}"
        ])

EXPERIMENTAL_FEATURE_COLUMNS = (
    FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS
)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_cmapss_file(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    dataframe = dataframe.dropna(
        axis=1,
        how="all"
    )

    if dataframe.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Unexpected columns in {file_path.name}. "
            f"Expected {len(COLUMN_NAMES)}, "
            f"found {dataframe.shape[1]}."
        )

    dataframe.columns = COLUMN_NAMES

    return dataframe


def load_all_data():

    print("\nLoading NASA FD001 datasets...")

    train_df = load_cmapss_file(TRAIN_FILE)
    test_df = load_cmapss_file(TEST_FILE)

    rul_df = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None
    )

    actual_rul = (
        rul_df.iloc[:, 0]
        .astype(float)
        .to_numpy()
    )

    test_engine_count = test_df["unit"].nunique()

    if len(actual_rul) != test_engine_count:
        raise ValueError(
            "Test RUL count does not match test engine count."
        )

    print("Training rows:", len(train_df))
    print("Training engines:", train_df["unit"].nunique())
    print("Test rows:", len(test_df))
    print("Test engines:", test_engine_count)

    return train_df, test_df, actual_rul


# ============================================================
# RUL LABEL GENERATION
# ============================================================

def add_rul_labels(dataframe):

    result = dataframe.copy()

    maximum_cycles = (
        result.groupby("unit")["cycle"]
        .transform("max")
    )

    result["RUL"] = (
        maximum_cycles - result["cycle"]
    ).clip(upper=RUL_CAP)

    return result


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def calculate_rolling_slope(values):

    values = np.asarray(
        values,
        dtype=float
    )

    count = len(values)

    if count < 2:
        return 0.0

    x = np.arange(
        count,
        dtype=float
    )

    x_centered = x - x.mean()
    y_centered = values - values.mean()

    denominator = np.dot(
        x_centered,
        x_centered
    )

    if denominator == 0:
        return 0.0

    slope_per_cycle = (
        np.dot(x_centered, y_centered)
        / denominator
    )

    return float(slope_per_cycle * 10.0)


def add_temporal_features(dataframe):

    print("\nGenerating historical sensor features...")

    if not set(
        ["unit", "cycle"] + TEMPORAL_SENSORS
    ).issubset(dataframe.columns):

        raise ValueError(
            "Required temporal feature columns are missing."
        )

    engine_parts = []

    for engine_id, engine in dataframe.groupby(
        "unit",
        sort=True
    ):

        engine = engine.sort_values(
            "cycle"
        ).copy()

        for sensor in TEMPORAL_SENSORS:

            values = pd.to_numeric(
                engine[sensor],
                errors="coerce"
            )

            for window in WINDOWS:

                rolling = values.rolling(
                    window=window,
                    min_periods=1
                )

                rolling_mean = rolling.mean()

                rolling_std = rolling.std(
                    ddof=0
                ).fillna(0.0)

                rolling_slope = values.rolling(
                    window=window,
                    min_periods=2
                ).apply(
                    calculate_rolling_slope,
                    raw=True
                ).fillna(0.0)

                engine[
                    f"{sensor}_rolling_mean_{window}"
                ] = rolling_mean

                engine[
                    f"{sensor}_rolling_std_{window}"
                ] = rolling_std

                engine[
                    f"{sensor}_slope_{window}"
                ] = rolling_slope

                engine[
                    f"{sensor}_deviation_{window}"
                ] = values - rolling_mean

        engine_parts.append(engine)

    result = pd.concat(
        engine_parts,
        axis=0
    ).sort_values(
        ["unit", "cycle"]
    )

    if not set(
        EXPERIMENTAL_FEATURE_COLUMNS
    ).issubset(result.columns):

        raise RuntimeError(
            "Temporal feature generation was incomplete."
        )

    print(
        "Original features:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Additional temporal features:",
        len(TEMPORAL_FEATURE_COLUMNS)
    )

    print(
        "Total experimental features:",
        len(EXPERIMENTAL_FEATURE_COLUMNS)
    )

    return result


# ============================================================
# ENGINE-LEVEL SPLIT
# ============================================================

def split_by_engine(dataframe):

    engine_ids = dataframe["unit"].unique()

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    shuffled_ids = rng.permutation(
        engine_ids
    )

    validation_count = max(
        1,
        int(round(len(shuffled_ids) * 0.20))
    )

    validation_ids = shuffled_ids[
        :validation_count
    ]

    training_ids = shuffled_ids[
        validation_count:
    ]

    if set(training_ids).intersection(
        set(validation_ids)
    ):

        raise RuntimeError(
            "Engine leakage detected."
        )

    train_part = dataframe[
        dataframe["unit"].isin(training_ids)
    ].copy()

    validation_part = dataframe[
        dataframe["unit"].isin(validation_ids)
    ].copy()

    print("\nEngine-level split:")
    print("Training engines:", len(training_ids))
    print("Validation engines:", len(validation_ids))

    return train_part, validation_part


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(
    dataframe,
    feature_columns,
    medians=None
):

    X = dataframe[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if medians is None:

        medians = X.median(
            numeric_only=True
        ).fillna(0.0)

    X = X.fillna(medians)

    X = X.fillna(0.0)

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "Non-finite values remain after feature preparation."
        )

    return X, medians


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_model():

    return HistGradientBoostingRegressor(
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=20,
        random_state=42
    )


# ============================================================
# METRICS
# ============================================================

def calculate_nasa_score(actual, predicted):

    errors = (
        np.asarray(predicted, dtype=float)
        - np.asarray(actual, dtype=float)
    )

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1,
        np.exp(errors / 10.0) - 1
    )

    return float(np.sum(scores))


def calculate_metrics(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.clip(
        np.asarray(predicted, dtype=float),
        0,
        None
    )

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
        "r2": float(
            r2_score(actual, predicted)
        ),
        "nasa_score": calculate_nasa_score(
            actual,
            predicted
        )
    }


# ============================================================
# MAINTENANCE CATEGORY AUDIT
# ============================================================

def classify_rul(value):

    if value <= 10:
        return "CRITICAL"

    if value <= 25:
        return "HIGH"

    if value <= 50:
        return "MEDIUM"

    return "LOW"


def calculate_maintenance_audit(
    actual,
    predicted
):

    actual_categories = [
        classify_rul(value)
        for value in actual
    ]

    predicted_categories = [
        classify_rul(value)
        for value in predicted
    ]

    priority = {
        "CRITICAL": 4,
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    actual_ranks = np.array([
        priority[value]
        for value in actual_categories
    ])

    predicted_ranks = np.array([
        priority[value]
        for value in predicted_categories
    ])

    dangerous = (
        predicted_ranks < actual_ranks
    )

    return {
        "category_mismatches": int(
            np.sum(
                np.array(actual_categories)
                != np.array(predicted_categories)
            )
        ),
        "dangerous_underprioritization": int(
            np.sum(dangerous)
        ),
        "actual_high_or_critical_engines": int(
            np.sum(np.asarray(actual) <= 25)
        ),
        "high_or_critical_overestimations": int(
            np.sum(
                (np.asarray(actual) <= 25)
                & (np.asarray(predicted) > np.asarray(actual))
            )
        )
    }


# ============================================================
# VALIDATION COMPARISON
# ============================================================

def compare_validation_models(
    train_df,
    validation_df
):

    print("\n========== VALIDATION COMPARISON ==========")

    y_train = train_df["RUL"].to_numpy()
    y_validation = validation_df["RUL"].to_numpy()

    baseline_train, baseline_medians = prepare_features(
        train_df,
        FEATURE_COLUMNS
    )

    baseline_validation, _ = prepare_features(
        validation_df,
        FEATURE_COLUMNS,
        baseline_medians
    )

    temporal_train, temporal_medians = prepare_features(
        train_df,
        EXPERIMENTAL_FEATURE_COLUMNS
    )

    temporal_validation, _ = prepare_features(
        validation_df,
        EXPERIMENTAL_FEATURE_COLUMNS,
        temporal_medians
    )

    baseline_model = build_model()

    temporal_model = build_model()

    print("\nTraining baseline validation model...")

    baseline_model.fit(
        baseline_train,
        y_train
    )

    baseline_predictions = np.clip(
        baseline_model.predict(
            baseline_validation
        ),
        0,
        None
    )

    baseline_metrics = calculate_metrics(
        y_validation,
        baseline_predictions
    )

    print("\nTraining temporal validation model...")

    temporal_model.fit(
        temporal_train,
        y_train
    )

    temporal_predictions = np.clip(
        temporal_model.predict(
            temporal_validation
        ),
        0,
        None
    )

    temporal_metrics = calculate_metrics(
        y_validation,
        temporal_predictions
    )

    print("\nBASELINE VALIDATION:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nTEMPORAL VALIDATION:")
    print(json.dumps(temporal_metrics, indent=4))

    return {
        "baseline": baseline_metrics,
        "temporal": temporal_metrics
    }


# ============================================================
# FINAL TEMPORAL MODEL TRAINING
# ============================================================

def train_final_temporal_model(train_df):

    print("\nTraining final experimental temporal model...")

    X_train, medians = prepare_features(
        train_df,
        EXPERIMENTAL_FEATURE_COLUMNS
    )

    y_train = train_df["RUL"].to_numpy()

    model = build_model()

    model.fit(
        X_train,
        y_train
    )

    print("Final temporal training rows:", len(X_train))
    print("Final temporal features:", len(EXPERIMENTAL_FEATURE_COLUMNS))

    return model, medians


# ============================================================
# TEST ENGINE EVALUATION
# ============================================================

def evaluate_test_engines(
    baseline_model,
    temporal_model,
    temporal_medians,
    test_df,
    actual_rul
):

    print("\n========== FINAL TEST ENGINE EVALUATION ==========")

    final_rows = (
        test_df.sort_values(
            ["unit", "cycle"]
        )
        .groupby("unit")
        .tail(1)
        .sort_values("unit")
        .copy()
    )

    actual_rul = np.asarray(
        actual_rul,
        dtype=float
    )

    if len(final_rows) != len(actual_rul):
        raise ValueError(
            "Final test engine count does not match reference RUL."
        )

    baseline_bundle = load_model_bundle(
        BASELINE_MODEL_NAME
    )

    registered_metadata = baseline_bundle["metadata"]

    registered_features = registered_metadata.get(
        "feature_names",
        FEATURE_COLUMNS
    )

    if registered_features != FEATURE_COLUMNS:
        raise ValueError(
            "Registered baseline feature schema differs "
            "from the expected FD001 feature schema."
        )

    registered_baseline = baseline_bundle["model"]

    X_baseline, _ = prepare_features(
        final_rows,
        FEATURE_COLUMNS
    )

    X_temporal, _ = prepare_features(
        final_rows,
        EXPERIMENTAL_FEATURE_COLUMNS,
        temporal_medians
    )

    baseline_predictions = np.clip(
        registered_baseline.predict(
            X_baseline[registered_features]
        ),
        0,
        None
    )

    temporal_predictions = np.clip(
        temporal_model.predict(
            X_temporal
        ),
        0,
        None
    )

    baseline_metrics = calculate_metrics(
        actual_rul,
        baseline_predictions
    )

    temporal_metrics = calculate_metrics(
        actual_rul,
        temporal_predictions
    )

    baseline_metrics["maintenance_audit"] = (
        calculate_maintenance_audit(
            actual_rul,
            baseline_predictions
        )
    )

    temporal_metrics["maintenance_audit"] = (
        calculate_maintenance_audit(
            actual_rul,
            temporal_predictions
        )
    )

    predictions = pd.DataFrame({
        "engine_id": final_rows["unit"].to_numpy(),
        "last_cycle": final_rows["cycle"].to_numpy(),
        "actual_rul": actual_rul,
        "baseline_predicted_rul": np.round(
            baseline_predictions,
            2
        ),
        "temporal_predicted_rul": np.round(
            temporal_predictions,
            2
        ),
        "baseline_error": np.round(
            baseline_predictions - actual_rul,
            2
        ),
        "temporal_error": np.round(
            temporal_predictions - actual_rul,
            2
        ),
        "baseline_absolute_error": np.round(
            np.abs(baseline_predictions - actual_rul),
            2
        ),
        "temporal_absolute_error": np.round(
            np.abs(temporal_predictions - actual_rul),
            2
        ),
        "actual_category": [
            classify_rul(value)
            for value in actual_rul
        ],
        "baseline_category": [
            classify_rul(value)
            for value in baseline_predictions
        ],
        "temporal_category": [
            classify_rul(value)
            for value in temporal_predictions
        ]
    })

    predictions_path = (
        REPORT_DIR
        / "cmapss_fd001_temporal_test_comparison.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    print("\nREGISTERED BASELINE TEST METRICS:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nEXPERIMENTAL TEMPORAL TEST METRICS:")
    print(json.dumps(temporal_metrics, indent=4))

    print("\nComparison report:", predictions_path)

    return (
        baseline_metrics,
        temporal_metrics,
        predictions
    )


# ============================================================
# SAVE EXPERIMENTAL ARTIFACTS
# ============================================================

def save_experimental_artifacts(
    model,
    medians,
    validation_results,
    baseline_metrics,
    temporal_metrics,
    predictions
):

    model_path = ARTIFACT_DIR / "model.joblib"

    metadata_path = ARTIFACT_DIR / "metadata.json"

    report_path = (
        REPORT_DIR
        / "cmapss_fd001_temporal_experiment_report.json"
    )

    model_tmp = ARTIFACT_DIR / "model.tmp.joblib"

    metadata_tmp = ARTIFACT_DIR / "metadata.tmp.json"

    joblib.dump(
        model,
        model_tmp
    )

    model_tmp.replace(
        model_path
    )

    metadata = {
        "model_name": "rul_cmapss_fd001_temporal_experiment",
        "model_type": "Experimental Regression",
        "dataset": "NASA C-MAPSS FD001",
        "baseline_model": BASELINE_MODEL_NAME,
        "algorithm": "HistGradientBoostingRegressor",
        "rul_cap": RUL_CAP,
        "original_feature_count": len(FEATURE_COLUMNS),
        "temporal_feature_count": len(TEMPORAL_FEATURE_COLUMNS),
        "total_feature_count": len(EXPERIMENTAL_FEATURE_COLUMNS),
        "feature_names": EXPERIMENTAL_FEATURE_COLUMNS,
        "temporal_sensors": TEMPORAL_SENSORS,
        "rolling_windows": WINDOWS,
        "imputation_medians": {
            key: float(value)
            for key, value in medians.items()
        },
        "model_parameters": model.get_params(),
        "validation_results": validation_results,
        "baseline_test_metrics": baseline_metrics,
        "temporal_test_metrics": temporal_metrics,
        "test_engine_count": int(len(predictions)),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifact_file": "model.joblib",
        "registered": False
    }

    with open(
        metadata_tmp,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False
        )

    metadata_tmp.replace(
        metadata_path
    )

    report = {
        "experiment": "FD001 Temporal Feature RUL Comparison",
        "baseline_model": BASELINE_MODEL_NAME,
        "validation_results": validation_results,
        "baseline_test_metrics": baseline_metrics,
        "temporal_test_metrics": temporal_metrics,
        "temporal_feature_count": len(TEMPORAL_FEATURE_COLUMNS),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "warning": (
            "Test results are comparative benchmark results, "
            "not independent real-world validation. "
            "The test dataset has already been inspected "
            "during prior error analysis."
        )
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False
        )

    print("\nExperimental model saved:", model_path)
    print("Experimental metadata saved:", metadata_path)
    print("Experimental report saved:", report_path)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 70)
    print("NASA FD001 TEMPORAL RUL EXPERIMENT")
    print("=" * 70)

    create_directories()

    train_df, test_df, actual_rul = load_all_data()

    train_df = add_rul_labels(
        train_df
    )

    print("\nGenerating training temporal features...")

    train_df = add_temporal_features(
        train_df
    )

    print("\nGenerating test temporal features...")

    test_df = add_temporal_features(
        test_df
    )

    training_part, validation_part = split_by_engine(
        train_df
    )

    validation_results = compare_validation_models(
        training_part,
        validation_part
    )

    temporal_model, temporal_medians = (
        train_final_temporal_model(
            train_df
        )
    )

    (
        baseline_test_metrics,
        temporal_test_metrics,
        predictions
    ) = evaluate_test_engines(
        None,
        temporal_model,
        temporal_medians,
        test_df,
        actual_rul
    )

    save_experimental_artifacts(
        temporal_model,
        temporal_medians,
        validation_results,
        baseline_test_metrics,
        temporal_test_metrics,
        predictions
    )

    print("\n" + "=" * 70)
    print("TEMPORAL RUL EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nValidation MAE:")
    print(
        "Baseline:",
        round(validation_results["baseline"]["mae"], 4)
    )
    print(
        "Temporal:",
        round(validation_results["temporal"]["mae"], 4)
    )

    print("\nTest MAE:")
    print(
        "Registered baseline:",
        round(baseline_test_metrics["mae"], 4)
    )
    print(
        "Temporal experiment:",
        round(temporal_test_metrics["mae"], 4)
    )

    print("\nDangerous under-prioritization:")
    print(
        "Baseline:",
        baseline_test_metrics[
            "maintenance_audit"
        ]["dangerous_underprioritization"]
    )
    print(
        "Temporal:",
        temporal_test_metrics[
            "maintenance_audit"
        ]["dangerous_underprioritization"]
    )

    print("\nExisting registered model was not modified.")

    return {
        "validation_results": validation_results,
        "baseline_test_metrics": baseline_test_metrics,
        "temporal_test_metrics": temporal_test_metrics
    }


if __name__ == "__main__":
    run_experiment()
