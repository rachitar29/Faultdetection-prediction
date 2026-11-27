
"""
NASA C-MAPSS FD001
Multi-Split Engine-Grouped RUL Safety Validation

Purpose:
1. Compare baseline and causal temporal RUL models.
2. Repeat validation across multiple engine-grouped splits.
3. Separate calibration engines from evaluation engines.
4. Select safety offsets using calibration engines only.
5. Evaluate selected offsets on different evaluation engines.
6. Measure maintenance-category errors and dangerous under-prioritisation.
7. Save experimental reports only.

IMPORTANT:
- Uses train_FD001.txt only.
- Does not load the NASA test dataset or RUL_FD001.txt.
- Does not register, overwrite, or save any ML model.
- Does not modify existing model artifacts.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split
)


warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"
TRAIN_FILE = DATA_DIR / "train_FD001.txt"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "reports"

RUL_CAP = 125
RANDOM_STATE = 42

N_SPLITS = 5
VALIDATION_SIZE = 0.20
EVALUATION_SIZE_WITHIN_VALIDATION = 0.50

WINDOWS = [10, 30]

# Includes late-life observation points to increase
# the chance of evaluating actual CRITICAL snapshots.
CENSOR_FRACTIONS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.92,
    0.94,
    0.96,
    0.98
]

SAFETY_OFFSETS = [0, 5, 10]

UNDERPRIORITIZATION_COST = 10
OVERPRIORITIZATION_COST = 1

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

PRIORITY = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4
}

METRIC_NAMES = [
    "mae",
    "rmse",
    "r2",
    "category_mismatches",
    "dangerous_underprioritization",
    "overprioritization",
    "actual_high_critical_snapshots",
    "high_critical_recall",
    "actual_critical_snapshots",
    "critical_recall",
    "false_high_critical_predictions",
    "high_critical_precision",
    "critical_precision",
    "weighted_safety_cost"
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"NASA FD001 training file not found: {TRAIN_FILE}"
        )

    dataframe = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None
    )

    dataframe = dataframe.dropna(
        axis=1,
        how="all"
    )

    if dataframe.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, "
            f"found {dataframe.shape[1]}."
        )

    dataframe.columns = COLUMN_NAMES

    if dataframe["unit"].isna().any():
        raise ValueError("Missing engine IDs found.")

    if dataframe["cycle"].isna().any():
        raise ValueError("Missing cycle values found.")

    if dataframe.duplicated(["unit", "cycle"]).any():
        raise ValueError(
            "Duplicate engine/cycle observations found."
        )

    dataframe = dataframe.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    print("\nNASA FD001 training data loaded.")
    print("Training rows:", len(dataframe))
    print("Training engines:", dataframe["unit"].nunique())

    return dataframe


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

    if len(values) < 2:
        return 0.0

    x = np.arange(
        len(values),
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

    slope = (
        np.dot(x_centered, y_centered)
        / denominator
    )

    return float(slope * 10.0)


def add_temporal_features(dataframe):

    print("\nGenerating causal temporal features...")

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
        ignore_index=True
    )

    result = result.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    missing = (
        set(EXPERIMENTAL_FEATURE_COLUMNS)
        - set(result.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing temporal features: {sorted(missing)}"
        )

    print("Original features:", len(FEATURE_COLUMNS))
    print("Temporal features:", len(TEMPORAL_FEATURE_COLUMNS))
    print("Total temporal model features:",
          len(EXPERIMENTAL_FEATURE_COLUMNS))

    return result


# ============================================================
# REPEATED ENGINE-GROUPED SPLITS
# ============================================================

def generate_engine_splits(dataframe):

    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    splits = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(
            dataframe,
            groups=dataframe["unit"]
        ),
        start=1
    ):

        training_part = dataframe.iloc[
            train_indices
        ].copy()

        validation_part = dataframe.iloc[
            validation_indices
        ].copy()

        training_engines = set(
            training_part["unit"].unique()
        )

        validation_engines = set(
            validation_part["unit"].unique()
        )

        if training_engines.intersection(validation_engines):
            raise RuntimeError(
                f"Engine leakage detected in fold {fold_number}."
            )

        if len(training_engines) < 2:
            raise RuntimeError(
                f"Insufficient training engines in fold {fold_number}."
            )

        if len(validation_engines) < 4:
            raise RuntimeError(
                f"Insufficient validation engines in fold {fold_number}."
            )

        # Split validation engines into separate calibration
        # and evaluation groups.
        validation_engine_ids = np.array(
            sorted(validation_engines)
        )

        calibration_ids, evaluation_ids = train_test_split(
            validation_engine_ids,
            test_size=EVALUATION_SIZE_WITHIN_VALIDATION,
            random_state=RANDOM_STATE + fold_number
        )

        calibration_part = validation_part[
            validation_part["unit"].isin(calibration_ids)
        ].copy()

        evaluation_part = validation_part[
            validation_part["unit"].isin(evaluation_ids)
        ].copy()

        calibration_engine_set = set(
            calibration_part["unit"].unique()
        )

        evaluation_engine_set = set(
            evaluation_part["unit"].unique()
        )

        if calibration_engine_set.intersection(evaluation_engine_set):
            raise RuntimeError(
                f"Calibration/evaluation engine leakage "
                f"in fold {fold_number}."
            )

        splits.append({
            "fold": fold_number,
            "training": training_part,
            "calibration": calibration_part,
            "evaluation": evaluation_part
        })

        print(f"\nFold {fold_number}")
        print("Training engines:", len(training_engines))
        print("Calibration engines:", len(calibration_engine_set))
        print("Evaluation engines:", len(evaluation_engine_set))

    return splits


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
            "Non-finite feature values remain."
        )

    return X, medians


# ============================================================
# MODEL
# ============================================================

def build_model(random_state):

    return HistGradientBoostingRegressor(
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=20,
        random_state=random_state
    )


# ============================================================
# PSEUDO-CENSORED SNAPSHOTS
# ============================================================

def create_validation_snapshots(dataframe):

    snapshots = []

    for engine_id, engine in dataframe.groupby(
        "unit",
        sort=True
    ):

        engine = engine.sort_values(
            "cycle"
        ).copy()

        maximum_cycle = int(
            engine["cycle"].max()
        )

        seen_cycles = set()

        for fraction in CENSOR_FRACTIONS:

            cutoff_cycle = int(
                round(maximum_cycle * fraction)
            )

            available_history = engine[
                engine["cycle"] <= cutoff_cycle
            ]

            if available_history.empty:
                continue

            snapshot = available_history.tail(
                1
            ).copy()

            observed_cycle = int(
                snapshot["cycle"].iloc[0]
            )

            # Different requested fractions can resolve to
            # the same observed cycle. Keep each cycle once.
            if observed_cycle in seen_cycles:
                continue

            seen_cycles.add(observed_cycle)

            actual_rul = min(
                maximum_cycle - observed_cycle,
                RUL_CAP
            )

            snapshot["validation_engine"] = int(
                engine_id
            )

            snapshot["requested_censor_fraction"] = float(
                fraction
            )

            snapshot["actual_censor_fraction"] = float(
                observed_cycle / maximum_cycle
            )

            snapshot["actual_rul_at_snapshot"] = float(
                actual_rul
            )

            snapshots.append(snapshot)

    if not snapshots:
        raise RuntimeError(
            "No validation snapshots were generated."
        )

    result = pd.concat(
        snapshots,
        ignore_index=True
    )

    if result.duplicated(
        ["validation_engine", "cycle"]
    ).any():

        raise RuntimeError(
            "Duplicate engine/cycle snapshots detected."
        )

    return result


# ============================================================
# RUL CATEGORIES
# ============================================================

def classify_rul(value):

    if value <= 10:
        return "CRITICAL"

    if value <= 25:
        return "HIGH"

    if value <= 50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# SAFETY METRICS
# ============================================================

def calculate_safety_metrics(
    actual,
    predicted
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    if len(actual) == 0:
        raise ValueError(
            "Cannot calculate metrics on an empty dataset."
        )

    if not np.isfinite(actual).all():
        raise ValueError("Actual RUL contains non-finite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain non-finite values.")

    predicted = np.clip(
        predicted,
        0,
        None
    )

    actual_categories = [
        classify_rul(value)
        for value in actual
    ]

    predicted_categories = [
        classify_rul(value)
        for value in predicted
    ]

    actual_ranks = np.array([
        PRIORITY[value]
        for value in actual_categories
    ])

    predicted_ranks = np.array([
        PRIORITY[value]
        for value in predicted_categories
    ])

    dangerous = predicted_ranks < actual_ranks
    overprioritized = predicted_ranks > actual_ranks

    high_critical_actual = actual <= 25
    critical_actual = actual <= 10

    high_critical_detected = predicted_ranks >= 3
    critical_detected = predicted_ranks >= 4

    high_critical_count = int(
        np.sum(high_critical_actual)
    )

    critical_count = int(
        np.sum(critical_actual)
    )

    predicted_high_critical_count = int(
        np.sum(high_critical_detected)
    )

    predicted_critical_count = int(
        np.sum(critical_detected)
    )

    high_critical_true_positive = int(
        np.sum(
            high_critical_detected
            & high_critical_actual
        )
    )

    critical_true_positive = int(
        np.sum(
            critical_detected
            & critical_actual
        )
    )

    dangerous_count = int(
        np.sum(dangerous)
    )

    overprioritized_count = int(
        np.sum(overprioritized)
    )

    weighted_cost = (
        UNDERPRIORITIZATION_COST * dangerous_count
        + OVERPRIORITIZATION_COST * overprioritized_count
    )

    if len(actual) >= 2 and np.unique(actual).size >= 2:
        r2 = float(
            r2_score(actual, predicted)
        )
    else:
        r2 = None

    return {
        "snapshot_count": int(len(actual)),

        "mae": float(
            mean_absolute_error(actual, predicted)
        ),

        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),

        "r2": r2,

        "category_mismatches": int(
            np.sum(actual_ranks != predicted_ranks)
        ),

        "dangerous_underprioritization": dangerous_count,

        "overprioritization": overprioritized_count,

        "actual_high_critical_snapshots": high_critical_count,

        "high_critical_recall": (
            float(
                high_critical_true_positive / high_critical_count
            )
            if high_critical_count else None
        ),

        "actual_critical_snapshots": critical_count,

        "critical_recall": (
            float(
                critical_true_positive / critical_count
            )
            if critical_count else None
        ),

        "false_high_critical_predictions": int(
            predicted_high_critical_count
            - high_critical_true_positive
        ),

        "high_critical_precision": (
            float(
                high_critical_true_positive
                / predicted_high_critical_count
            )
            if predicted_high_critical_count else None
        ),

        "critical_precision": (
            float(
                critical_true_positive
                / predicted_critical_count
            )
            if predicted_critical_count else None
        ),

        "weighted_safety_cost": int(weighted_cost)
    }


# ============================================================
# OFFSET EVALUATION
# ============================================================

def evaluate_offsets(
    actual,
    predictions,
    fold_number,
    partition_name
):

    rows = []

    for offset in SAFETY_OFFSETS:

        adjusted = np.maximum(
            predictions - offset,
            0
        )

        metrics = calculate_safety_metrics(
            actual,
            adjusted
        )

        rows.append({
            "fold": fold_number,
            "partition": partition_name,
            "safety_offset_cycles": offset,
            **metrics
        })

    return pd.DataFrame(rows)


def select_offset(calibration_results):

    # Selection uses calibration engines only.
    # Ties are resolved in favour of the smaller offset.

    ordered = calibration_results.sort_values(
        ["weighted_safety_cost", "safety_offset_cycles"],
        ascending=[True, True]
    )

    return int(
        ordered.iloc[0]["safety_offset_cycles"]
    )


# ============================================================
# MODEL TRAINING AND PREDICTION
# ============================================================

def train_and_predict(
    training_part,
    target_part,
    feature_columns,
    random_state
):

    X_train, medians = prepare_features(
        training_part,
        feature_columns
    )

    X_target, _ = prepare_features(
        target_part,
        feature_columns,
        medians
    )

    model = build_model(random_state)

    model.fit(
        X_train,
        training_part["RUL"].to_numpy(dtype=float)
    )

    predictions = model.predict(
        X_target
    )

    predictions = np.clip(
        predictions,
        0,
        None
    )

    return predictions


# ============================================================
# RUN ONE FOLD
# ============================================================

def run_fold(split):

    fold_number = split["fold"]

    training_part = split["training"]
    calibration_part = split["calibration"]
    evaluation_part = split["evaluation"]

    print("\n" + "=" * 65)
    print(f"RUNNING VALIDATION FOLD {fold_number}")
    print("=" * 65)

    calibration_snapshots = create_validation_snapshots(
        calibration_part
    )

    evaluation_snapshots = create_validation_snapshots(
        evaluation_part
    )

    if calibration_snapshots.empty or evaluation_snapshots.empty:
        raise RuntimeError(
            f"Empty calibration/evaluation snapshots in fold {fold_number}."
        )

    y_train = training_part["RUL"].to_numpy(
        dtype=float
    )

    if not np.isfinite(y_train).all():
        raise ValueError(
            f"Invalid training RUL values in fold {fold_number}."
        )

    # --------------------------------------------------------
    # Baseline model
    # --------------------------------------------------------

    print("\nTraining baseline model...")

    baseline_calibration_predictions = train_and_predict(
        training_part,
        calibration_snapshots,
        FEATURE_COLUMNS,
        RANDOM_STATE + fold_number
    )

    baseline_evaluation_predictions = train_and_predict(
        training_part,
        evaluation_snapshots,
        FEATURE_COLUMNS,
        RANDOM_STATE + fold_number
    )

    # --------------------------------------------------------
    # Temporal model
    # --------------------------------------------------------

    print("\nTraining temporal model...")

    temporal_calibration_predictions = train_and_predict(
        training_part,
        calibration_snapshots,
        EXPERIMENTAL_FEATURE_COLUMNS,
        RANDOM_STATE + fold_number
    )

    temporal_evaluation_predictions = train_and_predict(
        training_part,
        evaluation_snapshots,
        EXPERIMENTAL_FEATURE_COLUMNS,
        RANDOM_STATE + fold_number
    )

    calibration_actual = calibration_snapshots[
        "actual_rul_at_snapshot"
    ].to_numpy(dtype=float)

    evaluation_actual = evaluation_snapshots[
        "actual_rul_at_snapshot"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Select offset using calibration engines only
    # --------------------------------------------------------

    calibration_offset_results = evaluate_offsets(
        calibration_actual,
        temporal_calibration_predictions,
        fold_number,
        "calibration"
    )

    selected_offset = select_offset(
        calibration_offset_results
    )

    # --------------------------------------------------------
    # Evaluate every candidate on separate evaluation engines
    # --------------------------------------------------------

    evaluation_offset_results = evaluate_offsets(
        evaluation_actual,
        temporal_evaluation_predictions,
        fold_number,
        "evaluation"
    )

    selected_evaluation_row = evaluation_offset_results[
        evaluation_offset_results["safety_offset_cycles"]
        == selected_offset
    ].iloc[0]

    # --------------------------------------------------------
    # Baseline and raw temporal evaluation metrics
    # --------------------------------------------------------

    baseline_metrics = calculate_safety_metrics(
        evaluation_actual,
        baseline_evaluation_predictions
    )

    temporal_raw_metrics = calculate_safety_metrics(
        evaluation_actual,
        temporal_evaluation_predictions
    )

    selected_temporal_metrics = calculate_safety_metrics(
        evaluation_actual,
        np.maximum(
            temporal_evaluation_predictions - selected_offset,
            0
        )
    )

    # --------------------------------------------------------
    # Per-fold prediction report
    # --------------------------------------------------------

    prediction_report = pd.DataFrame({
        "fold": fold_number,
        "partition": "calibration",
        "engine_id": calibration_snapshots["unit"].to_numpy(),
        "last_observed_cycle": calibration_snapshots["cycle"].to_numpy(),
        "requested_censor_fraction": (
            calibration_snapshots["requested_censor_fraction"].to_numpy()
        ),
        "actual_censor_fraction": (
            calibration_snapshots["actual_censor_fraction"].to_numpy()
        ),
        "actual_rul": calibration_actual,
        "baseline_predicted_rul": np.round(
            baseline_calibration_predictions,
            2
        ),
        "temporal_raw_predicted_rul": np.round(
            temporal_calibration_predictions,
            2
        )
    })

    evaluation_report = pd.DataFrame({
        "fold": fold_number,
        "partition": "evaluation",
        "engine_id": evaluation_snapshots["unit"].to_numpy(),
        "last_observed_cycle": evaluation_snapshots["cycle"].to_numpy(),
        "requested_censor_fraction": (
            evaluation_snapshots["requested_censor_fraction"].to_numpy()
        ),
        "actual_censor_fraction": (
            evaluation_snapshots["actual_censor_fraction"].to_numpy()
        ),
        "actual_rul": evaluation_actual,
        "baseline_predicted_rul": np.round(
            baseline_evaluation_predictions,
            2
        ),
        "temporal_raw_predicted_rul": np.round(
            temporal_evaluation_predictions,
            2
        ),
        "temporal_selected_offset": selected_offset,
        "temporal_adjusted_predicted_rul": np.round(
            np.maximum(
                temporal_evaluation_predictions - selected_offset,
                0
            ),
            2
        )
    })

    prediction_report = pd.concat(
        [prediction_report, evaluation_report],
        ignore_index=True
    )

    for column, prediction_column in [
        ("baseline_category", "baseline_predicted_rul"),
        ("temporal_raw_category", "temporal_raw_predicted_rul")
    ]:

        prediction_report[column] = prediction_report[
            prediction_column
        ].apply(classify_rul)

    prediction_report["temporal_adjusted_category"] = (
        prediction_report["temporal_adjusted_predicted_rul"]
        .fillna(prediction_report["temporal_raw_predicted_rul"])
        .apply(classify_rul)
    )

    prediction_report["actual_category"] = (
        prediction_report["actual_rul"].apply(classify_rul)
    )

    # --------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------

    fold_summary = {
        "fold": fold_number,

        "training_engine_count": int(
            training_part["unit"].nunique()
        ),

        "calibration_engine_count": int(
            calibration_part["unit"].nunique()
        ),

        "evaluation_engine_count": int(
            evaluation_part["unit"].nunique()
        ),

        "calibration_snapshot_count": int(
            len(calibration_snapshots)
        ),

        "evaluation_snapshot_count": int(
            len(evaluation_snapshots)
        ),

        "selected_offset_cycles": selected_offset,

        "baseline_evaluation_metrics": baseline_metrics,

        "temporal_raw_evaluation_metrics": temporal_raw_metrics,

        "temporal_selected_offset_evaluation_metrics": (
            selected_temporal_metrics
        )
    }

    print("\nSelected offset from calibration engines:",
          selected_offset)

    print("\nBaseline evaluation metrics:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nTemporal raw evaluation metrics:")
    print(json.dumps(temporal_raw_metrics, indent=4))

    print("\nTemporal selected-offset evaluation metrics:")
    print(json.dumps(selected_temporal_metrics, indent=4))

    all_offset_results = pd.concat(
        [
            calibration_offset_results,
            evaluation_offset_results
        ],
        ignore_index=True
    )

    return {
        "fold_summary": fold_summary,
        "prediction_report": prediction_report,
        "offset_results": all_offset_results,
        "selected_evaluation_row": selected_evaluation_row.to_dict()
    }


# ============================================================
# AGGREGATE FOLD RESULTS
# ============================================================

def aggregate_fold_metrics(fold_results):

    model_sections = [
        (
            "baseline",
            "baseline_evaluation_metrics"
        ),
        (
            "temporal_raw",
            "temporal_raw_evaluation_metrics"
        ),
        (
            "temporal_selected_offset",
            "temporal_selected_offset_evaluation_metrics"
        )
    ]

    aggregate = {}

    for display_name, section_name in model_sections:

        aggregate[display_name] = {}

        for metric_name in METRIC_NAMES:

            values = []

            for result in fold_results:

                value = result["fold_summary"][
                    section_name
                ].get(metric_name)

                if value is not None and np.isfinite(value):
                    values.append(float(value))

            if values:

                aggregate[display_name][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(
                        np.std(values, ddof=1)
                        if len(values) > 1 else 0.0
                    ),
                    "fold_count": len(values)
                }

            else:

                aggregate[display_name][metric_name] = {
                    "mean": None,
                    "std": None,
                    "fold_count": 0
                }

    offsets = [
        result["fold_summary"]["selected_offset_cycles"]
        for result in fold_results
    ]

    aggregate["selected_offset_distribution"] = {
        str(offset): int(offsets.count(offset))
        for offset in sorted(set(offsets))
    }

    return aggregate


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    fold_results,
    aggregate_metrics
):

    predictions_path = (
        REPORT_DIR
        / "cmapss_fd001_multisplit_validation_predictions.csv"
    )

    offsets_path = (
        REPORT_DIR
        / "cmapss_fd001_multisplit_validation_offsets.csv"
    )

    folds_path = (
        REPORT_DIR
        / "cmapss_fd001_multisplit_validation_folds.csv"
    )

    report_path = (
        REPORT_DIR
        / "cmapss_fd001_multisplit_validation_report.json"
    )

    predictions_df = pd.concat(
        [
            result["prediction_report"]
            for result in fold_results
        ],
        ignore_index=True
    )

    offsets_df = pd.concat(
        [
            result["offset_results"]
            for result in fold_results
        ],
        ignore_index=True
    )

    folds_df = pd.DataFrame([
        result["fold_summary"]
        for result in fold_results
    ])

    # Save the detailed reports.
    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    offsets_df.to_csv(
        offsets_path,
        index=False
    )

    folds_df.to_csv(
        folds_path,
        index=False
    )

    report = {
        "experiment": (
            "NASA FD001 Multi-Split Engine-Grouped "
            "RUL Safety Validation"
        ),

        "dataset": "NASA C-MAPSS FD001",

        "data_source": "train_FD001.txt only",

        "test_dataset_used": False,

        "model_registry_modified": False,

        "experimental_model_saved": False,

        "validation_configuration": {
            "number_of_splits": N_SPLITS,
            "validation_size": VALIDATION_SIZE,
            "calibration_fraction_of_validation": (
                1.0 - EVALUATION_SIZE_WITHIN_VALIDATION
            ),
            "evaluation_fraction_of_validation": (
                EVALUATION_SIZE_WITHIN_VALIDATION
            ),
            "random_state": RANDOM_STATE
        },

        "feature_configuration": {
            "original_feature_count": len(FEATURE_COLUMNS),
            "temporal_feature_count": len(TEMPORAL_FEATURE_COLUMNS),
            "total_temporal_feature_count": (
                len(EXPERIMENTAL_FEATURE_COLUMNS)
            ),
            "temporal_sensors": TEMPORAL_SENSORS,
            "rolling_windows": WINDOWS
        },

        "snapshot_configuration": {
            "requested_censor_fractions": CENSOR_FRACTIONS,
            "duplicate_observed_cycles_removed": True,
            "actual_rul_definition": (
                "Maximum engine cycle minus last observed cycle, "
                "capped at the configured RUL cap."
            )
        },

        "rul_cap": RUL_CAP,

        "safety_configuration": {
            "offset_candidates": SAFETY_OFFSETS,
            "underprioritization_cost": UNDERPRIORITIZATION_COST,
            "overprioritization_cost": OVERPRIORITIZATION_COST,
            "weighted_cost_formula": (
                "10 * dangerous_underprioritization "
                "+ 1 * overprioritization"
            )
        },

        "fold_results": [
            result["fold_summary"]
            for result in fold_results
        ],

        "aggregate_fold_metrics": aggregate_metrics,

        "report_files": {
            "predictions_csv": predictions_path.name,
            "offsets_csv": offsets_path.name,
            "folds_csv": folds_path.name
        },

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "limitations": [
            (
                "Calibration and evaluation engines are separated "
                "within each fold."
            ),
            (
                "The same engine may appear in different roles "
                "across different repeated folds."
            ),
            (
                "Snapshots from the same engine are correlated "
                "and must not be treated as independent engines."
            ),
            (
                "Repeated validation folds do not replace an "
                "independent external test dataset."
            ),
            (
                "The weighted safety cost reflects an experimental "
                "cost assumption, not a certified industrial safety standard."
            ),
            (
                "Pseudo-censored RUL labels use the eventual maximum "
                "cycle retrospectively."
            ),
            (
                "An offset selected in calibration data is not "
                "automatically suitable for production."
            )
        ]
    }

    temporary_path = report_path.with_suffix(
        ".tmp.json"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False
        )

    temporary_path.replace(
        report_path
    )

    print("\n" + "=" * 65)
    print("MULTI-SPLIT REPORTS SAVED")
    print("=" * 65)

    print("Predictions:", predictions_path)
    print("Offset results:", offsets_path)
    print("Fold results:", folds_path)
    print("JSON report:", report_path)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    print("=" * 70)
    print("NASA FD001 MULTI-SPLIT RUL SAFETY VALIDATION")
    print("=" * 70)

    create_directories()

    dataframe = load_training_data()

    dataframe = add_rul_labels(
        dataframe
    )

    dataframe = add_temporal_features(
        dataframe
    )

    splits = generate_engine_splits(
        dataframe
    )

    fold_results = []

    for split in splits:

        result = run_fold(
            split
        )

        fold_results.append(
            result
        )

    aggregate_metrics = aggregate_fold_metrics(
        fold_results
    )

    print("\n" + "=" * 70)
    print("AGGREGATE FOLD METRICS")
    print("=" * 70)

    print(
        json.dumps(
            aggregate_metrics,
            indent=4
        )
    )

    save_reports(
        fold_results,
        aggregate_metrics
    )

    print("\n" + "=" * 70)
    print("MULTI-SPLIT VALIDATION COMPLETED")
    print("=" * 70)

    print(
        "\nSelected offset distribution:",
        aggregate_metrics["selected_offset_distribution"]
    )

    print(
        "\nNo registered model was modified."
    )

    print(
        "No experimental model artifact was overwritten."
    )

    print(
        "The NASA test dataset was not loaded."
    )

    return {
        "fold_results": fold_results,
        "aggregate_metrics": aggregate_metrics
    }


if __name__ == "__main__":
    run_experiment()
