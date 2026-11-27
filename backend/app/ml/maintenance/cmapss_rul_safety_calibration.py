
import json
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


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"
TRAIN_FILE = DATA_DIR / "train_FD001.txt"

REPORT_DIR = BASE_DIR / "ml_artifacts" / "reports"

RUL_CAP = 125
RANDOM_STATE = 42

WINDOWS = [10, 30]

CENSOR_FRACTIONS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

SAFETY_OFFSETS = [
    0,
    5,
    10,
    15,
    20,
    25,
    30
]

# Explicit experiment objective:
# One dangerous under-prioritisation costs 10 points.
# One unnecessary over-prioritisation costs 1 point.
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

    print("Training rows:", len(dataframe))
    print("Training engines:", dataframe["unit"].nunique())

    return dataframe


# ============================================================
# RUL LABELS
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

    missing = set(
        EXPERIMENTAL_FEATURE_COLUMNS
    ) - set(result.columns)

    if missing:
        raise RuntimeError(
            f"Missing temporal features: {sorted(missing)}"
        )

    print("Original features:", len(FEATURE_COLUMNS))
    print("Temporal features:", len(TEMPORAL_FEATURE_COLUMNS))
    print("Total features:", len(EXPERIMENTAL_FEATURE_COLUMNS))

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

    training_part = dataframe[
        dataframe["unit"].isin(training_ids)
    ].copy()

    validation_part = dataframe[
        dataframe["unit"].isin(validation_ids)
    ].copy()

    print("\nEngine-level split:")
    print("Training engines:", len(training_ids))
    print("Validation engines:", len(validation_ids))

    return training_part, validation_part


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

def build_model():

    return HistGradientBoostingRegressor(
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=20,
        random_state=RANDOM_STATE
    )


# ============================================================
# PSEUDO-CENSORED VALIDATION SNAPSHOTS
# ============================================================

def create_validation_snapshots(validation_df):

    print("\nGenerating pseudo-censored validation snapshots...")

    snapshots = []

    for engine_id, engine in validation_df.groupby(
        "unit",
        sort=True
    ):

        engine = engine.sort_values(
            "cycle"
        ).copy()

        maximum_cycle = int(
            engine["cycle"].max()
        )

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

            actual_rul = min(
                maximum_cycle - observed_cycle,
                RUL_CAP
            )

            snapshot["validation_engine"] = int(
                engine_id
            )

            snapshot["censor_fraction"] = float(
                fraction
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
        ["validation_engine", "censor_fraction"]
    ).any():
        raise RuntimeError(
            "Duplicate engine/censor snapshots detected."
        )

    print("Validation engines:", result["unit"].nunique())
    print("Censor fractions:", len(CENSOR_FRACTIONS))
    print("Total validation snapshots:", len(result))

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

    predicted = np.clip(
        np.asarray(predicted, dtype=float),
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

    high_critical_detected = (
        predicted_ranks >= 3
    )

    critical_detected = (
        predicted_ranks >= 4
    )

    high_critical_count = int(
        np.sum(high_critical_actual)
    )

    critical_count = int(
        np.sum(critical_actual)
    )

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

        "r2": float(
            r2_score(actual, predicted)
        ),

        "category_mismatches": int(
            np.sum(actual_ranks != predicted_ranks)
        ),

        "dangerous_underprioritization": int(
            np.sum(dangerous)
        ),

        "overprioritization": int(
            np.sum(overprioritized)
        ),

        "actual_high_critical_snapshots": high_critical_count,

        "high_critical_recall": (
            float(
                np.sum(
                    high_critical_detected
                    & high_critical_actual
                ) / high_critical_count
            )
            if high_critical_count else None
        ),

        "actual_critical_snapshots": critical_count,

        "critical_recall": (
            float(
                np.sum(
                    critical_detected
                    & critical_actual
                ) / critical_count
            )
            if critical_count else None
        ),

        "false_high_critical_predictions": int(
            np.sum(
                high_critical_detected
                & (~high_critical_actual)
            )
        )
    }


# ============================================================
# CONSERVATIVE OFFSET CALIBRATION
# ============================================================

def evaluate_offsets(
    actual,
    temporal_predictions
):

    results = []

    for offset in SAFETY_OFFSETS:

        adjusted_predictions = np.maximum(
            temporal_predictions - offset,
            0
        )

        metrics = calculate_safety_metrics(
            actual,
            adjusted_predictions
        )

        cost = (
            UNDERPRIORITIZATION_COST
            * metrics["dangerous_underprioritization"]
            +
            OVERPRIORITIZATION_COST
            * metrics["overprioritization"]
        )

        results.append({
            "safety_offset_cycles": offset,
            **metrics,
            "weighted_safety_cost": int(cost)
        })

    results_df = pd.DataFrame(
        results
    )

    # Lowest weighted cost wins.
    # If costs tie, prefer the smaller offset.
    selected = sorted(
        results,
        key=lambda item: (
            item["weighted_safety_cost"],
            item["safety_offset_cycles"]
        )
    )[0]

    return results_df, selected


# ============================================================
# VALIDATION EXPERIMENT
# ============================================================

def run_validation_experiment(
    training_part,
    validation_part
):

    print("\n========== VALIDATION SAFETY EXPERIMENT ==========")

    snapshots = create_validation_snapshots(
        validation_part
    )

    actual = snapshots[
        "actual_rul_at_snapshot"
    ].to_numpy(dtype=float)

    # ----------------------------------------
    # Baseline model
    # ----------------------------------------

    print("\nTraining baseline validation model...")

    X_baseline_train, baseline_medians = prepare_features(
        training_part,
        FEATURE_COLUMNS
    )

    X_baseline_validation, _ = prepare_features(
        snapshots,
        FEATURE_COLUMNS,
        baseline_medians
    )

    baseline_model = build_model()

    baseline_model.fit(
        X_baseline_train,
        training_part["RUL"].to_numpy(dtype=float)
    )

    baseline_predictions = np.clip(
        baseline_model.predict(
            X_baseline_validation
        ),
        0,
        None
    )

    baseline_metrics = calculate_safety_metrics(
        actual,
        baseline_predictions
    )

    # ----------------------------------------
    # Temporal model
    # ----------------------------------------

    print("\nTraining temporal validation model...")

    X_temporal_train, temporal_medians = prepare_features(
        training_part,
        EXPERIMENTAL_FEATURE_COLUMNS
    )

    X_temporal_validation, _ = prepare_features(
        snapshots,
        EXPERIMENTAL_FEATURE_COLUMNS,
        temporal_medians
    )

    temporal_model = build_model()

    temporal_model.fit(
        X_temporal_train,
        training_part["RUL"].to_numpy(dtype=float)
    )

    temporal_predictions = np.clip(
        temporal_model.predict(
            X_temporal_validation
        ),
        0,
        None
    )

    temporal_raw_metrics = calculate_safety_metrics(
        actual,
        temporal_predictions
    )

    # ----------------------------------------
    # Offset evaluation
    # ----------------------------------------

    print("\nEvaluating conservative RUL offsets...")

    offset_results, selected_offset = evaluate_offsets(
        actual,
        temporal_predictions
    )

    baseline_metrics["safety_offset_cycles"] = 0

    baseline_metrics["weighted_safety_cost"] = int(
        UNDERPRIORITIZATION_COST
        * baseline_metrics["dangerous_underprioritization"]
        +
        OVERPRIORITIZATION_COST
        * baseline_metrics["overprioritization"]
    )

    temporal_raw_metrics["safety_offset_cycles"] = 0

    selected_offset_value = int(
        selected_offset["safety_offset_cycles"]
    )

    adjusted_predictions = np.maximum(
        temporal_predictions - selected_offset_value,
        0
    )

    selected_metrics = calculate_safety_metrics(
        actual,
        adjusted_predictions
    )

    selected_metrics["safety_offset_cycles"] = (
        selected_offset_value
    )

    selected_metrics["weighted_safety_cost"] = int(
        selected_offset["weighted_safety_cost"]
    )

    # ----------------------------------------
    # Detailed snapshot report
    # ----------------------------------------

    prediction_report = pd.DataFrame({
        "engine_id": snapshots["unit"].to_numpy(),
        "last_observed_cycle": snapshots["cycle"].to_numpy(),
        "censor_fraction": snapshots["censor_fraction"].to_numpy(),
        "actual_rul": actual,
        "baseline_predicted_rul": np.round(
            baseline_predictions,
            2
        ),
        "temporal_raw_predicted_rul": np.round(
            temporal_predictions,
            2
        ),
        "temporal_adjusted_predicted_rul": np.round(
            adjusted_predictions,
            2
        ),
        "actual_category": [
            classify_rul(value)
            for value in actual
        ],
        "baseline_category": [
            classify_rul(value)
            for value in baseline_predictions
        ],
        "temporal_raw_category": [
            classify_rul(value)
            for value in temporal_predictions
        ],
        "temporal_adjusted_category": [
            classify_rul(value)
            for value in adjusted_predictions
        ]
    })

    print("\nBASELINE VALIDATION SAFETY:")
    print(json.dumps(baseline_metrics, indent=4))

    print("\nTEMPORAL VALIDATION SAFETY - NO OFFSET:")
    print(json.dumps(temporal_raw_metrics, indent=4))

    print("\nOFFSET COMPARISON:")
    print(
        offset_results.to_string(
            index=False
        )
    )

    print("\nSELECTED VALIDATION OFFSET:")
    print(json.dumps(selected_offset, indent=4))

    print("\nTEMPORAL VALIDATION SAFETY - SELECTED OFFSET:")
    print(json.dumps(selected_metrics, indent=4))

    return {
        "snapshot_predictions": prediction_report,
        "baseline_metrics": baseline_metrics,
        "temporal_raw_metrics": temporal_raw_metrics,
        "offset_results": offset_results,
        "selected_offset": selected_offset,
        "selected_metrics": selected_metrics
    }


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(results):

    predictions_path = (
        REPORT_DIR
        / "cmapss_fd001_validation_safety_snapshots.csv"
    )

    offsets_path = (
        REPORT_DIR
        / "cmapss_fd001_validation_safety_offsets.csv"
    )

    report_path = (
        REPORT_DIR
        / "cmapss_fd001_validation_safety_calibration.json"
    )

    results["snapshot_predictions"].to_csv(
        predictions_path,
        index=False
    )

    results["offset_results"].to_csv(
        offsets_path,
        index=False
    )

    report = {
        "experiment": "FD001 Validation-Only RUL Safety Calibration",

        "dataset": "NASA C-MAPSS FD001",

        "validation_method": (
            "Engine-level 80/20 split with five "
            "pseudo-censored observation points per validation engine."
        ),

        "censor_fractions": CENSOR_FRACTIONS,

        "validation_snapshot_count": int(
            len(results["snapshot_predictions"])
        ),

        "validation_engine_count": int(
            results["snapshot_predictions"]["engine_id"].nunique()
        ),

        "safety_offset_candidates": SAFETY_OFFSETS,

        "selection_objective": {
            "underprioritization_cost": UNDERPRIORITIZATION_COST,
            "overprioritization_cost": OVERPRIORITIZATION_COST,
            "formula": (
                "10 * dangerous_underprioritization "
                "+ 1 * overprioritization"
            )
        },

        "baseline_validation_metrics": (
            results["baseline_metrics"]
        ),

        "temporal_raw_validation_metrics": (
            results["temporal_raw_metrics"]
        ),

        "selected_validation_offset": (
            results["selected_offset"]
        ),

        "temporal_adjusted_validation_metrics": (
            results["selected_metrics"]
        ),

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "warning": (
            "The selected offset is a validation-derived experimental "
            "candidate, not a production setting. Validation snapshots "
            "from the same engine are correlated. The held-out test set "
            "was not loaded or used in this experiment. Independent "
            "evaluation is required before deployment."
        )
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

    print("\n========== REPORTS SAVED ==========")
    print("Snapshot predictions:", predictions_path)
    print("Offset comparison:", offsets_path)
    print("Validation report:", report_path)


# ============================================================
# MAIN
# ============================================================

def run_experiment():

    print("=" * 70)
    print("NASA FD001 VALIDATION-ONLY SAFETY CALIBRATION")
    print("=" * 70)

    create_directories()

    train_df = load_training_data()

    train_df = add_rul_labels(
        train_df
    )

    train_df = add_temporal_features(
        train_df
    )

    training_part, validation_part = split_by_engine(
        train_df
    )

    results = run_validation_experiment(
        training_part,
        validation_part
    )

    save_reports(
        results
    )

    print("\n" + "=" * 70)
    print("VALIDATION SAFETY CALIBRATION COMPLETED")
    print("=" * 70)

    print(
        "\nSelected experimental offset:",
        results["selected_offset"]["safety_offset_cycles"],
        "cycles"
    )

    print(
        "Validation dangerous under-prioritisation:",
        results["selected_metrics"]["dangerous_underprioritization"]
    )

    print(
        "Validation over-prioritisation:",
        results["selected_metrics"]["overprioritization"]
    )

    print(
        "\nNo registered model or existing experimental model was modified."
    )

    return results


if __name__ == "__main__":
    run_experiment()

