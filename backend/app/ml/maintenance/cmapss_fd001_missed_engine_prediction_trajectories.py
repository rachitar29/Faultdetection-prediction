"""
Read-only diagnostic for the three missed critical FD001 snapshots.

Reconstructs the exact five engine-grouped splits and temporal model
configuration from cmapss_rul_multisplit_validation.py. For the three
missed evaluation engines, it predicts RUL at every observed cycle so
we can inspect how predictions evolve before the missed snapshots.

This script:
- reads train_FD001.txt and the existing multi-split prediction CSV;
- trains models in memory only, using the original fold training engines;
- does not save, register, or modify any model;
- does not use NASA test data or RUL_FD001.txt;
- writes one new diagnostic CSV and does not overwrite existing reports.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
REPORT_DIR = BASE_DIR / "ml_artifacts" / "reports"

EXISTING_PREDICTIONS = (
    REPORT_DIR / "cmapss_fd001_multisplit_validation_predictions.csv"
)
OUTPUT_FILE = (
    REPORT_DIR / "cmapss_fd001_missed_engine_prediction_trajectories.csv"
)

RUL_CAP = 125
RANDOM_STATE = 42
N_SPLITS = 5
VALIDATION_SIZE = 0.20
EVALUATION_SIZE_WITHIN_VALIDATION = 0.50
WINDOWS = [10, 30]

TEMPORAL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_9", "sensor_12", "sensor_14", "sensor_17",
]

MISSED_CASES = [
    {"fold": 1, "engine_id": 77, "missed_snapshot_cycle": 145,
     "expected_actual_rul": 9, "expected_prediction": 15.47},
    {"fold": 4, "engine_id": 10, "missed_snapshot_cycle": 213,
     "expected_actual_rul": 9, "expected_prediction": 15.97},
    {"fold": 5, "engine_id": 59, "missed_snapshot_cycle": 222,
     "expected_actual_rul": 9, "expected_prediction": 16.12},
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
            f"{sensor}_deviation_{window}",
        ])

MODEL_FEATURES = FEATURE_COLUMNS + TEMPORAL_FEATURE_COLUMNS

DIAGNOSTIC_FEATURES = [
    "sensor_2", "sensor_2_rolling_mean_10",
    "sensor_2_slope_10", "sensor_2_deviation_10",
    "sensor_3", "sensor_3_rolling_mean_10",
    "sensor_3_rolling_std_10", "sensor_3_slope_10",
    "sensor_3_deviation_10", "sensor_3_slope_30",
    "sensor_3_deviation_30",
    "sensor_4", "sensor_4_rolling_mean_10",
    "sensor_4_slope_10", "sensor_4_deviation_10",
    "sensor_14", "sensor_14_rolling_mean_10",
    "sensor_14_rolling_std_10", "sensor_14_slope_10",
    "sensor_14_deviation_10", "sensor_14_slope_30",
    "sensor_14_deviation_30",
]


def calculate_rolling_slope(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denominator = np.dot(x_centered, x_centered)
    if denominator == 0:
        return 0.0
    return float(
        np.dot(x_centered, y_centered) / denominator * 10.0
    )


def load_and_prepare_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")
    if not EXISTING_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"Existing prediction report not found: {EXISTING_PREDICTIONS}"
        )

    data = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None)
    data = data.dropna(axis=1, how="all")
    if data.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, found {data.shape[1]}"
        )
    data.columns = COLUMN_NAMES
    data = data.sort_values(["unit", "cycle"]).reset_index(drop=True)

    maximum_cycles = data.groupby("unit")["cycle"].transform("max")
    data["RUL"] = (maximum_cycles - data["cycle"]).clip(upper=RUL_CAP)

    # Match the original script: trailing rolling windows are calculated
    # independently within each engine, using observations up to that row.
    engine_parts = []
    for _, engine in data.groupby("unit", sort=True):
        engine = engine.sort_values("cycle").copy()
        for sensor in TEMPORAL_SENSORS:
            values = pd.to_numeric(engine[sensor], errors="coerce")
            for window in WINDOWS:
                rolling = values.rolling(window=window, min_periods=1)
                rolling_mean = rolling.mean()
                rolling_std = rolling.std(ddof=0).fillna(0.0)
                rolling_slope = values.rolling(
                    window=window, min_periods=2
                ).apply(calculate_rolling_slope, raw=True).fillna(0.0)

                engine[f"{sensor}_rolling_mean_{window}"] = rolling_mean
                engine[f"{sensor}_rolling_std_{window}"] = rolling_std
                engine[f"{sensor}_slope_{window}"] = rolling_slope
                engine[f"{sensor}_deviation_{window}"] = (
                    values - rolling_mean
                )
        engine_parts.append(engine)

    prepared = pd.concat(engine_parts, ignore_index=True)
    prepared = prepared.sort_values(["unit", "cycle"]).reset_index(drop=True)

    existing = pd.read_csv(EXISTING_PREDICTIONS)
    required = {
        "fold", "partition", "engine_id", "last_observed_cycle",
        "actual_rul", "temporal_raw_predicted_rul",
    }
    missing = required - set(existing.columns)
    if missing:
        raise ValueError(
            f"Existing prediction report is missing columns: {sorted(missing)}"
        )

    return prepared, existing


def generate_exact_splits(data):
    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    splits = []
    for fold_number, (train_idx, validation_idx) in enumerate(
        splitter.split(data, groups=data["unit"]), start=1
    ):
        training = data.iloc[train_idx].copy()
        validation = data.iloc[validation_idx].copy()

        validation_ids = np.array(sorted(validation["unit"].unique()))
        calibration_ids, evaluation_ids = train_test_split(
            validation_ids,
            test_size=EVALUATION_SIZE_WITHIN_VALIDATION,
            random_state=RANDOM_STATE + fold_number,
        )
        evaluation = validation[
            validation["unit"].isin(evaluation_ids)
        ].copy()

        splits.append({
            "fold": fold_number,
            "training": training,
            "evaluation": evaluation,
        })
    return splits


def prepare_feature_matrices(training, target):
    x_train = training[MODEL_FEATURES].copy()
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    medians = x_train.median(numeric_only=True).fillna(0.0)
    x_train = x_train.fillna(medians).fillna(0.0)

    x_target = target[MODEL_FEATURES].copy()
    x_target = x_target.replace([np.inf, -np.inf], np.nan)
    x_target = x_target.fillna(medians).fillna(0.0)

    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain non-finite values.")
    if not np.isfinite(x_target.to_numpy(dtype=float)).all():
        raise ValueError("Target features contain non-finite values.")

    return x_train, x_target


def build_model(random_state):
    return HistGradientBoostingRegressor(
        l2_regularization=1.0,
        learning_rate=0.08,
        max_iter=150,
        max_leaf_nodes=20,
        random_state=random_state,
    )


def main():
    print("Loading FD001 data and existing validation predictions...")
    data, existing = load_and_prepare_data()
    splits = generate_exact_splits(data)

    output_parts = []

    for case in MISSED_CASES:
        fold_number = case["fold"]
        engine_id = case["engine_id"]
        split = next(s for s in splits if s["fold"] == fold_number)

        training = split["training"]
        evaluation = split["evaluation"]

        if engine_id not in set(evaluation["unit"].unique()):
            raise RuntimeError(
                f"Engine {engine_id} is not in fold {fold_number}'s "
                "evaluation partition. The reconstructed split does not "
                "match the saved experiment."
            )

        engine_rows = evaluation[
            evaluation["unit"] == engine_id
        ].sort_values("cycle").copy()

        if engine_rows.empty:
            raise RuntimeError(
                f"No evaluation rows found for fold {fold_number}, "
                f"engine {engine_id}."
            )

        x_train, x_target = prepare_feature_matrices(training, engine_rows)
        model = build_model(RANDOM_STATE + fold_number)
        model.fit(
            x_train,
            training["RUL"].to_numpy(dtype=float),
        )
        predictions = np.clip(model.predict(x_target), 0.0, None)

        original_match = existing[
            (existing["fold"] == fold_number)
            & (existing["partition"] == "evaluation")
            & (existing["engine_id"] == engine_id)
            & (
                existing["last_observed_cycle"]
                == case["missed_snapshot_cycle"]
            )
        ]

        if len(original_match) != 1:
            raise RuntimeError(
                f"Expected exactly one saved prediction for fold "
                f"{fold_number}, engine {engine_id}, cycle "
                f"{case['missed_snapshot_cycle']}; found {len(original_match)}."
            )

        saved_prediction = float(
            original_match["temporal_raw_predicted_rul"].iloc[0]
        )
        saved_actual = float(original_match["actual_rul"].iloc[0])

        trajectory = engine_rows[
            ["unit", "cycle", "RUL"] + DIAGNOSTIC_FEATURES
        ].copy()
        trajectory = trajectory.rename(columns={
            "unit": "engine_id",
            "cycle": "observed_cycle",
            "RUL": "actual_rul_retrospective",
        })
        trajectory.insert(0, "fold", fold_number)
        trajectory["predicted_rul"] = predictions
        trajectory["prediction_error"] = (
            trajectory["predicted_rul"]
            - trajectory["actual_rul_retrospective"]
        )
        trajectory["predicted_category"] = pd.cut(
            trajectory["predicted_rul"],
            bins=[-np.inf, 10, 25, 50, np.inf],
            labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            right=True,
        ).astype(str)
        trajectory["actual_category"] = pd.cut(
            trajectory["actual_rul_retrospective"],
            bins=[-np.inf, 10, 25, 50, np.inf],
            labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            right=True,
        ).astype(str)

        trajectory["is_original_missed_snapshot"] = (
            trajectory["observed_cycle"]
            == case["missed_snapshot_cycle"]
        )
        trajectory["original_saved_prediction"] = np.nan
        trajectory.loc[
            trajectory["is_original_missed_snapshot"],
            "original_saved_prediction",
        ] = saved_prediction

        trajectory["original_saved_actual_rul"] = np.nan
        trajectory.loc[
            trajectory["is_original_missed_snapshot"],
            "original_saved_actual_rul",
        ] = saved_actual

        # Use a boolean column from the start; assigning True/False into
        # a float column is rejected by newer pandas versions.
        trajectory["prediction_matches_saved_report"] = False
        reproduced = float(
            trajectory.loc[
                trajectory["is_original_missed_snapshot"], "predicted_rul"
            ].iloc[0]
        )
        trajectory.loc[
            trajectory["is_original_missed_snapshot"],
            "prediction_matches_saved_report",
        ] = abs(reproduced - saved_prediction) <= 0.02

        trajectory["missed_snapshot_cycle_reference"] = (
            case["missed_snapshot_cycle"]
        )
        output_parts.append(trajectory)

        print(
            f"\nFold {fold_number}, engine {engine_id}: "
            f"{len(trajectory)} observed cycles"
        )
        print(
            f"  Missed snapshot cycle: {case['missed_snapshot_cycle']}"
        )
        print(
            f"  Saved prediction: {saved_prediction:.2f}; "
            f"reproduced: {reproduced:.4f}; "
            f"saved actual RUL: {saved_actual:.1f}"
        )
        print(
            "  Reproduction check:",
            "PASS" if abs(reproduced - saved_prediction) <= 0.02
            else "WARNING - prediction differs from saved report",
        )

        at_miss = trajectory[
            trajectory["is_original_missed_snapshot"]
        ].iloc[0]
        earlier_critical = trajectory[
            (trajectory["observed_cycle"] < case["missed_snapshot_cycle"])
            & (trajectory["actual_rul_retrospective"] <= 10)
        ]
        predicted_critical_cycles = trajectory[
            trajectory["predicted_rul"] <= 10
        ]

        print(
            "  Prediction at missed snapshot:",
            f"{at_miss['predicted_rul']:.2f}",
        )
        print(
            "  Earlier observed cycles already actually critical:",
            len(earlier_critical),
        )
        print(
            "  All trajectory cycles predicted CRITICAL:",
            len(predicted_critical_cycles),
        )

    result = pd.concat(output_parts, ignore_index=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nDIAGNOSTIC COMPLETE")
    print("===================")
    print("Rows written:", len(result))
    print("Output:", OUTPUT_FILE)
    print("No model was saved, registered, or modified.")
    print("No existing report was overwritten.")
    print("NASA test data and RUL_FD001.txt were not used.")


if __name__ == "__main__":
    main()
