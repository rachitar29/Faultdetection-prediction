import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "ml_data" / "raw" / "CMAPSSData"

BASELINE_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "model_registry"
    / "rul_cmapss_fd001_best"
)

TEMPORAL_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "experimental"
    / "rul_cmapss_fd001_temporal"
)

REPORT_ROOT = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
    / "official_fd001_test_audit"
)

TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

BASELINE_MODEL_FILE = BASELINE_DIR / "model.joblib"
BASELINE_METADATA_FILE = BASELINE_DIR / "metadata.json"

TEMPORAL_MODEL_FILE = TEMPORAL_DIR / "model.joblib"
TEMPORAL_METADATA_FILE = TEMPORAL_DIR / "metadata.json"

FROZEN_OFFSET = 10.0
RUL_CAP = 125

TEMPORAL_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_12",
    "sensor_14",
    "sensor_17",
]

ROLLING_WINDOWS = [10, 30]

RAW_FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
] + [
    f"sensor_{i}"
    for i in range(1, 22)
]

COLUMN_NAMES = [
    "unit",
    "cycle",
] + RAW_FEATURE_COLUMNS

PRIORITY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}


# ============================================================
# GENERAL VALIDATION
# ============================================================

def require_file(path):

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )


def load_json(path):

    require_file(path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_finite_dataframe(dataframe, description):

    values = dataframe.to_numpy(dtype=float)

    if not np.isfinite(values).all():

        raise ValueError(
            f"{description} contains NaN or infinite values."
        )


# ============================================================
# NASA DATA LOADING
# ============================================================

def load_test_data():

    print("\nLoading official NASA FD001 test data...")

    require_file(TEST_FILE)
    require_file(RUL_FILE)

    test_df = pd.read_csv(
        TEST_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    actual_rul = np.loadtxt(
        RUL_FILE,
        dtype=float,
    )

    if actual_rul.ndim == 0:
        actual_rul = np.asarray([actual_rul])

    actual_rul = np.asarray(
        actual_rul,
        dtype=float,
    ).reshape(-1)

    if test_df.empty:
        raise ValueError("NASA test dataset is empty.")

    if not np.isfinite(actual_rul).all():
        raise ValueError("RUL reference file contains invalid values.")

    if not np.isfinite(
        test_df[["unit", "cycle"] + RAW_FEATURE_COLUMNS]
        .to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "NASA test dataset contains NaN or infinite values."
        )

    test_df["unit"] = test_df["unit"].astype(int)
    test_df["cycle"] = test_df["cycle"].astype(int)

    engine_ids = sorted(
        test_df["unit"].unique().tolist()
    )

    if len(engine_ids) != len(actual_rul):

        raise ValueError(
            "Test engine count does not match RUL reference count: "
            f"{len(engine_ids)} engines versus {len(actual_rul)} labels."
        )

    expected_ids = list(
        range(1, len(engine_ids) + 1)
    )

    if engine_ids != expected_ids:

        raise ValueError(
            "Unexpected engine ID sequence. "
            "Expected consecutive IDs starting at 1."
        )

    if len(actual_rul) != 100:

        raise ValueError(
            f"Expected 100 FD001 test engines; found {len(actual_rul)}."
        )

    print("Test rows:", len(test_df))
    print("Test engines:", len(engine_ids))
    print("Reference RUL labels:", len(actual_rul))

    return test_df, actual_rul


# ============================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================

def calculate_rolling_slope(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < 2:
        return 0.0

    x = np.arange(
        len(values),
        dtype=float,
    )

    slope = np.polyfit(
        x,
        values,
        1,
    )[0]

    return float(slope * 10.0)


def build_expected_temporal_feature_names():

    names = []

    for sensor in TEMPORAL_SENSORS:

        for window in ROLLING_WINDOWS:

            names.extend([
                f"{sensor}_rolling_mean_{window}",
                f"{sensor}_rolling_std_{window}",
                f"{sensor}_slope_{window}",
                f"{sensor}_deviation_{window}",
            ])

    return names


def add_temporal_features(dataframe):

    print("\nGenerating temporal features...")

    required = [
        "unit",
        "cycle",
    ] + TEMPORAL_SENSORS

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"Missing temporal input columns: {missing}"
        )

    engine_frames = []

    for _, engine_df in dataframe.groupby(
        "unit",
        sort=True,
    ):

        engine_df = (
            engine_df
            .sort_values("cycle")
            .copy()
        )

        for sensor in TEMPORAL_SENSORS:

            values = engine_df[sensor]

            for window in ROLLING_WINDOWS:

                rolling = values.rolling(
                    window=window,
                    min_periods=1,
                )

                rolling_mean = rolling.mean()

                rolling_std = (
                    rolling
                    .std(ddof=0)
                    .fillna(0)
                )

                rolling_slope = (
                    values
                    .rolling(
                        window=window,
                        min_periods=2,
                    )
                    .apply(
                        calculate_rolling_slope,
                        raw=True,
                    )
                    .fillna(0)
                )

                deviation = (
                    values - rolling_mean
                )

                engine_df[
                    f"{sensor}_rolling_mean_{window}"
                ] = rolling_mean

                engine_df[
                    f"{sensor}_rolling_std_{window}"
                ] = rolling_std

                engine_df[
                    f"{sensor}_slope_{window}"
                ] = rolling_slope

                engine_df[
                    f"{sensor}_deviation_{window}"
                ] = deviation

        engine_frames.append(engine_df)

    result = pd.concat(
        engine_frames,
        ignore_index=True,
    )

    result = result.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    expected_temporal = (
        build_expected_temporal_feature_names()
    )

    missing_features = [
        name
        for name in expected_temporal
        if name not in result.columns
    ]

    if missing_features:

        raise ValueError(
            f"Temporal feature generation failed: {missing_features}"
        )

    print("Temporal features generated:", len(expected_temporal))
    print(
        "Total model features:",
        len(RAW_FEATURE_COLUMNS) + len(expected_temporal),
    )

    return result


# ============================================================
# MODEL AND METADATA LOADING
# ============================================================

def load_artifacts():

    print("\nLoading saved model artifacts...")

    baseline_metadata = load_json(
        BASELINE_METADATA_FILE
    )

    temporal_metadata = load_json(
        TEMPORAL_METADATA_FILE
    )

    require_file(BASELINE_MODEL_FILE)
    require_file(TEMPORAL_MODEL_FILE)

    baseline_model = joblib.load(
        BASELINE_MODEL_FILE
    )

    temporal_model = joblib.load(
        TEMPORAL_MODEL_FILE
    )

    baseline_features = baseline_metadata.get(
        "feature_names"
    )

    temporal_features = temporal_metadata.get(
        "feature_names"
    )

    if baseline_features != RAW_FEATURE_COLUMNS:

        raise ValueError(
            "Baseline metadata feature order does not match "
            "the expected 24-feature FD001 schema."
        )

    expected_temporal_features = (
        RAW_FEATURE_COLUMNS
        + build_expected_temporal_feature_names()
    )

    if temporal_features != expected_temporal_features:

        raise ValueError(
            "Temporal metadata feature order does not match "
            "the expected 88-feature schema."
        )

    if len(baseline_features) != 24:
        raise ValueError("Baseline feature count is not 24.")

    if len(temporal_features) != 88:
        raise ValueError("Temporal feature count is not 88.")

    if temporal_metadata.get("registered") is not False:

        raise ValueError(
            "Expected the temporal artifact to remain unregistered."
        )

    if temporal_metadata.get("rul_cap") != RUL_CAP:

        raise ValueError(
            "Unexpected temporal model RUL cap."
        )

    temporal_medians = temporal_metadata.get(
        "imputation_medians"
    )

    if not isinstance(temporal_medians, dict):

        raise ValueError(
            "Temporal metadata does not contain imputation medians."
        )

    missing_medians = [
        name
        for name in temporal_features
        if name not in temporal_medians
    ]

    if missing_medians:

        raise ValueError(
            f"Missing temporal imputation medians: {missing_medians}"
        )

    median_values = np.asarray(
        [
            temporal_medians[name]
            for name in temporal_features
        ],
        dtype=float,
    )

    if not np.isfinite(median_values).all():

        raise ValueError(
            "Temporal imputation medians contain invalid values."
        )

    for model, expected_features, description in [
        (
            baseline_model,
            baseline_features,
            "Baseline",
        ),
        (
            temporal_model,
            temporal_features,
            "Temporal",
        ),
    ]:

        model_features = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if model_features is not None:

            if list(model_features) != expected_features:

                raise ValueError(
                    f"{description} model feature_names_in_ "
                    "does not match metadata."
                )

        model_count = getattr(
            model,
            "n_features_in_",
            None,
        )

        if model_count is not None:

            if int(model_count) != len(expected_features):

                raise ValueError(
                    f"{description} model feature count mismatch."
                )

    print("Baseline model:", baseline_metadata["model_name"])
    print("Baseline features:", len(baseline_features))
    print("Temporal model:", temporal_metadata["model_name"])
    print("Temporal features:", len(temporal_features))
    print("Temporal registered:", temporal_metadata["registered"])

    return (
        baseline_model,
        temporal_model,
        baseline_metadata,
        temporal_metadata,
        temporal_medians,
        baseline_features,
        temporal_features,
    )


# ============================================================
# FINAL ENGINE SNAPSHOTS
# ============================================================

def get_final_engine_rows(test_df):

    final_rows = (
        test_df
        .sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
        .copy()
    )

    if len(final_rows) != 100:

        raise ValueError(
            f"Expected 100 final engine rows; got {len(final_rows)}."
        )

    if final_rows["unit"].duplicated().any():

        raise ValueError(
            "Duplicate final engine snapshots detected."
        )

    return final_rows


# ============================================================
# PREDICTION
# ============================================================

def generate_predictions(
    baseline_model,
    temporal_model,
    temporal_medians,
    baseline_features,
    temporal_features,
    final_rows,
):

    print("\nPreparing baseline features...")

    X_baseline = (
        final_rows[baseline_features]
        .astype(float)
        .copy()
    )

    validate_finite_dataframe(
        X_baseline,
        "Baseline features",
    )

    print("Preparing temporal features...")

    X_temporal = (
        final_rows[temporal_features]
        .astype(float)
        .copy()
    )

    X_temporal = X_temporal.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_temporal = X_temporal.fillna(
        pd.Series(
            {
                name: float(temporal_medians[name])
                for name in temporal_features
            }
        )
    )

    X_temporal = X_temporal.fillna(0.0)

    validate_finite_dataframe(
        X_temporal,
        "Temporal features after training-median imputation",
    )

    print("\nPredicting with registered baseline model...")

    baseline_predictions = np.clip(
        baseline_model.predict(X_baseline),
        0,
        None,
    )

    print("Predicting with experimental temporal model...")

    temporal_raw_predictions = np.clip(
        temporal_model.predict(X_temporal),
        0,
        None,
    )

    temporal_adjusted_predictions = np.maximum(
        0,
        temporal_raw_predictions - FROZEN_OFFSET,
    )

    for name, predictions in [
        ("Baseline", baseline_predictions),
        ("Temporal raw", temporal_raw_predictions),
        ("Temporal adjusted", temporal_adjusted_predictions),
    ]:

        if not np.isfinite(predictions).all():

            raise ValueError(
                f"{name} predictions contain invalid values."
            )

    return (
        baseline_predictions,
        temporal_raw_predictions,
        temporal_adjusted_predictions,
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
        np.exp(errors / 10.0) - 1,
    )

    return float(np.sum(scores))


def calculate_metrics(actual, predicted):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.clip(
        np.asarray(predicted, dtype=float),
        0,
        None,
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
            predicted,
        ),
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


def calculate_maintenance_audit(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual_categories = [
        classify_rul(value)
        for value in actual
    ]

    predicted_categories = [
        classify_rul(value)
        for value in predicted
    ]

    actual_ranks = np.asarray([
        PRIORITY[value]
        for value in actual_categories
    ])

    predicted_ranks = np.asarray([
        PRIORITY[value]
        for value in predicted_categories
    ])

    dangerous = predicted_ranks < actual_ranks
    overprioritized = predicted_ranks > actual_ranks

    actual_critical = actual <= 10
    actual_high_or_critical = actual <= 25

    critical_recall = None

    if np.sum(actual_critical) > 0:

        critical_recall = float(
            np.sum(
                actual_critical
                & (predicted <= 10)
            )
            / np.sum(actual_critical)
        )

    return {
        "category_mismatches": int(
            np.sum(
                np.asarray(actual_categories)
                != np.asarray(predicted_categories)
            )
        ),
        "dangerous_underprioritization": int(
            np.sum(dangerous)
        ),
        "dangerous_underprioritization_rate": float(
            np.mean(dangerous)
        ),
        "category_overprioritization": int(
            np.sum(overprioritized)
        ),
        "actual_critical_engines": int(
            np.sum(actual_critical)
        ),
        "predicted_critical_engines": int(
            np.sum(predicted <= 10)
        ),
        "critical_recall": critical_recall,
        "actual_high_or_critical_engines": int(
            np.sum(actual_high_or_critical)
        ),
        "high_or_critical_numerical_overestimations": int(
            np.sum(
                actual_high_or_critical
                & (predicted > actual)
            )
        ),
    }


# ============================================================
# BUILD PREDICTION REPORT
# ============================================================

def build_prediction_dataframe(
    final_rows,
    actual_rul,
    baseline_predictions,
    temporal_raw_predictions,
    temporal_adjusted_predictions,
):

    baseline_categories = [
        classify_rul(value)
        for value in baseline_predictions
    ]

    temporal_raw_categories = [
        classify_rul(value)
        for value in temporal_raw_predictions
    ]

    temporal_adjusted_categories = [
        classify_rul(value)
        for value in temporal_adjusted_predictions
    ]

    actual_categories = [
        classify_rul(value)
        for value in actual_rul
    ]

    actual_ranks = np.asarray([
        PRIORITY[value]
        for value in actual_categories
    ])

    baseline_ranks = np.asarray([
        PRIORITY[value]
        for value in baseline_categories
    ])

    temporal_raw_ranks = np.asarray([
        PRIORITY[value]
        for value in temporal_raw_categories
    ])

    temporal_adjusted_ranks = np.asarray([
        PRIORITY[value]
        for value in temporal_adjusted_categories
    ])

    return pd.DataFrame({

        "engine_id": final_rows["unit"].to_numpy(),

        "last_cycle": final_rows["cycle"].to_numpy(),

        "actual_rul": actual_rul,

        "baseline_predicted_rul": np.round(
            baseline_predictions,
            4,
        ),

        "temporal_raw_predicted_rul": np.round(
            temporal_raw_predictions,
            4,
        ),

        "temporal_adjusted_predicted_rul": np.round(
            temporal_adjusted_predictions,
            4,
        ),

        "baseline_error": np.round(
            baseline_predictions - actual_rul,
            4,
        ),

        "temporal_raw_error": np.round(
            temporal_raw_predictions - actual_rul,
            4,
        ),

        "temporal_adjusted_error": np.round(
            temporal_adjusted_predictions - actual_rul,
            4,
        ),

        "baseline_absolute_error": np.round(
            np.abs(baseline_predictions - actual_rul),
            4,
        ),

        "temporal_raw_absolute_error": np.round(
            np.abs(temporal_raw_predictions - actual_rul),
            4,
        ),

        "temporal_adjusted_absolute_error": np.round(
            np.abs(temporal_adjusted_predictions - actual_rul),
            4,
        ),

        "actual_category": actual_categories,

        "baseline_category": baseline_categories,

        "temporal_raw_category": temporal_raw_categories,

        "temporal_adjusted_category": temporal_adjusted_categories,

        "baseline_dangerous_underprioritization": (
            baseline_ranks < actual_ranks
        ),

        "temporal_raw_dangerous_underprioritization": (
            temporal_raw_ranks < actual_ranks
        ),

        "temporal_adjusted_dangerous_underprioritization": (
            temporal_adjusted_ranks < actual_ranks
        ),
    })


# ============================================================
# REFERENCE METRIC REPRODUCTION CHECK
# ============================================================

def compare_with_saved_metadata(
    baseline_metrics,
    temporal_raw_metrics,
    baseline_metadata,
    temporal_metadata,
):

    checks = {}

    saved_baseline = (
        baseline_metadata
        .get("metrics", {})
        .get("test", {})
    )

    saved_temporal = (
        temporal_metadata
        .get("temporal_test_metrics", {})
    )

    for metric in ["mae", "rmse", "r2", "nasa_score"]:

        if metric in saved_baseline:

            difference = abs(
                baseline_metrics[metric]
                - float(saved_baseline[metric])
            )

            checks[f"baseline_{metric}_absolute_difference"] = (
                float(difference)
            )

        if metric in saved_temporal:

            difference = abs(
                temporal_raw_metrics[metric]
                - float(saved_temporal[metric])
            )

            checks[f"temporal_{metric}_absolute_difference"] = (
                float(difference)
            )

    return checks


# ============================================================
# MAIN AUDIT
# ============================================================

def run_audit():

    print("=" * 72)
    print("NASA FD001 OFFICIAL TEST AUDIT")
    print("=" * 72)

    print("\nFrozen safety adjustment:", FROZEN_OFFSET, "cycles")

    # Capture model and metadata hashes before evaluation.
    protected_files = [
        BASELINE_MODEL_FILE,
        BASELINE_METADATA_FILE,
        TEMPORAL_MODEL_FILE,
        TEMPORAL_METADATA_FILE,
    ]

    for path in protected_files:
        require_file(path)

    hashes_before = {
        str(path.relative_to(BASE_DIR)): sha256_file(path)
        for path in protected_files
    }

    (
        baseline_model,
        temporal_model,
        baseline_metadata,
        temporal_metadata,
        temporal_medians,
        baseline_features,
        temporal_features,
    ) = load_artifacts()

    test_df, actual_rul = load_test_data()

    # Generate causal temporal features from each engine's
    # observed history only. No future rows are used.
    test_with_temporal = add_temporal_features(
        test_df
    )

    final_rows = get_final_engine_rows(
        test_with_temporal
    )

    if len(final_rows) != len(actual_rul):

        raise ValueError(
            "Final snapshots and reference RUL labels do not align."
        )

    (
        baseline_predictions,
        temporal_raw_predictions,
        temporal_adjusted_predictions,
    ) = generate_predictions(
        baseline_model,
        temporal_model,
        temporal_medians,
        baseline_features,
        temporal_features,
        final_rows,
    )

    print("\nCalculating metrics...")

    baseline_metrics = calculate_metrics(
        actual_rul,
        baseline_predictions,
    )

    temporal_raw_metrics = calculate_metrics(
        actual_rul,
        temporal_raw_predictions,
    )

    temporal_adjusted_metrics = calculate_metrics(
        actual_rul,
        temporal_adjusted_predictions,
    )

    baseline_metrics["maintenance_audit"] = (
        calculate_maintenance_audit(
            actual_rul,
            baseline_predictions,
        )
    )

    temporal_raw_metrics["maintenance_audit"] = (
        calculate_maintenance_audit(
            actual_rul,
            temporal_raw_predictions,
        )
    )

    temporal_adjusted_metrics["maintenance_audit"] = (
        calculate_maintenance_audit(
            actual_rul,
            temporal_adjusted_predictions,
        )
    )

    predictions_df = build_prediction_dataframe(
        final_rows,
        actual_rul,
        baseline_predictions,
        temporal_raw_predictions,
        temporal_adjusted_predictions,
    )

    metadata_comparison = compare_with_saved_metadata(
        {
            key: baseline_metrics[key]
            for key in ["mae", "rmse", "r2", "nasa_score"]
        },
        {
            key: temporal_raw_metrics[key]
            for key in ["mae", "rmse", "r2", "nasa_score"]
        },
        baseline_metadata,
        temporal_metadata,
    )

    # Confirm that no protected model or metadata file changed.
    hashes_after = {
        str(path.relative_to(BASE_DIR)): sha256_file(path)
        for path in protected_files
    }

    protected_files_unchanged = (
        hashes_before == hashes_after
    )

    if not protected_files_unchanged:

        raise RuntimeError(
            "A protected model or metadata file changed during the audit."
        )

    # Create a unique output folder. Never overwrite an earlier audit.
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_dir = REPORT_ROOT / timestamp

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    predictions_path = (
        output_dir
        / "official_fd001_test_predictions.csv"
    )

    report_path = (
        output_dir
        / "official_fd001_test_audit_report.json"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    report = {
        "audit_name": "NASA FD001 Official Test Evaluation",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "dataset": "NASA C-MAPSS FD001",

        "test_engine_count": int(len(actual_rul)),

        "test_row_count": int(len(test_df)),

        "frozen_safety_adjustment_cycles": FROZEN_OFFSET,

        "adjustment_rule": (
            "max(0, temporal_raw_predicted_rul - 10)"
        ),

        "models": {
            "baseline": {
                "model_name": baseline_metadata.get("model_name"),
                "feature_count": len(baseline_features),
                "metrics": baseline_metrics,
            },

            "temporal_raw": {
                "model_name": temporal_metadata.get("model_name"),
                "registered": temporal_metadata.get("registered"),
                "feature_count": len(temporal_features),
                "metrics": temporal_raw_metrics,
            },

            "temporal_adjusted": {
                "model_name": temporal_metadata.get("model_name"),
                "registered": temporal_metadata.get("registered"),
                "feature_count": len(temporal_features),
                "frozen_offset_cycles": FROZEN_OFFSET,
                "metrics": temporal_adjusted_metrics,
            },
        },

        "saved_metadata_reproduction_check": metadata_comparison,

        "artifact_integrity": {
            "protected_files_unchanged": protected_files_unchanged,
            "sha256_before": hashes_before,
            "sha256_after": hashes_after,
        },

        "methodology": {
            "model_training_performed": False,
            "model_registration_performed": False,
            "test_set_used_for_offset_selection": False,
            "temporal_features_use_future_observations": False,
            "temporal_imputation": (
                "Saved training medians from temporal metadata"
            ),
            "baseline_imputation": (
                "No imputation performed; test features must be finite"
            ),
            "rul_reference": (
                "NASA RUL_FD001.txt, aligned to ascending engine IDs"
            ),
        },

        "important_limitation": (
            "The FD001 test set has already been evaluated and inspected "
            "during prior experiment and error analysis. This audit "
            "reproduces and compares frozen artifacts and a validation-"
            "derived fixed adjustment; it is not a virgin blind test."
        ),

        "output_files": {
            "predictions_csv": predictions_path.name,
            "report_json": report_path.name,
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n" + "=" * 72)
    print("OFFICIAL FD001 TEST RESULTS")
    print("=" * 72)

    for name, metrics in [
        ("BASELINE", baseline_metrics),
        ("TEMPORAL RAW", temporal_raw_metrics),
        ("TEMPORAL ADJUSTED (-10)", temporal_adjusted_metrics),
    ]:

        print(f"\n{name}")

        print(
            "MAE:",
            round(metrics["mae"], 4),
        )

        print(
            "RMSE:",
            round(metrics["rmse"], 4),
        )

        print(
            "R2:",
            round(metrics["r2"], 4),
        )

        print(
            "NASA score:",
            round(metrics["nasa_score"], 4),
        )

        audit = metrics["maintenance_audit"]

        print(
            "Category mismatches:",
            audit["category_mismatches"],
        )

        print(
            "Dangerous underprioritization:",
            audit["dangerous_underprioritization"],
        )

        print(
            "Category overprioritization:",
            audit["category_overprioritization"],
        )

        print(
            "Actual high/critical engines:",
            audit["actual_high_or_critical_engines"],
        )

        print(
            "Critical recall:",
            audit["critical_recall"],
        )

    print("\n" + "-" * 72)

    print("Saved metadata reproduction differences:")

    print(
        json.dumps(
            metadata_comparison,
            indent=4,
        )
    )

    print("\nProtected artifacts unchanged:", protected_files_unchanged)

    print("\nAudit output folder:")
    print(output_dir)

    print("\nPredictions CSV:")
    print(predictions_path)

    print("\nAudit JSON:")
    print(report_path)

    print("\nAUDIT COMPLETED SUCCESSFULLY.")

    return report


if __name__ == "__main__":

    run_audit()
