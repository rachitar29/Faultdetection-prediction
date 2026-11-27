
"""
NASA C-MAPSS FD001 - RUL Safety Sensitivity Audit

Uses existing multi-split validation predictions only.
Does not train models.
Does not modify production or experimental model artifacts.
Does not overwrite existing validation reports.

Analyses:
1. Safety cost ratios: 2:1, 5:1, 10:1.
2. Safety offsets: 0, 5, 10, 15, 20, 25 cycles.
3. Calibration-only offset selection.
4. Separate evaluation-engine assessment.
5. Engine-level bootstrap confidence intervals.
"""

from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
    / "cmapss_fd001_multisplit_validation_predictions.csv"
)

OUTPUT_ROOT = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
    / "safety_sensitivity_audit"
)

COST_RATIOS = [2, 5, 10]

OFFSET_CANDIDATES = [0, 5, 10, 15, 20, 25]

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_RANDOM_STATE = 42

PRIORITY = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# ============================================================
# INPUT VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "fold",
    "partition",
    "engine_id",
    "actual_rul",
    "baseline_predicted_rul",
    "temporal_raw_predicted_rul",
]


def load_predictions():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input predictions file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Prediction file is empty.")

    for column in [
        "fold",
        "engine_id",
        "actual_rul",
        "baseline_predicted_rul",
        "temporal_raw_predicted_rul",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    numeric_columns = [
        "fold",
        "engine_id",
        "actual_rul",
        "baseline_predicted_rul",
        "temporal_raw_predicted_rul",
    ]

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values found in required columns."
        )

    if not np.isfinite(
        df[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite numeric values found."
        )

    df["partition"] = (
        df["partition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed_partitions = {
        "calibration",
        "evaluation",
    }

    unexpected = (
        set(df["partition"].unique())
        - allowed_partitions
    )

    if unexpected:
        raise ValueError(
            f"Unexpected partitions found: {unexpected}"
        )

    if (df["actual_rul"] < 0).any():
        raise ValueError(
            "Negative actual RUL values found."
        )

    print("\nInput file loaded successfully.")
    print("Total prediction rows:", len(df))
    print("Folds:", sorted(df["fold"].unique().tolist()))
    print("Unique engines:", df["engine_id"].nunique())

    return df


# ============================================================
# ENGINE PARTITION VALIDATION
# ============================================================

def validate_engine_separation(df):

    for fold in sorted(df["fold"].unique()):

        fold_df = df[df["fold"] == fold]

        calibration_engines = set(
            fold_df.loc[
                fold_df["partition"] == "calibration",
                "engine_id"
            ].unique()
        )

        evaluation_engines = set(
            fold_df.loc[
                fold_df["partition"] == "evaluation",
                "engine_id"
            ].unique()
        )

        overlap = (
            calibration_engines
            & evaluation_engines
        )

        if overlap:
            raise RuntimeError(
                f"Engine leakage in fold {fold}: {sorted(overlap)}"
            )

        if not calibration_engines:
            raise RuntimeError(
                f"No calibration engines in fold {fold}."
            )

        if not evaluation_engines:
            raise RuntimeError(
                f"No evaluation engines in fold {fold}."
            )

        print(
            f"Fold {int(fold)}: "
            f"{len(calibration_engines)} calibration engines, "
            f"{len(evaluation_engines)} evaluation engines, "
            "no overlap."
        )


# ============================================================
# RUL CATEGORY AND SAFETY COST
# ============================================================

def classify_rul(value):

    if value <= 10:
        return "CRITICAL"

    if value <= 25:
        return "HIGH"

    if value <= 50:
        return "MEDIUM"

    return "LOW"


def calculate_row_costs(
    actual,
    predicted,
    under_cost,
    over_cost,
):

    actual = np.asarray(actual, dtype=float)

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    predicted = np.clip(
        predicted,
        0,
        None
    )

    actual_ranks = np.array([
        PRIORITY[classify_rul(value)]
        for value in actual
    ])

    predicted_ranks = np.array([
        PRIORITY[classify_rul(value)]
        for value in predicted
    ])

    dangerous = (
        predicted_ranks < actual_ranks
    )

    overprioritized = (
        predicted_ranks > actual_ranks
    )

    costs = (
        under_cost * dangerous.astype(int)
        + over_cost * overprioritized.astype(int)
    )

    return {
        "row_costs": costs,
        "dangerous": dangerous,
        "overprioritized": overprioritized,
        "actual_ranks": actual_ranks,
        "predicted_ranks": predicted_ranks,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
    under_cost,
    over_cost,
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

    details = calculate_row_costs(
        actual,
        predicted,
        under_cost,
        over_cost,
    )

    dangerous = details["dangerous"]
    over = details["overprioritized"]

    actual_critical = actual <= 10
    actual_high_critical = actual <= 25

    predicted_critical = (
        details["predicted_ranks"] == 4
    )

    predicted_high_critical = (
        details["predicted_ranks"] >= 3
    )

    critical_tp = int(
        np.sum(
            predicted_critical
            & actual_critical
        )
    )

    high_critical_tp = int(
        np.sum(
            predicted_high_critical
            & actual_high_critical
        )
    )

    predicted_critical_count = int(
        np.sum(predicted_critical)
    )

    predicted_high_critical_count = int(
        np.sum(predicted_high_critical)
    )

    actual_critical_count = int(
        np.sum(actual_critical)
    )

    actual_high_critical_count = int(
        np.sum(actual_high_critical)
    )

    return {
        "snapshot_count": int(len(actual)),

        "engine_count": None,

        "mae": float(
            np.mean(np.abs(actual - predicted))
        ),

        "rmse": float(
            np.sqrt(np.mean((actual - predicted) ** 2))
        ),

        "dangerous_underprioritization": int(
            np.sum(dangerous)
        ),

        "overprioritization": int(
            np.sum(over)
        ),

        "category_mismatches": int(
            np.sum(
                details["actual_ranks"]
                != details["predicted_ranks"]
            )
        ),

        "weighted_safety_cost": int(
            np.sum(details["row_costs"])
        ),

        "actual_critical_snapshots": (
            actual_critical_count
        ),

        "critical_recall": (
            float(critical_tp / actual_critical_count)
            if actual_critical_count else None
        ),

        "critical_precision": (
            float(
                critical_tp / predicted_critical_count
            )
            if predicted_critical_count else None
        ),

        "actual_high_critical_snapshots": (
            actual_high_critical_count
        ),

        "high_critical_recall": (
            float(
                high_critical_tp / actual_high_critical_count
            )
            if actual_high_critical_count else None
        ),

        "high_critical_precision": (
            float(
                high_critical_tp / predicted_high_critical_count
            )
            if predicted_high_critical_count else None
        ),
    }


# ============================================================
# CALIBRATION OFFSET SELECTION
# ============================================================

def select_offset(
    calibration_df,
    under_cost,
    over_cost,
):

    results = []

    actual = calibration_df[
        "actual_rul"
    ].to_numpy(dtype=float)

    raw_predictions = calibration_df[
        "temporal_raw_predicted_rul"
    ].to_numpy(dtype=float)

    for offset in OFFSET_CANDIDATES:

        adjusted = np.maximum(
            raw_predictions - offset,
            0
        )

        metrics = calculate_metrics(
            actual,
            adjusted,
            under_cost,
            over_cost,
        )

        results.append({
            "offset": offset,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    # Lowest cost wins. Ties prefer the smaller offset.
    selected = results_df.sort_values(
        [
            "weighted_safety_cost",
            "offset",
        ],
        ascending=[True, True]
    ).iloc[0]

    return (
        int(selected["offset"]),
        results_df,
    )


# ============================================================
# ENGINE-LEVEL BOOTSTRAP
# ============================================================

def engine_level_bootstrap(
    evaluation_df,
    selected_offset,
    under_cost,
    over_cost,
    rng,
):

    data = evaluation_df.copy()

    actual = data[
        "actual_rul"
    ].to_numpy(dtype=float)

    baseline = data[
        "baseline_predicted_rul"
    ].to_numpy(dtype=float)

    temporal = np.maximum(
        data["temporal_raw_predicted_rul"].to_numpy(dtype=float)
        - selected_offset,
        0,
    )

    baseline_details = calculate_row_costs(
        actual,
        baseline,
        under_cost,
        over_cost,
    )

    temporal_details = calculate_row_costs(
        actual,
        temporal,
        under_cost,
        over_cost,
    )

    data["baseline_row_cost"] = (
        baseline_details["row_costs"]
    )

    data["temporal_row_cost"] = (
        temporal_details["row_costs"]
    )

    # Average snapshot cost within each engine.
    # Engines, not individual snapshots, are resampled.
    engine_costs = data.groupby(
        "engine_id"
    ).agg(
        baseline_engine_cost=(
            "baseline_row_cost",
            "mean"
        ),
        temporal_engine_cost=(
            "temporal_row_cost",
            "mean"
        ),
    )

    engine_costs["difference"] = (
        engine_costs["temporal_engine_cost"]
        - engine_costs["baseline_engine_cost"]
    )

    values = engine_costs[
        "difference"
    ].to_numpy(dtype=float)

    engine_count = len(values)

    if engine_count < 2:
        raise ValueError(
            "At least two evaluation engines are required for bootstrap."
        )

    bootstrap_differences = np.empty(
        BOOTSTRAP_ITERATIONS,
        dtype=float
    )

    for i in range(BOOTSTRAP_ITERATIONS):

        sampled = rng.choice(
            values,
            size=engine_count,
            replace=True
        )

        bootstrap_differences[i] = np.mean(
            sampled
        )

    lower, upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5]
    )

    return {
        "evaluation_engine_count": engine_count,

        "mean_engine_cost_difference_temporal_minus_baseline": (
            float(np.mean(values))
        ),

        "bootstrap_95_percent_ci_lower": float(lower),

        "bootstrap_95_percent_ci_upper": float(upper),

        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

        "interpretation": (
            "Negative difference favours temporal; "
            "positive difference favours baseline."
        ),
    }


# ============================================================
# MAIN SENSITIVITY EXPERIMENT
# ============================================================

def run_audit():

    print("=" * 72)
    print("NASA FD001 SAFETY SENSITIVITY AUDIT")
    print("=" * 72)

    df = load_predictions()

    validate_engine_separation(df)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_dir = OUTPUT_ROOT / timestamp

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    calibration_rows = []
    evaluation_rows = []
    summary_rows = []
    bootstrap_rows = []

    rng = np.random.default_rng(
        BOOTSTRAP_RANDOM_STATE
    )

    folds = sorted(
        df["fold"].unique()
    )

    for fold in folds:

        print("\n" + "=" * 60)
        print(f"FOLD {int(fold)}")
        print("=" * 60)

        fold_df = df[
            df["fold"] == fold
        ].copy()

        calibration = fold_df[
            fold_df["partition"] == "calibration"
        ].copy()

        evaluation = fold_df[
            fold_df["partition"] == "evaluation"
        ].copy()

        for ratio in COST_RATIOS:

            under_cost = ratio
            over_cost = 1

            print(
                f"\nCost ratio: {under_cost}:1"
            )

            # ------------------------------------------------
            # Select offset using calibration engines only
            # ------------------------------------------------

            selected_offset, calibration_metrics = select_offset(
                calibration,
                under_cost,
                over_cost,
            )

            for _, row in calibration_metrics.iterrows():

                calibration_rows.append({
                    "fold": int(fold),
                    "underprioritization_cost": under_cost,
                    "overprioritization_cost": over_cost,
                    "offset": int(row["offset"]),
                    "selected": (
                        int(row["offset"]) == selected_offset
                    ),
                    **row.drop(
                        labels=["offset"]
                    ).to_dict(),
                })

            print(
                "Selected calibration offset:",
                selected_offset
            )

            # ------------------------------------------------
            # Evaluate every candidate on separate engines
            # ------------------------------------------------

            actual_eval = evaluation[
                "actual_rul"
            ].to_numpy(dtype=float)

            raw_temporal = evaluation[
                "temporal_raw_predicted_rul"
            ].to_numpy(dtype=float)

            for offset in OFFSET_CANDIDATES:

                adjusted = np.maximum(
                    raw_temporal - offset,
                    0,
                )

                metrics = calculate_metrics(
                    actual_eval,
                    adjusted,
                    under_cost,
                    over_cost,
                )

                evaluation_rows.append({
                    "fold": int(fold),
                    "underprioritization_cost": under_cost,
                    "overprioritization_cost": over_cost,
                    "offset": offset,
                    "selected_by_calibration": (
                        offset == selected_offset
                    ),
                    **metrics,
                })

            # ------------------------------------------------
            # Baseline and temporal comparison
            # ------------------------------------------------

            baseline_predictions = evaluation[
                "baseline_predicted_rul"
            ].to_numpy(dtype=float)

            adjusted_temporal = np.maximum(
                raw_temporal - selected_offset,
                0,
            )

            baseline_metrics = calculate_metrics(
                actual_eval,
                baseline_predictions,
                under_cost,
                over_cost,
            )

            temporal_raw_metrics = calculate_metrics(
                actual_eval,
                raw_temporal,
                under_cost,
                over_cost,
            )

            temporal_adjusted_metrics = calculate_metrics(
                actual_eval,
                adjusted_temporal,
                under_cost,
                over_cost,
            )

            baseline_metrics["engine_count"] = int(
                evaluation["engine_id"].nunique()
            )

            temporal_raw_metrics["engine_count"] = int(
                evaluation["engine_id"].nunique()
            )

            temporal_adjusted_metrics["engine_count"] = int(
                evaluation["engine_id"].nunique()
            )

            summary_rows.append({
                "fold": int(fold),
                "underprioritization_cost": under_cost,
                "overprioritization_cost": over_cost,
                "selected_offset": selected_offset,

                **{
                    f"baseline_{key}": value
                    for key, value in baseline_metrics.items()
                },

                **{
                    f"temporal_raw_{key}": value
                    for key, value in temporal_raw_metrics.items()
                },

                **{
                    f"temporal_adjusted_{key}": value
                    for key, value in temporal_adjusted_metrics.items()
                },
            })

            # ------------------------------------------------
            # Bootstrap by complete evaluation engines
            # ------------------------------------------------

            bootstrap = engine_level_bootstrap(
                evaluation,
                selected_offset,
                under_cost,
                over_cost,
                rng,
            )

            bootstrap_rows.append({
                "fold": int(fold),
                "underprioritization_cost": under_cost,
                "overprioritization_cost": over_cost,
                "selected_offset": selected_offset,
                **bootstrap,
            })

            print(
                "Baseline cost:",
                baseline_metrics["weighted_safety_cost"]
            )

            print(
                "Temporal raw cost:",
                temporal_raw_metrics["weighted_safety_cost"]
            )

            print(
                "Temporal adjusted cost:",
                temporal_adjusted_metrics["weighted_safety_cost"]
            )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    calibration_df.to_csv(
        output_dir / "calibration_offset_sensitivity.csv",
        index=False
    )

    evaluation_df.to_csv(
        output_dir / "evaluation_offset_sensitivity.csv",
        index=False
    )

    summary_df.to_csv(
        output_dir / "fold_model_comparison.csv",
        index=False
    )

    bootstrap_df.to_csv(
        output_dir / "engine_bootstrap_confidence_intervals.csv",
        index=False
    )

    report = {
        "experiment": (
            "NASA FD001 RUL Safety Sensitivity Audit"
        ),

        "input_file": str(INPUT_FILE),

        "test_dataset_used": False,

        "models_trained": False,

        "production_model_modified": False,

        "existing_reports_overwritten": False,

        "cost_ratios": COST_RATIOS,

        "offset_candidates": OFFSET_CANDIDATES,

        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,

        "bootstrap_unit": "complete evaluation engines",

        "fold_count": len(folds),

        "total_input_rows": len(df),

        "output_directory": str(output_dir),

        "files": [
            "calibration_offset_sensitivity.csv",
            "evaluation_offset_sensitivity.csv",
            "fold_model_comparison.csv",
            "engine_bootstrap_confidence_intervals.csv",
        ],

        "important_interpretation": [
            (
                "Offsets are selected using calibration engines only."
            ),
            (
                "Selected offsets are evaluated on separate evaluation engines."
            ),
            (
                "Bootstrap resamples complete engines, not individual snapshots."
            ),
            (
                "Fold results are not independent because repeated folds "
                "can reuse the same engines."
            ),
            (
                "The cost ratio is an experimental assumption, "
                "not an industrial safety certification."
            ),
            (
                "This audit does not evaluate the official NASA test set."
            ),
        ],

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with open(
        output_dir / "sensitivity_audit_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False
        )

    print("\n" + "=" * 72)
    print("SENSITIVITY AUDIT COMPLETED")
    print("=" * 72)

    print("Output folder:", output_dir)

    print("\nSelected offsets by fold and cost ratio:")

    print(
        summary_df[
            [
                "fold",
                "underprioritization_cost",
                "selected_offset",
                "baseline_weighted_safety_cost",
                "temporal_raw_weighted_safety_cost",
                "temporal_adjusted_weighted_safety_cost",
            ]
        ].to_string(index=False)
    )

    print("\nBootstrap confidence intervals:")

    print(
        bootstrap_df[
            [
                "fold",
                "underprioritization_cost",
                "selected_offset",
                "mean_engine_cost_difference_temporal_minus_baseline",
                "bootstrap_95_percent_ci_lower",
                "bootstrap_95_percent_ci_upper",
            ]
        ].to_string(index=False)
    )

    print("\nNo model was trained or modified.")
    print("No production artifact was changed.")
    print("Existing validation reports were preserved.")

    return output_dir


if __name__ == "__main__":
    run_audit()
