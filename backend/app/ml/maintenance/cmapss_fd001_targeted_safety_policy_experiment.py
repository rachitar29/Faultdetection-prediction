
"""
C-MAPSS FD001 targeted maintenance safety policy experiment.

Purpose
-------
Compare:
1. Original maintenance thresholds.
2. Uniform safety margins.
3. Targeted safety margins at CRITICAL, HIGH and MEDIUM boundaries.

Policy selection uses calibration snapshots only.
Selected policies are evaluated on separate evaluation snapshots.

This script:
- Does NOT train or retrain models.
- Does NOT load or modify model artifacts.
- Does NOT register models.
- Does NOT read the official NASA test set.
- Does NOT overwrite existing reports.
- Writes all outputs to a new timestamped report directory.

Run from:
D:\\Rachita\\faultdetectionml\\backend

Required input:
ml_artifacts\\reports\\cmapss_fd001_multisplit_validation_predictions.csv
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PATHS AND EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "ml_artifacts"
    / "reports"
    / "cmapss_fd001_multisplit_validation_predictions.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ml_artifacts"
    / "reports"
    / "targeted_safety_policy_experiment"
)

BASE_CRITICAL_THRESHOLD = 10.0
BASE_HIGH_THRESHOLD = 25.0
BASE_MEDIUM_THRESHOLD = 50.0

# Candidate safety margins, in predicted RUL cycles.
MARGIN_VALUES = [0, 5, 10, 15, 20, 25]

# Relative cost of underprioritization compared with overprioritization.
UNDERPRIORITIZATION_COSTS = [2, 5, 10]
OVERPRIORITIZATION_COST = 1

EXPECTED_PARTITIONS = {"calibration", "evaluation"}
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

REQUIRED_COLUMNS = {
    "fold",
    "partition",
    "engine_id",
    "actual_rul",
    "temporal_raw_predicted_rul",
}


# ---------------------------------------------------------------------
# CATEGORY AND METRIC FUNCTIONS
# ---------------------------------------------------------------------

CATEGORY_RANK = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def classify_rul(
    rul_values: np.ndarray,
    critical_threshold: float,
    high_threshold: float,
    medium_threshold: float,
) -> np.ndarray:
    """Convert RUL predictions into maintenance categories."""

    return np.select(
        [
            rul_values <= critical_threshold,
            rul_values <= high_threshold,
            rul_values <= medium_threshold,
        ],
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
        ],
        default="LOW",
    )


def calculate_metrics(
    frame: pd.DataFrame,
    critical_margin: int,
    high_margin: int,
    medium_margin: int,
    under_cost: int,
) -> dict[str, Any]:
    """Calculate maintenance-policy metrics for a snapshot partition."""

    actual_rul = frame["actual_rul"].to_numpy(dtype=float)
    predicted_rul = frame["temporal_raw_predicted_rul"].to_numpy(
        dtype=float
    )

    critical_threshold = (
        BASE_CRITICAL_THRESHOLD + critical_margin
    )
    high_threshold = BASE_HIGH_THRESHOLD + high_margin
    medium_threshold = BASE_MEDIUM_THRESHOLD + medium_margin

    actual_categories = classify_rul(
        actual_rul,
        BASE_CRITICAL_THRESHOLD,
        BASE_HIGH_THRESHOLD,
        BASE_MEDIUM_THRESHOLD,
    )

    predicted_categories = classify_rul(
        predicted_rul,
        critical_threshold,
        high_threshold,
        medium_threshold,
    )

    actual_ranks = np.array(
        [CATEGORY_RANK[value] for value in actual_categories]
    )
    predicted_ranks = np.array(
        [CATEGORY_RANK[value] for value in predicted_categories]
    )

    underprioritized = predicted_ranks < actual_ranks
    overprioritized = predicted_ranks > actual_ranks
    mismatches = predicted_ranks != actual_ranks

    actual_critical = actual_categories == "CRITICAL"
    predicted_critical = predicted_categories == "CRITICAL"

    actual_high_critical = actual_ranks >= CATEGORY_RANK["HIGH"]
    predicted_high_critical = (
        predicted_ranks >= CATEGORY_RANK["HIGH"]
    )

    critical_count = int(actual_critical.sum())
    high_critical_count = int(actual_high_critical.sum())

    predicted_critical_count = int(predicted_critical.sum())
    predicted_high_critical_count = int(
        predicted_high_critical.sum()
    )

    critical_true_positives = int(
        (actual_critical & predicted_critical).sum()
    )

    high_critical_true_positives = int(
        (actual_high_critical & predicted_high_critical).sum()
    )

    critical_recall = (
        critical_true_positives / critical_count
        if critical_count
        else None
    )

    critical_precision = (
        critical_true_positives / predicted_critical_count
        if predicted_critical_count
        else None
    )

    high_critical_recall = (
        high_critical_true_positives / high_critical_count
        if high_critical_count
        else None
    )

    high_critical_precision = (
        high_critical_true_positives / predicted_high_critical_count
        if predicted_high_critical_count
        else None
    )

    under_count = int(underprioritized.sum())
    over_count = int(overprioritized.sum())

    weighted_cost = (
        under_count * under_cost
        + over_count * OVERPRIORITIZATION_COST
    )

    errors = predicted_rul - actual_rul

    return {
        "snapshot_count": int(len(frame)),
        "engine_count": int(frame["engine_id"].nunique()),

        "critical_margin": int(critical_margin),
        "high_margin": int(high_margin),
        "medium_margin": int(medium_margin),

        "effective_critical_threshold": critical_threshold,
        "effective_high_threshold": high_threshold,
        "effective_medium_threshold": medium_threshold,

        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),

        "category_mismatches": int(mismatches.sum()),
        "dangerous_underprioritization": under_count,
        "overprioritization": over_count,

        "actual_critical_snapshots": critical_count,
        "predicted_critical_snapshots": predicted_critical_count,
        "critical_true_positives": critical_true_positives,
        "critical_recall": critical_recall,
        "critical_precision": critical_precision,

        "actual_high_critical_snapshots": high_critical_count,
        "predicted_high_critical_snapshots": (
            predicted_high_critical_count
        ),
        "high_critical_true_positives": (
            high_critical_true_positives
        ),
        "high_critical_recall": high_critical_recall,
        "high_critical_precision": high_critical_precision,

        "weighted_safety_cost": int(weighted_cost),
    }


# ---------------------------------------------------------------------
# CANDIDATE POLICY GENERATION
# ---------------------------------------------------------------------

def generate_candidate_policies() -> list[tuple[int, int, int]]:
    """
    Generate valid threshold-margin combinations.

    Effective thresholds must remain ordered:

        CRITICAL threshold <= HIGH threshold <= MEDIUM threshold

    Uniform offsets are included as a subset of these candidates.
    """

    policies = []

    for critical, high, medium in itertools.product(
        MARGIN_VALUES,
        repeat=3,
    ):
        critical_threshold = BASE_CRITICAL_THRESHOLD + critical
        high_threshold = BASE_HIGH_THRESHOLD + high
        medium_threshold = BASE_MEDIUM_THRESHOLD + medium

        if not (
            critical_threshold
            <= high_threshold
            <= medium_threshold
        ):
            continue

        policies.append((critical, high, medium))

    policies = sorted(set(policies))

    if (0, 0, 0) not in policies:
        raise RuntimeError("Original threshold policy is missing.")

    return policies


# ---------------------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------------------

def load_and_validate_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Validation predictions file not found:\n{INPUT_FILE}"
        )

    frame = pd.read_csv(INPUT_FILE)

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(
            "Validation CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()

    frame["fold"] = pd.to_numeric(
        frame["fold"],
        errors="raise",
    ).astype(int)

    frame["engine_id"] = pd.to_numeric(
        frame["engine_id"],
        errors="raise",
    ).astype(int)

    frame["actual_rul"] = pd.to_numeric(
        frame["actual_rul"],
        errors="raise",
    )

    frame["temporal_raw_predicted_rul"] = pd.to_numeric(
        frame["temporal_raw_predicted_rul"],
        errors="raise",
    )

    frame["partition"] = (
        frame["partition"].astype(str).str.strip().str.lower()
    )

    if frame.empty:
        raise ValueError("Validation predictions CSV is empty.")

    if not np.isfinite(
        frame[
            [
                "actual_rul",
                "temporal_raw_predicted_rul",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "RUL columns contain NaN or infinite values."
        )

    if (frame["actual_rul"] < 0).any():
        raise ValueError("Actual RUL contains negative values.")

    if (frame["temporal_raw_predicted_rul"] < 0).any():
        raise ValueError(
            "Temporal raw predictions contain negative values."
        )

    partitions = set(frame["partition"].unique())

    if partitions != EXPECTED_PARTITIONS:
        raise ValueError(
            f"Expected partitions {EXPECTED_PARTITIONS}; "
            f"found {partitions}."
        )

    folds = set(frame["fold"].unique())

    if folds != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected folds {EXPECTED_FOLDS}; found {folds}."
        )

    # Confirm that calibration and evaluation engines do not overlap
    # within an individual fold.
    for fold in sorted(EXPECTED_FOLDS):
        fold_data = frame[frame["fold"] == fold]

        calibration_engines = set(
            fold_data.loc[
                fold_data["partition"] == "calibration",
                "engine_id",
            ].unique()
        )

        evaluation_engines = set(
            fold_data.loc[
                fold_data["partition"] == "evaluation",
                "engine_id",
            ].unique()
        )

        overlap = calibration_engines & evaluation_engines

        if overlap:
            raise ValueError(
                f"Fold {fold} has engine leakage between "
                f"calibration and evaluation: {sorted(overlap)}"
            )

        if not calibration_engines or not evaluation_engines:
            raise ValueError(
                f"Fold {fold} has an empty calibration or evaluation set."
            )

    return frame


# ---------------------------------------------------------------------
# POLICY SELECTION
# ---------------------------------------------------------------------

def select_policy_on_calibration(
    calibration_frame: pd.DataFrame,
    policies: list[tuple[int, int, int]],
    under_cost: int,
) -> tuple[tuple[int, int, int], dict[str, Any]]:
    """
    Select the lowest-cost policy using calibration data only.

    Tie-breaking:
    1. Lowest weighted safety cost.
    2. Fewest category mismatches.
    3. Fewest total margin cycles.
    4. Lower critical margin.
    5. Lower high margin.
    6. Lower medium margin.
    """

    candidates = []

    for policy in policies:
        critical, high, medium = policy

        metrics = calculate_metrics(
            calibration_frame,
            critical,
            high,
            medium,
            under_cost,
        )

        tie_key = (
            metrics["weighted_safety_cost"],
            metrics["category_mismatches"],
            critical + high + medium,
            critical,
            high,
            medium,
        )

        candidates.append(
            {
                "policy": policy,
                "metrics": metrics,
                "tie_key": tie_key,
            }
        )

    selected = min(
        candidates,
        key=lambda item: item["tie_key"],
    )

    return selected["policy"], selected["metrics"]


# ---------------------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------------------

def run_experiment() -> Path:
    print("=" * 78)
    print("C-MAPSS FD001 TARGETED SAFETY POLICY EXPERIMENT")
    print("=" * 78)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Input file:   {INPUT_FILE}")

    frame = load_and_validate_data()
    policies = generate_candidate_policies()

    print(f"\nValidation snapshots: {len(frame)}")
    print(f"Validation engines:   {frame['engine_id'].nunique()}")
    print(f"Validation folds:     {len(EXPECTED_FOLDS)}")
    print(f"Candidate policies:   {len(policies)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    selected_rows = []
    calibration_rows = []
    evaluation_rows = []
    all_candidate_rows = []

    for fold in sorted(EXPECTED_FOLDS):
        fold_data = frame[frame["fold"] == fold]

        calibration = fold_data[
            fold_data["partition"] == "calibration"
        ].copy()

        evaluation = fold_data[
            fold_data["partition"] == "evaluation"
        ].copy()

        print(f"\n{'-' * 78}")
        print(f"FOLD {fold}")
        print(f"Calibration snapshots: {len(calibration)}")
        print(f"Evaluation snapshots:  {len(evaluation)}")

        for under_cost in UNDERPRIORITIZATION_COSTS:
            selected_policy, selected_calibration_metrics = (
                select_policy_on_calibration(
                    calibration,
                    policies,
                    under_cost,
                )
            )

            critical, high, medium = selected_policy

            evaluation_metrics = calculate_metrics(
                evaluation,
                critical,
                high,
                medium,
                under_cost,
            )

            selected_row = {
                "fold": fold,
                "underprioritization_cost": under_cost,
                "overprioritization_cost": (
                    OVERPRIORITIZATION_COST
                ),
                "selected_critical_margin": critical,
                "selected_high_margin": high,
                "selected_medium_margin": medium,
                "selection_source": "calibration_only",
                "calibration_weighted_cost": (
                    selected_calibration_metrics[
                        "weighted_safety_cost"
                    ]
                ),
                "evaluation_weighted_cost": (
                    evaluation_metrics["weighted_safety_cost"]
                ),
                "evaluation_dangerous_underprioritization": (
                    evaluation_metrics[
                        "dangerous_underprioritization"
                    ]
                ),
                "evaluation_overprioritization": (
                    evaluation_metrics["overprioritization"]
                ),
                "evaluation_category_mismatches": (
                    evaluation_metrics["category_mismatches"]
                ),
                "evaluation_critical_recall": (
                    evaluation_metrics["critical_recall"]
                ),
                "evaluation_critical_precision": (
                    evaluation_metrics["critical_precision"]
                ),
                "evaluation_high_critical_recall": (
                    evaluation_metrics["high_critical_recall"]
                ),
                "evaluation_high_critical_precision": (
                    evaluation_metrics["high_critical_precision"]
                ),
            }

            selected_rows.append(selected_row)

            calibration_rows.append(
                {
                    "fold": fold,
                    "underprioritization_cost": under_cost,
                    **selected_calibration_metrics,
                }
            )

            evaluation_rows.append(
                {
                    "fold": fold,
                    "underprioritization_cost": under_cost,
                    **evaluation_metrics,
                }
            )

            print(
                f"\nCost ratio {under_cost}:1"
                f"\n  Selected margins: "
                f"CRITICAL={critical}, HIGH={high}, MEDIUM={medium}"
                f"\n  Calibration cost: "
                f"{selected_calibration_metrics['weighted_safety_cost']}"
                f"\n  Evaluation cost:  "
                f"{evaluation_metrics['weighted_safety_cost']}"
                f"\n  Evaluation misses: "
                f"{evaluation_metrics['dangerous_underprioritization']}"
                f"\n  Evaluation overprioritization: "
                f"{evaluation_metrics['overprioritization']}"
                f"\n  Critical recall: "
                f"{evaluation_metrics['critical_recall']}"
            )

            # Save all calibration candidate results for auditability.
            for policy in policies:
                c, h, m = policy

                metrics = calculate_metrics(
                    calibration,
                    c,
                    h,
                    m,
                    under_cost,
                )

                all_candidate_rows.append(
                    {
                        "fold": fold,
                        "partition": "calibration",
                        "underprioritization_cost": under_cost,
                        "critical_margin": c,
                        "high_margin": h,
                        "medium_margin": m,
                        "selected": policy == selected_policy,
                        **metrics,
                    }
                )

    # -----------------------------------------------------------------
    # BASELINE AND UNIFORM POLICY EVALUATION
    # -----------------------------------------------------------------

    comparator_rows = []

    for fold in sorted(EXPECTED_FOLDS):
        evaluation = frame[
            (frame["fold"] == fold)
            & (frame["partition"] == "evaluation")
        ].copy()

        for under_cost in UNDERPRIORITIZATION_COSTS:
            comparator_policies = [
                ("original_no_adjustment", (0, 0, 0)),
                ("uniform_5_cycles", (5, 5, 5)),
                ("uniform_10_cycles", (10, 10, 10)),
                ("uniform_15_cycles", (15, 15, 15)),
                ("uniform_20_cycles", (20, 20, 20)),
                ("uniform_25_cycles", (25, 25, 25)),
            ]

            for name, policy in comparator_policies:
                critical, high, medium = policy

                metrics = calculate_metrics(
                    evaluation,
                    critical,
                    high,
                    medium,
                    under_cost,
                )

                comparator_rows.append(
                    {
                        "fold": fold,
                        "underprioritization_cost": under_cost,
                        "policy_name": name,
                        **metrics,
                    }
                )

    # -----------------------------------------------------------------
    # WRITE OUTPUTS
    # -----------------------------------------------------------------

    selected_df = pd.DataFrame(selected_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    evaluation_df = pd.DataFrame(evaluation_rows)
    candidates_df = pd.DataFrame(all_candidate_rows)
    comparators_df = pd.DataFrame(comparator_rows)

    selected_df.to_csv(
        output_dir / "selected_targeted_policies.csv",
        index=False,
    )

    calibration_df.to_csv(
        output_dir / "selected_policy_calibration_metrics.csv",
        index=False,
    )

    evaluation_df.to_csv(
        output_dir / "selected_policy_evaluation_metrics.csv",
        index=False,
    )

    candidates_df.to_csv(
        output_dir / "all_calibration_policy_candidates.csv",
        index=False,
    )

    comparators_df.to_csv(
        output_dir / "uniform_policy_comparison.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # CROSS-FOLD SUMMARY
    # -----------------------------------------------------------------

    summary_rows = []

    for under_cost in UNDERPRIORITIZATION_COSTS:
        subset = evaluation_df[
            evaluation_df["underprioritization_cost"] == under_cost
        ]

        summary_rows.append(
            {
                "underprioritization_cost": under_cost,
                "fold_count": int(subset["fold"].nunique()),
                "mean_evaluation_weighted_cost": float(
                    subset["weighted_safety_cost"].mean()
                ),
                "total_evaluation_weighted_cost": int(
                    subset["weighted_safety_cost"].sum()
                ),
                "mean_dangerous_underprioritization": float(
                    subset[
                        "dangerous_underprioritization"
                    ].mean()
                ),
                "total_dangerous_underprioritization": int(
                    subset[
                        "dangerous_underprioritization"
                    ].sum()
                ),
                "mean_overprioritization": float(
                    subset["overprioritization"].mean()
                ),
                "total_overprioritization": int(
                    subset["overprioritization"].sum()
                ),
                "mean_category_mismatches": float(
                    subset["category_mismatches"].mean()
                ),
                "mean_critical_recall": float(
                    subset["critical_recall"].mean()
                ),
                "mean_critical_precision": float(
                    subset["critical_precision"].mean()
                ),
                "mean_high_critical_recall": float(
                    subset["high_critical_recall"].mean()
                ),
                "mean_high_critical_precision": float(
                    subset["high_critical_precision"].mean()
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        output_dir / "cross_fold_policy_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # EXPERIMENT METADATA
    # -----------------------------------------------------------------

    metadata = {
        "experiment_name": (
            "C-MAPSS FD001 Targeted Safety Policy Experiment"
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(INPUT_FILE),
        "output_directory": str(output_dir),
        "validation_folds": sorted(EXPECTED_FOLDS),
        "candidate_margin_values": MARGIN_VALUES,
        "candidate_policy_count": len(policies),
        "underprioritization_costs": (
            UNDERPRIORITIZATION_COSTS
        ),
        "overprioritization_cost": OVERPRIORITIZATION_COST,
        "base_thresholds": {
            "critical": BASE_CRITICAL_THRESHOLD,
            "high": BASE_HIGH_THRESHOLD,
            "medium": BASE_MEDIUM_THRESHOLD,
        },
        "policy_selection": (
            "Minimum weighted safety cost on calibration snapshots "
            "only, with deterministic tie-breaking."
        ),
        "evaluation": (
            "Selected policy evaluated on separate evaluation "
            "snapshots within each fold."
        ),
        "engine_separation_validated": True,
        "model_training_performed": False,
        "model_registration_performed": False,
        "production_files_modified": False,
        "official_nasa_test_set_used": False,
        "thresholds_modified_in_production": False,
        "important_caveat": (
            "Repeated snapshots from the same engine are correlated. "
            "The five folds are not five fully independent datasets. "
            "Policy results are exploratory validation evidence and "
            "do not establish external generalization."
        ),
    }

    with open(
        output_dir / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            allow_nan=False,
        )

    # -----------------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # -----------------------------------------------------------------

    print("\n" + "=" * 78)
    print("CROSS-FOLD TARGETED POLICY SUMMARY")
    print("=" * 78)

    display_columns = [
        "underprioritization_cost",
        "total_evaluation_weighted_cost",
        "total_dangerous_underprioritization",
        "total_overprioritization",
        "mean_critical_recall",
        "mean_critical_precision",
        "mean_high_critical_recall",
        "mean_high_critical_precision",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT COMPLETED")
    print("=" * 78)

    print(f"\nOutput directory:\n{output_dir}")

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")

    print(
        "\nNo model was trained, registered, or modified. "
        "No official NASA test data was used."
    )

    return output_dir


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as exc:
        print("\nEXPERIMENT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
