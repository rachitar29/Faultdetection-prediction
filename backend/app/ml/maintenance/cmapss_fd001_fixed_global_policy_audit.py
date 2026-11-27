
"""
C-MAPSS FD001 fixed global maintenance policy audit.

Evaluates fixed maintenance thresholds on the existing five-fold
evaluation partitions. No policy is selected or tuned in this script.

Policies:
- Original:       10 / 25 / 50
- Uniform 5:      15 / 30 / 55
- Uniform 10:     20 / 35 / 60
- Fixed targeted: 15 / 30 / 65  (margins 5 / 5 / 15)
- Fold-specific targeted policy selected previously at 10:1,
  included only as a reference comparator.

No training, model loading, registration, production modification,
or official NASA test-set access.

Run from the backend directory.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "ml_artifacts"
    / "reports"
    / "cmapss_fd001_multisplit_validation_predictions.csv"
)

PREVIOUS_POLICY_FILE = (
    PROJECT_ROOT
    / "ml_artifacts"
    / "reports"
    / "targeted_safety_policy_experiment"
    / "20261126_161737"
    / "selected_targeted_policies.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ml_artifacts"
    / "reports"
    / "fixed_global_policy_audit"
)

BASE_THRESHOLDS = {
    "critical": 10.0,
    "high": 25.0,
    "medium": 50.0,
}

COST_RATIOS = [2, 5, 10]
OVERPRIORITIZATION_COST = 1
EXPECTED_FOLDS = {1, 2, 3, 4, 5}

REQUIRED_PREDICTION_COLUMNS = {
    "fold",
    "partition",
    "engine_id",
    "actual_rul",
    "temporal_raw_predicted_rul",
}

CATEGORY_RANK = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

# Margins are added to the original category thresholds.
FIXED_POLICIES = {
    "original_no_adjustment": (0, 0, 0),
    "uniform_5_cycles": (5, 5, 5),
    "uniform_10_cycles": (10, 10, 10),
    "fixed_targeted_5_5_15": (5, 5, 15),
}


def classify_rul(
    values: np.ndarray,
    critical: float,
    high: float,
    medium: float,
) -> np.ndarray:
    return np.select(
        [
            values <= critical,
            values <= high,
            values <= medium,
        ],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )


def calculate_metrics(
    frame: pd.DataFrame,
    margins: tuple[int, int, int],
    under_cost: int,
) -> dict[str, Any]:

    critical_margin, high_margin, medium_margin = margins

    actual = frame["actual_rul"].to_numpy(dtype=float)
    predicted = frame[
        "temporal_raw_predicted_rul"
    ].to_numpy(dtype=float)

    critical_threshold = (
        BASE_THRESHOLDS["critical"] + critical_margin
    )
    high_threshold = (
        BASE_THRESHOLDS["high"] + high_margin
    )
    medium_threshold = (
        BASE_THRESHOLDS["medium"] + medium_margin
    )

    if not (
        critical_threshold
        <= high_threshold
        <= medium_threshold
    ):
        raise ValueError(
            f"Invalid threshold order: "
            f"{critical_threshold}, {high_threshold}, "
            f"{medium_threshold}"
        )

    actual_categories = classify_rul(
        actual,
        BASE_THRESHOLDS["critical"],
        BASE_THRESHOLDS["high"],
        BASE_THRESHOLDS["medium"],
    )

    predicted_categories = classify_rul(
        predicted,
        critical_threshold,
        high_threshold,
        medium_threshold,
    )

    actual_ranks = np.array(
        [CATEGORY_RANK[x] for x in actual_categories]
    )
    predicted_ranks = np.array(
        [CATEGORY_RANK[x] for x in predicted_categories]
    )

    under = predicted_ranks < actual_ranks
    over = predicted_ranks > actual_ranks
    mismatch = predicted_ranks != actual_ranks

    actual_critical = actual_categories == "CRITICAL"
    predicted_critical = predicted_categories == "CRITICAL"

    actual_high_critical = actual_ranks >= CATEGORY_RANK["HIGH"]
    predicted_high_critical = (
        predicted_ranks >= CATEGORY_RANK["HIGH"]
    )

    critical_tp = int(
        (actual_critical & predicted_critical).sum()
    )
    actual_critical_count = int(actual_critical.sum())
    predicted_critical_count = int(predicted_critical.sum())

    high_critical_tp = int(
        (actual_high_critical & predicted_high_critical).sum()
    )
    actual_high_critical_count = int(
        actual_high_critical.sum()
    )
    predicted_high_critical_count = int(
        predicted_high_critical.sum()
    )

    under_count = int(under.sum())
    over_count = int(over.sum())

    errors = predicted - actual

    return {
        "snapshot_count": int(len(frame)),
        "engine_count": int(frame["engine_id"].nunique()),

        "critical_margin": critical_margin,
        "high_margin": high_margin,
        "medium_margin": medium_margin,

        "effective_critical_threshold": critical_threshold,
        "effective_high_threshold": high_threshold,
        "effective_medium_threshold": medium_threshold,

        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),

        "category_mismatches": int(mismatch.sum()),
        "dangerous_underprioritization": under_count,
        "overprioritization": over_count,

        "actual_critical_snapshots": actual_critical_count,
        "predicted_critical_snapshots": predicted_critical_count,
        "critical_true_positives": critical_tp,

        "critical_recall": (
            critical_tp / actual_critical_count
            if actual_critical_count else None
        ),
        "critical_precision": (
            critical_tp / predicted_critical_count
            if predicted_critical_count else None
        ),

        "actual_high_critical_snapshots": (
            actual_high_critical_count
        ),
        "predicted_high_critical_snapshots": (
            predicted_high_critical_count
        ),
        "high_critical_true_positives": high_critical_tp,

        "high_critical_recall": (
            high_critical_tp / actual_high_critical_count
            if actual_high_critical_count else None
        ),
        "high_critical_precision": (
            high_critical_tp / predicted_high_critical_count
            if predicted_high_critical_count else None
        ),

        "weighted_safety_cost": int(
            under_count * under_cost
            + over_count * OVERPRIORITIZATION_COST
        ),
    }


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing validation predictions file:\n"
            f"{PREDICTIONS_FILE}"
        )

    frame = pd.read_csv(PREDICTIONS_FILE)

    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    frame = frame.copy()

    for column in [
        "fold",
        "engine_id",
        "actual_rul",
        "temporal_raw_predicted_rul",
    ]:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame["fold"] = frame["fold"].astype(int)
    frame["engine_id"] = frame["engine_id"].astype(int)

    frame["partition"] = (
        frame["partition"].astype(str).str.strip().str.lower()
    )

    if set(frame["fold"].unique()) != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected folds {EXPECTED_FOLDS}; "
            f"found {set(frame['fold'].unique())}"
        )

    if set(frame["partition"].unique()) != {
        "calibration",
        "evaluation",
    }:
        raise ValueError(
            "Expected exactly calibration and evaluation partitions."
        )

    numeric = frame[
        ["actual_rul", "temporal_raw_predicted_rul"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("RUL columns contain NaN or infinity.")

    if (numeric < 0).any():
        raise ValueError("Negative RUL values detected.")

    # Ensure calibration and evaluation engines are disjoint per fold.
    for fold in sorted(EXPECTED_FOLDS):
        subset = frame[frame["fold"] == fold]

        calibration_ids = set(
            subset.loc[
                subset["partition"] == "calibration",
                "engine_id",
            ]
        )

        evaluation_ids = set(
            subset.loc[
                subset["partition"] == "evaluation",
                "engine_id",
            ]
        )

        if not calibration_ids or not evaluation_ids:
            raise ValueError(
                f"Fold {fold} has an empty partition."
            )

        overlap = calibration_ids & evaluation_ids

        if overlap:
            raise ValueError(
                f"Fold {fold} has engine overlap: {sorted(overlap)}"
            )

    return frame


def load_previous_selected_policies() -> pd.DataFrame:
    if not PREVIOUS_POLICY_FILE.exists():
        raise FileNotFoundError(
            "Previous selected-policy file not found:\n"
            f"{PREVIOUS_POLICY_FILE}\n"
            "Check that the earlier experiment folder exists."
        )

    policies = pd.read_csv(PREVIOUS_POLICY_FILE)

    required = {
        "fold",
        "underprioritization_cost",
        "selected_critical_margin",
        "selected_high_margin",
        "selected_medium_margin",
    }

    missing = required - set(policies.columns)

    if missing:
        raise ValueError(
            "Previous policy file is missing columns: "
            f"{sorted(missing)}"
        )

    policies["fold"] = pd.to_numeric(
        policies["fold"], errors="raise"
    ).astype(int)

    policies["underprioritization_cost"] = pd.to_numeric(
        policies["underprioritization_cost"],
        errors="raise",
    ).astype(int)

    for column in [
        "selected_critical_margin",
        "selected_high_margin",
        "selected_medium_margin",
    ]:
        policies[column] = pd.to_numeric(
            policies[column],
            errors="raise",
        ).astype(int)

    selected = policies[
        policies["underprioritization_cost"] == 10
    ].copy()

    if set(selected["fold"]) != EXPECTED_FOLDS:
        raise ValueError(
            "Expected one previously selected 10:1 policy "
            "for each of the five folds."
        )

    if selected["fold"].duplicated().any():
        raise ValueError(
            "Duplicate fold entries in previous selected policies."
        )

    return selected.set_index("fold")


def run_audit() -> Path:
    print("=" * 78)
    print("C-MAPSS FD001 FIXED GLOBAL POLICY AUDIT")
    print("=" * 78)

    print(f"\nPredictions:\n{PREDICTIONS_FILE}")
    print(f"\nPrevious policies:\n{PREVIOUS_POLICY_FILE}")

    frame = load_predictions()
    previous_policies = load_previous_selected_policies()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    result_rows = []

    for fold in sorted(EXPECTED_FOLDS):
        evaluation = frame[
            (frame["fold"] == fold)
            & (frame["partition"] == "evaluation")
        ].copy()

        previous = previous_policies.loc[fold]

        fold_policies = dict(FIXED_POLICIES)

        fold_policies["previous_fold_specific_targeted"] = (
            int(previous["selected_critical_margin"]),
            int(previous["selected_high_margin"]),
            int(previous["selected_medium_margin"]),
        )

        for under_cost in COST_RATIOS:
            for policy_name, margins in fold_policies.items():
                metrics = calculate_metrics(
                    evaluation,
                    margins,
                    under_cost,
                )

                result_rows.append({
                    "fold": fold,
                    "underprioritization_cost": under_cost,
                    "policy_name": policy_name,
                    **metrics,
                })

    results = pd.DataFrame(result_rows)

    results.to_csv(
        output_dir / "fixed_policy_fold_results.csv",
        index=False,
    )

    # Aggregate snapshot counts and confusion-related metrics across folds.
    summary_rows = []

    for under_cost in COST_RATIOS:
        subset = results[
            results["underprioritization_cost"] == under_cost
        ]

        for policy_name in fold_policies:
            policy_rows = subset[
                subset["policy_name"] == policy_name
            ].copy()

            total_actual_critical = int(
                policy_rows["actual_critical_snapshots"].sum()
            )
            total_predicted_critical = int(
                policy_rows["predicted_critical_snapshots"].sum()
            )
            total_critical_tp = int(
                policy_rows["critical_true_positives"].sum()
            )

            total_actual_hc = int(
                policy_rows["actual_high_critical_snapshots"].sum()
            )
            total_predicted_hc = int(
                policy_rows["predicted_high_critical_snapshots"].sum()
            )
            total_hc_tp = int(
                policy_rows["high_critical_true_positives"].sum()
            )

            summary_rows.append({
                "underprioritization_cost": under_cost,
                "policy_name": policy_name,
                "fold_count": int(policy_rows["fold"].nunique()),
                "total_snapshots": int(
                    policy_rows["snapshot_count"].sum()
                ),

                "total_weighted_safety_cost": int(
                    policy_rows["weighted_safety_cost"].sum()
                ),
                "total_dangerous_underprioritization": int(
                    policy_rows[
                        "dangerous_underprioritization"
                    ].sum()
                ),
                "total_overprioritization": int(
                    policy_rows["overprioritization"].sum()
                ),
                "total_category_mismatches": int(
                    policy_rows["category_mismatches"].sum()
                ),

                "total_actual_critical": total_actual_critical,
                "total_predicted_critical": total_predicted_critical,
                "total_critical_true_positives": total_critical_tp,

                "aggregate_critical_recall": (
                    total_critical_tp / total_actual_critical
                    if total_actual_critical else None
                ),
                "aggregate_critical_precision": (
                    total_critical_tp / total_predicted_critical
                    if total_predicted_critical else None
                ),

                "mean_fold_critical_recall": float(
                    policy_rows["critical_recall"].dropna().mean()
                ),
                "mean_fold_critical_precision": float(
                    policy_rows["critical_precision"].dropna().mean()
                ),

                "aggregate_high_critical_recall": (
                    total_hc_tp / total_actual_hc
                    if total_actual_hc else None
                ),
                "aggregate_high_critical_precision": (
                    total_hc_tp / total_predicted_hc
                    if total_predicted_hc else None
                ),

                "mean_fold_mae": float(
                    policy_rows["mae"].mean()
                ),
                "mean_fold_rmse": float(
                    policy_rows["rmse"].mean()
                ),
            })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output_dir / "fixed_policy_cross_fold_summary.csv",
        index=False,
    )

    # Save the exact fixed policy definitions and audit limitations.
    metadata = {
        "experiment_name": "C-MAPSS FD001 Fixed Global Policy Audit",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "predictions_file": str(PREDICTIONS_FILE),
        "previous_selected_policy_file": str(PREVIOUS_POLICY_FILE),
        "output_directory": str(output_dir),
        "base_thresholds": BASE_THRESHOLDS,
        "fixed_policy_margins": {
            name: {
                "critical": margins[0],
                "high": margins[1],
                "medium": margins[2],
            }
            for name, margins in FIXED_POLICIES.items()
        },
        "fixed_targeted_effective_thresholds": {
            "critical": 15,
            "high": 30,
            "medium": 65,
        },
        "cost_ratios": COST_RATIOS,
        "selection_or_tuning_performed": False,
        "evaluation_partitions_only": True,
        "calibration_partition_used": False,
        "model_training_performed": False,
        "model_registration_performed": False,
        "model_artifacts_loaded_or_modified": False,
        "production_files_modified": False,
        "official_nasa_test_set_used": False,
        "important_caveat": (
            "This is a fixed-policy comparison on previously examined "
            "validation predictions. Repeated snapshots from the same "
            "engine are correlated, and the five folds are not fully "
            "independent datasets. This is exploratory evidence, not "
            "an independent external validation."
        ),
    }

    with open(
        output_dir / "audit_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4, allow_nan=False)

    print("\n" + "=" * 78)
    print("CROSS-FOLD SUMMARY")
    print("=" * 78)

    display_columns = [
        "underprioritization_cost",
        "policy_name",
        "total_weighted_safety_cost",
        "total_dangerous_underprioritization",
        "total_overprioritization",
        "aggregate_critical_recall",
        "aggregate_critical_precision",
        "aggregate_high_critical_recall",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n" + "=" * 78)
    print("AUDIT COMPLETED")
    print("=" * 78)
    print(f"\nOutput directory:\n{output_dir}")

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")

    print(
        "\nNo model was trained, loaded, registered, or modified. "
        "No production thresholds were changed. "
        "No official NASA test data was used."
    )

    return output_dir


if __name__ == "__main__":
    try:
        run_audit()
    except Exception as exc:
        print("\nAUDIT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
