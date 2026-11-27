
"""
Data Drift Detection for Predictive Maintenance.

Compares a reference dataset with incoming production data.

Features:
- Numeric drift using PSI and Kolmogorov-Smirnov test
- Categorical drift using PSI
- Missing-value rate monitoring
- Schema drift detection
- Feature-level and overall drift status
- JSON report generation
- Built-in self-test

This module does not train or modify prediction models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


BACKEND_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BACKEND_DIR / "ml_artifacts" / "monitoring"

DEFAULT_PSI_WARNING = 0.10
DEFAULT_PSI_CRITICAL = 0.25
DEFAULT_KS_P_VALUE = 0.05
DEFAULT_MISSING_RATE_CHANGE = 0.10

EPSILON = 1e-6


def _json_safe(value: Any) -> Any:
    """Convert NumPy and pandas values into JSON-compatible values."""

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    if pd.isna(value):
        return None

    return value


def _calculate_psi(
    reference_values: np.ndarray,
    current_values: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index.

    Numeric features use reference quantile bins.
    Categorical features use the combined category set.
    """

    reference_values = np.asarray(reference_values)
    current_values = np.asarray(current_values)

    if len(reference_values) == 0 or len(current_values) == 0:
        return 0.0

    if (
        np.issubdtype(reference_values.dtype, np.number)
        and np.issubdtype(current_values.dtype, np.number)
    ):
        reference_values = reference_values.astype(float)
        current_values = current_values.astype(float)

        combined = np.concatenate(
            [reference_values, current_values]
        )

        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(
            np.quantile(reference_values, quantiles)
        )

        if len(edges) < 2:
            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_counts, _ = np.histogram(
            reference_values, bins=edges
        )
        current_counts, _ = np.histogram(
            current_values, bins=edges
        )

    else:
        reference_strings = pd.Series(
            reference_values
        ).astype(str)

        current_strings = pd.Series(
            current_values
        ).astype(str)

        categories = sorted(
            set(reference_strings) | set(current_strings)
        )

        reference_counts = np.array([
            (reference_strings == category).sum()
            for category in categories
        ])

        current_counts = np.array([
            (current_strings == category).sum()
            for category in categories
        ])

    reference_percent = (
        reference_counts + EPSILON
    ) / (
        reference_counts.sum()
        + EPSILON * len(reference_counts)
    )

    current_percent = (
        current_counts + EPSILON
    ) / (
        current_counts.sum()
        + EPSILON * len(current_counts)
    )

    psi = np.sum(
        (current_percent - reference_percent)
        * np.log(current_percent / reference_percent)
    )

    return float(psi)


def _psi_status(
    psi: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Classify PSI severity."""

    if psi >= critical_threshold:
        return "CRITICAL"

    if psi >= warning_threshold:
        return "WARNING"

    return "STABLE"


class DataDriftDetector:
    """
    Compare reference and current datasets for data drift.
    """

    def __init__(
        self,
        psi_warning: float = DEFAULT_PSI_WARNING,
        psi_critical: float = DEFAULT_PSI_CRITICAL,
        ks_p_value: float = DEFAULT_KS_P_VALUE,
        missing_rate_change: float = DEFAULT_MISSING_RATE_CHANGE,
        ignored_columns: Optional[list[str]] = None,
        bins: int = 10,
    ):
        if not 0 <= psi_warning < psi_critical:
            raise ValueError(
                "PSI thresholds must satisfy "
                "0 <= warning < critical."
            )

        if not 0 < ks_p_value < 1:
            raise ValueError(
                "KS p-value threshold must be between 0 and 1."
            )

        if not 0 <= missing_rate_change <= 1:
            raise ValueError(
                "Missing-rate change must be between 0 and 1."
            )

        if bins < 2:
            raise ValueError("At least two PSI bins are required.")

        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_p_value = ks_p_value
        self.missing_rate_change = missing_rate_change
        self.ignored_columns = set(ignored_columns or [])
        self.bins = bins

    def _numeric_feature_report(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> dict:
        """Calculate drift metrics for a numeric feature."""

        ref = pd.to_numeric(
            reference, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        cur = pd.to_numeric(
            current, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ref) < 2 or len(cur) < 2:
            return {
                "feature_type": "numeric",
                "status": "INSUFFICIENT_DATA",
                "psi": None,
                "ks_statistic": None,
                "ks_p_value": None,
                "reference_mean": (
                    float(ref.mean()) if len(ref) else None
                ),
                "current_mean": (
                    float(cur.mean()) if len(cur) else None
                ),
                "reference_std": (
                    float(ref.std()) if len(ref) > 1 else None
                ),
                "current_std": (
                    float(cur.std()) if len(cur) > 1 else None
                ),
                "reference_valid_count": int(len(ref)),
                "current_valid_count": int(len(cur)),
            }

        psi = _calculate_psi(
            ref.to_numpy(),
            cur.to_numpy(),
            self.bins,
        )

        ks_result = ks_2samp(
            ref.to_numpy(),
            cur.to_numpy(),
        )

        status = _psi_status(
            psi,
            self.psi_warning,
            self.psi_critical,
        )

        if ks_result.pvalue < self.ks_p_value:
            if status == "STABLE":
                status = "WARNING"

        return {
            "feature_type": "numeric",
            "status": status,
            "psi": round(psi, 6),
            "ks_statistic": round(
                float(ks_result.statistic), 6
            ),
            "ks_p_value": round(
                float(ks_result.pvalue), 8
            ),
            "reference_mean": float(ref.mean()),
            "current_mean": float(cur.mean()),
            "reference_std": float(ref.std()),
            "current_std": float(cur.std()),
            "reference_valid_count": int(len(ref)),
            "current_valid_count": int(len(cur)),
        }

    def _categorical_feature_report(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> dict:
        """Calculate PSI for a categorical feature."""

        ref = reference.dropna().astype(str)
        cur = current.dropna().astype(str)

        if len(ref) == 0 or len(cur) == 0:
            return {
                "feature_type": "categorical",
                "status": "INSUFFICIENT_DATA",
                "psi": None,
                "reference_valid_count": int(len(ref)),
                "current_valid_count": int(len(cur)),
            }

        psi = _calculate_psi(
            ref.to_numpy(),
            cur.to_numpy(),
            self.bins,
        )

        return {
            "feature_type": "categorical",
            "status": _psi_status(
                psi,
                self.psi_warning,
                self.psi_critical,
            ),
            "psi": round(psi, 6),
            "reference_unique_values": int(ref.nunique()),
            "current_unique_values": int(cur.nunique()),
            "reference_valid_count": int(len(ref)),
            "current_valid_count": int(len(cur)),
        }

    def compare(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """
        Compare two DataFrames and return a drift report.

        Target columns and identifier columns should be excluded
        using ignored_columns when creating the detector.
        """

        if not isinstance(reference_df, pd.DataFrame):
            raise TypeError("Reference input must be a DataFrame.")

        if not isinstance(current_df, pd.DataFrame):
            raise TypeError("Current input must be a DataFrame.")

        if reference_df.empty or current_df.empty:
            raise ValueError("Both datasets must contain rows.")

        if not reference_df.columns.is_unique:
            raise ValueError(
                "Reference dataset contains duplicate column names."
            )

        if not current_df.columns.is_unique:
            raise ValueError(
                "Current dataset contains duplicate column names."
            )

        reference_columns = set(reference_df.columns)
        current_columns = set(current_df.columns)

        missing_columns = sorted(
            reference_columns - current_columns
        )

        new_columns = sorted(
            current_columns - reference_columns
        )

        common_columns = sorted(
            (reference_columns & current_columns)
            - self.ignored_columns
        )

        feature_reports = {}
        missing_reports = {}

        for column in common_columns:
            reference = reference_df[column]
            current = current_df[column]

            reference_missing = float(
                reference.isna().mean()
            )

            current_missing = float(
                current.isna().mean()
            )

            missing_change = (
                current_missing - reference_missing
            )

            missing_status = (
                "WARNING"
                if missing_change >= self.missing_rate_change
                else "STABLE"
            )

            missing_reports[column] = {
                "reference_missing_rate": round(
                    reference_missing, 6
                ),
                "current_missing_rate": round(
                    current_missing, 6
                ),
                "missing_rate_change": round(
                    missing_change, 6
                ),
                "status": missing_status,
            }

            if pd.api.types.is_numeric_dtype(reference):
                report = self._numeric_feature_report(
                    reference, current
                )
            else:
                report = self._categorical_feature_report(
                    reference, current
                )

            report["missing_value_status"] = missing_status

            feature_reports[column] = report

        statuses = [
            item["status"]
            for item in feature_reports.values()
        ]

        statuses.extend(
            item["status"]
            for item in missing_reports.values()
        )

        if missing_columns or new_columns:
            overall_status = "CRITICAL"
        elif "CRITICAL" in statuses:
            overall_status = "CRITICAL"
        elif "WARNING" in statuses:
            overall_status = "WARNING"
        elif "INSUFFICIENT_DATA" in statuses:
            overall_status = "WARNING"
        else:
            overall_status = "STABLE"

        drifted_features = [
            name
            for name, item in feature_reports.items()
            if item["status"] in ("WARNING", "CRITICAL")
        ]

        return _json_safe({
            "report_type": "data_drift",
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "overall_status": overall_status,
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "reference_columns": int(len(reference_columns)),
            "current_columns": int(len(current_columns)),
            "ignored_columns": sorted(
                self.ignored_columns
            ),
            "schema_drift": {
                "missing_columns": missing_columns,
                "new_columns": new_columns,
                "has_schema_drift": bool(
                    missing_columns or new_columns
                ),
            },
            "summary": {
                "features_compared": len(common_columns),
                "features_with_drift": len(drifted_features),
                "features_stable": sum(
                    item["status"] == "STABLE"
                    for item in feature_reports.values()
                ),
                "features_with_insufficient_data": sum(
                    item["status"] == "INSUFFICIENT_DATA"
                    for item in feature_reports.values()
                ),
            },
            "thresholds": {
                "psi_warning": self.psi_warning,
                "psi_critical": self.psi_critical,
                "ks_p_value": self.ks_p_value,
                "missing_rate_change": (
                    self.missing_rate_change
                ),
            },
            "feature_reports": feature_reports,
            "missing_value_reports": missing_reports,
            "drifted_features": drifted_features,
        })

    def compare_csv(
        self,
        reference_path: str | Path,
        current_path: str | Path,
    ) -> dict:
        """Load two CSV files and compare their distributions."""

        reference_path = Path(reference_path)
        current_path = Path(current_path)

        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference CSV not found: {reference_path}"
            )

        if not current_path.is_file():
            raise FileNotFoundError(
                f"Current CSV not found: {current_path}"
            )

        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)

        report = self.compare(
            reference_df,
            current_df,
        )

        report["reference_file"] = str(reference_path)
        report["current_file"] = str(current_path)

        return report

    @staticmethod
    def save_report(
        report: dict,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Save the drift report as JSON."""

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_path is None:
            timestamp = datetime.now(
                timezone.utc
            ).strftime("%Y%m%d_%H%M%S")

            output_path = (
                REPORT_DIR
                / f"data_drift_report_{timestamp}.json"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _json_safe(report),
                file,
                indent=4,
                allow_nan=False,
            )

        return output_path


def compare_datasets(
    reference_path: str | Path,
    current_path: str | Path,
    ignored_columns: Optional[list[str]] = None,
    output_path: Optional[str | Path] = None,
) -> dict:
    """Convenience function for comparing two CSV datasets."""

    detector = DataDriftDetector(
        ignored_columns=ignored_columns
    )

    report = detector.compare_csv(
        reference_path,
        current_path,
    )

    saved_path = detector.save_report(
        report,
        output_path,
    )

    report["report_path"] = str(saved_path)

    return report


def run_self_test() -> None:
    """Run controlled tests for stable data and detected drift."""

    rng = np.random.default_rng(42)

    reference = pd.DataFrame({
        "vibration": rng.normal(5.0, 1.0, 1000),
        "pressure": rng.normal(100.0, 5.0, 1000),
        "machine_type": rng.choice(
            ["A", "B", "C"], 1000
        ),
        "machine_id": np.arange(1000),
        "machine_failure": rng.integers(0, 2, 1000),
    })

    stable = reference.copy()

    shifted = pd.DataFrame({
        "vibration": rng.normal(9.0, 1.5, 1000),
        "pressure": rng.normal(120.0, 8.0, 1000),
        "machine_type": rng.choice(
            ["A", "B", "D"], 1000
        ),
        "machine_id": np.arange(1000, 2000),
        "machine_failure": rng.integers(0, 2, 1000),
    })

    detector = DataDriftDetector(
        ignored_columns=[
            "machine_id",
            "machine_failure",
        ]
    )

    stable_report = detector.compare(
        reference,
        stable,
    )

    shifted_report = detector.compare(
        reference,
        shifted,
    )

    print("\nDATA DRIFT DETECTOR SELF-TEST")
    print("=" * 55)

    print(
        "Stable dataset status:",
        stable_report["overall_status"],
    )

    print(
        "Shifted dataset status:",
        shifted_report["overall_status"],
    )

    print(
        "Shifted features:",
        shifted_report["drifted_features"],
    )

    assert stable_report["overall_status"] == "STABLE", (
        "Stable dataset test failed."
    )

    assert shifted_report["overall_status"] in (
        "WARNING",
        "CRITICAL",
    ), "Shifted dataset test failed."

    saved_path = detector.save_report(
        shifted_report,
        REPORT_DIR / "data_drift_self_test.json",
    )

    print("\nSelf-test report saved:", saved_path)
    print("\nALL DATA DRIFT SELF-TESTS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predictive maintenance data drift detector"
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the built-in drift detection tests",
    )

    parser.add_argument(
        "--reference",
        type=str,
        help="Path to the reference CSV",
    )

    parser.add_argument(
        "--current",
        type=str,
        help="Path to the incoming CSV",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    elif args.reference and args.current:
        result = compare_datasets(
            args.reference,
            args.current,
            ignored_columns=[
                "UDI",
                "Product ID",
                "Machine failure",
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF",
            ],
        )

        print("\nDATA DRIFT ANALYSIS")
        print("=" * 55)
        print("Overall status:", result["overall_status"])
        print("Features compared:", result["summary"]["features_compared"])
        print("Features with drift:", result["summary"]["features_with_drift"])
        print("Drifted features:", result["drifted_features"])
        print("Report:", result["report_path"])

    else:
        parser.print_help()

