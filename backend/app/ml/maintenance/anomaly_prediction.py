
"""
Anomaly prediction service for AI4I predictive maintenance.

Uses registered Isolation Forest and Local Outlier Factor models.
Uses optimized validation thresholds saved in model metadata.
Supports single and batch sensor readings.
Does not retrain models or modify the model registry.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.algorithms.model_registry import load_model_bundle


ISOLATION_FOREST_NAME = "Isolation Forest"
LOF_NAME = "Local Outlier Factor"

NORMAL = "NORMAL"
WARNING = "WARNING"
ANOMALY = "ANOMALY"


class AnomalyPredictionService:
    """Detect unusual equipment readings using registered models."""

    def __init__(self):
        self.isolation_bundle = load_model_bundle(ISOLATION_FOREST_NAME)
        self.lof_bundle = load_model_bundle(LOF_NAME)

        self.isolation_model = self.isolation_bundle["model"]
        self.lof_model = self.lof_bundle["model"]

        isolation_metadata = self.isolation_bundle["metadata"]
        lof_metadata = self.lof_bundle["metadata"]

        self.isolation_features = isolation_metadata.get("feature_names", [])
        self.lof_features = lof_metadata.get("feature_names", [])

        self.isolation_threshold = isolation_metadata.get("threshold")
        self.lof_threshold = lof_metadata.get("threshold")

        if not self.isolation_features or not self.lof_features:
            raise ValueError("Feature names are missing from anomaly model metadata.")

        if self.isolation_features != self.lof_features:
            raise ValueError("Anomaly models expect different feature schemas.")

        if self.isolation_threshold is None or self.lof_threshold is None:
            raise ValueError("Optimized thresholds are missing from model metadata.")

        self.isolation_threshold = float(self.isolation_threshold)
        self.lof_threshold = float(self.lof_threshold)

        if not np.isfinite(self.isolation_threshold):
            raise ValueError("Isolation Forest threshold must be finite.")

        if not np.isfinite(self.lof_threshold):
            raise ValueError("LOF threshold must be finite.")

        self.feature_names = self.isolation_features

    def _prepare_input(self, sensor_data: Any) -> pd.DataFrame:
        """Convert supported inputs into a validated DataFrame."""

        if isinstance(sensor_data, pd.DataFrame):
            dataframe = sensor_data.copy()
        elif isinstance(sensor_data, dict):
            dataframe = pd.DataFrame([sensor_data])
        elif isinstance(sensor_data, list):
            dataframe = pd.DataFrame(sensor_data)
        else:
            raise TypeError(
                "Input must be a pandas DataFrame, dictionary, or list."
            )

        if dataframe.empty:
            raise ValueError("No sensor readings were provided.")

        missing = [
            feature for feature in self.feature_names
            if feature not in dataframe.columns
        ]

        if missing:
            raise ValueError(f"Missing required sensor features: {missing}")

        dataframe = dataframe[self.feature_names].copy()

        numeric_features = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        for feature in numeric_features:
            if feature not in dataframe.columns:
                raise ValueError(f"Required numeric feature is missing: {feature}")

            if dataframe[feature].map(
                lambda value: isinstance(value, (bool, np.bool_))
            ).any():
                raise ValueError(f"{feature} must contain numeric values, not booleans.")

            dataframe[feature] = pd.to_numeric(
                dataframe[feature], errors="coerce"
            )

            if not np.isfinite(dataframe[feature].to_numpy(dtype=float)).all():
                raise ValueError(
                    f"{feature} contains missing or non-finite values."
                )

        if "Type" in dataframe.columns:
            if dataframe["Type"].isna().any():
                raise ValueError("Machine Type contains missing values.")

            dataframe["Type"] = dataframe["Type"].map(
                lambda value: value.strip().upper()
                if isinstance(value, str) else value
            )

            invalid_types = ~dataframe["Type"].isin(["L", "M", "H"])

            if invalid_types.any():
                raise ValueError("Machine Type must be L, M, or H.")

        return dataframe

    @staticmethod
    def _get_anomaly_scores(model, dataframe) -> np.ndarray:
        """Return scores where larger values indicate greater abnormality."""

        scores = -np.asarray(
            model.decision_function(dataframe),
            dtype=float
        ).reshape(-1)

        if not np.isfinite(scores).all():
            raise RuntimeError("Anomaly model returned non-finite scores.")

        return scores

    def predict_anomalies(self, sensor_data: Any) -> dict:
        """Run both detectors using their saved optimized thresholds."""

        dataframe = self._prepare_input(sensor_data)

        isolation_scores = self._get_anomaly_scores(
            self.isolation_model, dataframe
        )
        lof_scores = self._get_anomaly_scores(
            self.lof_model, dataframe
        )

        isolation_anomaly = isolation_scores >= self.isolation_threshold
        lof_anomaly = lof_scores >= self.lof_threshold

        combined_status = []

        for isolation_flag, lof_flag in zip(
            isolation_anomaly, lof_anomaly
        ):
            if isolation_flag and lof_flag:
                combined_status.append(ANOMALY)
            elif isolation_flag or lof_flag:
                combined_status.append(WARNING)
            else:
                combined_status.append(NORMAL)

        results = pd.DataFrame({
            "isolation_forest_score": isolation_scores,
            "isolation_forest_threshold": self.isolation_threshold,
            "isolation_forest_anomaly": isolation_anomaly,
            "lof_score": lof_scores,
            "lof_threshold": self.lof_threshold,
            "lof_anomaly": lof_anomaly,
            "combined_anomaly_status": combined_status,
        })

        results.insert(0, "reading_id", range(1, len(results) + 1))

        total = len(results)

        summary = {
            "total_readings": total,
            "normal_readings": int(
                (results["combined_anomaly_status"] == NORMAL).sum()
            ),
            "warning_readings": int(
                (results["combined_anomaly_status"] == WARNING).sum()
            ),
            "anomaly_readings": int(
                (results["combined_anomaly_status"] == ANOMALY).sum()
            ),
            "isolation_forest_anomalies": int(isolation_anomaly.sum()),
            "lof_anomalies": int(lof_anomaly.sum()),
        }

        return {
            "models": {
                "isolation_forest": ISOLATION_FOREST_NAME,
                "local_outlier_factor": LOF_NAME,
            },
            "features_used": self.feature_names,
            "thresholds": {
                "isolation_forest": self.isolation_threshold,
                "local_outlier_factor": self.lof_threshold,
            },
            "threshold_source": "Saved validation thresholds",
            "summary": summary,
            "predictions": results,
        }


def predict_anomalies(sensor_data: Any) -> dict:
    """Convenience function for other project modules."""
    service = AnomalyPredictionService()
    return service.predict_anomalies(sensor_data)


def run_self_test():
    """Test the service using AI4I sensor rows."""

    backend_dir = Path(__file__).resolve().parents[3]

    dataset_path = backend_dir / "ml_data" / "raw" / "ai4i2020.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"AI4I dataset not found: {dataset_path}")

    dataframe = pd.read_csv(dataset_path)

    ignored_columns = [
        "UDI", "Product ID", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF",
    ]

    sensor_dataframe = dataframe.drop(
        columns=ignored_columns, errors="ignore"
    ).head(10)

    service = AnomalyPredictionService()
    result = service.predict_anomalies(sensor_dataframe)

    print("\nANOMALY DETECTION SELF-TEST")
    print("=" * 60)
    print("Models:", result["models"])
    print("Features:", result["features_used"])
    print("Thresholds:", result["thresholds"])
    print("Threshold source:", result["threshold_source"])

    print("\nSUMMARY")
    for key, value in result["summary"].items():
        print(f"{key}: {value}")

    print("\nROW-WISE RESULTS")
    print(result["predictions"].to_string(index=False))

    print("\nANOMALY DETECTION SELF-TEST COMPLETED")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_self_test()
