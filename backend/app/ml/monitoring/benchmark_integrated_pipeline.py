
"""Benchmark the complete integrated equipment prediction pipeline."""

import argparse
import json
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.ml.data.data_loader import load_dataset
from app.ml.data.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from app.ml.ingestion.prediction_bridge import (
    predict_uploaded_file,
)


BACKEND_DIR = Path(__file__).resolve().parents[3]

REPORT_PATH = (
    BACKEND_DIR
    / "ml_artifacts"
    / "monitoring"
    / "integrated_pipeline_benchmark.json"
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def benchmark_pipeline(sizes):
    """Benchmark end-to-end ingestion and integrated prediction."""

    print("\nINTEGRATED PIPELINE PERFORMANCE BENCHMARK")
    print("=" * 65)

    # Load source dataset once.
    dataset = load_dataset()

    missing = [
        feature for feature in FEATURES
        if feature not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"AI4I dataset is missing features: {missing}"
        )

    sensor_dataset = dataset.loc[:, FEATURES].copy()

    results = []

    with tempfile.TemporaryDirectory() as temp_directory:

        temp_file = Path(temp_directory) / "benchmark_sensor_data.csv"

        # Warm up model and SHAP caches.
        print("\nWARM-UP")

        warmup_data = sensor_dataset.head(1)
        warmup_data.to_csv(temp_file, index=False)

        warmup_start = time.perf_counter()

        predict_uploaded_file(temp_file)

        warmup_seconds = time.perf_counter() - warmup_start

        print(f"Warm-up completed: {warmup_seconds:.4f} seconds")

        # Run requested benchmark sizes.
        for size in sizes:

            if size <= 0 or size > len(sensor_dataset):
                raise ValueError(
                    f"Invalid benchmark size: {size}. "
                    f"Available rows: {len(sensor_dataset)}"
                )

            sample = sensor_dataset.sample(
                n=size,
                random_state=42
            ).copy()

            sample.to_csv(
                temp_file,
                index=False
            )

            print("\n" + "-" * 65)
            print(f"BENCHMARK SIZE: {size} READINGS")

            start_time = time.perf_counter()

            prediction = predict_uploaded_file(temp_file)

            elapsed = time.perf_counter() - start_time

            throughput = (
                size / elapsed if elapsed > 0 else 0
            )

            average_time = elapsed / size

            result = {
                "readings": size,
                "total_seconds": round(elapsed, 6),
                "average_seconds_per_reading": round(
                    average_time, 6
                ),
                "readings_per_second": round(
                    throughput, 3
                ),
                "predicted_failures": (
                    prediction["predicted_failures"]
                ),
                "predicted_normal": (
                    prediction["predicted_normal"]
                ),
                "readings_with_anomalies": (
                    prediction["readings_with_anomalies"]
                ),
                "risk_distribution": (
                    prediction["risk_distribution"]
                ),
            }

            results.append(result)

            print("Total time:", result["total_seconds"], "seconds")
            print(
                "Average per reading:",
                result["average_seconds_per_reading"],
                "seconds"
            )
            print(
                "Throughput:",
                result["readings_per_second"],
                "readings/second"
            )
            print(
                "Predicted failures:",
                result["predicted_failures"]
            )
            print(
                "Readings with anomalies:",
                result["readings_with_anomalies"]
            )

    report = {
        "benchmark_type": "integrated_equipment_prediction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "AI4I 2020",
        "warmup_seconds": round(warmup_seconds, 6),
        "model_thresholds_unchanged": True,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("Report saved:", REPORT_PATH)
    print("=" * 65)

    return report


def main():

    parser = argparse.ArgumentParser(
        description="Integrated prediction pipeline benchmark"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[3, 100, 1000],
        help="Number of sensor readings to benchmark"
    )

    args = parser.parse_args()

    benchmark_pipeline(args.sizes)


if __name__ == "__main__":
    main()
