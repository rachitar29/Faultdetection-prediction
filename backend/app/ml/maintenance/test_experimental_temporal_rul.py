"""
Smoke test for experimental NASA C-MAPSS FD001 temporal RUL inference.

Compares:
1. Production RUL prediction using the latest sensor snapshot.
2. Experimental temporal RUL prediction using complete engine history.

This is a smoke/comparison test, not independent validation.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.maintenance.experimental_temporal_rul import (
    predict_experimental_temporal_rul,
)

from app.ml.maintenance.rul_prediction import (
    predict_single_machine_rul,
)


BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    BASE_DIR
    / "ml_data"
    / "raw"
    / "CMAPSSData"
    / "test_FD001.txt"
)

FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CMAPSS_COLUMNS = ["unit", "cycle"] + FEATURE_COLUMNS


class TestExperimentalTemporalRUL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"NASA dataset not found: {DATASET_PATH}"
            )

        cls.dataset = pd.read_csv(
            DATASET_PATH,
            sep=r"\s+",
            header=None,
            names=CMAPSS_COLUMNS,
        )

        if cls.dataset.empty:
            raise ValueError("NASA FD001 dataset is empty.")

        if cls.dataset[CMAPSS_COLUMNS].isna().any().any():
            raise ValueError(
                "NASA dataset contains missing values."
            )

    def test_temporal_and_production_prediction(self):

        engine_id = self.dataset["unit"].iloc[0]

        history = (
            self.dataset[
                self.dataset["unit"] == engine_id
            ]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        self.assertGreater(len(history), 1)

        # Experimental temporal prediction using full history.
        temporal_result = predict_experimental_temporal_rul(
            history
        )

        temporal_rul = float(
            temporal_result["remaining_useful_life"]
        )

        self.assertTrue(np.isfinite(temporal_rul))
        self.assertGreaterEqual(temporal_rul, 0.0)
        self.assertTrue(temporal_result["experimental"])
        self.assertFalse(temporal_result["production_approved"])
        self.assertTrue(
            temporal_result["artifact_integrity_verified"]
        )
        self.assertEqual(
            temporal_result["feature_count"],
            88,
        )
        self.assertEqual(
            temporal_result["history_rows_used"],
            len(history),
        )

        # Production baseline using latest 24-feature snapshot.
        latest_snapshot = (
            history.tail(1)[FEATURE_COLUMNS].copy()
        )

        production_result = predict_single_machine_rul(
            latest_snapshot
        )

        if not isinstance(production_result, dict):
            raise TypeError(
                "Expected production prediction to be a dictionary; "
                f"received {type(production_result).__name__}."
            )

        production_rul = float(
            production_result["remaining_useful_life"]
        )

        self.assertTrue(np.isfinite(production_rul))
        self.assertGreaterEqual(production_rul, 0.0)

        # Display comparison.
        print("\n")
        print("=" * 65)
        print("NASA FD001 TEMPORAL RUL SMOKE TEST")
        print("=" * 65)

        print(f"Engine ID:                 {engine_id}")
        print(f"Cycles used:               {len(history)}")
        print(
            f"Latest cycle:              "
            f"{int(history['cycle'].iloc[-1])}"
        )

        print("-" * 65)

        print(
            f"Production baseline RUL:   "
            f"{production_rul:.2f} cycles"
        )

        print(
            f"Experimental temporal RUL: "
            f"{temporal_rul:.2f} cycles"
        )

        print(
            f"Production category:       "
            f"{production_result['rul_category']}"
        )

        print(
            f"Experimental category:     "
            f"{temporal_result['rul_category']}"
        )

        print(
            f"Experimental priority:     "
            f"{temporal_result['maintenance_priority']}"
        )

        print(
            f"Temporal features:         "
            f"{temporal_result['feature_count']}"
        )

        print(
            f"Artifact integrity:        "
            f"{temporal_result['artifact_integrity_verified']}"
        )

        print("=" * 65)
        print(
            "Comparison only; not independent validation "
            "or production approval."
        )
        print("=" * 65)


if __name__ == "__main__":
    unittest.main(verbosity=2)
