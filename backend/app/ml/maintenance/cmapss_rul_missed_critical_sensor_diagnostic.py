from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_FILE = (
    BASE_DIR
    / "ml_data"
    / "raw"
    / "CMAPSSData"
    / "train_FD001.txt"
)

PREDICTIONS_FILE = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
    / "cmapss_fd001_multisplit_validation_predictions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "ml_artifacts"
    / "reports"
    / "cmapss_fd001_missed_critical_sensor_diagnostic.csv"
)

SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_9",
    "sensor_12",
    "sensor_14",
    "sensor_17",
]

WINDOWS = [10, 30]

MISSED_CASES = [
    (1, 77, 145),
    (4, 10, 213),
    (5, 59, 222),
]

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    data = pd.read_csv(
        DATA_FILE,
        sep=r"\s+",
        header=None
    )

    data = data.dropna(axis=1, how="all")
    data.columns = COLUMN_NAMES

    return data.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)


def rolling_slope(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    y = values - values.mean()

    denominator = np.dot(x, x)

    if denominator == 0:
        return 0.0

    return float(10.0 * np.dot(x, y) / denominator)


def calculate_features(history):
    row = {
        "observed_cycle": int(history["cycle"].iloc[-1]),
        "history_length": len(history),
    }

    for sensor in SENSORS:
        values = history[sensor].astype(float)

        row[f"{sensor}_current"] = float(values.iloc[-1])

        for window in WINDOWS:
            recent = values.tail(window)

            row[f"{sensor}_mean_{window}"] = float(
                recent.mean()
            )

            row[f"{sensor}_std_{window}"] = float(
                recent.std(ddof=0)
            )

            row[f"{sensor}_slope_{window}"] = rolling_slope(
                recent.to_numpy()
            )

            row[f"{sensor}_deviation_{window}"] = float(
                recent.iloc[-1] - recent.mean()
            )

    return row


def main():
    print("Loading original FD001 training data...")
    data = load_data()

    print("Loading existing validation predictions...")
    predictions = pd.read_csv(PREDICTIONS_FILE)

    predictions["fold"] = predictions["fold"].astype(int)
    predictions["engine_id"] = predictions["engine_id"].astype(int)
    predictions["last_observed_cycle"] = (
        predictions["last_observed_cycle"].astype(int)
    )

    evaluation = predictions[
        predictions["partition"] == "evaluation"
    ].copy()

    evaluation["actual_rul"] = pd.to_numeric(
        evaluation["actual_rul"]
    )

    evaluation["temporal_raw_predicted_rul"] = pd.to_numeric(
        evaluation["temporal_raw_predicted_rul"]
    )

    evaluation["threshold_15_detected"] = (
        evaluation["temporal_raw_predicted_rul"] <= 15
    )

    selected_rows = []

    for fold, engine, cycle in MISSED_CASES:
        selected_rows.append({
            "fold": fold,
            "engine_id": engine,
            "last_observed_cycle": cycle,
            "comparison_group": "MISSED_CRITICAL",
        })

    # Select correctly detected critical snapshots from the same folds.
    for fold in sorted({item[0] for item in MISSED_CASES}):
        controls = evaluation[
            (evaluation["fold"] == fold)
            & (evaluation["actual_rul"] <= 10)
            & (evaluation["temporal_raw_predicted_rul"] <= 15)
        ].copy()

        # Choose up to three controls nearest to the missed cases'
        # actual RUL and predicted RUL ranges.
        missed_predictions = evaluation[
            (evaluation["fold"] == fold)
            & (evaluation["actual_rul"] <= 10)
            & (evaluation["temporal_raw_predicted_rul"] > 15)
        ]

        if missed_predictions.empty:
            continue

        reference_prediction = float(
            missed_predictions["temporal_raw_predicted_rul"].iloc[0]
        )

        controls["comparison_distance"] = (
            (controls["actual_rul"] - 9).abs()
            + 0.25 * (
                controls["temporal_raw_predicted_rul"]
                - reference_prediction
            ).abs()
        )

        controls = controls.sort_values(
            "comparison_distance"
        ).drop_duplicates("engine_id").head(3)

        for _, control in controls.iterrows():
            selected_rows.append({
                "fold": fold,
                "engine_id": int(control["engine_id"]),
                "last_observed_cycle": int(
                    control["last_observed_cycle"]
                ),
                "comparison_group": "DETECTED_CRITICAL_CONTROL",
            })

    results = []

    for selected in selected_rows:
        engine_id = selected["engine_id"]
        cycle = selected["last_observed_cycle"]

        history = data[
            (data["unit"] == engine_id)
            & (data["cycle"] <= cycle)
        ].sort_values("cycle")

        if history.empty or int(history["cycle"].iloc[-1]) != cycle:
            print(
                f"WARNING: Exact history unavailable for "
                f"engine {engine_id}, cycle {cycle}"
            )
            continue

        features = calculate_features(history)

        results.append({
            **selected,
            **features,
        })

    report = pd.DataFrame(results)

    if report.empty:
        raise RuntimeError("No diagnostic rows were generated.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nDIAGNOSTIC SUMMARY")
    print("==================")

    print(
        report[
            [
                "fold",
                "engine_id",
                "last_observed_cycle",
                "comparison_group",
                "history_length",
            ]
        ].to_string(index=False)
    )

    print("\nSENSOR FEATURE COMPARISON")
    print("=========================")

    summary_columns = [
        column
        for column in report.columns
        if any(
            column.startswith(sensor + "_")
            for sensor in SENSORS
        )
    ]

    grouped = report.groupby(
        "comparison_group"
    )[summary_columns].mean().T

    print(grouped.round(4).to_string())

    print("\nSaved diagnostic report:")
    print(OUTPUT_FILE)

    print("\nNo model was trained or modified.")
    print("No existing report was overwritten.")


if __name__ == "__main__":
    main()
