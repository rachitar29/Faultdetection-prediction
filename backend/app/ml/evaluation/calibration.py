
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)


BASE_DIR = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "calibration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# PROBABILITY CALIBRATION ANALYSIS
# ==========================================================

def evaluate_calibration(
    model_name,
    y_test,
    probabilities,
    n_bins=10
):

    y_true = np.asarray(y_test)
    probabilities = np.asarray(probabilities)

    probabilities = np.clip(
        probabilities, 1e-7, 1 - 1e-7
    )

    if len(y_true) != len(probabilities):
        raise ValueError(
            "Labels and probabilities must have equal length."
        )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Test data must contain both classes."
        )

    # Brier score
    brier = brier_score_loss(
        y_true,
        probabilities
    )

    # Log loss
    loss = log_loss(
        y_true,
        probabilities
    )

    # Calibration curve
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="uniform"
    )

    # Expected calibration error
    bin_ids = np.minimum(
        (probabilities * n_bins).astype(int),
        n_bins - 1
    )

    ece = 0.0

    for bin_id in range(n_bins):

        mask = bin_ids == bin_id

        if np.any(mask):

            actual_rate = np.mean(y_true[mask])

            predicted_rate = np.mean(
                probabilities[mask]
            )

            ece += (
                np.sum(mask) / len(y_true)
            ) * abs(
                actual_rate - predicted_rate
            )

    results = {
        "model": model_name,
        "brier_score": float(brier),
        "log_loss": float(loss),
        "expected_calibration_error": float(ece)
    }

    print("\n" + "=" * 55)
    print("CALIBRATION ANALYSIS:", model_name)
    print("=" * 55)

    for name, value in results.items():

        if name != "model":
            print(f"{name}: {value:.5f}")

    return {
        "results": results,
        "fraction_positive": fraction_positive,
        "mean_predicted": mean_predicted
    }


# ==========================================================
# RELIABILITY DIAGRAM
# ==========================================================

def plot_reliability_diagram(
    model_name,
    calibration_data
):

    fraction_positive = calibration_data[
        "fraction_positive"
    ]

    mean_predicted = calibration_data[
        "mean_predicted"
    ]

    plt.figure(figsize=(8, 6))

    # Perfect calibration reference
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    # Actual model calibration
    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label=model_name
    )

    plt.xlabel("Mean Predicted Failure Probability")

    plt.ylabel("Actual Failure Frequency")

    plt.title(
        f"Reliability Diagram - {model_name}"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_reliability.png"
    )

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print("RELIABILITY DIAGRAM SAVED:", output_path)
