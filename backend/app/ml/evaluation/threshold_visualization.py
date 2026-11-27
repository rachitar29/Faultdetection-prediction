
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "ml_artifacts" / "trained_models"

CSV_PATH = MODEL_DIR / "model_comparison.csv"

OUTPUT_DIR = (
    BASE_DIR
    / "ml_artifacts"
    / "threshold_visualizations"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD RESULTS
# ==========================================================

def load_comparison_results():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Comparison CSV not found: {CSV_PATH}\n"
            "Run comparison.py first."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "threshold",
        "threshold_type"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in comparison CSV: {missing}"
        )

    print("\nCOMPARISON RESULTS LOADED")
    print(df.to_string(index=False))

    return df


# ==========================================================
# 1. F1-SCORE COMPARISON
# ==========================================================

def plot_f1_comparison(df):

    pivot = df.pivot(
        index="model",
        columns="threshold_type",
        values="f1_score"
    )

    pivot = pivot.reindex(
        columns=["Default", "Optimized"]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
        width=0.7
    )

    ax.set_title(
        "Default vs Optimized Threshold - F1 Score"
    )

    ax.set_xlabel("Classification Model")
    ax.set_ylabel("F1 Score")

    ax.set_ylim(0, 1)
    ax.legend(title="Threshold")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "f1_threshold_comparison.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print("F1 CHART SAVED:", path)


# ==========================================================
# 2. PRECISION AND RECALL COMPARISON
# ==========================================================

def plot_precision_recall(df):

    metrics = ["precision", "recall"]

    models = df["model"].unique()

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(12, 5),
        squeeze=False
    )

    for ax, model_name in zip(axes[0], models):

        model_df = df[df["model"] == model_name]

        pivot = model_df.set_index(
            "threshold_type"
        )[metrics]

        pivot = pivot.reindex(
            ["Default", "Optimized"]
        )

        pivot.plot(
            kind="bar",
            ax=ax,
            width=0.7
        )

        ax.set_title(model_name)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        ax.tick_params(axis="x", rotation=0)

    plt.suptitle(
        "Precision and Recall Comparison"
    )

    plt.tight_layout()

    path = OUTPUT_DIR / "precision_recall_comparison.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print("PRECISION RECALL CHART SAVED:", path)


# ==========================================================
# 3. OPTIMAL THRESHOLD COMPARISON
# ==========================================================

def plot_threshold_comparison(df):

    pivot = df.pivot(
        index="model",
        columns="threshold_type",
        values="threshold"
    )

    pivot = pivot.reindex(
        columns=["Default", "Optimized"]
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
        width=0.7
    )

    ax.set_title(
        "Default vs Optimized Classification Threshold"
    )

    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Threshold")

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "threshold_comparison.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print("THRESHOLD CHART SAVED:", path)


# ==========================================================
# 4. OVERALL MODEL PERFORMANCE
# ==========================================================

def plot_model_performance(df):

    optimized_df = df[
        df["threshold_type"] == "Optimized"
    ].copy()

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    ax = optimized_df.set_index(
        "model"
    )[metrics].plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75
    )

    ax.set_title(
        "Model Performance Using Optimized Thresholds"
    )

    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Score")

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "optimized_model_performance.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print("MODEL PERFORMANCE CHART SAVED:", path)


# ==========================================================
# 5. ROC-AUC AND PR-AUC COMPARISON
# ==========================================================

def plot_auc_comparison(df):

    metrics = ["roc_auc", "pr_auc"]

    if not all(col in df.columns for col in metrics):
        print("ROC-AUC / PR-AUC columns missing. Skipping.")
        return

    optimized_df = df[
        df["threshold_type"] == "Optimized"
    ]

    ax = optimized_df.set_index(
        "model"
    )[metrics].plot(
        kind="bar",
        figsize=(9, 5),
        width=0.7
    )

    ax.set_title(
        "ROC-AUC and PR-AUC Model Comparison"
    )

    ax.set_xlabel("Classification Model")
    ax.set_ylabel("Score")

    ax.set_ylim(0, 1)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    path = OUTPUT_DIR / "auc_comparison.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print("AUC CHART SAVED:", path)


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def run_threshold_visualization():

    print("\n" + "=" * 60)
    print("THRESHOLD VISUALIZATION")
    print("=" * 60)

    df = load_comparison_results()

    plot_f1_comparison(df)

    plot_precision_recall(df)

    plot_threshold_comparison(df)

    plot_model_performance(df)

    plot_auc_comparison(df)

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED")
    print("=" * 60)

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run_threshold_visualization()
