
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)

BASE_DIR = Path(__file__).resolve().parents[3]
PLOT_DIR = BASE_DIR / "ml_artifacts" / "evaluation_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_curve(model_name, y_true, probabilities):
    fpr, tpr, _ = roc_curve(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.tight_layout()

    path = PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_roc.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved ROC curve: {path}")


def plot_precision_recall(model_name, y_true, probabilities):
    precision, recall, _ = precision_recall_curve(
        y_true, probabilities
    )
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    path = PLOT_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved PR curve: {path}")


def plot_confusion_matrix(model_name, y_true, predictions):
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Normal", "Failure"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(f"{model_name} Confusion Matrix")
    plt.tight_layout()

    path = PLOT_DIR / (
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {path}")
