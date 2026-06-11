"""
Phase 4 plotting utilities.

All figures use the Agg backend so this module runs cleanly on Windows,
headless terminals, CI jobs, and background scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


sns.set_theme(style="whitegrid", context="notebook")


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_class_distribution(y: pd.Series | np.ndarray, output_dir: str | Path) -> Path:
    """Save a countplot showing the original class imbalance."""

    output_path = _prepare_output_dir(output_dir)
    class_series = pd.Series(y, name="Class").map({0: "Legitimate", 1: "Fraud"})

    plt.figure(figsize=(7, 5))
    axis = sns.countplot(x=class_series, hue=class_series, palette=["#4C78A8", "#E45756"], legend=False)
    axis.set_title("Class Distribution Before Balancing")
    axis.set_xlabel("Transaction Class")
    axis.set_ylabel("Count")

    total = len(class_series)
    for patch in axis.patches:
        count = int(patch.get_height())
        percent = count / total * 100
        axis.annotate(
            f"{count:,}\n({percent:.2f}%)",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    file_path = output_path / "class_distribution.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_confusion_matrix(cm: np.ndarray, output_dir: str | Path) -> Path:
    """Save a confusion matrix heatmap with original values."""

    output_path = _prepare_output_dir(output_dir)

    plt.figure(figsize=(6, 5))
    axis = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    axis.set_title("Confusion Matrix - Random Forest")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    plt.tight_layout()
    file_path = output_path / "confusion_matrix_heatmap.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_roc_curve(y_true: pd.Series | np.ndarray, y_score: np.ndarray, output_dir: str | Path) -> Path:
    """Save ROC-AUC curve."""

    output_path = _prepare_output_dir(output_dir)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1F77B4", linewidth=2.5, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random baseline")
    plt.title("ROC-AUC Curve - Random Forest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    file_path = output_path / "roc_auc_curve.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_precision_recall_curve(y_true: pd.Series | np.ndarray, y_score: np.ndarray, output_dir: str | Path) -> Path:
    """Save precision-recall curve, which is critical for imbalanced datasets."""

    output_path = _prepare_output_dir(output_dir)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    fraud_rate = float(np.mean(y_true))

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#F58518", linewidth=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    plt.axhline(fraud_rate, color="gray", linestyle="--", linewidth=1.5, label=f"Fraud rate = {fraud_rate:.4f}")
    plt.title("Precision-Recall Curve - Random Forest")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.tight_layout()

    file_path = output_path / "precision_recall_curve.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_feature_importance(model, feature_names: list[str], output_dir: str | Path, top_n: int = 20) -> Path:
    """Save top feature importances from RandomForestClassifier."""

    output_path = _prepare_output_dir(output_dir)
    importances = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(9, max(5, top_n * 0.35)))
    axis = sns.barplot(data=importances, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    axis.set_title(f"Top {top_n} Feature Importances - Random Forest")
    axis.set_xlabel("Mean Decrease in Impurity")
    axis.set_ylabel("Feature")
    plt.tight_layout()

    file_path = output_path / "feature_importance.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path


def plot_error_curve(final_error: float, output_dir: str | Path, epochs: int = 20) -> Path:
    """Save a simulated training error reduction curve for report visualization."""

    output_path = _prepare_output_dir(output_dir)
    epoch_values = np.arange(1, epochs + 1)
    start_error = min(0.5, max(final_error + 0.20, 0.25))
    decay = np.exp(-np.linspace(0, 4, epochs))
    errors = final_error + (start_error - final_error) * decay

    plt.figure(figsize=(7, 5))
    plt.plot(epoch_values, errors, marker="o", color="#54A24B", linewidth=2)
    plt.title("Simulated Training Error Reduction")
    plt.xlabel("Training Iteration")
    plt.ylabel("Error Rate")
    plt.xticks(epoch_values)
    plt.tight_layout()

    file_path = output_path / "loss_error_graph.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path
