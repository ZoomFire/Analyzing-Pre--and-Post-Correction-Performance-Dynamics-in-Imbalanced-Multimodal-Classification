"""
Model comparison utilities for Phase 4.

The comparison graph benchmarks the baseline, Random Forest, and major
class-balancing strategies using Recall and F1-score, the two most relevant
metrics for highly imbalanced fraud detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.phase4_evaluation.metrics import evaluate_binary_classifier, metrics_row


def _load_resamplers() -> dict[str, Callable]:
    """Import imbalanced-learn lazily so the error message is clear."""

    try:
        from imblearn.combine import SMOTEENN, SMOTETomek
        from imblearn.over_sampling import ADASYN, SMOTE
    except ImportError as exc:
        raise ImportError(
            "Phase 4 comparison requires imbalanced-learn. Install it with: "
            "python -m pip install imbalanced-learn"
        ) from exc

    return {
        "SMOTE": SMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42),
    }


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create the final evaluation model with stable, reproducible settings."""

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def run_model_comparison(
    X_train,
    X_test,
    y_train,
    y_test,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train comparison models and return a metrics table."""

    rows = []

    baseline = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    baseline.fit(X_train, y_train)
    rows.append(metrics_row("Baseline", evaluate_binary_classifier(baseline, X_test, y_test)))

    random_forest = build_random_forest(random_state)
    random_forest.fit(X_train, y_train)
    rows.append(metrics_row("Random Forest", evaluate_binary_classifier(random_forest, X_test, y_test)))

    for technique, sampler in _load_resamplers().items():
        print(f"\nApplying {technique} and training Random Forest...")
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        model = build_random_forest(random_state)
        model.fit(X_resampled, y_resampled)
        rows.append(metrics_row(technique, evaluate_binary_classifier(model, X_test, y_test)))

    return pd.DataFrame(rows)


def save_comparison_table(comparison_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save comparison metrics as CSV."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "phase4_model_comparison.csv"
    comparison_df.to_csv(file_path, index=False)
    return file_path


def plot_comparison_graph(comparison_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save Recall and F1-score comparison graph."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plot_df = comparison_df.melt(
        id_vars="Model",
        value_vars=["Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6))
    axis = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette=["#E45756", "#4C78A8"])
    axis.set_title("Phase 4 Comparison: Recall and F1-score")
    axis.set_xlabel("Model / Balancing Technique")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")

    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    plt.tight_layout()
    file_path = output_path / "comparison_recall_f1.png"
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path
