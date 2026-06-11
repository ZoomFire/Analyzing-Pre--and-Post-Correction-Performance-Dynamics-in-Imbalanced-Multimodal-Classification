"""
Phase 4 metric utilities for the Credit Card Fraud Detection project.

This module keeps all evaluation logic separate from plotting and training so
the results can be reused in reports, notebooks, and comparison charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_positive_scores(model: Any, X_test: np.ndarray) -> np.ndarray:
    """Return fraud-class scores for models with predict_proba or decision_function."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    return model.predict(X_test)


def evaluate_binary_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Compute classification report, confusion matrix, and headline metrics."""

    y_pred = model.predict(X_test)
    y_score = predict_positive_scores(model, X_test)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": report,
        "y_pred": y_pred,
        "y_score": y_score,
    }

    return metrics


def metrics_row(model_name: str, evaluation: dict[str, Any]) -> dict[str, float | str]:
    """Convert an evaluation dictionary into one row for comparison tables."""

    return {
        "Model": model_name,
        "Accuracy": evaluation["accuracy"],
        "Precision": evaluation["precision"],
        "Recall": evaluation["recall"],
        "F1-score": evaluation["f1_score"],
        "ROC-AUC": evaluation["roc_auc"],
    }


def save_classification_outputs(
    evaluation: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "random_forest",
) -> tuple[Path, Path]:
    """Save classification report and summary metrics as CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(evaluation["classification_report"]).transpose()
    report_file = output_path / f"{prefix}_classification_report.csv"
    report_df.to_csv(report_file, index=True)

    summary_df = pd.DataFrame(
        [
            {
                "accuracy": evaluation["accuracy"],
                "precision": evaluation["precision"],
                "recall": evaluation["recall"],
                "f1_score": evaluation["f1_score"],
                "roc_auc": evaluation["roc_auc"],
            }
        ]
    )
    summary_file = output_path / f"{prefix}_summary_metrics.csv"
    summary_df.to_csv(summary_file, index=False)

    return report_file, summary_file
