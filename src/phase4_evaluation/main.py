"""
Phase 4 Evaluation Pipeline - Credit Card Fraud Detection.

Run from the project root:
    python -m src.phase4_evaluation.main

Expected dataset:
    datasets/tabular/creditcard.csv

The target column must be named:
    Class
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.phase4_evaluation.comparison import (
    plot_comparison_graph,
    run_model_comparison,
    save_comparison_table,
)
from src.phase4_evaluation.metrics import evaluate_binary_classifier, save_classification_outputs
from src.phase4_evaluation.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_error_curve,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "tabular" / "creditcard.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TARGET_COLUMN = "Class"


def resolve_path(path: str | Path) -> Path:
    """Resolve relative paths from the project root."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def load_credit_card_data(data_path: str | Path) -> pd.DataFrame:
    """Load and validate the Credit Card Fraud Detection CSV dataset."""

    csv_path = resolve_path(data_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download creditcard.csv and place it at datasets/tabular/creditcard.csv, "
            "or pass --data-path with the correct CSV path."
        )

    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {csv_path}.")

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Phase 4 expects binary classification with Class values 0 and 1.")

    return df


def preprocess_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str]]:
    """Split and scale numeric features while preserving class stratification."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"All features must be numeric. Non-numeric columns found: {non_numeric}")

    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


def train_final_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train the final Phase 4 RandomForestClassifier."""

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise imbalance summary for terminal output."""

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    print("\n========== DATASET SUMMARY ==========")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Features: {df.shape[1] - 1:,}")
    print("\nClass distribution:")
    for class_id in counts.index:
        label = "Legitimate" if class_id == 0 else "Fraud"
        print(f"  {class_id} ({label}): {counts[class_id]:,} ({percentages[class_id]:.4f}%)")


def print_phase4_metrics(evaluation: dict) -> None:
    """Print headline Phase 4 evaluation metrics."""

    print("\n========== PHASE 4 FINAL RANDOM FOREST METRICS ==========")
    print(f"Accuracy : {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall   : {evaluation['recall']:.4f}")
    print(f"F1-score : {evaluation['f1_score']:.4f}")
    print(f"ROC-AUC  : {evaluation['roc_auc']:.4f}")

    print("\nClassification Report:")
    report_df = pd.DataFrame(evaluation["classification_report"]).transpose()
    print(report_df.round(4))


def run_phase4(args: argparse.Namespace) -> None:
    """Execute complete Phase 4 evaluation workflow."""

    plots_dir = resolve_path(args.plots_dir)
    reports_dir = resolve_path(args.reports_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== PHASE 4 : MODEL EVALUATION ==========")
    df = load_credit_card_data(args.data_path)
    print_dataset_summary(df)

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nTraining final RandomForestClassifier...")
    final_model = train_final_random_forest(X_train, y_train, args.random_state)
    evaluation = evaluate_binary_classifier(final_model, X_test, y_test)
    print_phase4_metrics(evaluation)

    report_file, summary_file = save_classification_outputs(evaluation, reports_dir)

    print("\nSaving Phase 4 plots...")
    generated_plots = [
        plot_class_distribution(df[TARGET_COLUMN], plots_dir),
        plot_confusion_matrix(evaluation["confusion_matrix"], plots_dir),
        plot_roc_curve(y_test, evaluation["y_score"], plots_dir),
        plot_precision_recall_curve(y_test, evaluation["y_score"], plots_dir),
        plot_feature_importance(final_model, feature_names, plots_dir, top_n=args.top_features),
        plot_error_curve(1.0 - evaluation["accuracy"], plots_dir),
    ]

    print("\nRunning comparison benchmark...")
    comparison_df = run_model_comparison(X_train, X_test, y_train, y_test, args.random_state)
    comparison_file = save_comparison_table(comparison_df, reports_dir)
    comparison_plot = plot_comparison_graph(comparison_df, plots_dir)

    print("\n========== SAVED OUTPUTS ==========")
    print(f"Classification report: {report_file}")
    print(f"Summary metrics      : {summary_file}")
    print(f"Comparison CSV       : {comparison_file}")
    print(f"Comparison plot      : {comparison_plot}")
    for plot_file in generated_plots:
        print(f"Plot                 : {plot_file}")

    print("\n========== PHASE 4 COMPLETED ==========")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line interface for package-safe execution."""

    parser = argparse.ArgumentParser(description="Run Phase 4 evaluation for Credit Card Fraud Detection.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH), help="Path to creditcard.csv.")
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR), help="Directory for PNG plots.")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR), help="Directory for CSV reports.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Stratified test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--top-features", type=int, default=20, help="Number of feature importances to plot.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase4(args)


if __name__ == "__main__":
    main()
