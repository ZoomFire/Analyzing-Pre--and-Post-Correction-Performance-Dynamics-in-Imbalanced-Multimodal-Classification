from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = project_root()
TEXT_MODULE_ROOT = ROOT / "text-module"
REPORTS_DIR = ROOT / "outputs" / "reports"
TEXT_ARTIFACTS_DIR = REPORTS_DIR / "text_artifacts"
PHASE4_TEXT_DIR = REPORTS_DIR / "phase4_text"
TEXT_DATA_PATH = TEXT_MODULE_ROOT / "data" / "processed" / "cleaned_train.csv"


def add_project_to_path() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_output_dir(path: str | Path | None = None) -> Path:
    output_dir = Path(path) if path is not None else PHASE4_TEXT_DIR
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def existing_path(value: str | Path, description: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def read_report(path: str | Path | None = None) -> pd.DataFrame:
    report_path = existing_path(path or REPORTS_DIR / "text_classification_report.csv", "classification report")
    report = pd.read_csv(report_path)
    report.columns = [str(column).strip().lower() for column in report.columns]
    required = {"class", "precision", "recall", "f1", "support"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"{report_path} is missing columns: {', '.join(sorted(missing))}")
    return report


def minority_class(report: pd.DataFrame, class_name: str | None = None) -> str:
    if class_name:
        return class_name

    supported = report[pd.to_numeric(report["support"], errors="coerce").fillna(0) > 0].copy()
    if supported.empty:
        raise ValueError("No class with support > 0 found in classification report.")
    supported["support"] = pd.to_numeric(supported["support"], errors="coerce")
    return str(supported.sort_values(["support", "f1"], ascending=[True, True]).iloc[0]["class"])


def label_names(path: str | Path | None = None) -> list[str]:
    label_path = existing_path(path or REPORTS_DIR / "text_label_names.csv", "label names file")
    labels = pd.read_csv(label_path)
    if "label" not in labels.columns:
        raise ValueError(f"{label_path} must contain a label column.")
    if "label_id" in labels.columns:
        labels = labels.sort_values("label_id")
    return labels["label"].astype(str).tolist()


def parse_output_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=str(PHASE4_TEXT_DIR),
        help="Directory where the visualization image will be saved.",
    )
