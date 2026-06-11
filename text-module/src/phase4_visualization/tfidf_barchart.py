from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import (
    REPORTS_DIR,
    TEXT_DATA_PATH,
    add_project_to_path,
    ensure_output_dir,
    existing_path,
    minority_class,
    parse_output_dir_argument,
    read_report,
)


def best_model_path() -> str:
    comparison = pd.read_csv(REPORTS_DIR / "text_model_comparison.csv")
    best = comparison.sort_values("f1", ascending=False).iloc[0]
    return str(best["Source"])


def load_log_odds(class_name: str | None, top_n: int) -> tuple[str, pd.DataFrame]:
    add_project_to_path()
    report = read_report()
    selected_class = minority_class(report, class_name)
    model_path = best_model_path()
    with open(existing_path(model_path, "model artifact"), "rb") as handle:
        artifact = pickle.load(handle)

    model = artifact["model"]
    vocabulary = artifact["vocabulary"]
    terms = np.array([term for term, _ in sorted(vocabulary.items(), key=lambda item: item[1])])
    classes = list(model.classes_)
    if selected_class not in classes:
        raise ValueError(f"Class {selected_class!r} is not present in model classes.")

    class_index = classes.index(selected_class)
    target_log_prob = model.feature_log_prob_[class_index]
    other_log_prob = np.delete(model.feature_log_prob_, class_index, axis=0).mean(axis=0)
    scores = target_log_prob - other_log_prob
    top_indices = np.argsort(scores)[::-1][:top_n]
    rows = pd.DataFrame({"term": terms[top_indices], "score": scores[top_indices]})
    return selected_class, rows


def load_tfidf_terms(data_file: str, text_column: str, class_name: str | None, top_n: int) -> tuple[str, pd.DataFrame]:
    data = pd.read_csv(existing_path(data_file, "text data"))
    if text_column not in data.columns:
        raise ValueError(f"{data_file} is missing text column: {text_column}")

    label_columns = [
        column
        for column in data.columns
        if column not in {"id", text_column, "comment_text"} and set(data[column].dropna().unique()).issubset({0, 1})
    ]
    if not label_columns:
        raise ValueError("No binary class columns found for TF-IDF mode.")

    counts = data[label_columns].sum().sort_values()
    selected_class = class_name or str(counts.index[0])
    class_text = data.loc[data[selected_class] == 1, text_column].fillna("").astype(str)
    vectorizer = TfidfVectorizer(max_features=max(top_n * 20, top_n), stop_words="english")
    matrix = vectorizer.fit_transform(class_text)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_indices = np.argsort(scores)[::-1][:top_n]
    rows = pd.DataFrame({"term": terms[top_indices], "score": scores[top_indices]})
    return selected_class, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top log-odds or TF-IDF words for a text class.")
    parser.add_argument("--mode", choices=["log-odds", "tfidf"], default="log-odds", help="Scoring method.")
    parser.add_argument("--class-name", default=None, help="Class to explain. Defaults to the lowest-support class.")
    parser.add_argument("--data-file", default=str(TEXT_DATA_PATH), help="CSV file used in TF-IDF mode.")
    parser.add_argument("--text-column", default="clean_text", help="Text column used in TF-IDF mode.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of terms to plot.")
    parser.add_argument("--output", default="text_log_odds_barchart.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    if args.mode == "tfidf":
        selected_class, scores = load_tfidf_terms(args.data_file, args.text_column, args.class_name, args.top_n)
        y_label = "Mean TF-IDF Score"
        title = f"Top TF-IDF Terms | Class: {selected_class}"
    else:
        selected_class, scores = load_log_odds(args.class_name, args.top_n)
        y_label = "Log Odds Ratio"
        title = f"Top Log-Odds Terms | Class: {selected_class}"

    plt.figure(figsize=(12, 6))
    plt.bar(scores["term"], scores["score"])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Term")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
