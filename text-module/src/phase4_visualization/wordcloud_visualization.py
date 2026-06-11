from __future__ import annotations

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from common import TEXT_DATA_PATH, add_project_to_path, ensure_output_dir, existing_path, parse_output_dir_argument


def load_reuters_texts(data_dir: str | None, class_name: str | None) -> tuple[str, str]:
    add_project_to_path()
    from src.text.reuters_pipeline import DEFAULT_DATA_DIR, load_reuters_dataset

    train_examples, test_examples = load_reuters_dataset(data_dir or DEFAULT_DATA_DIR)
    examples = train_examples + test_examples
    counts = Counter(example.label for example in examples)
    selected_class = class_name or min(counts, key=lambda label: (counts[label], label))
    text = " ".join(example.text for example in examples if example.label == selected_class)
    return selected_class, text


def load_csv_texts(data_file: str, text_column: str, class_name: str | None) -> tuple[str, str]:
    data = pd.read_csv(existing_path(data_file, "text data"))
    if text_column not in data.columns:
        raise ValueError(f"{data_file} is missing text column: {text_column}")

    label_columns = [
        column
        for column in data.columns
        if column not in {"id", text_column, "comment_text"} and set(data[column].dropna().unique()).issubset({0, 1})
    ]
    if not label_columns:
        raise ValueError("No binary class columns found. Pass Reuters data or a CSV with 0/1 label columns.")

    counts = data[label_columns].sum().sort_values()
    selected_class = class_name or str(counts.index[0])
    text = " ".join(data.loc[data[selected_class] == 1, text_column].fillna("").astype(str))
    return selected_class, text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a word cloud for the minority text class.")
    parser.add_argument("--source", choices=["reuters", "csv"], default="csv", help="Text source to use.")
    parser.add_argument("--data-dir", default=None, help="Reuters data directory.")
    parser.add_argument("--data-file", default=str(TEXT_DATA_PATH), help="CSV file used when --source csv.")
    parser.add_argument("--text-column", default="clean_text", help="Text column used when --source csv.")
    parser.add_argument("--class-name", default=None, help="Class to plot. Defaults to the minority class.")
    parser.add_argument("--output", default="text_wordcloud.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    if args.source == "csv":
        selected_class, text = load_csv_texts(args.data_file, args.text_column, args.class_name)
    else:
        selected_class, text = load_reuters_texts(args.data_dir, args.class_name)

    if not text.strip():
        raise ValueError(f"No text found for class: {selected_class}")

    wordcloud = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud | Class: {selected_class}")
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
