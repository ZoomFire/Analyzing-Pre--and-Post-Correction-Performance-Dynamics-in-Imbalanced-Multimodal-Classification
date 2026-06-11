from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ensure_output_dir, minority_class, parse_output_dir_argument, read_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a class-wise classification report heatmap.")
    parser.add_argument("--report", default=None, help="CSV with class, precision, recall, f1 and support columns.")
    parser.add_argument("--minority-class", default=None, help="Class to highlight. Defaults to the lowest-support class.")
    parser.add_argument("--top-n", type=int, default=30, help="Maximum classes to display, sorted by support.")
    parser.add_argument("--output", default="text_classification_heatmap.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    report = read_report(args.report)
    metrics = ["precision", "recall", "f1"]
    if "accuracy" in report.columns:
        metrics.append("accuracy")

    highlighted_class = minority_class(report, args.minority_class)
    report[metrics + ["support"]] = report[metrics + ["support"]].apply(lambda column: column.astype(float))
    plot_df = report.sort_values(["support", "f1"], ascending=[True, True]).head(args.top_n).copy()
    if highlighted_class not in set(plot_df["class"].astype(str)):
        highlight_row = report[report["class"].astype(str) == highlighted_class]
        plot_df = pd.concat([highlight_row, plot_df], ignore_index=True).drop_duplicates("class").head(args.top_n)

    y_labels = [
        f"{row['class']} (n={int(row['support'])})" + (" *" if str(row["class"]) == highlighted_class else "")
        for _, row in plot_df.iterrows()
    ]
    heatmap_data = plot_df.set_index("class")[metrics]

    plt.figure(figsize=(10, max(6, len(plot_df) * 0.35)))
    axis = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_yticklabels(y_labels, rotation=0)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Class")
    axis.set_title(f"Class-wise Classification Report Heatmap | Minority: {highlighted_class}")
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
