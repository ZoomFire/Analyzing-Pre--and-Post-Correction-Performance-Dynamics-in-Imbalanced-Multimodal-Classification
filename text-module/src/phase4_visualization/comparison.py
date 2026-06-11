from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import REPORTS_DIR, ensure_output_dir, existing_path, parse_output_dir_argument


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot text model comparison metrics.")
    parser.add_argument("--comparison", default=str(REPORTS_DIR / "text_model_comparison.csv"), help="Model comparison CSV.")
    parser.add_argument("--metric", default="f1", help="Metric column to plot.")
    parser.add_argument("--output", default="text_model_comparison.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    comparison = pd.read_csv(existing_path(args.comparison, "model comparison"))
    if args.metric not in comparison.columns:
        raise ValueError(f"{args.comparison} is missing metric column: {args.metric}")

    label_column = "Technique" if "Technique" in comparison.columns else comparison.columns[0]
    plot_df = comparison.sort_values(args.metric, ascending=True)

    plt.figure(figsize=(10, max(5, len(plot_df) * 0.45)))
    plt.barh(plot_df[label_column], plot_df[args.metric].astype(float))
    plt.xlabel(args.metric)
    plt.ylabel(label_column)
    plt.title(f"Text Model Comparison by {args.metric}")
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
