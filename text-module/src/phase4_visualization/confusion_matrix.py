from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix

from common import (
    TEXT_ARTIFACTS_DIR,
    ensure_output_dir,
    label_names,
    minority_class,
    parse_output_dir_argument,
    read_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a confusion matrix from saved labels and predictions.")
    parser.add_argument("--labels", default=str(TEXT_ARTIFACTS_DIR / "text_labels.npy"), help="Numpy file with true labels.")
    parser.add_argument("--predictions", default=str(TEXT_ARTIFACTS_DIR / "text_predictions.npy"), help="Numpy file with predicted labels.")
    parser.add_argument("--label-names", default=None, help="CSV with label_id and label columns.")
    parser.add_argument("--report", default=None, help="Classification report CSV used to keep the minority class visible.")
    parser.add_argument("--minority-class", default=None, help="Minority class to include. Defaults to the lowest-support class.")
    parser.add_argument("--top-n", type=int, default=15, help="Show the most frequent true classes.")
    parser.add_argument(
        "--value-mode",
        choices=["normalized", "raw", "log"],
        default="raw",
        help="Cell values to display. raw shows original counts on the full matrix.",
    )
    parser.add_argument("--hide-zeros", action="store_true", help="Hide zero-value cell annotations.")
    parser.add_argument("--output", default="text_confusion_matrix.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    y_true = np.load(args.labels)
    y_pred = np.load(args.predictions)
    names = label_names(args.label_names)
    classes = np.arange(len(names))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    support = cm.sum(axis=1)
    selected = np.argsort(support)[::-1][: args.top_n]
    selected = selected[support[selected] > 0]

    report = read_report(args.report)
    class_to_index = {name: index for index, name in enumerate(names)}
    highlighted_class = minority_class(report, args.minority_class)
    highlighted_index = class_to_index.get(highlighted_class)
    if highlighted_index is not None and support[highlighted_index] > 0 and highlighted_index not in selected:
        selected = np.append(selected[:-1], highlighted_index) if len(selected) >= args.top_n else np.append(selected, highlighted_index)

    cm = cm[np.ix_(selected, selected)]
    selected_names = [names[index] for index in selected]

    if args.value_mode == "normalized":
        row_totals = cm.sum(axis=1, keepdims=True)
        plot_values = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
        annotations = np.vectorize(lambda value: f"{value:.0%}")(plot_values)
        if args.hide_zeros:
            annotations = np.where(plot_values > 0, annotations, "")
        fmt = ""
        colorbar_label = "Percent of actual class"
        norm = None
        title_suffix = "Row-normalized"
    else:
        plot_values = cm.astype(float)
        annotations = cm.astype(str)
        if args.hide_zeros:
            annotations = np.where(cm > 0, annotations, "")
        fmt = ""
        colorbar_label = "Count"
        norm = LogNorm(vmin=1, vmax=max(1, int(cm.max()))) if args.value_mode == "log" and cm.max() > 0 else None
        title_suffix = "Original counts" if args.value_mode == "raw" else "Original counts, log-scaled color"

    mask = np.zeros_like(plot_values, dtype=bool)
    if args.value_mode == "log":
        mask = plot_values == 0
    width = max(8, len(selected_names) * 0.62)
    height = max(6, len(selected_names) * 0.55)
    plt.figure(figsize=(width, height))
    axis = sns.heatmap(
        plot_values,
        annot=annotations,
        fmt=fmt,
        cmap="YlGnBu",
        mask=mask,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"label": colorbar_label, "shrink": 0.85},
        norm=norm,
    )
    x_labels = [f"{name} *" if name == highlighted_class else name for name in selected_names]
    y_labels = [f"{name} *" if name == highlighted_class else name for name in selected_names]
    axis.set_xticklabels(x_labels, rotation=35, ha="right")
    axis.set_yticklabels(y_labels, rotation=0)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"Text Confusion Matrix ({title_suffix}) | * minority class: {highlighted_class}")
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
