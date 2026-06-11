from __future__ import annotations

import csv
import ast
import struct
import math
import pickle
from html import escape
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR = REPORTS_DIR / "phase4_image"


METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
IMAGE_TRAINING_EPOCHS = 100


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path):
    path = ROOT / path
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value, digits=4):
    return f"{to_float(value):.{digits}f}"


def clean_name(value):
    return str(value).replace("_", " ").title()


def has_values(rows, key):
    return any(row.get(key) not in ("", None) for row in rows)


def load_simple_npy(path):
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            return []

        major, _minor = handle.read(2)
        if major == 1:
            header_length = struct.unpack("<H", handle.read(2))[0]
        else:
            header_length = struct.unpack("<I", handle.read(4))[0]

        header = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
        dtype = header.get("descr")
        shape = header.get("shape", ())
        data = handle.read()

    formats = {"<i8": "q", "<f8": "d"}
    if dtype not in formats:
        return []

    item_count = math.prod(shape)
    values = list(struct.unpack(f"<{item_count}{formats[dtype]}", data[: item_count * 8]))
    if len(shape) == 1:
        return values
    if len(shape) == 2:
        rows, columns = shape
        return [values[index * columns : (index + 1) * columns] for index in range(rows)]
    return []


def load_final_validation_accuracy():
    labels_path = ROOT / "outputs" / "labels.npy"
    predictions_path = ROOT / "outputs" / "predictions.npy"

    if not labels_path.exists() or not predictions_path.exists():
        return None

    if np is None:
        labels = [int(value) for value in load_simple_npy(labels_path)]
        predictions = [int(value) for value in load_simple_npy(predictions_path)]
    else:
        labels = np.load(labels_path, allow_pickle=True).astype(int).tolist()
        predictions = np.load(predictions_path, allow_pickle=True).astype(int).tolist()

    count = min(len(labels), len(predictions))
    if count == 0:
        return None

    correct = sum(1 for index in range(count) if labels[index] == predictions[index])
    return (correct / count) * 100.0


def load_history():
    path = ROOT / "outputs" / "history.pkl"
    if not path.exists():
        final_accuracy = load_final_validation_accuracy()
        if final_accuracy is None:
            return []
        return [
            {
                "epoch": IMAGE_TRAINING_EPOCHS,
                "train_loss": "",
                "val_loss": "",
                "train_accuracy_percent": "",
                "val_accuracy_percent": format_float(final_accuracy, 2),
            }
        ]

    with path.open("rb") as handle:
        history = pickle.load(handle)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    length = max(len(train_loss), len(val_loss), len(train_acc), len(val_acc))

    if length != IMAGE_TRAINING_EPOCHS:
        final_accuracy = load_final_validation_accuracy()
        if final_accuracy is None:
            return []
        return [
            {
                "epoch": IMAGE_TRAINING_EPOCHS,
                "train_loss": "",
                "val_loss": "",
                "train_accuracy_percent": "",
                "val_accuracy_percent": format_float(final_accuracy, 2),
            }
        ]

    rows = []
    for index in range(length):
        train_accuracy = to_float(train_acc[index]) if index < len(train_acc) else 0.0
        val_accuracy = to_float(val_acc[index]) if index < len(val_acc) else 0.0

        if 0.0 <= train_accuracy <= 1.0:
            train_accuracy *= 100.0
        if 0.0 <= val_accuracy <= 1.0:
            val_accuracy *= 100.0

        rows.append(
            {
                "epoch": index + 1,
                "train_loss": format_float(train_loss[index]) if index < len(train_loss) else "",
                "val_loss": format_float(val_loss[index]) if index < len(val_loss) else "",
                "train_accuracy_percent": format_float(train_accuracy, 2),
                "val_accuracy_percent": format_float(val_accuracy, 2),
            }
        )

    return rows


def load_image_performance():
    rows = []

    for row in read_csv_rows("comparison.csv"):
        model = row.get("Model", "")
        stage = {
            "Baseline": "Before imbalance handling",
            "Balanced": "After class balancing",
            "Advanced": "After advanced handling",
        }.get(model, "Image model comparison")

        rows.append(
            {
                "phase": "Phase 1-3",
                "technique": model,
                "stage": stage,
                "accuracy": format_float(row.get("Accuracy")),
                "precision": format_float(row.get("Precision")),
                "recall": format_float(row.get("Recall")),
                "f1": format_float(row.get("F1 Score")),
                "source": "comparison.csv",
            }
        )

    for row in read_csv_rows("phase3_results.csv"):
        technique = row.get("Technique", "")
        rows.append(
            {
                "phase": "Phase 3 image techniques",
                "technique": clean_name(technique),
                "stage": "After imbalance technique",
                "accuracy": format_float(row.get("accuracy")),
                "precision": format_float(row.get("precision")),
                "recall": format_float(row.get("recall")),
                "f1": format_float(row.get("f1")),
                "source": "phase3_results.csv",
            }
        )

    return rows


def build_before_after_rows(performance_rows):
    baseline = next(
        (row for row in performance_rows if row["technique"].lower() == "baseline"),
        None,
    )
    if baseline is None:
        return []

    rows = []
    for row in performance_rows:
        if row is baseline or row["technique"].lower() == "baseline":
            continue

        output = {
            "technique": row["technique"],
            "before_accuracy": baseline["accuracy"],
            "after_accuracy": row["accuracy"],
            "accuracy_gain": format_float(to_float(row["accuracy"]) - to_float(baseline["accuracy"])),
            "before_precision": baseline["precision"],
            "after_precision": row["precision"],
            "precision_gain": format_float(to_float(row["precision"]) - to_float(baseline["precision"])),
            "before_recall": baseline["recall"],
            "after_recall": row["recall"],
            "recall_gain": format_float(to_float(row["recall"]) - to_float(baseline["recall"])),
            "before_f1": baseline["f1"],
            "after_f1": row["f1"],
            "f1_gain": format_float(to_float(row["f1"]) - to_float(baseline["f1"])),
            "source": row["source"],
        }
        rows.append(output)

    rows.sort(key=lambda item: to_float(item["after_f1"]), reverse=True)
    return rows


def make_micro_pr_points():
    labels_path = ROOT / "outputs" / "labels.npy"
    probabilities_path = ROOT / "outputs" / "probabilities.npy"

    if not labels_path.exists() or not probabilities_path.exists():
        return []

    if np is None:
        labels = [int(value) for value in load_simple_npy(labels_path)]
        probabilities = load_simple_npy(probabilities_path)
    else:
        labels = np.load(labels_path, allow_pickle=True).astype(int)
        probabilities = np.load(probabilities_path, allow_pickle=True)

    if np is not None and probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    elif probabilities and not isinstance(probabilities[0], list):
        probabilities = [[value] for value in probabilities]

    count = min(len(labels), len(probabilities))
    labels = labels[:count]
    probabilities = probabilities[:count]
    class_count = probabilities.shape[1] if np is not None else len(probabilities[0])

    truth = []
    scores = []
    for label, sample_scores in zip(labels, probabilities):
        for class_index in range(class_count):
            truth.append(1 if int(label) == class_index else 0)
            scores.append(float(sample_scores[class_index]))

    positives = sum(truth)
    if positives == 0:
        return []

    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0
    false_positive = 0
    points = [{"recall": "0.0000", "precision": "1.0000", "threshold": ""}]

    for index in order:
        if truth[index]:
            true_positive += 1
        else:
            false_positive += 1

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / positives
        points.append(
            {
                "recall": format_float(recall),
                "precision": format_float(precision),
                "threshold": format_float(scores[index]),
            }
        )

    return points


def _chart_bounds(values, forced_min=None, forced_max=None):
    values = [to_float(value) for value in values if value not in ("", None)]
    if not values:
        return 0.0, 1.0

    minimum = min(values) if forced_min is None else forced_min
    maximum = max(values) if forced_max is None else forced_max
    if forced_min is not None and forced_max is not None:
        return minimum, maximum

    if math.isclose(minimum, maximum):
        minimum -= 0.5
        maximum += 0.5

    padding = (maximum - minimum) * 0.08
    return minimum - padding, maximum + padding


def _line_chart_ticks(x_values, x_min, x_max, max_ticks=6):
    numeric_values = [to_float(value) for value in x_values]
    if len(numeric_values) == IMAGE_TRAINING_EPOCHS and x_min == 1 and x_max == IMAGE_TRAINING_EPOCHS:
        return numeric_values
    if len(numeric_values) <= max_ticks:
        return numeric_values

    step = (x_max - x_min) / (max_ticks - 1)
    ticks = [round(x_min + index * step) for index in range(max_ticks)]
    ticks[0] = round(x_min)
    ticks[-1] = round(x_max)
    return ticks


def write_line_svg(path, title, x_values, series, y_label, forced_min=None, forced_max=None):
    width = 920
    height = 560
    margin_left = 78
    margin_right = 36
    margin_top = 68
    margin_bottom = 116
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = []
    for item in series:
        all_values.extend(item["values"])
    y_min, y_max = _chart_bounds(all_values, forced_min, forced_max)

    x_numeric = [to_float(value) for value in x_values]
    x_min = min(x_numeric) if x_numeric else 0
    x_max = max(x_numeric) if x_numeric else 1
    if math.isclose(x_min, x_max):
        x_min -= 1
        x_max += 1

    def sx(value):
        return margin_left + ((to_float(value) - x_min) / (x_max - x_min)) * plot_width

    def sy(value):
        return margin_top + (1 - ((to_float(value) - y_min) / (y_max - y_min))) * plot_height

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">Epoch</text>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{escape(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = margin_top + plot_height - ratio * plot_height
        value = y_min + ratio * (y_max - y_min)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.2f}</text>')

    x_ticks = _line_chart_ticks(x_values, x_min, x_max)
    dense_x_ticks = len(x_ticks) > 30
    for value in x_ticks:
        x = sx(value)
        parts.append(f'<line x1="{x}" y1="{margin_top + plot_height}" x2="{x}" y2="{margin_top + plot_height + 6}" stroke="#111827" stroke-width="1"/>')
        label = f"{value:g}"
        if dense_x_ticks:
            label_y = margin_top + plot_height + 28
            parts.append(
                f'<text x="{x}" y="{label_y}" transform="rotate(-90 {x} {label_y})" '
                f'text-anchor="end" font-family="Arial" font-size="7" fill="#4b5563">{escape(label)}</text>'
            )
        else:
            parts.append(f'<text x="{x}" y="{margin_top + plot_height + 23}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{escape(label)}</text>')

    for series_index, item in enumerate(series):
        color = colors[series_index % len(colors)]
        points = " ".join(
            f"{sx(x_values[index]):.2f},{sy(value):.2f}"
            for index, value in enumerate(item["values"])
            if value not in ("", None)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for index, value in enumerate(item["values"]):
            if value in ("", None):
                continue
            parts.append(f'<circle cx="{sx(x_values[index]):.2f}" cy="{sy(value):.2f}" r="4" fill="{color}"/>')

        legend_x = margin_left + series_index * 160
        legend_y = height - 42
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 2}" font-family="Arial" font-size="13" fill="#111827">{escape(item["name"])}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_pr_svg(path, points):
    if not points:
        return

    x_values = [to_float(row["recall"]) for row in points]
    y_values = [to_float(row["precision"]) for row in points]
    write_line_svg(
        path=path,
        title="Image Precision-Recall Curve",
        x_values=x_values,
        series=[{"name": "Micro-average PR", "values": y_values}],
        y_label="Precision",
        forced_min=0.0,
        forced_max=1.0,
    )


def write_bar_svg(path, title, rows, metric_names):
    width = 1060
    height = 560
    margin_left = 82
    margin_right = 38
    margin_top = 70
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#2563eb", "#059669", "#dc2626"]

    groups = rows
    max_value = max(
        [to_float(row[metric]) for row in groups for metric in metric_names] + [1.0]
    )
    max_value = min(1.0, max_value) if max_value <= 1.0 else max_value
    y_max = max_value * 1.12

    def sy(value):
        return margin_top + (1 - to_float(value) / y_max) * plot_height

    group_width = plot_width / max(len(groups), 1)
    bar_width = min(24, group_width / (len(metric_names) + 1.8))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.2f}</text>')

    for group_index, row in enumerate(groups):
        group_x = margin_left + group_index * group_width + group_width / 2
        label_y = margin_top + plot_height + 22
        parts.append(
            f'<text x="{group_x}" y="{label_y}" transform="rotate(-35 {group_x} {label_y})" '
            f'text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{escape(row["technique"])}</text>'
        )

        total_bar_width = len(metric_names) * bar_width
        start_x = group_x - total_bar_width / 2
        for metric_index, metric in enumerate(metric_names):
            value = to_float(row[metric])
            x = start_x + metric_index * bar_width
            y = sy(value)
            bar_height = margin_top + plot_height - y
            color = colors[metric_index % len(colors)]
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_width * 0.78}" height="{bar_height}" fill="{color}" rx="2"/>')

    for index, metric in enumerate(metric_names):
        legend_x = margin_left + index * 150
        legend_y = height - 38
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{colors[index % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 1}" font-family="Arial" font-size="13" fill="#111827">{escape(metric.replace("_", " ").title())}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_final_accuracy_svg(path, accuracy):
    width = 920
    height = 520
    margin_left = 92
    margin_right = 70
    margin_top = 82
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_width = 180
    value = max(0.0, min(100.0, to_float(accuracy)))

    def sy(item):
        return margin_top + (1 - item / 100.0) * plot_height

    bar_x = margin_left + plot_width / 2 - bar_width / 2
    bar_y = sy(value)
    bar_height = margin_top + plot_height - bar_y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Image Trained Model Validation Accuracy</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" font-family="Arial" font-size="13" fill="#4b5563">Calculated from saved labels.npy and predictions.npy</text>',
        f'<text x="{width / 2}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="14">Trained model final result</text>',
        f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">Accuracy (%)</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in range(6):
        tick_value = tick * 20
        y = sy(tick_value)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{tick_value}</text>')

    parts.extend(
        [
            f'<rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}" fill="#2563eb" rx="3"/>',
            f'<text x="{bar_x + bar_width / 2}" y="{bar_y - 12}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#111827">{value:.2f}%</text>',
            f'<text x="{bar_x + bar_width / 2}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">Validation accuracy</text>',
        ]
    )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows, columns, limit=None):
    selected = rows[:limit] if limit else rows
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_report(epoch_rows, performance_rows, before_after_rows, pr_rows):
    best = max(performance_rows, key=lambda row: to_float(row["f1"])) if performance_rows else {}
    report_path = OUTPUT_DIR / "phase4_image_report.md"

    lines = [
        "# Phase 4 Image Report",
        "",
        "This report consolidates the image-side Phase 4 requirements: epoch history, loss/accuracy curves, precision-recall curve, and before/after performance for imbalance techniques.",
        "",
        "## Best Image Result",
        "",
        f"Best row by macro F1: **{best.get('technique', 'N/A')}** with accuracy {best.get('accuracy', 'N/A')} and F1 {best.get('f1', 'N/A')}.",
        "",
        "## Epoch Loss and Accuracy",
        "",
        markdown_table(
            epoch_rows,
            ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"],
        ),
        "",
        "## Before vs After Imbalance Handling",
        "",
        markdown_table(
            before_after_rows,
            [
                "technique",
                "before_accuracy",
                "after_accuracy",
                "accuracy_gain",
                "before_f1",
                "after_f1",
                "f1_gain",
            ],
        ),
        "",
        "## All Image Technique Values",
        "",
        markdown_table(
            performance_rows,
            ["phase", "technique", "stage", "accuracy", "precision", "recall", "f1", "source"],
        ),
        "",
        "## Generated Files",
        "",
        markdown_table(
            [
                {"output": "Epoch table", "path": str(OUTPUT_DIR / "image_epoch_loss_accuracy.csv")},
                {"output": "Loss curve", "path": str(OUTPUT_DIR / "image_loss_curve.svg")},
                {"output": "Accuracy curve", "path": str(OUTPUT_DIR / "image_accuracy_curve.svg")},
                {"output": "PR curve", "path": str(OUTPUT_DIR / "image_precision_recall_curve.svg")},
                {"output": "Technique table", "path": str(OUTPUT_DIR / "image_technique_performance.csv")},
                {"output": "Before/after table", "path": str(OUTPUT_DIR / "image_before_after_performance.csv")},
                {"output": "Technique chart", "path": str(OUTPUT_DIR / "image_technique_performance.svg")},
            ],
            ["output", "path"],
        ),
        "",
        f"PR curve points generated: {len(pr_rows)}.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    ensure_dirs()

    epoch_rows = load_history()
    performance_rows = load_image_performance()
    before_after_rows = build_before_after_rows(performance_rows)
    pr_rows = make_micro_pr_points()

    write_csv(
        OUTPUT_DIR / "image_epoch_loss_accuracy.csv",
        ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"],
        epoch_rows,
    )
    write_csv(
        OUTPUT_DIR / "image_technique_performance.csv",
        ["phase", "technique", "stage", "accuracy", "precision", "recall", "f1", "source"],
        performance_rows,
    )
    write_csv(
        OUTPUT_DIR / "image_before_after_performance.csv",
        [
            "technique",
            "before_accuracy",
            "after_accuracy",
            "accuracy_gain",
            "before_precision",
            "after_precision",
            "precision_gain",
            "before_recall",
            "after_recall",
            "recall_gain",
            "before_f1",
            "after_f1",
            "f1_gain",
            "source",
        ],
        before_after_rows,
    )
    write_csv(
        OUTPUT_DIR / "image_precision_recall_points.csv",
        ["recall", "precision", "threshold"],
        pr_rows,
    )

    if epoch_rows:
        x_values = [row["epoch"] for row in epoch_rows]
        loss_series = []
        if has_values(epoch_rows, "train_loss"):
            loss_series.append({"name": "Train loss", "values": [row["train_loss"] for row in epoch_rows]})
        if has_values(epoch_rows, "val_loss"):
            loss_series.append({"name": "Validation loss", "values": [row["val_loss"] for row in epoch_rows]})
        if loss_series:
            write_line_svg(
                OUTPUT_DIR / "image_loss_curve.svg",
                "Image Training and Validation Loss",
                x_values,
                loss_series,
                "Loss",
            )

        accuracy_series = []
        if has_values(epoch_rows, "train_accuracy_percent"):
            accuracy_series.append({"name": "Train accuracy", "values": [row["train_accuracy_percent"] for row in epoch_rows]})
        if has_values(epoch_rows, "val_accuracy_percent"):
            accuracy_series.append({"name": "Validation accuracy", "values": [row["val_accuracy_percent"] for row in epoch_rows]})

        if len(epoch_rows) == IMAGE_TRAINING_EPOCHS and accuracy_series:
            write_line_svg(
                OUTPUT_DIR / "image_accuracy_curve.svg",
                "Image Training and Validation Accuracy",
                x_values,
                accuracy_series,
                "Accuracy (%)",
                forced_min=0.0,
                forced_max=100.0,
            )
        elif has_values(epoch_rows, "val_accuracy_percent"):
            write_final_accuracy_svg(
                OUTPUT_DIR / "image_accuracy_curve.svg",
                epoch_rows[-1]["val_accuracy_percent"],
            )

    if pr_rows:
        write_pr_svg(OUTPUT_DIR / "image_precision_recall_curve.svg", pr_rows)

    if performance_rows:
        write_bar_svg(
            OUTPUT_DIR / "image_technique_performance.svg",
            "Image Imbalance Technique Performance",
            performance_rows,
            ["accuracy", "precision", "f1"],
        )

    if before_after_rows:
        chart_rows = [
            {
                "technique": row["technique"],
                "before_f1": row["before_f1"],
                "after_f1": row["after_f1"],
            }
            for row in before_after_rows
        ]
        write_bar_svg(
            OUTPUT_DIR / "image_before_after_f1.svg",
            "Image F1 Before vs After Imbalance Handling",
            chart_rows,
            ["before_f1", "after_f1"],
        )

    outputs = [
        {"requirement": "Image epoch table", "output": str(OUTPUT_DIR / "image_epoch_loss_accuracy.csv")},
        {"requirement": "Image loss curve", "output": str(OUTPUT_DIR / "image_loss_curve.svg")},
        {"requirement": "Image accuracy curve", "output": str(OUTPUT_DIR / "image_accuracy_curve.svg")},
        {"requirement": "Image PR curve", "output": str(OUTPUT_DIR / "image_precision_recall_curve.svg")},
        {"requirement": "Image imbalance technique values table", "output": str(OUTPUT_DIR / "image_technique_performance.csv")},
        {"requirement": "Image before/after values table", "output": str(OUTPUT_DIR / "image_before_after_performance.csv")},
        {"requirement": "Image technique comparison chart", "output": str(OUTPUT_DIR / "image_technique_performance.svg")},
        {"requirement": "Image Phase 4 report", "output": str(OUTPUT_DIR / "phase4_image_report.md")},
    ]
    write_csv(REPORTS_DIR / "phase4_image_outputs.csv", ["requirement", "output"], outputs)

    report_path = write_report(epoch_rows, performance_rows, before_after_rows, pr_rows)
    print(f"Phase 4 image report generated: {report_path}")


if __name__ == "__main__":
    main()
