from __future__ import annotations

import csv
import math
import pickle
import shutil
from html import escape
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "outputs" / "reports"
OUTPUT_DIR = REPORTS_DIR / "phase4_audio"


METRICS = ["accuracy", "precision", "recall", "f1"]
COLORS = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c"]


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def reset_output_dir():
    if not OUTPUT_DIR.exists():
        return

    reports_root = REPORTS_DIR.resolve()
    output_root = OUTPUT_DIR.resolve()

    if OUTPUT_DIR.name != "phase4_audio" or reports_root not in output_root.parents:
        raise RuntimeError(f"Refusing to clean unexpected output directory: {OUTPUT_DIR}")

    for item in OUTPUT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def read_csv_rows(path):
    path = ROOT / path if not isinstance(path, Path) else path
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


def chart_bounds(values, forced_min=None, forced_max=None):
    values = [to_float(value) for value in values if value not in ("", None)]
    if not values:
        return 0.0, 1.0

    minimum = min(values) if forced_min is None else forced_min
    maximum = max(values) if forced_max is None else forced_max

    if math.isclose(minimum, maximum):
        minimum -= 0.5
        maximum += 0.5

    padding = (maximum - minimum) * 0.08
    if forced_min is not None:
        minimum = forced_min
    else:
        minimum -= padding

    if forced_max is not None:
        maximum = forced_max
    else:
        maximum += padding

    return minimum, maximum


def downsample_points(points, max_points=850):
    if len(points) <= max_points:
        return points

    stride = max(1, math.ceil(len(points) / max_points))
    sampled = points[::stride]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def write_line_svg(
    path,
    title,
    x_values,
    series,
    x_label,
    y_label,
    forced_y_min=None,
    forced_y_max=None,
    forced_x_min=None,
    forced_x_max=None,
):
    width = 960
    height = 540
    margin_left = 82
    margin_right = 36
    margin_top = 68
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_y = []
    for item in series:
        all_y.extend(item["values"])

    y_min, y_max = chart_bounds(all_y, forced_y_min, forced_y_max)
    x_min, x_max = chart_bounds(x_values, forced_x_min, forced_x_max)

    def sx(value):
        return margin_left + ((to_float(value) - x_min) / (x_max - x_min)) * plot_width

    def sy(value):
        return margin_top + (1 - ((to_float(value) - y_min) / (y_max - y_min))) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">{escape(x_label)}</text>',
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

        x = margin_left + ratio * plot_width
        x_value = x_min + ratio * (x_max - x_min)
        parts.append(f'<line x1="{x}" y1="{margin_top + plot_height}" x2="{x}" y2="{margin_top + plot_height + 6}" stroke="#111827" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{margin_top + plot_height + 23}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{x_value:.2f}</text>')

    for series_index, item in enumerate(series):
        color = COLORS[series_index % len(COLORS)]
        paired_points = [
            (x_values[index], value)
            for index, value in enumerate(item["values"])
            if value not in ("", None)
        ]
        polyline = " ".join(
            f"{sx(x):.2f},{sy(y):.2f}" for x, y in paired_points
        )
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')

        if len(paired_points) <= 140:
            for x, y in paired_points:
                parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.8" fill="{color}"/>')

        legend_x = margin_left + series_index * 180
        legend_y = height - 48
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 2}" font-family="Arial" font-size="13" fill="#111827">{escape(item["name"])}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_scatter_svg(path, title, rows):
    width = 960
    height = 540
    margin_left = 82
    margin_right = 36
    margin_top = 68
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [to_float(row["sample_index"]) for row in rows]
    y_values = [to_float(row["residual"]) for row in rows]
    x_min, x_max = chart_bounds(x_values)
    y_min, y_max = chart_bounds(y_values, 0.0, 1.0)

    def sx(value):
        return margin_left + ((to_float(value) - x_min) / (x_max - x_min)) * plot_width

    def sy(value):
        return margin_top + (1 - ((to_float(value) - y_min) / (y_max - y_min))) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="14">Validation sample index</text>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">1 - P(true class)</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in range(6):
        ratio = tick / 5
        y = margin_top + plot_height - ratio * plot_height
        value = y_min + ratio * (y_max - y_min)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.2f}</text>')

    for row in rows:
        color = "#059669" if row["correct"] == "1" else "#dc2626"
        parts.append(
            f'<circle cx="{sx(row["sample_index"]):.2f}" cy="{sy(row["residual"]):.2f}" r="3.2" fill="{color}" opacity="0.72"/>'
        )

    legend_y = height - 48
    parts.append(f'<circle cx="{margin_left}" cy="{legend_y - 3}" r="5" fill="#059669" opacity="0.72"/>')
    parts.append(f'<text x="{margin_left + 14}" y="{legend_y + 2}" font-family="Arial" font-size="13" fill="#111827">Correct</text>')
    parts.append(f'<circle cx="{margin_left + 100}" cy="{legend_y - 3}" r="5" fill="#dc2626" opacity="0.72"/>')
    parts.append(f'<text x="{margin_left + 114}" y="{legend_y + 2}" font-family="Arial" font-size="13" fill="#111827">Incorrect</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_bar_svg(path, title, rows, label_key, metric_keys, y_label="Score"):
    width = 1040
    height = 570
    margin_left = 82
    margin_right = 38
    margin_top = 70
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_max = max([to_float(row[key]) for row in rows for key in metric_keys] + [1.0]) * 1.12

    def sy(value):
        return margin_top + (1 - to_float(value) / y_max) * plot_height

    group_width = plot_width / max(len(rows), 1)
    bar_width = min(28, group_width / (len(metric_keys) + 1.8))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{escape(y_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.4"/>',
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{margin_left + plot_width}" y2="{y}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.2f}</text>')

    for row_index, row in enumerate(rows):
        group_x = margin_left + row_index * group_width + group_width / 2
        label_y = margin_top + plot_height + 26
        parts.append(
            f'<text x="{group_x}" y="{label_y}" transform="rotate(-35 {group_x} {label_y})" '
            f'text-anchor="end" font-family="Arial" font-size="12" fill="#374151">{escape(str(row[label_key]))}</text>'
        )

        start_x = group_x - (len(metric_keys) * bar_width) / 2
        for metric_index, metric in enumerate(metric_keys):
            value = to_float(row[metric])
            x = start_x + metric_index * bar_width
            y = sy(value)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width * 0.78}" height="{margin_top + plot_height - y}" '
                f'fill="{COLORS[metric_index % len(COLORS)]}" rx="2"/>'
            )

    for index, metric in enumerate(metric_keys):
        legend_x = margin_left + index * 160
        legend_y = height - 40
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{COLORS[index % len(COLORS)]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 1}" font-family="Arial" font-size="13" fill="#111827">{escape(metric.replace("_", " ").title())}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_horizontal_grouped_bar_svg(path, title, rows, label_key, metric_keys):
    row_height = 24
    width = 1160
    height = 110 + row_height * len(rows)
    margin_left = 190
    margin_right = 60
    margin_top = 62
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    max_value = max([to_float(row[key]) for row in rows for key in metric_keys] + [1.0])
    bar_height = 8

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        x = margin_left + (value / max_value) * plot_width
        parts.append(f'<line x1="{x}" y1="{margin_top - 10}" x2="{x}" y2="{height - margin_bottom}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{value:.0f}</text>')

    for row_index, row in enumerate(rows):
        y = margin_top + row_index * row_height
        parts.append(f'<text x="{margin_left - 10}" y="{y + 12}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{escape(str(row[label_key]))}</text>')
        for metric_index, metric in enumerate(metric_keys):
            value = to_float(row[metric])
            x = margin_left
            bar_y = y + metric_index * (bar_height + 2)
            parts.append(
                f'<rect x="{x}" y="{bar_y}" width="{(value / max_value) * plot_width}" height="{bar_height}" '
                f'fill="{COLORS[metric_index % len(COLORS)]}" rx="2"/>'
            )

    for index, metric in enumerate(metric_keys):
        legend_x = margin_left + index * 170
        legend_y = 52
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{COLORS[index % len(COLORS)]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 1}" font-family="Arial" font-size="13" fill="#111827">{escape(metric.replace("_", " ").title())}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_horizontal_bar_svg(path, title, rows, label_key, value_key, x_label):
    row_height = 22
    width = 1080
    height = 105 + row_height * len(rows)
    margin_left = 190
    margin_right = 70
    margin_top = 60
    margin_bottom = 45
    plot_width = width - margin_left - margin_right
    max_value = max([to_float(row[value_key]) for row in rows] + [1.0])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">{escape(x_label)}</text>',
    ]

    for tick in range(6):
        value = max_value * tick / 5
        x = margin_left + (value / max_value) * plot_width
        parts.append(f'<line x1="{x}" y1="{margin_top - 8}" x2="{x}" y2="{height - margin_bottom}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{value:.2f}</text>')

    for row_index, row in enumerate(rows):
        y = margin_top + row_index * row_height
        value = to_float(row[value_key])
        parts.append(f'<text x="{margin_left - 10}" y="{y + 12}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{escape(str(row[label_key]))}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y}" width="{(value / max_value) * plot_width}" height="14" fill="#2563eb" rx="2"/>')
        parts.append(f'<text x="{margin_left + (value / max_value) * plot_width + 6}" y="{y + 12}" font-family="Arial" font-size="10" fill="#111827">{value:.2f}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_heatmap_svg(path, title, rows, row_key, metric_keys):
    cell_width = 142
    cell_height = 58
    label_width = 150
    width = label_width + cell_width * len(metric_keys) + 60
    height = 110 + cell_height * len(rows)
    margin_top = 70
    min_value = min([to_float(row[key]) for row in rows for key in metric_keys] + [0.0])
    max_value = max([to_float(row[key]) for row in rows for key in metric_keys] + [1.0])
    max_abs = max(abs(min_value), abs(max_value), 0.0001)

    def color(value):
        value = to_float(value)
        ratio = min(abs(value) / max_abs, 1.0)
        if value >= 0:
            red = int(237 - 182 * ratio)
            green = int(247 - 89 * ratio)
            blue = int(239 - 125 * ratio)
        else:
            red = int(254 - 69 * ratio)
            green = int(242 - 116 * ratio)
            blue = int(242 - 108 * ratio)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for index, metric in enumerate(metric_keys):
        x = label_width + index * cell_width + cell_width / 2
        parts.append(f'<text x="{x}" y="62" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">{escape(metric.title())}</text>')

    for row_index, row in enumerate(rows):
        y = margin_top + row_index * cell_height
        parts.append(f'<text x="{label_width - 12}" y="{y + cell_height / 2 + 5}" text-anchor="end" font-family="Arial" font-size="13" fill="#111827">{escape(str(row[row_key]))}</text>')
        for metric_index, metric in enumerate(metric_keys):
            x = label_width + metric_index * cell_width
            value = to_float(row[metric])
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{color(value)}" stroke="#ffffff" stroke-width="2"/>')
            parts.append(f'<text x="{x + cell_width / 2}" y="{y + cell_height / 2 + 5}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{value:+.4f}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_radar_svg(path, title, rows, label_key, metric_keys):
    width = 720
    height = 620
    center_x = width / 2
    center_y = height / 2 + 20
    radius = 210

    def point(index, value):
        angle = -math.pi / 2 + (2 * math.pi * index / len(metric_keys))
        return (
            center_x + math.cos(angle) * radius * to_float(value),
            center_y + math.sin(angle) * radius * to_float(value),
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{escape(title)}</text>',
    ]

    for level in [0.25, 0.5, 0.75, 1.0]:
        points = " ".join(f"{point(index, level)[0]:.2f},{point(index, level)[1]:.2f}" for index in range(len(metric_keys)))
        parts.append(f'<polygon points="{points}" fill="none" stroke="#e5e7eb" stroke-width="1"/>')

    for index, metric in enumerate(metric_keys):
        axis_x, axis_y = point(index, 1.0)
        label_x, label_y = point(index, 1.12)
        parts.append(f'<line x1="{center_x}" y1="{center_y}" x2="{axis_x}" y2="{axis_y}" stroke="#d1d5db" stroke-width="1"/>')
        parts.append(f'<text x="{label_x}" y="{label_y + 4}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{escape(metric.title())}</text>')

    for row_index, row in enumerate(rows):
        color = COLORS[row_index % len(COLORS)]
        points = " ".join(
            f"{point(index, row[metric])[0]:.2f},{point(index, row[metric])[1]:.2f}"
            for index, metric in enumerate(metric_keys)
        )
        parts.append(f'<polygon points="{points}" fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="3"/>')

        legend_x = 70 + row_index * 210
        legend_y = height - 48
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 1}" font-family="Arial" font-size="13" fill="#111827">{escape(str(row[label_key]))}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def load_history():
    path = ROOT / "outputs" / "phase2_audio_history.pkl"
    if not path.exists():
        return []

    with path.open("rb") as handle:
        history = pickle.load(handle)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])
    length = max(len(train_loss), len(val_loss), len(train_acc), len(val_acc))

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


def load_audio_performance():
    rows = []
    for row in read_csv_rows("outputs/reports/audio_model_comparison.csv"):
        rows.append(
            {
                "phase": row.get("Phase", ""),
                "technique": row.get("Technique", ""),
                "accuracy": format_float(row.get("accuracy")),
                "precision": format_float(row.get("precision")),
                "recall": format_float(row.get("recall")),
                "f1": format_float(row.get("f1")),
                "source": row.get("Source", ""),
            }
        )
    return rows


def load_modality_summary():
    rows = []
    for row in read_csv_rows("outputs/reports/modality_summary.csv"):
        rows.append(
            {
                "modality": row.get("Modality", ""),
                "best_model": row.get("Best Model", ""),
                "accuracy": format_float(row.get("accuracy")),
                "precision": format_float(row.get("precision")),
                "recall": format_float(row.get("recall")),
                "f1": format_float(row.get("f1")),
            }
        )
    return rows


def load_class_balance_rows():
    rows = []
    for row in read_csv_rows("outputs/reports/class_balance_counts.csv"):
        rows.append(
            {
                "class": row.get("Class", ""),
                "before_balancing": str(int(to_float(row.get("Before Balancing")))),
                "after_balancing": str(int(to_float(row.get("After Balancing")))),
            }
        )
    return rows


def load_class_names():
    class_names = {}
    for row in read_csv_rows("outputs/reports/audio_imbalance_plan.csv"):
        if row.get("target") not in (None, ""):
            class_names[int(to_float(row.get("target")))] = row.get("category", str(row.get("target")))

    if class_names:
        return class_names

    balance_rows = load_class_balance_rows()
    return {index: row["class"] for index, row in enumerate(balance_rows)}


def load_prediction_arrays():
    labels_path = ROOT / "outputs" / "phase2_audio_labels.npy"
    predictions_path = ROOT / "outputs" / "phase2_audio_predictions.npy"
    probabilities_path = ROOT / "outputs" / "phase2_audio_probabilities.npy"

    labels = np.load(labels_path, allow_pickle=True).astype(int)
    predictions = np.load(predictions_path, allow_pickle=True).astype(int)
    probabilities = np.load(probabilities_path, allow_pickle=True)

    count = min(len(labels), len(predictions), len(probabilities))
    return labels[:count], predictions[:count], probabilities[:count]


def compute_curve_points(labels, probabilities, curve_type):
    class_count = probabilities.shape[1]
    truth = []
    scores = []

    for label, sample_scores in zip(labels, probabilities):
        for class_index in range(class_count):
            truth.append(1 if int(label) == class_index else 0)
            scores.append(float(sample_scores[class_index]))

    positives = sum(truth)
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return [], 0.0

    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    true_positive = 0
    false_positive = 0

    if curve_type == "pr":
        points = [{"recall": "0.0000", "precision": "1.0000", "threshold": ""}]
    else:
        points = [{"fpr": "0.0000", "tpr": "0.0000", "threshold": ""}]

    for index in order:
        if truth[index]:
            true_positive += 1
        else:
            false_positive += 1

        if curve_type == "pr":
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / positives
            points.append(
                {
                    "recall": format_float(recall),
                    "precision": format_float(precision),
                    "threshold": format_float(scores[index]),
                }
            )
        else:
            tpr = true_positive / positives
            fpr = false_positive / negatives
            points.append(
                {
                    "fpr": format_float(fpr),
                    "tpr": format_float(tpr),
                    "threshold": format_float(scores[index]),
                }
            )

    if curve_type == "pr":
        x_key = "recall"
        y_key = "precision"
    else:
        x_key = "fpr"
        y_key = "tpr"

    auc = 0.0
    previous_x = to_float(points[0][x_key])
    previous_y = to_float(points[0][y_key])
    for row in points[1:]:
        current_x = to_float(row[x_key])
        current_y = to_float(row[y_key])
        auc += (current_x - previous_x) * (current_y + previous_y) / 2
        previous_x = current_x
        previous_y = current_y

    return points, auc


def compute_per_class_accuracy(labels, predictions, class_names):
    rows = []
    for label in sorted(set(labels.tolist()) | set(predictions.tolist())):
        label = int(label)
        mask = labels == label
        support = int(mask.sum())
        correct = int(((predictions == label) & mask).sum())
        accuracy = correct / support if support else 0.0
        rows.append(
            {
                "class_id": label,
                "class": class_names.get(label, str(label)),
                "support": support,
                "correct": correct,
                "accuracy": format_float(accuracy),
            }
        )
    return rows


def compute_residuals(labels, predictions, probabilities, class_names):
    rows = []
    for index, (label, prediction) in enumerate(zip(labels, predictions)):
        label = int(label)
        prediction = int(prediction)
        true_probability = (
            float(probabilities[index, label])
            if probabilities.ndim == 2 and label < probabilities.shape[1]
            else 0.0
        )
        rows.append(
            {
                "sample_index": index,
                "true_label": label,
                "true_class": class_names.get(label, str(label)),
                "predicted_label": prediction,
                "predicted_class": class_names.get(prediction, str(prediction)),
                "true_class_probability": format_float(true_probability),
                "residual": format_float(1.0 - true_probability),
                "correct": "1" if label == prediction else "0",
            }
        )
    return rows


def build_before_after_rows(performance_rows):
    baseline = next(
        (
            row for row in performance_rows
            if row["phase"] == "Phase 3" and row["technique"].lower().endswith("baseline")
        ),
        None,
    )
    if baseline is None:
        return []

    rows = []
    for row in performance_rows:
        if row["phase"] != "Phase 3" or row is baseline:
            continue

        rows.append(
            {
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
            }
        )

    rows.sort(key=lambda item: to_float(item["after_f1"]), reverse=True)
    return rows


def build_modality_gap_rows(modality_rows):
    by_modality = {row["modality"].lower(): row for row in modality_rows}
    audio = by_modality.get("audio")
    image = by_modality.get("image")
    if not audio or not image:
        return []

    return [
        {
            "comparison": "Image - Audio",
            "accuracy": format_float(to_float(image["accuracy"]) - to_float(audio["accuracy"])),
            "precision": format_float(to_float(image["precision"]) - to_float(audio["precision"])),
            "recall": format_float(to_float(image["recall"]) - to_float(audio["recall"])),
            "f1": format_float(to_float(image["f1"]) - to_float(audio["f1"])),
        }
    ]


def markdown_table(rows, columns, limit=None):
    selected = rows[:limit] if limit else rows
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    history_rows,
    performance_rows,
    before_after_rows,
    class_balance_rows,
    per_class_rows,
    modality_rows,
    gap_rows,
    pr_auc,
    roc_auc,
):
    best = max(performance_rows, key=lambda row: to_float(row["f1"])) if performance_rows else {}
    report_path = OUTPUT_DIR / "phase4_audio_report.md"

    lines = [
        "# Phase 4 Audio Report",
        "",
        "This report uses saved audio training/evaluation artifacts already present in the project. No retraining is performed.",
        "",
        "## Best Audio Result",
        "",
        f"Best row by macro F1: **{best.get('technique', 'N/A')}** with accuracy {best.get('accuracy', 'N/A')} and F1 {best.get('f1', 'N/A')}.",
        "",
        "## Audio Technique Values",
        "",
        markdown_table(performance_rows, ["phase", "technique", "accuracy", "precision", "recall", "f1", "source"]),
        "",
        "## Phase 3 Before vs After Imbalance Techniques",
        "",
        markdown_table(before_after_rows, ["technique", "before_accuracy", "after_accuracy", "accuracy_gain", "before_f1", "after_f1", "f1_gain"]),
        "",
        "## Epoch Summary",
        "",
        markdown_table(history_rows[:5] + history_rows[-5:], ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"]),
        "",
        "## Class Balance Sample",
        "",
        markdown_table(class_balance_rows[:10], ["class", "before_balancing", "after_balancing"]),
        "",
        "## Per-Class Accuracy Sample",
        "",
        markdown_table(per_class_rows[:10], ["class_id", "class", "support", "correct", "accuracy"]),
        "",
        "## Modality Summary",
        "",
        markdown_table(modality_rows, ["modality", "best_model", "accuracy", "precision", "recall", "f1"]),
        "",
        "## Modality Gap",
        "",
        markdown_table(gap_rows, ["comparison", "accuracy", "precision", "recall", "f1"]),
        "",
        "## Curve Areas",
        "",
        f"Micro-average PR AUC: {pr_auc:.4f}",
        "",
        f"Micro-average ROC AUC: {roc_auc:.4f}",
        "",
        "## Generated Files",
        "",
        markdown_table(
            [
                {"output": "Audio Phase 4 output index", "path": str(REPORTS_DIR / "phase4_audio_outputs.csv")},
                {"output": "Audio performance table", "path": str(OUTPUT_DIR / "audio_technique_performance.csv")},
                {"output": "Before/after table", "path": str(OUTPUT_DIR / "audio_before_after_performance.csv")},
                {"output": "Class balance table", "path": str(OUTPUT_DIR / "audio_class_balance_counts.csv")},
                {"output": "Epoch history table", "path": str(OUTPUT_DIR / "audio_epoch_loss_accuracy.csv")},
                {"output": "PR curve", "path": str(OUTPUT_DIR / "audio_precision_recall_curve.svg")},
                {"output": "ROC curve", "path": str(OUTPUT_DIR / "audio_roc_curve.svg")},
                {"output": "Loss curve", "path": str(OUTPUT_DIR / "audio_loss_curve.svg")},
                {"output": "Residual plot", "path": str(OUTPUT_DIR / "audio_residual_plot.svg")},
                {"output": "Per-class accuracy", "path": str(OUTPUT_DIR / "audio_per_class_accuracy.svg")},
                {"output": "Precision vs recall per modality", "path": str(OUTPUT_DIR / "audio_precision_recall_by_modality.svg")},
                {"output": "Radar chart", "path": str(OUTPUT_DIR / "audio_modality_radar_chart.svg")},
                {"output": "Modality gap heatmap", "path": str(OUTPUT_DIR / "audio_modality_gap_heatmap.svg")},
            ],
            ["output", "path"],
        ),
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    reset_output_dir()
    ensure_dirs()

    history_rows = load_history()
    performance_rows = load_audio_performance()
    before_after_rows = build_before_after_rows(performance_rows)
    class_balance_rows = load_class_balance_rows()
    modality_rows = load_modality_summary()
    gap_rows = build_modality_gap_rows(modality_rows)
    class_names = load_class_names()
    labels, predictions, probabilities = load_prediction_arrays()

    pr_rows, pr_auc = compute_curve_points(labels, probabilities, "pr")
    roc_rows, roc_auc = compute_curve_points(labels, probabilities, "roc")
    per_class_rows = compute_per_class_accuracy(labels, predictions, class_names)
    residual_rows = compute_residuals(labels, predictions, probabilities, class_names)

    write_csv(
        OUTPUT_DIR / "audio_epoch_loss_accuracy.csv",
        ["epoch", "train_loss", "val_loss", "train_accuracy_percent", "val_accuracy_percent"],
        history_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_technique_performance.csv",
        ["phase", "technique", "accuracy", "precision", "recall", "f1", "source"],
        performance_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_before_after_performance.csv",
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
        ],
        before_after_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_class_balance_counts.csv",
        ["class", "before_balancing", "after_balancing"],
        class_balance_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_precision_recall_points.csv",
        ["recall", "precision", "threshold"],
        pr_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_roc_points.csv",
        ["fpr", "tpr", "threshold"],
        roc_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_per_class_accuracy.csv",
        ["class_id", "class", "support", "correct", "accuracy"],
        per_class_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_residuals.csv",
        [
            "sample_index",
            "true_label",
            "true_class",
            "predicted_label",
            "predicted_class",
            "true_class_probability",
            "residual",
            "correct",
        ],
        residual_rows,
    )
    write_csv(
        OUTPUT_DIR / "audio_modality_gap.csv",
        ["comparison", "accuracy", "precision", "recall", "f1"],
        gap_rows,
    )

    if history_rows:
        epochs = [row["epoch"] for row in history_rows]
        write_line_svg(
            OUTPUT_DIR / "audio_loss_curve.svg",
            "Audio Training and Validation Loss",
            epochs,
            [
                {"name": "Train loss", "values": [row["train_loss"] for row in history_rows]},
                {"name": "Validation loss", "values": [row["val_loss"] for row in history_rows]},
            ],
            "Epoch",
            "Loss",
        )
        write_line_svg(
            OUTPUT_DIR / "audio_accuracy_curve.svg",
            "Audio Training and Validation Accuracy",
            epochs,
            [
                {"name": "Train accuracy", "values": [row["train_accuracy_percent"] for row in history_rows]},
                {"name": "Validation accuracy", "values": [row["val_accuracy_percent"] for row in history_rows]},
            ],
            "Epoch",
            "Accuracy (%)",
            forced_y_min=0.0,
            forced_y_max=100.0,
        )

    if pr_rows:
        sampled = downsample_points(pr_rows)
        write_line_svg(
            OUTPUT_DIR / "audio_precision_recall_curve.svg",
            "Audio Precision-Recall Curve",
            [row["recall"] for row in sampled],
            [{"name": "Micro-average PR", "values": [row["precision"] for row in sampled]}],
            "Recall",
            "Precision",
            forced_y_min=0.0,
            forced_y_max=1.0,
            forced_x_min=0.0,
            forced_x_max=1.0,
        )

    if roc_rows:
        sampled = downsample_points(roc_rows)
        write_line_svg(
            OUTPUT_DIR / "audio_roc_curve.svg",
            "Audio ROC Curve",
            [row["fpr"] for row in sampled],
            [{"name": "Micro-average ROC", "values": [row["tpr"] for row in sampled]}],
            "False positive rate",
            "True positive rate",
            forced_y_min=0.0,
            forced_y_max=1.0,
            forced_x_min=0.0,
            forced_x_max=1.0,
        )

    if class_balance_rows:
        write_horizontal_grouped_bar_svg(
            OUTPUT_DIR / "audio_class_balance_before_after.svg",
            "Audio Class Counts Before and After Balancing",
            class_balance_rows,
            "class",
            ["before_balancing", "after_balancing"],
        )
        write_horizontal_bar_svg(
            OUTPUT_DIR / "audio_class_counts_before_balance.svg",
            "Audio Class Counts Before Balancing",
            class_balance_rows,
            "class",
            "before_balancing",
            "Samples",
        )
        write_horizontal_bar_svg(
            OUTPUT_DIR / "audio_class_counts_after_balance.svg",
            "Audio Class Counts After Balancing",
            class_balance_rows,
            "class",
            "after_balancing",
            "Samples",
        )

    if per_class_rows:
        write_horizontal_bar_svg(
            OUTPUT_DIR / "audio_per_class_accuracy.svg",
            "Audio Per-Class Accuracy",
            per_class_rows,
            "class",
            "accuracy",
            "Accuracy",
        )

    if residual_rows:
        write_scatter_svg(
            OUTPUT_DIR / "audio_residual_plot.svg",
            "Audio Classification Residuals",
            residual_rows,
        )

    if modality_rows:
        write_grouped_bar_svg(
            OUTPUT_DIR / "audio_precision_recall_by_modality.svg",
            "Precision vs Recall per Modality",
            modality_rows,
            "modality",
            ["precision", "recall"],
            "Score",
        )
        write_radar_svg(
            OUTPUT_DIR / "audio_modality_radar_chart.svg",
            "Modality Radar Chart",
            modality_rows,
            "modality",
            METRICS,
        )

    if gap_rows:
        write_heatmap_svg(
            OUTPUT_DIR / "audio_modality_gap_heatmap.svg",
            "Modality Performance Gap",
            gap_rows,
            "comparison",
            METRICS,
        )

    outputs = [
        {"requirement": "Audio count plot before balancing", "output": str(OUTPUT_DIR / "audio_class_counts_before_balance.svg")},
        {"requirement": "Audio count plot after balancing", "output": str(OUTPUT_DIR / "audio_class_counts_after_balance.svg")},
        {"requirement": "Audio count plot before vs after", "output": str(OUTPUT_DIR / "audio_class_balance_before_after.svg")},
        {"requirement": "Audio PR curve", "output": str(OUTPUT_DIR / "audio_precision_recall_curve.svg")},
        {"requirement": "Audio ROC curve", "output": str(OUTPUT_DIR / "audio_roc_curve.svg")},
        {"requirement": "Audio loss graph", "output": str(OUTPUT_DIR / "audio_loss_curve.svg")},
        {"requirement": "Audio accuracy graph", "output": str(OUTPUT_DIR / "audio_accuracy_curve.svg")},
        {"requirement": "Audio residual plot", "output": str(OUTPUT_DIR / "audio_residual_plot.svg")},
        {"requirement": "Audio per-class accuracy", "output": str(OUTPUT_DIR / "audio_per_class_accuracy.svg")},
        {"requirement": "Audio precision vs recall per modality", "output": str(OUTPUT_DIR / "audio_precision_recall_by_modality.svg")},
        {"requirement": "Audio radar chart", "output": str(OUTPUT_DIR / "audio_modality_radar_chart.svg")},
        {"requirement": "Audio modality gap heatmap", "output": str(OUTPUT_DIR / "audio_modality_gap_heatmap.svg")},
        {"requirement": "Audio Phase 4 report", "output": str(OUTPUT_DIR / "phase4_audio_report.md")},
    ]
    write_csv(REPORTS_DIR / "phase4_audio_outputs.csv", ["requirement", "output"], outputs)

    report_path = write_report(
        history_rows,
        performance_rows,
        before_after_rows,
        class_balance_rows,
        per_class_rows,
        modality_rows,
        gap_rows,
        pr_auc,
        roc_auc,
    )
    print(f"Phase 4 audio report generated: {report_path}")


if __name__ == "__main__":
    main()
