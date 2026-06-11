from collections import defaultdict
from html import escape
from pathlib import Path
import re


def _slugify(value):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "confusion_matrix"


def _cell_color(value, max_value):
    if max_value <= 0:
        intensity = 0
    else:
        intensity = value / max_value

    red = int(242 - 178 * intensity)
    green = int(247 - 112 * intensity)
    blue = int(255 - 55 * intensity)
    return f"rgb({red},{green},{blue})"


def plot_confusion_matrix(y_true, y_pred, title, save_path=None):
    """Save a dependency-light SVG confusion matrix and return its path."""
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {label: index for index, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for actual, predicted in zip(y_true, y_pred):
        matrix[label_to_index[actual]][label_to_index[predicted]] += 1

    if save_path is None:
        save_dir = Path("outputs") / "phase4" / "confusion_matrices"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{_slugify(title)}_confusion_matrix.svg"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    max_value = max((value for row in matrix for value in row), default=0)
    cell = 32 if size <= 15 else 18
    left = 120
    top = 90
    width = left + size * cell + 40
    height = top + size * cell + 90

    label_font = 11 if size <= 15 else 8
    value_font = 11 if size <= 15 else 7

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{escape(str(title))}</text>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Predicted label</text>',
        f'<text x="22" y="{top + (size * cell) / 2}" transform="rotate(-90 22 {top + (size * cell) / 2})" text-anchor="middle" font-family="Arial" font-size="13">Actual label</text>',
    ]

    for index, label in enumerate(labels):
        x = left + index * cell + cell / 2
        y = top - 10
        parts.append(
            f'<text x="{x}" y="{y}" transform="rotate(-45 {x} {y})" '
            f'text-anchor="start" font-family="Arial" font-size="{label_font}">{escape(str(label))}</text>'
        )

        row_y = top + index * cell + cell / 2 + 4
        parts.append(
            f'<text x="{left - 10}" y="{row_y}" text-anchor="end" '
            f'font-family="Arial" font-size="{label_font}">{escape(str(label))}</text>'
        )

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            x = left + column_index * cell
            y = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_cell_color(value, max_value)}" stroke="#d6dee8" stroke-width="1"/>'
            )
            if size <= 25:
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                    f'font-family="Arial" font-size="{value_font}" fill="#111827">{value}</text>'
                )

    parts.append("</svg>")
    save_path.write_text("\n".join(parts), encoding="utf-8")
    return str(save_path)


def confusion_counts(y_true, y_pred):
    counts = defaultdict(int)
    for actual, predicted in zip(y_true, y_pred):
        counts[(actual, predicted)] += 1
    return counts

