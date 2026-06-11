from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import PHASE4_TEXT_DIR


SCRIPTS = [
    "classification_heatmap.py",
    "confusion_matrix.py",
    "loss_graph.py",
    "tfidf_barchart.py",
    "wordcloud_visualization.py",
    "comparison.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Phase 4 text visualizations.")
    parser.add_argument("--output-dir", default=str(PHASE4_TEXT_DIR), help="Directory where images will be saved.")
    parser.add_argument("--skip-wordcloud", action="store_true", help="Skip word cloud generation if Reuters data is unavailable.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        if args.skip_wordcloud and script == "wordcloud_visualization.py":
            continue
        command = [sys.executable, str(script_dir / script), "--output-dir", args.output_dir]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
