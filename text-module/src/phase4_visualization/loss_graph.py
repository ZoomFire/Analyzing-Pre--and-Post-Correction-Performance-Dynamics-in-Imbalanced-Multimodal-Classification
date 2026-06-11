from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from common import REPORTS_DIR, ensure_output_dir, existing_path, parse_output_dir_argument


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and validation loss from a history CSV.")
    parser.add_argument("--history", default=str(REPORTS_DIR / "text_learning_history.csv"), help="CSV with epoch and loss columns.")
    parser.add_argument("--output", default="text_loss_curve.png", help="Output image filename.")
    parse_output_dir_argument(parser)
    args = parser.parse_args()

    history = pd.read_csv(existing_path(args.history, "learning history"))
    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"{args.history} is missing columns: {', '.join(sorted(missing))}")

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"].astype(float), label="Train Loss", marker="o")
    plt.plot(history["epoch"], history["val_loss"].astype(float), label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = ensure_output_dir(args.output_dir) / args.output
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
