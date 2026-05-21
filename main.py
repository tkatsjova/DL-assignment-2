"""
Entry points for the MEG classification project.

EDA only:
    python main.py

Training (intra + cross) followed by plots:
    python main.py --train
"""

import argparse
from pathlib import Path

from src.data.data_loading import inspect_folder
from src.data.eda import (
    plot_raw_vs_normalised,
    plot_class_balance,
    plot_signal_per_class,
    plot_value_distribution,
)

DATA_DIR = Path("Final Project data")


def main() -> None:
    intra_train  = DATA_DIR / "Intra" / "train"
    intra_test   = DATA_DIR / "Intra" / "test"
    cross_train  = DATA_DIR / "Cross" / "train"
    cross_test1  = DATA_DIR / "Cross" / "test1"
    cross_test2  = DATA_DIR / "Cross" / "test2"
    cross_test3  = DATA_DIR / "Cross" / "test3"

    # ── 1. Dataset inspection ──────────────────────────────────────────────────
    print("=== Dataset Inspection ===")
    inspect_folder(intra_train)
    inspect_folder(intra_test)
    inspect_folder(cross_train)
    inspect_folder(cross_test1)
    inspect_folder(cross_test2)
    inspect_folder(cross_test3)

    # ── 2. EDA plots ───────────────────────────────────────────────────────────
    print("\n=== Generating EDA Plots ===\n")

    print("Plot 1: Raw vs Normalised signal...")
    plot_raw_vs_normalised(intra_train)

    print("Plot 2: Class balance...")
    plot_class_balance({
        "Intra/train":  intra_train,
        "Intra/test":   intra_test,
        "Cross/train":  cross_train,
        "Cross/test1":  cross_test1,
        "Cross/test2":  cross_test2,
        "Cross/test3":  cross_test3,
    })

    print("Plot 3: Signal per class...")
    plot_signal_per_class(intra_train)

    print("Plot 4: Value distribution...")
    plot_value_distribution(intra_train)

    print(f"\nDone. Plots saved to: {OUTPUT_DIR}/")


def train_and_plot() -> None:
    import src.models.train as intra
    import src.models.train_cross as cross
    import src.models.plot_results as plots

    print("=" * 60)
    print("Step 1/3 — Intra-subject training")
    print("=" * 60)
    intra.main()

    print("\n" + "=" * 60)
    print("Step 2/3 — Cross-subject training")
    print("=" * 60)
    cross.main()

    print("\n" + "=" * 60)
    print("Step 3/3 — Generating plots")
    print("=" * 60)
    plots.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Run intra + cross training, then generate plots")
    args = parser.parse_args()

    if args.train:
        train_and_plot()
    else:
        main()
