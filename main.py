"""
Main entry point for the MEG classification project.

EDA only (default):
    python main.py

Train all models — intra only:
    python main.py --train --intra

Train all models — cross only:
    python main.py --train --cross

Train all models — intra + cross:
    python main.py --train
"""

import argparse
import importlib
from pathlib import Path

DATA_DIR   = Path("Final Project data")
OUTPUT_DIR = Path("outputs/eda")

MODELS = [
    "simple_cnn",
    "resnet",
    "cnn_gru",
    "eegnet",
    "cnn_lstm_attn",
    "meg_graphnet",
]


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def run_eda() -> None:
    from src.data.data_loading import inspect_folder
    from src.data.eda import (
        plot_raw_vs_normalised,
        plot_class_balance,
        plot_signal_per_class,
        plot_value_distribution,
    )

    intra_train = DATA_DIR / "Intra"  / "train"
    intra_test  = DATA_DIR / "Intra"  / "test"
    cross_train = DATA_DIR / "Cross"  / "train"
    cross_test1 = DATA_DIR / "Cross"  / "test1"
    cross_test2 = DATA_DIR / "Cross"  / "test2"
    cross_test3 = DATA_DIR / "Cross"  / "test3"

    print("=== Dataset Inspection ===")
    for folder in (intra_train, intra_test, cross_train, cross_test1, cross_test2, cross_test3):
        inspect_folder(folder)

    print("\n=== Generating EDA Plots ===\n")
    plot_raw_vs_normalised(intra_train)
    plot_class_balance({
        "Intra/train": intra_train,
        "Intra/test":  intra_test,
        "Cross/train": cross_train,
        "Cross/test1": cross_test1,
        "Cross/test2": cross_test2,
        "Cross/test3": cross_test3,
    })
    plot_signal_per_class(intra_train)
    plot_value_distribution(intra_train)
    print(f"\nDone. Plots saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_module(module_path: str, model_name: str) -> None:
    mod = importlib.import_module(module_path)
    original = mod.MODEL_NAME
    mod.MODEL_NAME = model_name
    try:
        print(f"\n{'=' * 70}")
        print(f"  {module_path.split('.')[-1]}  |  model: {model_name}")
        print(f"{'=' * 70}\n")
        mod.main()
    finally:
        mod.MODEL_NAME = original


def run_training(intra: bool, cross: bool) -> None:
    for model in MODELS:
        if intra:
            _run_module("src.models.train", model)
        if cross:
            _run_module("src.models.train_cross", model)

    print(f"\n{'=' * 70}")
    print("All models done. Generating training plots...")
    print(f"{'=' * 70}\n")
    import src.evaluate.train_plots as plots
    plots.main()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--intra", action="store_true", help="Intra-subject only (requires --train)")
    parser.add_argument("--cross", action="store_true", help="Cross-subject only (requires --train)")
    args = parser.parse_args()

    if args.train:
        # If neither flag given, run both
        do_intra = args.intra or (not args.intra and not args.cross)
        do_cross = args.cross or (not args.intra and not args.cross)
        run_training(intra=do_intra, cross=do_cross)
    else:
        run_eda()


if __name__ == "__main__":
    main()
