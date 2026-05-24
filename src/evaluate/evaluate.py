"""
Evaluate saved checkpoints on held-out test sets and generate report figures.

Intra-subject:
    python evaluate.py

Cross-subject (run after python run_all_models.py --cross):
    python evaluate.py --cross

Both:
    python evaluate.py --both
"""

import argparse
from pathlib import Path

DATA_DIR   = Path("Final Project data")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--cross", action="store_true", help="Evaluate cross-subject checkpoints")
    group.add_argument("--both",  action="store_true", help="Evaluate intra then cross")
    args = parser.parse_args()

    run_intra = not args.cross
    run_cross = args.cross or args.both

    if run_intra:
        from src.evaluate.intra import run as run_intra_eval
        run_intra_eval(DATA_DIR, OUTPUT_DIR)

    if run_cross:
        from src.evaluate.cross import run as run_cross_eval
        run_cross_eval(DATA_DIR, OUTPUT_DIR)

    # Regenerate training plots now that eval JSONs are available;
    # this produces true_train_test_gap.png combining train acc with test acc.
    from src.evaluate.train_plots import main as update_train_plots
    print("\nUpdating training plots with test results ...")
    update_train_plots()


if __name__ == "__main__":
    main()
