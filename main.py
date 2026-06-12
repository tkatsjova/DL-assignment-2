"""
Main entry point for the MEG classification project.

EDA only (default):
    python main.py

Train all models with default hyperparameters — intra + cross:
    python main.py --train
    python main.py --train --intra   # intra only
    python main.py --train --cross   # cross only

Run curated report configs (comparison ablations + best models):
    python main.py --final

Evaluate saved checkpoints on held-out test sets and generate report figures:
    python main.py --evaluate              # both scenarios (default)
    python main.py --evaluate --intra      # intra only
    python main.py --evaluate --cross      # cross only
"""

import argparse
import importlib
from pathlib import Path

DATA_DIR = Path("Final Project data")
OUTPUT_DIR = Path("outputs")

MODELS = [
    "simple_cnn",
    "resnet",
    "cnn_gru",
    "eegnet",
    "cnn_gru_attn",
]

FINAL_CONFIGS = [
    # ── WD ablation — CNN-GRU-Attn, intra ───────────────────────────────────
    # Three WD values show why WD=5e-3 was chosen (tighter train-val gap).
    # Each run gets its own learning-curve plot so you can compare convergence.
    {"module": "src.models.train",       "model": "cnn_gru_attn", "weight_decay": 1e-4},   # baseline
    {"module": "src.models.train",       "model": "cnn_gru_attn", "weight_decay": 1e-3},   # medium
    {"module": "src.models.train",       "model": "cnn_gru_attn", "weight_decay": 5e-3},   # best ←

    # ── Remaining best intra models ──────────────────────────────────────────
    {"module": "src.models.train",       "model": "cnn_gru",      "weight_decay": 5e-3},
    {"module": "src.models.train",       "model": "simple_cnn",   "weight_decay": 1e-4},

    # ── WD + dropout ablation — CNN-GRU, cross ───────────────────────────────
    # Three configs isolate the contribution of each regulariser:
    #   1) no WD boost, no extra dropout → weakest baseline
    #   2) WD=1e-3, no extra dropout     → WD effect only
    #   3) WD=1e-3, dropout=0.3          → WD + dropout (expected best)
    {"module": "src.models.train_cross", "model": "cnn_gru", "weight_decay": 1e-4},                      # baseline
    {"module": "src.models.train_cross", "model": "cnn_gru", "weight_decay": 1e-3},                      # WD only
    {"module": "src.models.train_cross", "model": "cnn_gru", "weight_decay": 1e-3, "dropout": 0.3},      # best ←

    # ── Remaining best cross models ───────────────────────────────────────────
    {"module": "src.models.train_cross", "model": "simple_cnn",   "weight_decay": 1e-4},
    {"module": "src.models.train_cross", "model": "cnn_gru_attn", "weight_decay": 1e-4},
]


def run_eda() -> None:
    from src.data.data_loading import inspect_folder
    from src.data.eda import (
        plot_raw_vs_normalised,
        plot_class_balance,
        plot_signal_per_class,
        plot_value_distribution,
    )

    intra_train = DATA_DIR / "Intra" / "train"
    intra_test  = DATA_DIR / "Intra" / "test"
    cross_train = DATA_DIR / "Cross" / "train"
    cross_test1 = DATA_DIR / "Cross" / "test1"
    cross_test2 = DATA_DIR / "Cross" / "test2"
    cross_test3 = DATA_DIR / "Cross" / "test3"

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
    print(f"\nDone. Plots saved to {OUTPUT_DIR}/eda/")


def _run_module(
    module_path: str,
    model_name: str,
    weight_decay: float = 1e-4,
    dropout: float | None = None,
) -> None:

    mod = importlib.import_module(module_path)

    wd_tag = f"_wd{weight_decay:.0e}" if abs(weight_decay - 1e-4) > 1e-10 else ""
    do_tag = f"_do{dropout}" if dropout is not None else ""
    new_suffix = f"_lr{mod.LR:.0e}_bs{mod.BATCH_SIZE}{wd_tag}{do_tag}"

    scenario = "cross" if "cross" in module_path else "intra"
    json_path = OUTPUT_DIR / f"results_{scenario}_{model_name}{new_suffix}.json"
    if json_path.exists():
        print(f"\n[SKIP] {json_path.name} — already done\n")
        return

    orig = {
        "MODEL_NAME": mod.MODEL_NAME,
        "WEIGHT_DECAY": getattr(mod, "WEIGHT_DECAY", 1e-4),
        "DROPOUT": getattr(mod, "DROPOUT", None),
        "RUN_SUFFIX": mod.RUN_SUFFIX,
    }

    mod.MODEL_NAME = model_name
    mod.WEIGHT_DECAY = weight_decay
    mod.DROPOUT = dropout
    mod.RUN_SUFFIX = new_suffix

    script = module_path.split(".")[-1]
    wd_str = f"wd={weight_decay:.0e}"
    do_str = f"  dropout={dropout}" if dropout is not None else ""
    print(f"\n{'=' * 70}")
    print(f"  {script}  |  model: {model_name}  |  {wd_str}{do_str}")
    print(f"{'=' * 70}\n")

    try:
        mod.main()
    finally:
        for k, v in orig.items():
            setattr(mod, k, v)


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


def run_evaluate(intra: bool, cross: bool) -> None:
    """Evaluate saved checkpoints on test sets and regenerate all report plots."""
    if intra:
        from src.evaluate.intra import run as run_intra_eval
        run_intra_eval(DATA_DIR, OUTPUT_DIR)

    if cross:
        from src.evaluate.cross import run as run_cross_eval
        run_cross_eval(DATA_DIR, OUTPUT_DIR)

    # Regenerate training plots now that eval JSONs exist;
    # this produces true_train_test_gap.png combining train acc with test acc.
    from src.evaluate.train_plots import main as update_train_plots
    print(f"\n{'=' * 70}")
    print("Updating training plots with test results...")
    print(f"{'=' * 70}\n")
    update_train_plots()


def run_final() -> None:
    """Run the FINAL_CONFIGS list then regenerate all report plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    n = len(FINAL_CONFIGS)
    for i, cfg in enumerate(FINAL_CONFIGS, 1):
        print(f"\n[{i}/{n}]", end="")
        _run_module(
            module_path=cfg["module"],
            model_name=cfg["model"],
            weight_decay=cfg.get("weight_decay", 1e-4),
            dropout=cfg.get("dropout"),
        )

    print(f"\n{'=' * 70}")
    print("All final configs done. Generating report plots...")
    print(f"{'=' * 70}\n")
    import src.evaluate.train_plots as plots
    plots.main()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train all models with default hyperparameters")
    parser.add_argument("--final", action="store_true",
                        help="Run curated report configs (WD/dropout ablations + best models)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate saved checkpoints on test sets and generate report figures")
    parser.add_argument("--intra", action="store_true",
                        help="Intra-subject only (used with --train / --evaluate)")
    parser.add_argument("--cross", action="store_true",
                        help="Cross-subject only (used with --train / --evaluate)")
    args = parser.parse_args()

    do_intra = args.intra or (not args.intra and not args.cross)
    do_cross = args.cross or (not args.intra and not args.cross)

    if args.final:
        run_final()
    elif args.train:
        run_training(intra=do_intra, cross=do_cross)
    elif args.evaluate:
        run_evaluate(intra=do_intra, cross=do_cross)
    else:
        run_eda()


if __name__ == "__main__":
    main()
