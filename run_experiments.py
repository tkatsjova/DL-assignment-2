"""
Experiment runner — trains multiple model/hyperparameter combinations sequentially.

Usage:
    python run_experiments.py                  # baseline intra-subject only
    python run_experiments.py --cross          # baseline intra + cross-subject
    python run_experiments.py --improve        # improvement experiments (intra)
    python run_experiments.py --improve --cross  # improvement intra + cross
    python run_experiments.py --summary        # print summary of existing results only

Results are saved to outputs/ as JSON files (one per experiment).
A summary table is printed when all experiments finish.
Experiments that already have a saved JSON are skipped automatically,
so you can safely re-run after an interruption.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.data_loading import list_h5_files
from src.data.dataset import MEGWindowDataset, create_dataloader
from src.models.train import (
    set_seed, get_device, get_model,
    stratified_split, train_one_epoch, check_accuracy,
)

# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------

# Core architecture comparison.
# These are the clean report baselines.
MODEL_EXPERIMENTS = [
    {"model": "simple_cnn",   "lr": 1e-3, "batch_size": 16},
    {"model": "cnn_gru",      "lr": 1e-3, "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16},
]


# Larger but targeted hyperparameter search.
# Goal:
# - keep SimpleCNN as baseline,
# - tune CNN-GRU fairly,
# - tune CNN-GRU-Attention fairly,
# - avoid only optimizing one model family.
HYPERPARAM_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # SimpleCNN baseline tuning
    # -----------------------------------------------------------------------
    {"model": "simple_cnn", "lr": 2e-3,   "batch_size": 16},
    {"model": "simple_cnn", "lr": 1.5e-3, "batch_size": 16},
    {"model": "simple_cnn", "lr": 7e-4,   "batch_size": 16},
    {"model": "simple_cnn", "lr": 5e-4,   "batch_size": 16},
    {"model": "simple_cnn", "lr": 3e-4,   "batch_size": 16},
    {"model": "simple_cnn", "lr": 1e-4,   "batch_size": 16},

    {"model": "simple_cnn", "lr": 1e-3, "batch_size": 8},
    {"model": "simple_cnn", "lr": 1e-3, "batch_size": 32},

    # small regularization check for baseline
    {"model": "simple_cnn", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "simple_cnn", "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "simple_cnn", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-3},

    # -----------------------------------------------------------------------
    # CNN-GRU learning-rate tuning
    # -----------------------------------------------------------------------
    {"model": "cnn_gru", "lr": 2e-3,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 1.5e-3, "batch_size": 16},
    {"model": "cnn_gru", "lr": 1.2e-3, "batch_size": 16},
    {"model": "cnn_gru", "lr": 1e-3,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 8e-4,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 7e-4,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 5e-4,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 3e-4,   "batch_size": 16},
    {"model": "cnn_gru", "lr": 1e-4,   "batch_size": 16},

    # CNN-GRU batch-size check
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 8},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 32},
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 8},
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 32},

    # CNN-GRU weight decay around good learning rates
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-5},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-5},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-3},

    {"model": "cnn_gru", "lr": 7e-4, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru", "lr": 7e-4, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru", "lr": 7e-4, "batch_size": 16, "weight_decay": 5e-3},

    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 16, "weight_decay": 5e-3},

    # -----------------------------------------------------------------------
    # CNN-GRU-Attention learning-rate tuning
    # -----------------------------------------------------------------------
    {"model": "cnn_gru_attn", "lr": 2e-3,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1.5e-3, "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1.2e-3, "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1e-3,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 8e-4,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 7e-4,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 5e-4,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 3e-4,   "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1e-4,   "batch_size": 16},

    # CNN-GRU-Attention batch-size check
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 8},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 32},
    {"model": "cnn_gru_attn", "lr": 5e-4, "batch_size": 8},
    {"model": "cnn_gru_attn", "lr": 5e-4, "batch_size": 32},

    # CNN-GRU-Attention weight decay around good learning rates
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-5},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-5},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "weight_decay": 5e-3},

    {"model": "cnn_gru_attn", "lr": 7e-4, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru_attn", "lr": 7e-4, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru_attn", "lr": 7e-4, "batch_size": 16, "weight_decay": 5e-3},

    {"model": "cnn_gru_attn", "lr": 5e-4, "batch_size": 16, "weight_decay": 5e-4},
    {"model": "cnn_gru_attn", "lr": 5e-4, "batch_size": 16, "weight_decay": 1e-3},
    {"model": "cnn_gru_attn", "lr": 5e-4, "batch_size": 16, "weight_decay": 5e-3},
]


# Improvement experiments.
# These focus on overfitting and window correlation:
# - dropout
# - stronger weight decay
# - reduced overlap
# - longer windows
# - combined regularization + preprocessing
IMPROVEMENT_EXPERIMENTS = [
    # -----------------------------------------------------------------------
    # CNN-GRU-Attention regularization
    # Your previous run showed 0.5/0.6 dropout did not help much,
    # so include milder dropout too.
    # -----------------------------------------------------------------------
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.2, "weight_decay": 1e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.5, "weight_decay": 1e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.6, "weight_decay": 1e-4},

    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.5, "weight_decay": 5e-4},

    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.5, "weight_decay": 1e-3},

    # -----------------------------------------------------------------------
    # CNN-GRU regularization too, because CNN-GRU was competitive before.
    # -----------------------------------------------------------------------
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.2, "weight_decay": 1e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.5, "weight_decay": 1e-4},

    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3},

    # -----------------------------------------------------------------------
    # Preprocessing-only sweep for CNN-GRU-Attention
    # -----------------------------------------------------------------------
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "overlap": 0.25},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "overlap": 0.0},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "window_seconds": 1.5},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "window_seconds": 2.5},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "window_seconds": 3.0},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "window_seconds": 4.0},

    # -----------------------------------------------------------------------
    # Preprocessing-only sweep for CNN-GRU
    # -----------------------------------------------------------------------
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "overlap": 0.25},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "overlap": 0.0},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "window_seconds": 1.5},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "window_seconds": 2.5},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "window_seconds": 3.0},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "window_seconds": 4.0},

    # -----------------------------------------------------------------------
    # Combined regularization + reduced overlap
    # -----------------------------------------------------------------------
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4, "overlap": 0.25},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4, "overlap": 0.25},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3, "overlap": 0.25},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3, "overlap": 0.25},

    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4, "overlap": 0.25},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4, "overlap": 0.25},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3, "overlap": 0.25},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3, "overlap": 0.25},

    # -----------------------------------------------------------------------
    # Combined regularization + longer windows
    # -----------------------------------------------------------------------
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4, "window_seconds": 3.0},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4, "window_seconds": 3.0},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3, "window_seconds": 3.0},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3, "window_seconds": 3.0},

    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 5e-4, "window_seconds": 3.0},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 5e-4, "window_seconds": 3.0},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.3, "weight_decay": 1e-3, "window_seconds": 3.0},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 16, "dropout": 0.4, "weight_decay": 1e-3, "window_seconds": 3.0},
]


# Remove exact duplicates while keeping order.
_seen = set()
ALL_EXPERIMENTS = []

for cfg in MODEL_EXPERIMENTS + HYPERPARAM_EXPERIMENTS:
    key = (
        cfg.get("model"),
        cfg.get("lr"),
        cfg.get("batch_size"),
        cfg.get("dropout"),
        cfg.get("weight_decay", 1e-4),
        cfg.get("overlap"),
        cfg.get("window_seconds"),
    )

    if key not in _seen:
        ALL_EXPERIMENTS.append(cfg)
        _seen.add(key)

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

SEED        = 42
N_EPOCHS    = 100
ES_PATIENCE = 15
LR_PATIENCE = 7
DATA_DIR    = Path("Final Project data") / "Final Project data"
OUTPUT_DIR  = Path("outputs")

DEFAULT_DATASET_PARAMS = dict(
    original_sampling_rate=ORIGINAL_SAMPLING_RATE,
    downsample_factor=DOWNSAMPLE_FACTOR,
    window_seconds=WINDOW_SECONDS,
    overlap=OVERLAP,
)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_one(
    model_name: str,
    lr: float,
    batch_size: int,
    train_folder: Path,
    scenario: str,
    device: torch.device,
    dropout: float | None = None,
    weight_decay: float = 1e-4,
    overlap: float | None = None,
    window_seconds: float | None = None,
) -> dict:

    # Build a unique run key that encodes every non-default param
    run_key = f"{scenario}_{model_name}_lr{lr:.1e}_bs{batch_size}"
    if dropout is not None:
        run_key += f"_do{dropout}"
    if weight_decay != 1e-4:
        run_key += f"_wd{weight_decay:.0e}"
    if overlap is not None:
        run_key += f"_ov{overlap}"
    if window_seconds is not None:
        run_key += f"_win{window_seconds}"

    json_path = OUTPUT_DIR / f"results_{run_key}.json"

    if json_path.exists():
        print(f"\n[SKIP] {run_key} — already done\n")
        with open(json_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  {run_key}")
    print(f"{'='*60}")

    set_seed(SEED)

    dataset_params = dict(DEFAULT_DATASET_PARAMS)
    if overlap is not None:
        dataset_params["overlap"] = overlap
    if window_seconds is not None:
        dataset_params["window_seconds"] = window_seconds

    all_files              = list_h5_files(train_folder)
    train_files, val_files = stratified_split(all_files, val_ratio=0.2, seed=SEED)

    train_data   = MEGWindowDataset(files=train_files, **dataset_params)
    val_data     = MEGWindowDataset(files=val_files,   **dataset_params)
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(val_data,   batch_size=batch_size, shuffle=False)

    model     = get_model(model_name, device, dropout=dropout)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE,
    )

    best_val_acc = 0.0
    es_counter   = 0
    history      = []
    save_path    = OUTPUT_DIR / f"best_{run_key}.pt"

    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss,   val_acc   = check_accuracy(model, val_loader,   loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{N_EPOCHS} | "
            f"train {train_acc:.3f} | val {val_acc:.3f} | "
            f"LR {current_lr:.1e}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
        })

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            es_counter   = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy":     best_val_acc,
                "epoch":            epoch,
                "model_name":       model_name,
            }, save_path)
            print(f"  → New best: {best_val_acc:.4f}")
        else:
            es_counter += 1
            if es_counter >= ES_PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    elapsed     = time.time() - t_start
    best_epoch  = max(history, key=lambda h: h["val_acc"])["epoch"] if history else 0
    final_train = history[-1]["train_acc"] if history else 0.0

    results = {
        "model_name":      model_name,
        "scenario":        scenario,
        "lr":              lr,
        "batch_size":      batch_size,
        "dropout":         dropout,
        "weight_decay":    weight_decay,
        "overlap":         dataset_params["overlap"],
        "window_seconds":  dataset_params["window_seconds"],
        "n_params":        sum(p.numel() for p in model.parameters()),
        "best_val_acc":    round(best_val_acc, 4),
        "final_train_acc": round(final_train, 4),
        "best_epoch":      best_epoch,
        "training_time_s": round(elapsed, 1),
        "history":         history,
    }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinished in {elapsed/60:.1f} min — best val acc: {best_val_acc:.4f}")
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    if not all_results:
        print("No results to show.")
        return

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Scenario':<8} {'Model':<14} {'LR':<8} {'BS':<4} {'DO':<5} {'WD':<8} "
          f"{'Val Acc':>8} {'Train Acc':>10} {'Gap':>7} {'Ep':>4}")
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: x["best_val_acc"], reverse=True):
        gap = r["final_train_acc"] - r["best_val_acc"]
        do  = r.get("dropout") or "-"
        wd  = r.get("weight_decay", 1e-4)
        print(
            f"{r.get('scenario', '?'):<8} "
            f"{r['model_name']:<14} "
            f"{r['lr']:<8.1e} "
            f"{r['batch_size']:<4} "
            f"{str(do):<5} "
            f"{wd:<8.0e} "
            f"{r['best_val_acc']:>8.4f} "
            f"{r['final_train_acc']:>10.4f} "
            f"{gap:>+7.3f} "
            f"{r['best_epoch']:>4}"
        )

    print("=" * 90)
    print("Gap = train_acc - val_acc  (positive = overfitting)")


def load_existing_results() -> list[dict]:
    results = []
    for path in sorted(OUTPUT_DIR.glob("results_*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross",   action="store_true",
                        help="Also run cross-subject experiments")
    parser.add_argument("--improve", action="store_true",
                        help="Run improvement experiments (regularisation + preprocessing sweep on CNN-GRU-Attn)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing results without running anything")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.summary:
        print_summary(load_existing_results())
        return

    device = get_device()
    print(f"Device : {device}")

    experiments = IMPROVEMENT_EXPERIMENTS if args.improve else ALL_EXPERIMENTS
    label       = "improvement" if args.improve else "baseline"
    print(f"Running {len(experiments)} {label} intra-subject experiment(s)...")
    if args.cross:
        cross_exps = IMPROVEMENT_EXPERIMENTS if args.improve else MODEL_EXPERIMENTS
        print(f"  + {len(cross_exps)} cross-subject experiment(s)")

    all_results  = []
    intra_folder = DATA_DIR / "Intra" / "train"

    for cfg in experiments:
        result = run_one(
            model_name     = cfg["model"],
            lr             = cfg["lr"],
            batch_size     = cfg["batch_size"],
            train_folder   = intra_folder,
            scenario       = "intra",
            device         = device,
            dropout        = cfg.get("dropout"),
            weight_decay   = cfg.get("weight_decay", 1e-4),
            overlap        = cfg.get("overlap"),
            window_seconds = cfg.get("window_seconds"),
        )
        all_results.append(result)

    if args.cross:
        cross_folder = DATA_DIR / "Cross" / "train"
        cross_exps   = IMPROVEMENT_EXPERIMENTS if args.improve else MODEL_EXPERIMENTS
        for cfg in cross_exps:
            result = run_one(
                model_name     = cfg["model"],
                lr             = cfg["lr"],
                batch_size     = cfg["batch_size"],
                train_folder   = cross_folder,
                scenario       = "cross",
                device         = device,
                dropout        = cfg.get("dropout"),
                weight_decay   = cfg.get("weight_decay", 1e-4),
                overlap        = cfg.get("overlap"),
                window_seconds = cfg.get("window_seconds"),
            )
            all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
