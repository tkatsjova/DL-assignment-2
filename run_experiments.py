"""
Experiment runner — trains multiple model/hyperparameter combinations sequentially.

Usage:
    uv run python run_experiments.py              # intra-subject only
    uv run python run_experiments.py --cross      # intra + cross-subject
    uv run python run_experiments.py --summary    # print summary of existing results only

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
# Experiment grid — edit this to add/remove experiments
# ---------------------------------------------------------------------------

# Three models, default hyperparameters — covers assignment tasks (a) and (b)
MODEL_EXPERIMENTS = [
    {"model": "simple_cnn",   "lr": 1e-3, "batch_size": 16},
    {"model": "cnn_gru",      "lr": 1e-3, "batch_size": 16},
    {"model": "cnn_gru_attn", "lr": 1e-3, "batch_size": 16},
]

# Vary LR and batch size on CNN-GRU — covers assignment task (c)
HYPERPARAM_EXPERIMENTS = [
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 16},
    {"model": "cnn_gru", "lr": 1e-4, "batch_size": 16},
    {"model": "cnn_gru", "lr": 1e-3, "batch_size": 32},
    {"model": "cnn_gru", "lr": 5e-4, "batch_size": 32},
    {"model": "cnn_gru", "lr": 1e-4, "batch_size": 32},
]

ALL_EXPERIMENTS = MODEL_EXPERIMENTS + HYPERPARAM_EXPERIMENTS

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

SEED        = 42
N_EPOCHS    = 100
ES_PATIENCE = 15
LR_PATIENCE = 7
DATA_DIR    = Path("Final Project data")
OUTPUT_DIR  = Path("outputs")

DATASET_PARAMS = dict(
    original_sampling_rate=ORIGINAL_SAMPLING_RATE,
    downsample_factor=DOWNSAMPLE_FACTOR,
    window_seconds=WINDOW_SECONDS,
    overlap=OVERLAP,
)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_one(model_name: str, lr: float, batch_size: int,
            train_folder: Path, scenario: str, device: torch.device) -> dict:

    run_key   = f"{scenario}_{model_name}_lr{lr:.0e}_bs{batch_size}"
    json_path = OUTPUT_DIR / f"results_{run_key}.json"

    if json_path.exists():
        print(f"\n[SKIP] {run_key} — already done\n")
        with open(json_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  {run_key}")
    print(f"{'='*60}")

    set_seed(SEED)

    all_files                = list_h5_files(train_folder)
    train_files, val_files   = stratified_split(all_files, val_ratio=0.2, seed=SEED)

    train_data   = MEGWindowDataset(files=train_files, **DATASET_PARAMS)
    val_data     = MEGWindowDataset(files=val_files,   **DATASET_PARAMS)
    train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = create_dataloader(val_data,   batch_size=batch_size, shuffle=False)

    model     = get_model(model_name, device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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

    elapsed      = time.time() - t_start
    best_epoch   = max(history, key=lambda h: h["val_acc"])["epoch"] if history else 0
    final_train  = history[-1]["train_acc"] if history else 0.0

    results = {
        "model_name":      model_name,
        "scenario":        scenario,
        "lr":              lr,
        "batch_size":      batch_size,
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

    print("\n" + "=" * 76)
    print("RESULTS SUMMARY")
    print("=" * 76)
    print(f"{'Scenario':<8} {'Model':<14} {'LR':<8} {'BS':<4} "
          f"{'Val Acc':>8} {'Train Acc':>10} {'Gap':>7} {'Best Ep':>8}")
    print("-" * 76)

    for r in sorted(all_results, key=lambda x: x["best_val_acc"], reverse=True):
        gap = r["final_train_acc"] - r["best_val_acc"]
        print(
            f"{r.get('scenario', '?'):<8} "
            f"{r['model_name']:<14} "
            f"{r['lr']:<8.0e} "
            f"{r['batch_size']:<4} "
            f"{r['best_val_acc']:>8.4f} "
            f"{r['final_train_acc']:>10.4f} "
            f"{gap:>+7.3f} "
            f"{r['best_epoch']:>8}"
        )

    print("=" * 76)
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
                        help="Also run cross-subject experiments for the 3 main models")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing results without running anything")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.summary:
        print_summary(load_existing_results())
        return

    device = get_device()
    print(f"Device : {device}")
    print(f"Running {len(ALL_EXPERIMENTS)} intra-subject experiment(s)...")
    if args.cross:
        print(f"  + {len(MODEL_EXPERIMENTS)} cross-subject experiment(s)")

    all_results = []

    # Intra-subject — all 6 experiments
    intra_folder = DATA_DIR / "Intra" / "train"
    for cfg in ALL_EXPERIMENTS:
        result = run_one(
            cfg["model"], cfg["lr"], cfg["batch_size"],
            intra_folder, "intra", device,
        )
        all_results.append(result)

    # Cross-subject — 3 main models only (hyperparameter sweeps not needed here)
    if args.cross:
        cross_folder = DATA_DIR / "Cross" / "train"
        for cfg in MODEL_EXPERIMENTS:
            result = run_one(
                cfg["model"], cfg["lr"], cfg["batch_size"],
                cross_folder, "cross", device,
            )
            all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()
