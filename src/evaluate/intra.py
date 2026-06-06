"""Evaluate all intra-subject checkpoints on Intra/test and Intra/train (for gap analysis)."""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.dataset import MEGWindowDataset
from src.evaluate.core import (
    find_checkpoint, load_model, predict, majority_vote, compute_metrics, get_device,
)
from src.evaluate.plots import generate_all_intra_plots

MODELS = ["simple_cnn", "resnet", "cnn_gru", "eegnet", "cnn_gru_attn", "meg_graphnet"]

DATASET_PARAMS = dict(
    original_sampling_rate=ORIGINAL_SAMPLING_RATE,
    downsample_factor=DOWNSAMPLE_FACTOR,
    window_seconds=WINDOW_SECONDS,
    overlap=OVERLAP,
)


def run(data_dir: Path, output_dir: Path) -> dict:
    device = get_device()
    test_folder = data_dir / "Intra" / "test"
    train_folder = data_dir / "Intra" / "train"

    print(f"Device: {device}")
    print(f"\nLoading Intra/test from {test_folder} ...")
    test_data = MEGWindowDataset(folder=test_folder, **DATASET_PARAMS)

    # Load training data once for eval-mode gap analysis (no dropout — true train acc)
    print(f"Loading Intra/train from {train_folder} for gap analysis ...")
    train_data_gap = MEGWindowDataset(folder=train_folder, **DATASET_PARAMS)

    all_results = {}

    for name in MODELS:
        ckpt_path = find_checkpoint(output_dir, "intra", name)
        if ckpt_path is None:
            print(f"\n[SKIP] No checkpoint for {name}")
            continue

        model, val_acc, best_epoch = load_model(name, ckpt_path, device)
        print(f"\n--- {name}  (val_acc={val_acc:.4f}, best_epoch={best_epoch}) ---")

        # Test evaluation
        preds, labels = predict(model, test_data, device)
        window_metrics = compute_metrics(labels, preds, label=f"{name} | Intra/test")

        mv_preds, mv_labels = majority_vote(preds, labels, test_data.file_names)
        mv_acc = round(float(accuracy_score(mv_labels, mv_preds)), 4)
        print(f"  File majority-vote accuracy (test): {mv_acc:.4f}")

        # Train evaluation in eval mode — for true overfitting gap
        tr_preds, tr_labels = predict(model, train_data_gap, device)
        train_eval_acc = round(float(accuracy_score(tr_labels, tr_preds)), 4)
        print(f"  Train eval accuracy (eval mode, for gap): {train_eval_acc:.4f}")

        all_results[name] = {
            "val_accuracy": val_acc,
            "best_epoch": best_epoch,
            "train_eval_accuracy": train_eval_acc,
            "window_level": window_metrics,
            "file_majority_vote_accuracy": mv_acc,
        }

    out_path = output_dir / "eval_intra.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    generate_all_intra_plots(out_path)
    return all_results
