"""Evaluate all cross-subject checkpoints on Cross/test1, test2, test3."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.dataset import MEGWindowDataset
from src.evaluate.core import (
    find_checkpoint, load_model, predict, majority_vote, compute_metrics, get_device,
)
from src.evaluate.plots import generate_all_cross_plots

MODELS = ["simple_cnn", "resnet", "cnn_gru", "eegnet", "cnn_lstm_attn", "meg_graphnet"]

TEST_SPLITS = ["test1", "test2", "test3"]

DATASET_PARAMS = dict(
    original_sampling_rate=ORIGINAL_SAMPLING_RATE,
    downsample_factor=DOWNSAMPLE_FACTOR,
    window_seconds=WINDOW_SECONDS,
    overlap=OVERLAP,
)


def run(data_dir: Path, output_dir: Path) -> dict:
    device = get_device()
    print(f"Device: {device}")

    all_results = {}

    for name in MODELS:
        ckpt_path = find_checkpoint(output_dir, "cross", name)
        if ckpt_path is None:
            print(f"\n[SKIP] No checkpoint for {name} (cross)")
            continue

        model, val_acc, best_epoch = load_model(name, ckpt_path, device)
        print(f"\n{'#'*60}")
        print(f"  {name}  (cross val_acc={val_acc:.4f}, best_epoch={best_epoch})")
        print(f"{'#'*60}")

        model_results = {
            "val_accuracy": val_acc,
            "best_epoch":   best_epoch,
            "test_sets":    {},
        }

        for split in TEST_SPLITS:
            folder = data_dir / "Cross" / split
            print(f"\nLoading {split} from {folder} ...")
            test_data = MEGWindowDataset(folder=folder, **DATASET_PARAMS)

            preds, labels = predict(model, test_data, device)
            split_metrics = compute_metrics(labels, preds, label=f"{name} | {split}")

            mv_preds, mv_labels = majority_vote(preds, labels, test_data.file_names)
            mv_acc = round(float(accuracy_score(mv_labels, mv_preds)), 4)
            print(f"  File majority-vote accuracy: {mv_acc:.4f}")

            split_metrics["file_majority_vote_accuracy"] = mv_acc
            model_results["test_sets"][split] = split_metrics

        # Summary across test sets
        accs  = [model_results["test_sets"][s]["accuracy"]                    for s in TEST_SPLITS if s in model_results["test_sets"]]
        mv_accs = [model_results["test_sets"][s]["file_majority_vote_accuracy"] for s in TEST_SPLITS if s in model_results["test_sets"]]
        if accs:
            model_results["avg_test_window_accuracy"]    = round(float(np.mean(accs)),    4)
            model_results["avg_test_file_vote_accuracy"] = round(float(np.mean(mv_accs)), 4)
            print(f"\n  Avg window accuracy  (test1-3): {model_results['avg_test_window_accuracy']:.4f}")
            print(f"  Avg file vote accuracy (test1-3): {model_results['avg_test_file_vote_accuracy']:.4f}")

        all_results[name] = model_results

    out_path = output_dir / "eval_cross.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    intra_path = output_dir / "eval_intra.json"
    if intra_path.exists():
        generate_all_cross_plots(intra_path, out_path)

    return all_results
