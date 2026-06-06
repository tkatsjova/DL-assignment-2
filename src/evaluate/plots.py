"""
Evaluation plots for the MEG classification report.

Figures produced:
  confusion_matrix_<model>_<split>.png  — per-model confusion matrix
  test_acc_comparison.png               — file-level test accuracy across models
  per_class_f1_intra.png                — per-class F1 bar chart (intra)
  per_class_f1_cross.png                — per-class F1 bar chart (cross, averaged over test sets)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.data_loading import ID_TO_LABEL

CLASS_NAMES = [ID_TO_LABEL[i] for i in range(4)]

MODEL_LABELS = {
    "simple_cnn":    "SimpleCNN",
    "resnet":        "ResNet",
    "cnn_gru":       "CNN-GRU",
    "eegnet":        "EEGNet",
    "cnn_gru_attn":  "CNN-GRU-Attn",
    "meg_graphnet":  "MEGGraphNet",
}

PLOT_DIR = Path("outputs/plots")


def plot_confusion_matrix(cm: list[list[int]], model_name: str, split: str) -> None:
    cm_arr = np.array(cm)
    # Normalise rows to proportions for readability
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)} — {split}", fontsize=10)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            count = cm_arr[i, j]
            pct   = cm_norm[i, j]
            color = "white" if pct > 0.55 else "black"
            ax.text(j, i, f"{count}\n({pct:.0%})", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    out = PLOT_DIR / f"confusion_matrix_{model_name}_{split}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_test_acc_comparison(eval_intra: dict, eval_cross: dict | None = None) -> None:
    present = set(eval_intra) | (set(eval_cross) if eval_cross else set())
    models  = [m for m in MODEL_LABELS if m in present]
    if not models:
        print("[skip] plot_test_acc_comparison — no eval results found")
        return
    labels  = [MODEL_LABELS[m] for m in models]

    intra_win  = [eval_intra.get(m, {}).get("window_level",  {}).get("accuracy", 0.0) for m in models]
    intra_file = [eval_intra.get(m, {}).get("file_majority_vote_accuracy", 0.0)        for m in models]

    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - w * 1.5, intra_win,  w, label="Intra — window",    color="#4C72B0", alpha=0.9)
    b2 = ax.bar(x - w * 0.5, intra_file, w, label="Intra — file vote", color="#4C72B0", alpha=0.55)

    if eval_cross:
        cross_win  = []
        cross_file = []
        for m in models:
            sets = eval_cross.get(m, {}).get("test_sets", {})
            if sets:
                cross_win.append(np.mean([s.get("accuracy", 0.0) for s in sets.values()]))
                cross_file.append(np.mean([s.get("file_majority_vote_accuracy", 0.0) for s in sets.values()]))
            else:
                cross_win.append(0.0)
                cross_file.append(0.0)

        b3 = ax.bar(x + w * 0.5, cross_win,  w, label="Cross — window",    color="#DD8452", alpha=0.9)
        b4 = ax.bar(x + w * 1.5, cross_file, w, label="Cross — file vote", color="#DD8452", alpha=0.55)
        for b in (b3, b4):
            ax.bar_label(b, fmt="%.2f", fontsize=6, padding=2)

    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", fontsize=6, padding=2)

    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Test accuracy: window-level vs file majority vote")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out = PLOT_DIR / "test_acc_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def plot_per_class_f1(eval_results: dict, split_label: str, filename: str) -> None:
    models = [m for m in MODEL_LABELS if m in eval_results]
    if not models:
        return

    x      = np.arange(len(CLASS_NAMES))
    n      = len(models)
    width  = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 0.6, n))

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(models):
        per_class = eval_results[model].get("window_level", eval_results[model]).get("per_class", {})
        f1_values = [per_class.get(cls, {}).get("f1", 0.0) for cls in CLASS_NAMES]
        offset    = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, f1_values, width, label=MODEL_LABELS[model],
               color=colors[i], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class F1 score — {split_label}")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.7, label="Chance")
    plt.tight_layout()

    out = PLOT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def generate_all_intra_plots(eval_path: Path) -> None:
    with open(eval_path) as f:
        data = json.load(f)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for model, results in data.items():
        cm = results.get("window_level", {}).get("confusion_matrix")
        if cm:
            plot_confusion_matrix(cm, model, "intra")

    plot_per_class_f1(
        {m: r.get("window_level", r) for m, r in data.items()},
        split_label="intra-subject",
        filename="per_class_f1_intra.png",
    )
    plot_test_acc_comparison(data)
    print("Intra evaluation plots done.")


def generate_all_cross_plots(eval_intra_path: Path, eval_cross_path: Path) -> None:
    with open(eval_intra_path) as f:
        intra_data = json.load(f)
    with open(eval_cross_path) as f:
        cross_data = json.load(f)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for model, results in cross_data.items():
        for split_name, split_results in results.get("test_sets", {}).items():
            cm = split_results.get("confusion_matrix")
            if cm:
                plot_confusion_matrix(cm, model, f"cross_{split_name}")

    # Average F1 across test sets for the cross per-class chart
    cross_avg: dict = {}
    for model, results in cross_data.items():
        sets = results.get("test_sets", {})
        if not sets:
            continue
        avg_per_class: dict = {}
        for cls in CLASS_NAMES:
            avg_per_class[cls] = {
                "f1": np.mean([s.get("per_class", {}).get(cls, {}).get("f1", 0.0) for s in sets.values()])
            }
        cross_avg[model] = {"per_class": avg_per_class}

    plot_per_class_f1(cross_avg, split_label="cross-subject (avg over test sets)", filename="per_class_f1_cross.png")
    plot_test_acc_comparison(intra_data, cross_data)
    print("Cross evaluation plots done.")
