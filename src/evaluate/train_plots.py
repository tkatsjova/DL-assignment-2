"""
Plot training-phase results for the MEG classification assignment report.

Loads all outputs/results_intra_*.json and outputs/results_cross_*.json files
and generates PNG figures to outputs/plots/.

Figures produced:
  (a) model_params.png        — parameter count per model
  (b) val_acc_comparison.png  — intra vs cross best_val_acc bar chart
  (c) learning_curves_*.png   — per-model train/val loss and acc over epochs
  (d) overfit_gap.png         — final_train_acc vs best_val_acc (overfitting gap)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUTPUT_DIR = Path("outputs")
PLOT_DIR   = OUTPUT_DIR / "plots"

# Pretty display names for model keys
MODEL_LABELS = {
    "simple_cnn":    "SimpleCNN1D",
    "resnet":        "ResNet1D",
    "cnn_gru":       "CNN-GRU",
    "eegnet":        "EEGNet",
    "cnn_lstm_attn": "CNN-LSTM-Attn",
    "meg_graphnet":  "MEGGraphNet",
}


def load_results(pattern: str) -> dict[str, dict]:
    """Return {run_key: result_dict} where run_key is the model+suffix part of the filename.

    Strips the leading scenario prefix so intra and cross results share the same key:
      results_intra_cnn_gru_lr1e-03_bs16  →  cnn_gru_lr1e-03_bs16
      results_cross_cnn_gru_lr1e-03_bs16  →  cnn_gru_lr1e-03_bs16
    This allows the comparison plots to pair them correctly.
    """
    results = {}
    for p in sorted(OUTPUT_DIR.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        # strip "results_intra_" or "results_cross_" prefix
        stem = p.stem
        for prefix in ("results_intra_", "results_cross_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        results[stem] = d
    return results


def plot_param_count(intra: dict, cross: dict) -> None:
    merged = {**intra, **cross}
    if not merged:
        print("[skip] plot_param_count — no result files found")
        return

    # Deduplicate by model_name — n_params doesn't change across hyperparameter runs
    seen = {}
    for d in merged.values():
        seen.setdefault(d["model_name"], d["n_params"])
    models = sorted(seen.items(), key=lambda x: x[1])
    names  = [MODEL_LABELS.get(m, m) for m, _ in models]
    params = [p for _, p in models]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, params, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, labels=[f"{p:,}" for p in params], padding=4, fontsize=8)
    ax.set_xlabel("Trainable parameters")
    ax.set_title("(a) Model size comparison")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.margins(x=0.15)
    plt.tight_layout()
    out = PLOT_DIR / "model_params.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ── (b) intra vs cross accuracy bar chart ──────────────────────────────────────

def _run_label(d: dict) -> str:
    """Short label for a result dict — includes LR and batch size if present."""
    name = MODEL_LABELS.get(d["model_name"], d["model_name"])
    if "lr" in d and "batch_size" in d:
        return f"{name}\nlr={d['lr']:.0e} bs={d['batch_size']}"
    return name


def plot_val_acc_comparison(intra: dict, cross: dict) -> None:
    all_keys = sorted(set(intra) | set(cross))
    if not all_keys:
        print("[skip] plot_val_acc_comparison — no result files found")
        return

    x        = np.arange(len(all_keys))
    w        = 0.35
    intra_acc = [intra[k]["best_val_acc"] if k in intra else 0.0 for k in all_keys]
    cross_acc = [cross[k]["best_val_acc"] if k in cross else 0.0 for k in all_keys]
    labels    = [_run_label(intra.get(k) or cross.get(k)) for k in all_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, intra_acc, w, label="Intra-subject", color="#4C72B0")
    b2 = ax.bar(x + w / 2, cross_acc, w, label="Cross-subject", color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=2)
    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Best validation accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Intra-subject vs cross-subject validation accuracy")
    ax.legend()
    plt.tight_layout()
    out = PLOT_DIR / "val_acc_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ── (c) learning curves ────────────────────────────────────────────────────────

def plot_learning_curves(results: dict, scenario: str) -> None:
    if not results:
        print(f"[skip] plot_learning_curves ({scenario}) — no result files found")
        return

    for stem, d in results.items():
        history = d.get("history", [])
        if not history:
            print(f"  [skip] {stem} — empty history")
            continue

        epochs     = [h["epoch"]      for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss   = [h["val_loss"]   for h in history]
        train_acc  = [h["train_acc"]  for h in history]
        val_acc    = [h["val_acc"]    for h in history]

        fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

        ax_loss.plot(epochs, train_loss, label="Train", color="#4C72B0")
        ax_loss.plot(epochs, val_loss,   label="Val",   color="#DD8452")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Cross-entropy loss")
        ax_loss.set_title("Loss")
        ax_loss.legend()

        ax_acc.plot(epochs, train_acc, label="Train", color="#4C72B0")
        ax_acc.plot(epochs, val_acc,   label="Val",   color="#DD8452")
        ax_acc.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_title("Accuracy")
        ax_acc.legend()

        best_epoch = d.get("best_epoch", None)
        if best_epoch:
            for ax in (ax_loss, ax_acc):
                ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1,
                           label=f"Best (ep {best_epoch})")

        fig.suptitle(f"(c) Learning curves — {_run_label(d)} [{scenario}]", fontsize=12)
        plt.tight_layout()
        out = PLOT_DIR / f"learning_curves_{scenario}_{stem}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  saved {out}")


# ── (d) train–test gap ─────────────────────────────────────────────────────────

def plot_overfit_gap(intra: dict, cross: dict) -> None:
    all_keys = sorted(set(intra) | set(cross))
    if not all_keys:
        print("[skip] plot_overfit_gap — no result files found")
        return

    x      = np.arange(len(all_keys))
    w      = 0.2
    labels = [_run_label(intra.get(k) or cross.get(k)) for k in all_keys]

    def _get(d, k, key): return d[k][key] if k in d else None

    intra_train = [_get(intra, k, "final_train_acc") for k in all_keys]
    intra_val   = [_get(intra, k, "best_val_acc")    for k in all_keys]
    cross_train = [_get(cross, k, "final_train_acc") for k in all_keys]
    cross_val   = [_get(cross, k, "best_val_acc")    for k in all_keys]

    fig, ax = plt.subplots(figsize=(10, 5))

    def _bar(pos, values, color, label, hatch=""):
        vals  = [v if v is not None else 0.0 for v in values]
        valid = [v is not None for v in values]
        b = ax.bar(pos, vals, w, color=color, label=label,
                   hatch=hatch, edgecolor="white", alpha=0.85)
        for bar, ok in zip(b, valid):
            if not ok:
                bar.set_visible(False)
        return b

    _bar(x - 1.5 * w, intra_train, "#4C72B0", "Intra — train acc",  hatch="")
    _bar(x - 0.5 * w, intra_val,   "#4C72B0", "Intra — val acc",    hatch="///")
    _bar(x + 0.5 * w, cross_train, "#DD8452", "Cross — train acc",  hatch="")
    _bar(x + 1.5 * w, cross_val,   "#DD8452", "Cross — val acc",    hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Train acc vs val acc gap (proxy for overfitting; see true_train_test_gap for test)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = PLOT_DIR / "overfit_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ── (d) true train → val → test gap ───────────────────────────────────────────

def plot_true_train_test_gap(
    intra_train: dict,
    cross_train: dict,
    output_dir: Path | None = None,
) -> None:
    """(d) True train → val → test accuracy gap.

    Combines training-phase JSONs (final_train_acc, best_val_acc) with
    evaluation-phase JSONs (window-level test accuracy from eval_intra.json /
    eval_cross.json).  Silently skips if no eval JSON is found yet.
    """
    output_dir = output_dir or OUTPUT_DIR
    eval_intra_path = output_dir / "eval_intra.json"
    eval_cross_path = output_dir / "eval_cross.json"

    eval_intra: dict = {}
    eval_cross: dict = {}
    if eval_intra_path.exists():
        with open(eval_intra_path) as f:
            eval_intra = json.load(f)
    if eval_cross_path.exists():
        with open(eval_cross_path) as f:
            eval_cross = json.load(f)

    if not eval_intra and not eval_cross:
        print("[skip] plot_true_train_test_gap — no eval JSON files found yet")
        return

    # For each scenario keep only the run with the highest best_val_acc per model
    def best_run_by_model(results: dict) -> dict[str, dict]:
        best: dict[str, dict] = {}
        for d in results.values():
            name = d.get("model_name", "")
            if name and (name not in best or d["best_val_acc"] > best[name]["best_val_acc"]):
                best[name] = d
        return best

    intra_best = best_run_by_model(intra_train)
    cross_best = best_run_by_model(cross_train)

    all_models = sorted(
        set(intra_best) | set(cross_best) | set(eval_intra) | set(eval_cross)
    )
    if not all_models:
        return

    x     = np.arange(len(all_models))
    w     = 0.13
    names = [MODEL_LABELS.get(m, m) for m in all_models]

    # Prefer eval-mode train accuracy (no dropout) over training-phase final_train_acc
    def _train_acc(eval_d: dict, train_d: dict) -> float | None:
        v = eval_d.get("train_eval_accuracy")
        if v is None:
            v = train_d.get("final_train_acc")
        return v

    intra_tr   = [_train_acc(eval_intra.get(m, {}), intra_best.get(m, {})) for m in all_models]
    intra_val  = [intra_best.get(m, {}).get("best_val_acc")                 for m in all_models]
    intra_test = [eval_intra.get(m, {}).get("window_level", {}).get("accuracy") for m in all_models]

    cross_tr  = [_train_acc(eval_cross.get(m, {}), cross_best.get(m, {})) for m in all_models]
    cross_val = [cross_best.get(m, {}).get("best_val_acc")                 for m in all_models]

    # Use precomputed avg if available, otherwise compute from test_sets
    cross_test = []
    for m in all_models:
        r   = eval_cross.get(m, {})
        avg = r.get("avg_test_window_accuracy")
        if avg is None:
            accs = [s.get("accuracy", 0.0) for s in r.get("test_sets", {}).values()]
            avg  = float(np.mean(accs)) if accs else None
        cross_test.append(avg)

    palette = [
        "#2166ac",  # intra train  — dark blue
        "#6baed6",  # intra val    — light blue
        "#084594",  # intra test   — navy
        "#d94801",  # cross train  — dark orange
        "#fd8d3c",  # cross val    — light orange
        "#7f2704",  # cross test   — maroon
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    def _bar(offset, values, color, label):
        vals  = [v if v is not None else 0.0 for v in values]
        valid = [v is not None for v in values]
        bars = ax.bar(x + offset * w, vals, w * 0.92,
                      color=color, label=label, alpha=0.88, edgecolor="white")
        for bar, ok in zip(bars, valid):
            if not ok:
                bar.set_visible(False)
        return bars

    _bar(-2.5, intra_tr,   palette[0], "Intra — train")
    _bar(-1.5, intra_val,  palette[1], "Intra — val")
    _bar(-0.5, intra_test, palette[2], "Intra — test")
    _bar( 0.5, cross_tr,   palette[3], "Cross — train")
    _bar( 1.5, cross_val,  palette[4], "Cross — val")
    _bar( 2.5, cross_test, palette[5], "Cross — test (avg)")

    ax.axhline(0.25, color="grey", linestyle="--", linewidth=0.8, label="Chance (25%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("(d) True train → val → test accuracy gap  (train = eval mode, no dropout)")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    plt.tight_layout()

    out = PLOT_DIR / "true_train_test_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    intra = load_results("results_intra_*.json")
    cross = load_results("results_cross_*.json")

    print(f"Loaded {len(intra)} intra result(s): {list(intra)}")
    print(f"Loaded {len(cross)} cross result(s): {list(cross)}")
    print(f"Saving plots to {PLOT_DIR}/\n")

    plot_param_count(intra, cross)
    plot_val_acc_comparison(intra, cross)
    plot_learning_curves(intra, scenario="intra")
    plot_learning_curves(cross, scenario="cross")
    plot_overfit_gap(intra, cross)
    plot_true_train_test_gap(intra, cross)   # no-op until eval JSONs exist

    print("\nDone.")


if __name__ == "__main__":
    main()
