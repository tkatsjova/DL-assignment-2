"""
Exploratory Data Analysis for MEG dataset.
Run directly: python -m src.data.eda
"""

from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR
from src.data.data_loading import (
    list_h5_files,
    load_h5_file,
    load_folder_metadata,
    extract_label_from_filename,
    ID_TO_LABEL,
)
from src.data.preprocessing import zscore_per_sensor, downsample_signal

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("Final Project data")
OUTPUT_DIR = Path("outputs") / "eda"
SENSOR_IDX = 0              # which sensor to plot in signal examples

CLASS_COLORS = {
    "rest":   "#4C72B0",
    "math":   "#DD8452",
    "memory": "#55A868",
    "motor":  "#C44E52",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def time_axis(n_samples: int, sampling_rate: float) -> np.ndarray:
    return np.arange(n_samples) / sampling_rate


# ── Plot 1: Raw vs Normalised signal ──────────────────────────────────────────

def plot_raw_vs_normalised(folder: Path) -> None:
    """Show one sensor before and after Z-score normalisation."""
    file = list_h5_files(folder)[0]
    raw  = load_h5_file(file)
    norm = zscore_per_sensor(raw)

    t = time_axis(raw.shape[1], ORIGINAL_SAMPLING_RATE)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("Raw vs Z-score Normalised Signal (sensor 0)", fontsize=13)

    axes[0].plot(t, raw[SENSOR_IDX], color="#4C72B0", linewidth=0.6)
    axes[0].set_ylabel("Amplitude (T)")
    axes[0].set_title("Raw")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    axes[1].plot(t, norm[SENSOR_IDX], color="#DD8452", linewidth=0.6)
    axes[1].set_ylabel("Z-score")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("After Z-score normalisation")

    plt.tight_layout()
    save(fig, "01_raw_vs_normalised.png")


# ── Plot 2: Class balance ──────────────────────────────────────────────────────

def plot_class_balance(folders: dict[str, Path]) -> None:
    """Bar chart of file counts per class for each folder."""
    fig, axes = plt.subplots(1, len(folders), figsize=(5 * len(folders), 5))
    if len(folders) == 1:
        axes = [axes]

    fig.suptitle("Class Balance (number of files per class)", fontsize=13)

    for ax, (name, folder) in zip(axes, folders.items()):
        metadata = load_folder_metadata(folder)
        counts   = Counter(item["label_name"] for item in metadata)

        labels = list(CLASS_COLORS.keys())
        values = [counts.get(l, 0) for l in labels]
        colors = [CLASS_COLORS[l] for l in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
        ax.set_title(name)
        ax.set_ylabel("Number of files")
        ax.set_ylim(0, max(values) * 1.2)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val),
                ha="center", va="bottom", fontsize=10,
            )

    plt.tight_layout()
    save(fig, "02_class_balance.png")


# ── Plot 3: One signal per class ──────────────────────────────────────────────

def plot_signal_per_class(folder: Path) -> None:
    """Show normalised + downsampled signal for one file per class."""
    metadata      = load_folder_metadata(folder)
    sampling_rate = ORIGINAL_SAMPLING_RATE / DOWNSAMPLE_FACTOR

    seen, chosen = set(), []
    for item in metadata:
        if item["label_name"] not in seen:
            seen.add(item["label_name"])
            chosen.append(item)
        if len(chosen) == 4:
            break

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Normalised & Downsampled MEG Signal per Class (sensor 0)", fontsize=13)

    for ax, item in zip(axes, chosen):
        raw   = load_h5_file(item["file_path"])
        norm  = zscore_per_sensor(raw)
        down  = downsample_signal(norm, factor=DOWNSAMPLE_FACTOR)
        t     = time_axis(down.shape[1], sampling_rate)
        label = item["label_name"]

        ax.plot(t, down[SENSOR_IDX], color=CLASS_COLORS[label], linewidth=0.7)
        ax.set_ylabel("Z-score")
        ax.set_title(f"Class: {label}")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    save(fig, "03_signal_per_class.png")


# ── Plot 4: Value distribution after normalisation ────────────────────────────

def plot_value_distribution(folder: Path, n_files: int = 4) -> None:
    """Histogram of normalised values across a few files."""
    files = list_h5_files(folder)[:n_files]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"Distribution of Z-score Values After Normalisation ({n_files} files)",
        fontsize=13,
    )

    for file in files:
        raw   = load_h5_file(file)
        norm  = zscore_per_sensor(raw)
        label = ID_TO_LABEL[extract_label_from_filename(file)]

        ax.hist(
            norm.flatten(),
            bins=100,
            alpha=0.4,
            label=f"{file.name[:20]}… ({label})",
            density=True,
        )

    ax.set_xlabel("Z-score value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    save(fig, "04_value_distribution.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    intra_train = DATA_DIR / "Intra" / "train"
    intra_test  = DATA_DIR / "Intra" / "test"
    cross_train = DATA_DIR / "Cross" / "train"

    print("=== EDA ===\n")

    print("Plot 1: Raw vs Normalised...")
    plot_raw_vs_normalised(intra_train)

    print("Plot 2: Class balance...")
    plot_class_balance({
        "Intra/train": intra_train,
        "Intra/test":  intra_test,
        "Cross/train": cross_train,
    })

    print("Plot 3: Signal per class...")
    plot_signal_per_class(intra_train)

    print("Plot 4: Value distribution...")
    plot_value_distribution(intra_train)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
