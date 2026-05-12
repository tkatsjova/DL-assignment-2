"""
MEG Brain State Classification — INFOMDLR Deep Learning Project
4 classes: rest, math, memory, motor
"""

import os
import glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT     = "data"
DOWNSAMPLE    = 4          # keep every Nth time step (2034 → ~508 Hz)
WINDOW_SIZE   = 256        # samples per segment after downsampling
WINDOW_STEP   = 128        # hop size (50 % overlap)
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"rest": 0, "task_story_math": 1, "task_working_memory": 2, "task_motor": 3}

# ── Data utilities ─────────────────────────────────────────────────────────────

def get_dataset_name(filepath: str) -> str:
    base = os.path.basename(filepath)
    parts = base.split("_")[:-1]
    return "_".join(parts)


def load_h5(filepath: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        name = get_dataset_name(filepath)
        return f.get(name)[()]   # shape: (248, 35624)


def label_from_path(filepath: str) -> int:
    base = os.path.basename(filepath)
    for key in LABEL_MAP:
        if base.startswith(key):
            return LABEL_MAP[key]
    raise ValueError(f"Unknown task type in filename: {base}")


def preprocess(matrix: np.ndarray, downsample: int = DOWNSAMPLE) -> np.ndarray:
    """Z-score normalise time-wise, then downsample."""
    mean = matrix.mean(axis=1, keepdims=True)
    std  = matrix.std(axis=1, keepdims=True) + 1e-8
    matrix = (matrix - mean) / std
    return matrix[:, ::downsample]   # (248, T//downsample)


def make_windows(matrix: np.ndarray, size: int = WINDOW_SIZE,
                 step: int = WINDOW_STEP) -> np.ndarray:
    """Slide a window over the time axis → (N, 248, size)."""
    windows = []
    T = matrix.shape[1]
    for start in range(0, T - size + 1, step):
        windows.append(matrix[:, start:start + size])
    return np.stack(windows)   # (N, 248, size)


def load_folder(folder: str, chunk_size: int = 8):
    """Yield batches of (X, y) arrays from all .h5 files in folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.h5")))
    for i in range(0, len(files), chunk_size):
        Xs, ys = [], []
        for fp in files[i:i + chunk_size]:
            matrix  = load_h5(fp)
            matrix  = preprocess(matrix)
            windows = make_windows(matrix)
            label   = label_from_path(fp)
            Xs.append(windows)
            ys.append(np.full(len(windows), label, dtype=np.int64))
        yield np.concatenate(Xs), np.concatenate(ys)


# ── Dataset ────────────────────────────────────────────────────────────────────

class MEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model (1D CNN) ─────────────────────────────────────────────────────────────

class MEGClassifier(nn.Module):
    def __init__(self, n_channels: int = 248, n_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.conv(x))


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_on_folder(model: nn.Module, folder: str, optimizer, criterion):
    model.train()
    for X_batch, y_batch in load_folder(folder):
        ds = MEGDataset(X_batch, y_batch)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        for X, y in dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()


def evaluate_folder(model: nn.Module, folder: str) -> float:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in load_folder(folder):
            ds = MEGDataset(X_batch, y_batch)
            dl = DataLoader(ds, batch_size=BATCH_SIZE)
            for X, y in dl:
                preds = model(X.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(classification_report(all_labels, all_preds,
                                target_names=list(LABEL_MAP.keys())))
    return acc


def run_experiment(name: str, train_folder: str, test_folders: list[str]):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    model     = MEGClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_on_folder(model, train_folder, optimizer, criterion)
        print(f"Epoch {epoch}/{EPOCHS} done")

    for tf in test_folders:
        print(f"\nTest folder: {tf}")
        acc = evaluate_folder(model, tf)
        print(f"Accuracy: {acc:.4f}")

    return model


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Intra-subject
    run_experiment(
        name         = "Intra-subject classification",
        train_folder = os.path.join(DATA_ROOT, "Intra", "train"),
        test_folders = [os.path.join(DATA_ROOT, "Intra", "test")],
    )

    # Cross-subject
    run_experiment(
        name         = "Cross-subject classification",
        train_folder = os.path.join(DATA_ROOT, "Cross", "train"),
        test_folders = [
            os.path.join(DATA_ROOT, "Cross", "test1"),
            os.path.join(DATA_ROOT, "Cross", "test2"),
            os.path.join(DATA_ROOT, "Cross", "test3"),
        ],
    )
