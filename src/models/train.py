import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.data_loading import ID_TO_LABEL, list_h5_files, extract_label_from_filename
from src.data.dataset import MEGWindowDataset, create_dataloader
from src.models.model import SimpleCNN1D, ResNet1D, CNNGRU, EEGNet, CNNGRUAttention, MEGGraphNet

N_TIMEPOINTS = int(WINDOW_SECONDS * ORIGINAL_SAMPLING_RATE / DOWNSAMPLE_FACTOR)


MODEL_NAME = "cnn_gru"
# options: "simple_cnn" | "resnet" | "cnn_gru" | "eegnet" | "cnn_gru_attn" | "meg_graphnet"

SEED       = 42
N_EPOCHS   = 100   # early stopping decides the actual stopping point
BATCH_SIZE = 16
LR         = 1e-3
# early stopping: stop if val_acc hasn't improved for this many epochs
ES_PATIENCE  = 15
# LR scheduler: halve LR if val_loss doesn't improve for this many epochs
LR_PATIENCE  = 7

# Auto-suffix encodes key hyperparameters so each experiment saves to its own
# file — change LR or BATCH_SIZE and results won't overwrite each other.
RUN_SUFFIX = f"_lr{LR:.0e}_bs{BATCH_SIZE}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Stratified file-level train / val split
# Splits files per class so that every class is equally represented in val.
# With 8 files per class and val_ratio=0.2 → 6 train + 2 val per class.
# ---------------------------------------------------------------------------

def stratified_split(
    files: list[Path],
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[list[Path], list[Path]]:
    by_label: dict[int, list[Path]] = defaultdict(list)
    for f in files:
        by_label[extract_label_from_filename(f)].append(f)

    rng = random.Random(seed)
    train_files, val_files = [], []

    for label in sorted(by_label):
        label_files = by_label[label][:]
        rng.shuffle(label_files)
        n_val = max(1, round(len(label_files) * val_ratio))
        val_files.extend(label_files[:n_val])
        train_files.extend(label_files[n_val:])

    return train_files, val_files


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(name: str, device: torch.device, dropout: float | None = None) -> nn.Module:
    def make():
        if name == "simple_cnn":
            return SimpleCNN1D(num_channels=248, num_classes=4)
        if name == "resnet":
            return ResNet1D(num_channels=248, num_classes=4)
        if name == "cnn_gru":
            return CNNGRU(num_channels=248, num_classes=4)
        if name == "eegnet":
            return EEGNet(n_channels=248, n_timepoints=N_TIMEPOINTS, n_classes=4)
        if name == "cnn_gru_attn":
            kwargs = {"n_channels": 248, "n_classes": 4}
            if dropout is not None:
                kwargs["dropout_rate"] = dropout
            return CNNGRUAttention(**kwargs)
        if name == "meg_graphnet":
            return MEGGraphNet(n_nodes=248, n_timepoints=N_TIMEPOINTS, n_classes=4)
        raise ValueError(f"Unknown model '{name}'. Options: simple_cnn, resnet, cnn_gru, eegnet, cnn_gru_attn, meg_graphnet")
    return make().to(device)


def get_save_path(name: str, output_dir: Path) -> Path:
    return output_dir / f"best_intra_{name}{RUN_SUFFIX}.pt"


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = total_correct = total_items = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        n = x.size(0)
        total_loss    += loss.item() * n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_items   += n

    return total_loss / total_items, total_correct / total_items


def check_accuracy(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_correct = total_items = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)

            n = x.size(0)
            total_loss    += loss.item() * n
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_items   += n

    return total_loss / total_items, total_correct / total_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    data_dir   = Path("Final Project data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_folder = data_dir / "Intra" / "train"

    device    = get_device()
    save_path = get_save_path(MODEL_NAME, output_dir)

    print(f"Device:  {device}")
    print(f"Model:   {MODEL_NAME}")
    print(f"Seed:    {SEED}")

    dataset_params = dict(
        original_sampling_rate=ORIGINAL_SAMPLING_RATE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        window_seconds=WINDOW_SECONDS,
        overlap=OVERLAP,
    )

    # Stratified 80/20 split — each class contributes equally to val
    all_train_files          = list_h5_files(train_folder)
    train_files, val_files   = stratified_split(all_train_files, val_ratio=0.2, seed=SEED)

    print(f"\nTrain files: {len(train_files)} | Val files: {len(val_files)}")
    val_label_counts = Counter(extract_label_from_filename(f) for f in val_files)
    print(f"Val class distribution: { {ID_TO_LABEL[k]: v for k, v in sorted(val_label_counts.items())} }")

    train_data = MEGWindowDataset(files=train_files, **dataset_params)
    val_data   = MEGWindowDataset(files=val_files,   **dataset_params)

    train_loader = create_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = create_dataloader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    model   = get_model(MODEL_NAME, device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Halve LR when val_loss stops improving for LR_PATIENCE epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
    )

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    es_counter    = 0          # epochs since last val_acc improvement
    history       = []         # one dict per epoch, for plotting / reporting

    print(f"\nStarting intra-subject training (max {N_EPOCHS} epochs, "
          f"early stop patience={ES_PATIENCE})...\n")

    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss,   val_acc   = check_accuracy(model, val_loader,   loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{N_EPOCHS} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        history.append(dict(
            epoch=epoch,
            train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss,     val_acc=val_acc,
        ))

        # Step scheduler on val_loss
        scheduler.step(val_loss)

        # Save best model (tracked by val_acc)
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            es_counter    = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy":     best_val_acc,
                    "val_loss":         best_val_loss,
                    "epoch":            epoch,
                    "model_name":       MODEL_NAME,
                    "history":          history,
                },
                save_path,
            )
            print(f"  → Saved best model (val acc: {best_val_acc:.4f})")
        else:
            es_counter += 1
            if es_counter >= ES_PATIENCE:
                print(f"\nEarly stopping: val_acc hasn't improved for {ES_PATIENCE} epochs.")
                break

    training_time_s = time.time() - t_start
    # train acc at the last completed epoch — used for (d): train vs test gap
    final_train_acc = history[-1]["train_acc"] if history else 0.0

    print(f"\nTraining finished in {training_time_s:.1f}s. Best val accuracy: {best_val_acc:.4f}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    n_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_name":       MODEL_NAME,
        "n_params":         n_params,
        "seed":             SEED,
        "lr":               LR,
        "batch_size":       BATCH_SIZE,
        "best_epoch":       checkpoint["epoch"],
        "training_time_s":  round(training_time_s, 1),
        "final_train_acc":  round(final_train_acc, 4),
        "best_val_acc":     round(best_val_acc, 4),
        "history":          history,
    }

    json_path = output_dir / f"results_intra_{MODEL_NAME}{RUN_SUFFIX}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
