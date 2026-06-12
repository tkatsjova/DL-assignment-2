import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.data_loading import ID_TO_LABEL, list_h5_files, extract_label_from_filename
from src.data.dataset import MEGWindowDataset, create_dataloader
from src.models.train import set_seed, get_device, get_model, stratified_split, check_accuracy

MODEL_NAME = "cnn_gru"
# options: "simple_cnn" | "resnet" | "cnn_gru" | "eegnet" | "cnn_gru_attn"

SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
CHUNK_SIZE = 8  # files loaded into RAM at once
ES_PATIENCE = 15
LR_PATIENCE = 7

WEIGHT_DECAY: float = 1e-4
DROPOUT: float | None = None

RUN_SUFFIX = f"_lr{LR:.0e}_bs{BATCH_SIZE}"


def get_save_path(name: str, output_dir: Path) -> Path:
    return output_dir / f"best_cross_{name}{RUN_SUFFIX}.pt"


# ---------------------------------------------------------------------------
# Chunk-based training epoch
# Cross/train has 64 files — loading all at once requires ~14 GB RAM.
# Instead we iterate over chunks of CHUNK_SIZE files, fitting the model on
# each chunk before moving to the next. Weighted accumulation keeps the
# epoch-level loss/acc accurate even when the last chunk is smaller.
# ---------------------------------------------------------------------------

def run_epoch_chunked(
    model, train_files, loss_fn, optimizer, device, dataset_params
) -> tuple[float, float]:
    model.train()

    random.shuffle(train_files)   # different order every epoch → regularisation
    chunks = [
        train_files[i:i + CHUNK_SIZE]
        for i in range(0, len(train_files), CHUNK_SIZE)
    ]

    epoch_loss = epoch_correct = epoch_total = 0

    for chunk_idx, chunk_files in enumerate(chunks):
        chunk_dataset = MEGWindowDataset(files=chunk_files, **dataset_params)
        chunk_loader = create_dataloader(chunk_dataset, batch_size=BATCH_SIZE, shuffle=True)

        chunk_loss = chunk_correct = chunk_total = 0

        for x, y in tqdm(chunk_loader, desc=f"  Chunk {chunk_idx+1}/{len(chunks)}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            n = x.size(0)
            chunk_loss += loss.item() * n
            chunk_correct += (logits.argmax(dim=1) == y).sum().item()
            chunk_total += n

        epoch_loss += chunk_loss
        epoch_correct += chunk_correct
        epoch_total += chunk_total

    return epoch_loss / epoch_total, epoch_correct / epoch_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)

    data_dir = Path("Final Project data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_folder = data_dir / "Cross" / "train"

    device = get_device()
    save_path = get_save_path(MODEL_NAME, output_dir)

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seed: {SEED}")

    dataset_params = dict(
        original_sampling_rate=ORIGINAL_SAMPLING_RATE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        window_seconds=WINDOW_SECONDS,
        overlap=OVERLAP,
    )

    all_train_files = list_h5_files(train_folder)
    train_files, val_files = stratified_split(all_train_files, val_ratio=0.2, seed=SEED)

    print(f"\nTrain files: {len(train_files)} | Val files: {len(val_files)}")
    val_label_counts = Counter(extract_label_from_filename(f) for f in val_files)
    print(f"Val class distribution: { {ID_TO_LABEL[k]: v for k, v in sorted(val_label_counts.items())} }")
    print(f"Chunk size: {CHUNK_SIZE} files per chunk\n")

    val_data = MEGWindowDataset(files=val_files, **dataset_params)
    val_loader = create_dataloader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(MODEL_NAME, device, dropout=DROPOUT)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE,
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    es_counter = 0
    history = []

    print(f"Starting cross-subject training (max {N_EPOCHS} epochs, "
          f"early stop patience={ES_PATIENCE})...\n")

    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = run_epoch_chunked(
            model, train_files, loss_fn, optimizer, device, dataset_params
        )
        val_loss, val_acc = check_accuracy(model, val_loader, loss_fn, device)

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
            val_loss=val_loss, val_acc=val_acc,
        ))

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            es_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best_val_acc,
                    "val_loss": best_val_loss,
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "history": history,
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
    final_train_acc = history[-1]["train_acc"] if history else 0.0

    print(f"\nTraining finished in {training_time_s:.1f}s. Best val accuracy: {best_val_acc:.4f}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    n_params = sum(p.numel() for p in model.parameters())

    results = {
        "model_name": MODEL_NAME,
        "n_params": n_params,
        "seed": SEED,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "best_epoch": checkpoint["epoch"],
        "training_time_s": round(training_time_s, 1),
        "final_train_acc": round(final_train_acc, 4),
        "best_val_acc": round(best_val_acc, 4),
        "history": history,
    }

    json_path = output_dir / f"results_cross_{MODEL_NAME}{RUN_SUFFIX}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
