import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.config import ORIGINAL_SAMPLING_RATE, DOWNSAMPLE_FACTOR, WINDOW_SECONDS, OVERLAP
from src.data.data_loading import list_h5_files
from src.data.dataset import MEGWindowDataset, create_dataloader
from src.models.model import SimpleCNN1D, ResNet1D, CNNGRU


MODEL_NAME = "cnn_gru"   # options: "simple_cnn", "resnet", "cnn_gru"


def get_model(name: str, device: torch.device) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN1D(num_channels=248, num_classes=4).to(device)

    if name == "resnet":
        return ResNet1D(num_channels=248, num_classes=4).to(device)

    if name == "cnn_gru":
        return CNNGRU(num_channels=248, num_classes=4).to(device)

    raise ValueError(f"Unknown model: {name}")


def get_save_path(name: str, output_dir: Path) -> Path:
    if name == "simple_cnn":
        return output_dir / "best_cross_cnn1d.pt"

    if name == "resnet":
        return output_dir / "best_cross_resnet1d.pt"

    if name == "cnn_gru":
        return output_dir / "best_cross_cnngru.pt"

    raise ValueError(f"Unknown model: {name}")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0
    total_correct = 0
    total_items = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def check_accuracy(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def main():
    data_dir   = Path("Final Project data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_folder = data_dir / "Cross" / "train"

    test_folders = {
        "test1": data_dir / "Cross" / "test1",
        "test2": data_dir / "Cross" / "test2",
        "test3": data_dir / "Cross" / "test3",
    }

    device = get_device()

    # chunk_size=8 follows the PDF recommendation: load ~8 files at a time
    # to avoid holding all 64 train files in RAM simultaneously (~14 GB)
    chunk_size = 8
    dataset_params = dict(
        original_sampling_rate=ORIGINAL_SAMPLING_RATE,
        downsample_factor=DOWNSAMPLE_FACTOR,
        window_seconds=WINDOW_SECONDS,
        overlap=OVERLAP,
    )

    print(f"Using device: {device}")
    print(f"Training cross-subject model: {MODEL_NAME}")

    # Split Cross/train files 80/20 into train and val by file —
    # same reasoning as intra: windows from the same file are very
    # similar, so we split by file to avoid leakage
    all_train_files = list_h5_files(train_folder)
    random.seed(42)
    random.shuffle(all_train_files)  # shuffle before split so val isn't just the last chunk
    split       = int(len(all_train_files) * 0.8)
    train_files = all_train_files[:split]
    val_files   = all_train_files[split:]

    print(f"\nTotal train files: {len(train_files)} | Val files: {len(val_files)}")
    print(f"Chunk size: {chunk_size} files per chunk")

    # Val set loaded once in full — it's 20% of train so still manageable
    val_data   = MEGWindowDataset(files=val_files, **dataset_params)
    val_loader = create_dataloader(val_data, batch_size=16, shuffle=False)

    # Test sets loaded once — touched only at the end
    test_data = {
        name: MEGWindowDataset(folder=folder, **dataset_params)
        for name, folder in test_folders.items()
    }
    test_loaders = {
        name: create_dataloader(data, batch_size=16, shuffle=False)
        for name, data in test_data.items()
    }

    model     = get_model(MODEL_NAME, device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_epochs     = 40
    best_val_acc = 0.0
    save_path    = get_save_path(MODEL_NAME, output_dir)

    print("\nStarting cross-subject training...")

    for epoch in range(1, n_epochs + 1):

        # Shuffle train files every epoch so the model sees data
        # in a different order — acts as regularisation and
        # prevents overfitting to a fixed chunk ordering
        random.shuffle(train_files)

        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0

        # Iterate over chunks — each chunk lives in RAM only while
        # training on it, then Python's GC frees the memory
        chunks = [
            train_files[i:i + chunk_size]
            for i in range(0, len(train_files), chunk_size)
        ]

        for chunk_idx, chunk_files in enumerate(chunks):
            chunk_dataset = MEGWindowDataset(files=chunk_files, **dataset_params)
            chunk_loader  = create_dataloader(chunk_dataset, batch_size=16, shuffle=True)

            chunk_loss, chunk_acc = run_epoch(
                model=model,
                loader=chunk_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            # Weighted accumulation — last chunk may be smaller than chunk_size
            # so a simple average across chunks would be inaccurate
            n = len(chunk_dataset)
            epoch_loss    += chunk_loss * n
            epoch_correct += chunk_acc * n
            epoch_total   += n

            print(
                f"  Epoch {epoch:02d} | Chunk {chunk_idx + 1}/{len(chunks)} | "
                f"loss: {chunk_loss:.4f} | acc: {chunk_acc:.4f}"
            )

        train_loss = epoch_loss / epoch_total
        train_acc  = epoch_correct / epoch_total

        # Evaluate on val set — used to select the best model
        val_loss, val_acc = check_accuracy(
            model=model, loader=val_loader, loss_fn=loss_fn, device=device,
        )

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best_val_acc,
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                },
                save_path,
            )

            print(f"Saved new best model (val acc: {best_val_acc:.4f}) to: {save_path}")

    print("\nCross-subject training complete.")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # Load best checkpoint and evaluate on all three unseen test subjects
    print("\nLoading best checkpoint for final evaluation on test subjects...")
    checkpoint = torch.load(save_path, map_location=device)
    model = get_model(MODEL_NAME, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Val accuracy at that epoch: {checkpoint['val_accuracy']:.4f}")

    print("\nFinal test results:")
    for test_name, test_loader in test_loaders.items():
        _, test_acc = check_accuracy(
            model=model, loader=test_loader, loss_fn=loss_fn, device=device,
        )
        print(f"  {test_name}: acc={test_acc:.4f}")


if __name__ == "__main__":
    main()