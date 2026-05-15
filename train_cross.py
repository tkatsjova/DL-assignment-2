from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import MEGWindowDataset, create_dataloader
from src.model import SimpleCNN1D, ResNet1D, CNNGRU


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
    data_dir = Path("Final Project data") / "Final Project data"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_folder = data_dir / "Cross" / "train"

    test_folders = {
        "test1": data_dir / "Cross" / "test1",
        "test2": data_dir / "Cross" / "test2",
        "test3": data_dir / "Cross" / "test3",
    }

    device = get_device()

    print(f"Using device: {device}")
    print(f"Training cross-subject model: {MODEL_NAME}")

    train_data = MEGWindowDataset(
        folder=train_folder,
        original_sampling_rate=2034,
        downsample_factor=4,
        window_seconds=2.0,
        overlap=0.5,
    )

    test_data = {}

    for name, folder in test_folders.items():
        test_data[name] = MEGWindowDataset(
            folder=folder,
            original_sampling_rate=2034,
            downsample_factor=4,
            window_seconds=2.0,
            overlap=0.5,
        )

    train_loader = create_dataloader(
        train_data,
        batch_size=16,
        shuffle=True,
    )

    test_loaders = {
        name: create_dataloader(data, batch_size=16, shuffle=False)
        for name, data in test_data.items()
    }

    model = get_model(MODEL_NAME, device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    n_epochs = 40
    best_mean_acc = 0.0
    save_path = get_save_path(MODEL_NAME, output_dir)

    print("\nStarting cross-subject training...")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_results = {}

        for test_name, test_loader in test_loaders.items():
            test_loss, test_acc = check_accuracy(
                model=model,
                loader=test_loader,
                loss_fn=loss_fn,
                device=device,
            )

            test_results[test_name] = {
                "loss": test_loss,
                "acc": test_acc,
            }

        mean_acc = sum(result["acc"] for result in test_results.values()) / len(test_results)

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Mean test acc: {mean_acc:.4f}"
        )

        for test_name, result in test_results.items():
            print(
                f"  {test_name}: "
                f"loss={result['loss']:.4f}, "
                f"acc={result['acc']:.4f}"
            )

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean_test_accuracy": best_mean_acc,
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "test_results": test_results,
                },
                save_path,
            )

            print(f"Saved new best cross-subject model to: {save_path}")

    print("\nCross-subject training complete.")
    print(f"Best mean cross-subject accuracy: {best_mean_acc:.4f}")


if __name__ == "__main__":
    main()