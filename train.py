from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.data_loading import ID_TO_LABEL
from src.dataset import MEGWindowDataset, create_dataloader
from src.model import SimpleCNN1D, ResNet1D, CNNGRU


MODEL_NAME = "cnn_gru"   # options: "simple_cnn", "resnet", "cnn_gru"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(name: str, device: torch.device):
    if name == "simple_cnn":
        return SimpleCNN1D(num_channels=248, num_classes=4).to(device)

    if name == "resnet":
        return ResNet1D(num_channels=248, num_classes=4).to(device)

    if name == "cnn_gru":
        return CNNGRU(num_channels=248, num_classes=4).to(device)


def get_save_path(name: str, output_dir: Path):
    if name == "simple_cnn":
        return output_dir / "best_intra_cnn1d.pt"

    if name == "resnet":
        return output_dir / "best_intra_resnet1d.pt"

    if name == "cnn_gru":
        return output_dir / "best_intra_cnngru.pt"

def train_one_epoch(model, loader, loss_fn, optimizer, device):
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


def predict_windows(model, dataset, device, batch_size=16):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))

            x = torch.from_numpy(dataset.X[start:stop]).to(device)
            y = dataset.y[start:stop]

            logits = model(x)
            batch_preds = logits.argmax(dim=1).cpu().numpy()

            preds.extend(batch_preds)
            labels.extend(y)

    return np.array(preds), np.array(labels)


def majority_vote_by_file(preds, labels, file_names):
    file_preds = defaultdict(list)
    file_labels = {}

    for pred, label, file_name in zip(preds, labels, file_names):
        file_preds[file_name].append(pred)
        file_labels[file_name] = label

    final_preds = []
    final_labels = []

    for file_name, pred_list in file_preds.items():
        vote = Counter(pred_list).most_common(1)[0][0]
        final_preds.append(vote)
        final_labels.append(file_labels[file_name])

    return np.array(final_preds), np.array(final_labels)


def show_results(title, y_true, y_pred):
    class_names = [ID_TO_LABEL[i] for i in range(4)]

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    print("\nPrediction counts:")
    for class_id, count in sorted(Counter(y_pred).items()):
        print(f"{ID_TO_LABEL[class_id]}: {count}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def main():
    data_dir = Path("Final Project data") / "Final Project data"
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_folder = data_dir / "Intra" / "train"
    test_folder = data_dir / "Intra" / "test"

    device = get_device()
    save_path = get_save_path(MODEL_NAME, output_dir)

    print(f"Using device: {device}")
    print(f"Training model: {MODEL_NAME}")

    train_data = MEGWindowDataset(
        folder=train_folder,
        original_sampling_rate=2034,
        downsample_factor=4,
        window_seconds=2.0,
        overlap=0.5,
    )

    test_data = MEGWindowDataset(
        folder=test_folder,
        original_sampling_rate=2034,
        downsample_factor=4,
        window_seconds=2.0,
        overlap=0.5,
    )

    train_loader = create_dataloader(train_data, batch_size=16, shuffle=True)
    test_loader = create_dataloader(test_data, batch_size=16, shuffle=False)

    model = get_model(MODEL_NAME, device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    n_epochs = 40
    best_acc = 0.0

    print("\nStarting intra-subject training...")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc = check_accuracy(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_accuracy": best_acc,
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                },
                save_path,
            )

            print(f"Saved new best model to: {save_path}")

    print("\nTraining complete.")
    print(f"Best window-level test accuracy during training: {best_acc:.4f}")

    print("\nLoading best checkpoint for final evaluation...")
    checkpoint = torch.load(save_path, map_location=device)

    model = get_model(MODEL_NAME, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint: {save_path}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best saved test accuracy: {checkpoint['test_accuracy']:.4f}")

    window_preds, window_labels = predict_windows(
        model=model,
        dataset=test_data,
        device=device,
        batch_size=16,
    )

    show_results(
        title="Final window-level evaluation",
        y_true=window_labels,
        y_pred=window_preds,
    )

    file_preds, file_labels = majority_vote_by_file(
        preds=window_preds,
        labels=window_labels,
        file_names=test_data.file_names,
    )

    show_results(
        title="Final file-level majority-vote evaluation",
        y_true=file_labels,
        y_pred=file_preds,
    )


if __name__ == "__main__":
    main()