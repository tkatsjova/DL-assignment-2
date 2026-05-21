from pathlib import Path

import torch

from src.dataset import MEGWindowDataset, create_dataloader
from src.model import ResNet1D


def main():
    data_dir = Path("Final Project data")
    train_folder = data_dir / "Intra" / "train"

    dataset = MEGWindowDataset(
        folder=train_folder,
        original_sampling_rate=2034,
        downsample_factor=4,
        window_seconds=2.0,
        overlap=0.5,
    )

    loader = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    x, y = next(iter(loader))

    print("\nBatch check")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"y values: {y}")

    model = ResNet1D(num_channels=248, num_classes=4)
    logits = model(x)

    print("\nModel check")
    print(f"logits shape: {logits.shape}")
    print(f"predictions: {torch.argmax(logits, dim=1)}")


if __name__ == "__main__":
    main()