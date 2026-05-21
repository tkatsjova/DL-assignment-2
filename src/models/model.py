import torch
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    def __init__(self, num_channels: int = 248, num_classes: int = 4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),

            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x) + self.skip(x)
        x = self.relu(x)
        return self.dropout(x)


class ResNet1D(nn.Module):
    def __init__(self, num_channels: int = 248, num_classes: int = 4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


class CNNGRU(nn.Module):
    def __init__(
        self,
        num_channels: int = 248,
        num_classes: int = 4,
        cnn_channels: int = 64,
        gru_hidden_size: int = 64,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(gru_hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_output, _ = self.gru(x)
        x = gru_output.mean(dim=1)
        return self.classifier(x)
