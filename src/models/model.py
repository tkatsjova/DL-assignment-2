import torch
import torch.nn as nn


# Block 1: temporal conv → depthwise (spatial) conv
# Block 2: separable conv
class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 248,
        n_timepoints: int = 1017,  # 508 Hz × 2 s after downsampling
        n_classes: int = 4,  # rest / math / memory / motor
        F1: int = 8,  # temporal filters in Block 1
        D: int = 2,  # depth multiplier → spatial filters = F1 * D
        F2: int = 16,  # pointwise filters in Block 2
        kernel_length: int = 128,  # ≈ 0.25 s at 508 Hz
        dropout_rate: float = 0.5,  # paper recommends 0.5 for cross-subject
    ):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # dry run to get flatten size — padding makes a closed-form formula fragile
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            dummy = self.temporal_conv(dummy)
            dummy = self.spatial_conv(dummy)
            dummy = self.separable_conv(dummy)
            flatten_size = dummy.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return self.classifier(x)


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

        self.relu = nn.ReLU()
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
        dropout_rate: float = 0.2,
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
            nn.Dropout(dropout_rate),
            nn.Linear(gru_hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.transpose(1, 2)
        gru_output, _ = self.gru(x)
        x = gru_output.mean(dim=1)
        return self.classifier(x)


# CNN + GRU + Attention
# Three stages: spatial projection → temporal CNN → bidirectional GRU with additive attention.
# Attention lets the model focus on the time steps where the brain state is most expressed,
# which is more flexible than simple mean pooling used in CNNGRU.
class CNNGRUAttention(nn.Module):
    def __init__(
        self,
        n_channels: int = 248,  # MEG sensors
        n_classes: int = 4,  # rest / math / memory / motor
        spatial_hidden: int = 64,
        cnn_channels: int = 64,
        gru_hidden: int = 64,  # per direction
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # kernel_size=1: linear mix of sensors at each time step independently
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(n_channels, spatial_hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(spatial_hidden),
            nn.ReLU(),
        )

        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(spatial_hidden, cnn_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate),
        )

        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        gru_out_size = gru_hidden * 2

        self.attention = nn.Linear(gru_out_size, 1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gru_out_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_proj(x)
        x = self.temporal_cnn(x)

        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)

        scores = self.attention(gru_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * gru_out).sum(dim=1)

        return self.classifier(context)
