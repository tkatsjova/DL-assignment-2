import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# EEGNet
# Source: https://arxiv.org/abs/1611.08024
#
# Three-block design that separates spatial and temporal learning:
#   Block 1 — Temporal conv:    learns frequency-band filters across time
#   Block 1 — Depthwise conv:   learns one spatial filter per temporal filter (which sensors fire together?)
#   Block 2 — Separable conv:   efficiently refines temporal features
#
# =============================================================================
class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 248,       # number of MEG sensors
        n_timepoints: int = 1017,    # time steps after downsampling (508 Hz × 2 s)
        n_classes: int = 4,          # rest / math / memory / motor
        F1: int = 8,                 # number of temporal filters in Block 1
        D: int = 2,                  # depth multiplier → spatial filters = F1 * D
        F2: int = 16,                # number of pointwise filters in Block 2
        kernel_length: int = 128,    # temporal filter size ≈ 0.25 s at 508 Hz
        dropout_rate: float = 0.5,   # paper recommends 0.5 for cross-subject
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Temporal Convolution
        # Filter shape (1, kernel_length): slides along the time axis only.
        # Each of the F1 filters learns to detect a particular frequency band across ALL sensors simultaneously.
        # Same padding keeps the time dimension unchanged.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Depthwise (Spatial) Convolution
        # Filter shape (n_channels, 1): spans ALL 248 sensors at one time step.
        # groups=F1 means each temporal filter gets its own spatial filter →
        # the model learns "for this frequency band, which sensor combination is most informative?"
        # Collapses the sensor dimension to 1.
        # AvgPool2d reduces time by 4× (1017 → 254) to cut computation.
        # ------------------------------------------------------------------
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(n_channels, 1),
                groups=F1,           # one spatial filter per temporal filter
                bias=False,
            ),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # 1017 → 254
            nn.Dropout(dropout_rate),
        )

        # ------------------------------------------------------------------
        # Separable Convolution
        # Two-step convolution that is more parameter-efficient than a plain
        # Conv2d: first a depthwise pass (each channel independently), then
        # a pointwise (1×1) pass that mixes channels.
        # AvgPool2d reduces time by 8× (254 → 31).
        # ------------------------------------------------------------------
        self.separable_conv = nn.Sequential(
            # depthwise: refines each feature map along time independently
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            # pointwise: mixes the F1*D feature maps into F2 outputs
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # 254 → 31
            nn.Dropout(dropout_rate),
        )

        # ------------------------------------------------------------------
        # Classifier
        # After Block 2: shape is (batch, F2, 1, T').
        # T' is computed via a dry run because the temporal conv uses
        # padding = kernel_length // 2 which, for even kernel_length,
        # adds 1 extra time-step (output = n_timepoints + 1), making a
        # closed-form formula fragile. The dry run is exact for any input.
        # ------------------------------------------------------------------
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
        # x arrives as (batch, n_channels, n_timepoints)
        # add a "colour channel" dim so Conv2d sees (batch, 1, n_channels, n_timepoints)
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)   # → (batch, F1,   n_channels, T)
        x = self.spatial_conv(x)    # → (batch, F1*D, 1,          T//4)
        x = self.separable_conv(x)  # → (batch, F2,   1,          T//32)
        return self.classifier(x)   # → (batch, n_classes)


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


# =============================================================================
# CNN + GRU + Attention
#
# Hybrid architecture for multivariate time-series classification.
# General pattern used across EEG: https://arxiv.org/abs/2007.00897
#
# Three stages that mirror how neuroscientists think about MEG data:
#   Stage 1 — Spatial projection:  which sensors carry useful information?
#   Stage 2 — Temporal CNN:        what local patterns appear in the signal?
#   Stage 3 — GRU + Attention:     how do those patterns unfold over 2 seconds,
#                                   and at which moment is the brain state clearest?
# =============================================================================
class CNNGRUAttention(nn.Module):
    def __init__(
        self,
        n_channels: int = 248,       # MEG sensors
        n_classes: int = 4,          # rest / math / memory / motor
        spatial_hidden: int = 64,    # sensors projected to this many features
        cnn_channels: int = 64,      # CNN feature maps
        gru_hidden: int = 64,        # GRU hidden size (per direction)
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Spatial Projection
        # kernel_size=1 means no temporal mixing — purely a linear combination
        # of the 248 sensor signals at each time step independently.
        # 248 → spatial_hidden cuts the channel dimension before the CNN,
        # which keeps parameter count manageable on our small dataset.
        # ------------------------------------------------------------------
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(n_channels, spatial_hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(spatial_hidden),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Temporal CNN
        # Two Conv1d layers slide a small window (~14 ms at 508 Hz) along the
        # time axis to detect local bursts and oscillatory patterns.
        # MaxPool1d(4) compresses 1017 → 254 time steps so the GRU receives
        # a shorter but information-dense sequence.
        # ------------------------------------------------------------------
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(spatial_hidden, cnn_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(4),           # 1017 → 254
            nn.Dropout(dropout_rate),
        )

        # ------------------------------------------------------------------
        # Bidirectional GRU
        # Reads the 254-step sequence in both directions so every time step
        # has context from its past AND its future within the 2-second window.
        # Output at each step: gru_hidden * 2 (forward + backward concat).
        # GRU is used instead of LSTM for consistency with CNNGRU and to keep
        # the recurrent unit identical across the model progression.
        # ------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,        # expects (batch, seq, features)
            bidirectional=True,
        )

        gru_out_size = gru_hidden * 2  # bidirectional doubles the output dim

        # ------------------------------------------------------------------
        # Additive Attention
        # Learns a scalar score for each of the 254 time steps.
        # Steps where the brain state is most expressed get higher weight.
        # The weighted sum collapses the sequence into one fixed-size vector.
        # This is strictly more expressive than mean pooling used in CNNGRU.
        # ------------------------------------------------------------------
        self.attention = nn.Linear(gru_out_size, 1)

        # ------------------------------------------------------------------
        # Classifier
        # Single dropout + linear layer on top of the attention context vector.
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gru_out_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 248, 1017)

        x = self.spatial_proj(x)    # → (batch, spatial_hidden, 1017)
        x = self.temporal_cnn(x)    # → (batch, cnn_channels,   254)

        # GRU expects (batch, seq_len, features)
        x = x.transpose(1, 2)      # → (batch, 254, cnn_channels)
        gru_out, _ = self.gru(x)   # → (batch, 254, gru_hidden*2)

        # Attention: score each time step, then softmax → weights
        scores = self.attention(gru_out)          # → (batch, 254, 1)
        weights = torch.softmax(scores, dim=1)    # → (batch, 254, 1)  (sum to 1 over time)
        context = (weights * gru_out).sum(dim=1)  # → (batch, gru_hidden*2)

        return self.classifier(context)           # → (batch, n_classes)


# =============================================================================
# GCN + Learnable Adjacency + Node Attention  (MEGGraphNet)
#
# Treats the 248 MEG sensors as nodes in a graph whose connectivity
# structure is learned from data rather than fixed by physical distance.
#
# Key idea — learnable adjacency:
#   Each node gets a small embedding vector E_i.
#   A_ij = softmax(ReLU(E @ E^T))  ← scalar "how much does sensor i
#   attend to sensor j?" — learned end-to-end via back-propagation.
#   This lets the model discover *functional* connectivity (e.g. that
#   two distant sensors synchronise during the math task) rather than
#   relying on anatomy.
#
# Why this fits MEG:
#   Different brain states activate different sensor networks.
#   A fixed graph (k-nearest neighbours on the scalp) cannot adapt to
#   this; a learnable graph can specialise per task automatically.
#
# Sources / inspiration:
#   - https://arxiv.org/abs/1609.02907
#   - https://arxiv.org/abs/2007.02842
#   - Stanczyk & Mehrkanoon, "Deep GCNs for Wind Speed Prediction",
#     ESANN 2021  (WeatherGCNet)
# =============================================================================


class GCNLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # W projects node features; shared across all 248 nodes (weight tying)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # LayerNorm over the feature dimension — stable with any batch/graph size
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_nodes, in_features)
        # A: (n_nodes, n_nodes)  — row-stochastic after softmax
        x = self.linear(x)          # → (batch, n_nodes, out_features)
        x = torch.matmul(A, x)      # → (batch, n_nodes, out_features)
        # matmul broadcasts A over the batch dimension automatically
        return F.relu(self.norm(x))


class MEGGraphNet(nn.Module):
    def __init__(
        self,
        n_nodes: int = 248,         # MEG sensors = graph nodes
        n_timepoints: int = 1017,   # raw time steps per window
        n_classes: int = 4,         # rest / math / memory / motor
        node_feat_dim: int = 64,    # each node's time series projected to this size
        emb_dim: int = 16,          # dimension of learnable node embeddings for A
        gcn_hidden: int = 64,       # hidden size inside GCN layers
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Node Feature Projection
        # Each of the 248 sensors has a 1017-dimensional time series.
        # A plain Linear(1017, node_feat_dim) applied to the last dim
        # projects every node independently: (batch,248,1017)→(batch,248,64).
        # This step is *essential* — without it the GCN would operate on
        # 1017-d vectors, creating an unmanageable number of parameters.
        # ------------------------------------------------------------------
        self.node_proj = nn.Sequential(
            nn.Linear(n_timepoints, node_feat_dim, bias=False),
            nn.LayerNorm(node_feat_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Learnable Node Embeddings  (the heart of the learnable adjacency)
        # E ∈ R^(n_nodes × emb_dim) is a free parameter matrix.
        # The adjacency is re-computed every forward pass as:
        #   A = softmax( ReLU( E @ E^T ) )
        # ReLU removes negative inner products (no inhibitory edges).
        # Row-wise softmax turns raw scores into a probability distribution:
        # A_i sums to 1, so the GCN aggregation is a weighted average of
        # neighbouring node features — numerically stable and interpretable.
        # ------------------------------------------------------------------
        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim))

        # ------------------------------------------------------------------
        # GCN Layers
        # Two stacked layers give each node a receptive field of depth 2 —
        # it "sees" its direct neighbours AND their neighbours.
        # For 248 densely-connected nodes two layers is usually sufficient;
        # deeper stacks risk over-smoothing (all nodes converge to the same
        # representation).
        # ------------------------------------------------------------------
        self.gcn1 = GCNLayer(node_feat_dim, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden,    gcn_hidden)
        self.dropout = nn.Dropout(dropout_rate)

        # ------------------------------------------------------------------
        # Node-level Attention (readout)
        # After two GCN layers we have (batch, 248, gcn_hidden).
        # We need one fixed-size vector per sample for the classifier.
        # A linear layer scores each node; softmax turns scores into weights;
        # the weighted sum is the final graph-level representation.
        # This is more informative than simple mean-pooling: sensors that are
        # most discriminative for the current sample get higher weight.
        # ------------------------------------------------------------------
        self.node_attention = nn.Linear(gcn_hidden, 1)

        # ------------------------------------------------------------------
        # Classifier
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(gcn_hidden, n_classes),
        )

    def _build_adjacency(self) -> torch.Tensor:
        # Compute the adjacency matrix from learnable node embeddings.
        # Shape: (n_nodes, n_nodes)
        raw = torch.matmul(self.node_emb, self.node_emb.T)  # (248, 248)
        raw = F.relu(raw)           # remove negative similarities
        A   = F.softmax(raw, dim=1) # row-normalise → weighted average in GCN
        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_nodes, n_timepoints) = (batch, 248, 1017)

        # Project each node's time series to a compact feature vector
        h = self.node_proj(x)           # → (batch, 248, node_feat_dim)

        # Build adjacency once per forward pass (shared across batch)
        A = self._build_adjacency()     # → (248, 248)

        # Two rounds of neighbourhood aggregation
        h = self.gcn1(h, A)             # → (batch, 248, gcn_hidden)
        h = self.dropout(h)
        h = self.gcn2(h, A)             # → (batch, 248, gcn_hidden)

        # Weighted readout: which sensors are most informative?
        scores  = self.node_attention(h)            # → (batch, 248, 1)
        weights = torch.softmax(scores, dim=1)      # → (batch, 248, 1)
        graph_repr = (weights * h).sum(dim=1)       # → (batch, gcn_hidden)

        return self.classifier(graph_repr)          # → (batch, n_classes)
