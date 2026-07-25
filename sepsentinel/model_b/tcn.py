# Temporal Convolutional Network for per-timestep sepsis prediction.
#
# Architecture (v1 single-branch):
#   Input: (batch, seq_len, n_features)
#   -> Causal dilated convolutions (exponentially increasing dilation)
#   -> Per-timestep hidden representations
#   -> Linear head -> (batch, seq_len, 1) per-timestep logits
#
# The TCN encoder implements SequenceEncoder so it can later be used
# as one branch in a dual-branch architecture (v2).

import torch
import torch.nn as nn

from sepsentinel.model_b.base import SequenceEncoder


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Two causal convolutions with residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual projection if channel dims differ
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + self.residual(x))


class TCNEncoder(SequenceEncoder, nn.Module):
    """Temporal Convolutional Network encoder.

    Stacks TemporalBlocks with exponentially increasing dilation
    to cover long-range dependencies without recurrence.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=4,
                 kernel_size=3, dropout=0.2):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim

        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, hidden_dim, kernel_size,
                                        dilation, dropout))
        self.network = nn.Sequential(*layers)

    def get_encoding_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: ignored (TCN handles variable length via masking)

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Conv1d expects (batch, channels, seq_len)
        out = self.network(x.transpose(1, 2))
        return out.transpose(1, 2)  # back to (batch, seq_len, hidden_dim)


class SepsisTCN(nn.Module):
    """Per-timestep sepsis prediction using TCN encoder + linear head."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=4,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, hidden_dim, num_layers,
                                  kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) — not used by TCN but kept for API compatibility

        Returns:
            (batch, seq_len) — logits at each timestep
        """
        h = self.encoder(x, lengths)       # (batch, seq_len, hidden_dim)
        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)  # (batch, seq_len)
        return logits
