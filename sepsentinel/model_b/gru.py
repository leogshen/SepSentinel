# GRU baseline for per-timestep sepsis prediction.
#
# Why GRU over LSTM/TCN for the initial baseline:
#   - Fewer parameters than LSTM (2 gates vs 3), faster training
#   - Strong empirical performance on clinical time series
#   - Handles variable-length sequences via pack_padded_sequence
#   - Simple enough to verify the data pipeline before adding complexity
#   - Modular: the GRU encoder implements SequenceEncoder and can be swapped
#     for a Transformer or TCN without changing the training loop
#
# Architecture:
#   Input: (batch, seq_len, n_features)
#   -> GRU encoder -> (batch, seq_len, hidden_dim)
#   -> Linear head -> (batch, seq_len, 1) per-timestep probability
#
# Future channels (Lactate, pH, IL-6) simply increase n_features.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sepsentinel.model_b.base import SequenceEncoder


class GRUEncoder(SequenceEncoder, nn.Module):
    """GRU-based sequence encoder."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def get_encoding_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) original sequence lengths for packing

        Returns:
            (batch, seq_len, hidden_dim) — hidden state at every timestep
        """
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                         enforce_sorted=True)
            output, _ = self.gru(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, _ = self.gru(x)
        return output


class SepsisGRU(nn.Module):
    """Per-timestep sepsis prediction using GRU encoder + linear head."""

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = GRUEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) for packing

        Returns:
            (batch, seq_len) — P(sepsis) at each timestep
        """
        h = self.encoder(x, lengths)        # (batch, seq_len, hidden_dim)
        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)   # (batch, seq_len)
        return logits
