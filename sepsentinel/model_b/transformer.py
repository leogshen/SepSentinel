# Transformer encoder for per-timestep sepsis prediction.
#
# Architecture (v1 single-branch):
#   Input: (batch, seq_len, n_features)
#   -> Linear projection to d_model
#   -> Positional encoding (sinusoidal)
#   -> Transformer encoder layers (causal self-attention)
#   -> Linear head -> (batch, seq_len, 1) per-timestep logits
#
# Uses causal masking so each timestep can only attend to past/current,
# matching the clinical real-time prediction setting.

import math

import torch
import torch.nn as nn

from sepsentinel.model_b.base import SequenceEncoder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(SequenceEncoder, nn.Module):
    """Transformer encoder with causal self-attention."""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2):
        nn.Module.__init__(self)
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def get_encoding_dim(self) -> int:
        return self.d_model

    def _generate_causal_mask(self, seq_len, device):
        """Upper-triangular mask: each position attends only to past/current."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) original lengths for padding mask

        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # Padding mask: True = ignore
        pad_mask = None
        if lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        out = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return out


class SepsisTransformer(nn.Module):
    """Per-timestep sepsis prediction using Transformer encoder + linear head."""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, nhead, num_layers,
                                          dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) for padding mask

        Returns:
            (batch, seq_len) — logits at each timestep
        """
        h = self.encoder(x, lengths)       # (batch, seq_len, d_model)
        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)  # (batch, seq_len)
        return logits
