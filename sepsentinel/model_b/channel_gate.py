# Dynamic channel gating for feature-group-level attention.
#
# Produces learnable, time-varying weights for each feature group,
# allowing the model to emphasize different biomarkers for different
# patients and timepoints.
#
# Architecture:  input -> ChannelGate -> gated_input -> Transformer -> head
#
# Gate weights are in [0,1] via sigmoid, applied multiplicatively.
# Weights can be extracted during evaluation for interpretability
# via forward_with_weights().
#
# MAE compatibility: self.encoder can accept pretrained weights via
# self.encoder.load_state_dict(pretrained_dict) before fine-tuning.

import torch
import torch.nn as nn

from sepsentinel.model_b.transformer import TransformerEncoder


class ChannelGate(nn.Module):
    """Per-timestep feature group gating.

    All channels belonging to the same feature group (value, mask, delta,
    trajectory) share one scalar gate weight per timestep.

    Args:
        input_dim: Total number of input channels.
        n_groups: Number of feature groups (= number of raw features).
        channel_to_group: List[int] of length input_dim mapping each channel
            to its group index in [0, n_groups).
        hidden_dim: Hidden layer size in the gating network.
    """

    def __init__(self, input_dim, n_groups, channel_to_group, hidden_dim=32):
        super().__init__()
        self.n_groups = n_groups
        self.register_buffer(
            "channel_to_group",
            torch.tensor(channel_to_group, dtype=torch.long),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_groups),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            x_gated: (batch, seq_len, input_dim) — input scaled by gate weights
            group_weights: (batch, seq_len, n_groups) — gate weights in [0, 1]
        """
        group_weights = self.gate_net(x)
        channel_weights = group_weights[:, :, self.channel_to_group]
        return x * channel_weights, group_weights


class SepsisTransformerGated(nn.Module):
    """Transformer with dynamic channel gating for sepsis prediction.

    The ChannelGate produces interpretable per-feature-group weights that
    can be extracted during evaluation via forward_with_weights().
    """

    def __init__(self, input_dim, n_groups, channel_to_group,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                 dropout=0.2, gate_hidden=32):
        super().__init__()
        self.channel_gate = ChannelGate(
            input_dim, n_groups, channel_to_group, gate_hidden,
        )
        self.encoder = TransformerEncoder(
            input_dim, d_model, nhead, num_layers, dim_feedforward, dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths=None):
        """Standard forward pass (training-compatible with existing Trainer)."""
        x_gated, _ = self.channel_gate(x)
        h = self.encoder(x_gated, lengths)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)

    def forward_with_weights(self, x, lengths=None):
        """Forward pass returning gate weights for interpretability."""
        x_gated, gate_weights = self.channel_gate(x)
        h = self.encoder(x_gated, lengths)
        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)
        return logits, gate_weights
