# Transformer encoder — Module 7 implementation.
#
# Architecture (v1 single-branch):
#   Input: (batch, timesteps, n_features)  — n_features is dynamic per stage
#   -> Positional encoding
#   -> Transformer encoder layers
#   -> Pooling -> (batch, encoding_dim)
#   -> Dense head -> (batch, 1) sigmoid
#
# The encoder implements SequenceEncoder, same swap/compose pattern as TCN.

# from sepsentinel.model_b.base import SepsisModel, SequenceEncoder
# from sepsentinel.model_b.registry import register
#
# class TransformerEncoder(SequenceEncoder, nn.Module):
#     ...
#
# @register
# class TransformerModel(SepsisModel):
#     name = "transformer"
#     requires_sequences = True
