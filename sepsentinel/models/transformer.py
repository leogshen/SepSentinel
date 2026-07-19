# Transformer encoder — Module 7 implementation.
#
# Architecture (v1 single-branch):
#   Input: (batch, timesteps, 7)
#   -> Positional encoding
#   -> Transformer encoder layers
#   -> Pooling -> (batch, encoding_dim)
#   -> Dense head -> (batch, 1) sigmoid
#
# The encoder implements SequenceEncoder, same swap/compose pattern as TCN.

# from sepsentinel.models.base import SepsisModel, SequenceEncoder
# from sepsentinel.models.registry import register
#
# class TransformerEncoder(SequenceEncoder, nn.Module):
#     ...
#
# @register
# class TransformerModel(SepsisModel):
#     name = "transformer"
#     requires_sequences = True
