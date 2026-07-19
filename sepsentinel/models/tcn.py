# Temporal Convolutional Network — Module 7 implementation.
#
# Architecture (v1 single-branch):
#   Input: (batch, timesteps, 7)
#   -> TCN encoder -> (batch, encoding_dim)
#   -> Dense head -> (batch, 1) sigmoid
#
# The TCN encoder implements SequenceEncoder so it can later be used
# as one branch in a dual-branch architecture (v2):
#   physio_encoder = TCNEncoder(input_dim=4, ...)   # HR, RR, Temp, SpO2
#   bio_encoder    = TCNEncoder(input_dim=3, ...)   # pH, Lactate, IL-6
#   fused = cat(physio_encoder(x_physio), bio_encoder(x_bio))
#   output = head(fused)

# from sepsentinel.models.base import SepsisModel, SequenceEncoder
# from sepsentinel.models.registry import register
#
# class TCNEncoder(SequenceEncoder, nn.Module):
#     ...
#
# @register
# class TCNModel(SepsisModel):
#     name = "tcn"
#     requires_sequences = True
