# Model B: Sepsis risk prediction
#
# Inputs: Estimated biomarker concentrations (from Model A) + physiological signals
# Output: Continuous probability of sepsis
#
# Staged development:
#   Stage 1: HR, SpO2, Temperature, Respiratory Rate
#   Stage 2: + Lactate, pH
#   Stage 3: + IL-6 (once Model A is operational)

from sepsentinel.model_b.registry import get_model, list_models
from sepsentinel.model_b.base import SepsisModel, SequenceEncoder
