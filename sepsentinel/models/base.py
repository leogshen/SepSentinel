# Abstract base classes for all sepsis prediction models.
#
# Design: encoder and prediction head are separated so that:
#   v1: single encoder processes all 7 features
#   v2: two encoders (physiological + biomarker) with fusion before the head
#
# Flat models (RF, XGBoost) ignore the encoder abstraction and work directly
# on feature vectors, flattening temporal windows internally.

from abc import ABC, abstractmethod

import numpy as np


class SequenceEncoder(ABC):
    """Encodes a temporal sequence into a fixed-size representation.

    Input:  (batch, timesteps, features)
    Output: (batch, encoding_dim)

    Implementations: TCN, Transformer encoder, etc.
    Separating the encoder from the prediction head allows:
    - Swapping backbones (TCN <-> Transformer) without changing the rest
    - Composing multiple encoders for dual-branch architectures
    """

    @abstractmethod
    def get_encoding_dim(self) -> int:
        """Return the dimensionality of the encoder's output."""

    # forward() is defined by nn.Module in PyTorch implementations


class SepsisModel(ABC):
    """Base class for all sepsis prediction models.

    All models — flat (RF, XGBoost) and sequential (TCN, Transformer) —
    implement this interface so they can be trained, evaluated, and swapped
    through a single API.
    """

    name: str = "base"
    requires_sequences: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """Train the model. Returns a metrics dict.

        X: (n_samples, n_features) for flat models
           (n_samples, n_timesteps, n_features) for sequential models
        y: (n_samples,) binary labels (0=healthy, 1=septic)
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(sepsis) for each sample. Shape: (n_samples,)"""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        """Return 0-100 risk scores."""
        return np.round(self.predict_proba(X) * 100, 1)

    @abstractmethod
    def save(self, directory: str) -> None:
        """Save model artifacts to a directory."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> "SepsisModel":
        """Load a saved model from a directory."""


def rule_based_risk(values: dict) -> float:
    """Rule-based fallback when no ML model is available.

    Accepts a dict with any subset of signal keys. Returns 0-100 risk score.
    Weights: lactate 25%, il6 25%, ph 15%, heart_rate 10%, respiratory_rate 10%,
             temperature 10%, spo2 5%.
    """
    score = 0.0

    lactate = values.get("lactate")
    if lactate is not None:
        if lactate <= 2.0:
            score += 0
        elif lactate <= 4.0:
            score += (lactate - 2.0) / 2.0 * 25
        else:
            score += 25

    il6 = values.get("il6")
    if il6 is not None:
        if il6 <= 7:
            score += 0
        elif il6 <= 100:
            score += (il6 - 7) / 93 * 25
        else:
            score += 25

    ph = values.get("ph")
    if ph is not None:
        if ph >= 7.35:
            score += 0
        elif ph >= 7.25:
            score += (7.35 - ph) / 0.10 * 15
        else:
            score += 15

    hr = values.get("heart_rate")
    if hr is not None:
        if hr <= 100:
            score += 0
        elif hr <= 130:
            score += (hr - 100) / 30 * 10
        else:
            score += 10

    rr = values.get("respiratory_rate")
    if rr is not None:
        if rr <= 22:
            score += 0
        elif rr <= 30:
            score += (rr - 22) / 8 * 10
        else:
            score += 10

    temp = values.get("temperature")
    if temp is not None:
        if temp <= 38.0:
            score += 0
        elif temp <= 40.0:
            score += (temp - 38.0) / 2.0 * 10
        else:
            score += 10

    spo2 = values.get("spo2")
    if spo2 is not None:
        if spo2 >= 95:
            score += 0
        elif spo2 >= 88:
            score += (95 - spo2) / 7 * 5
        else:
            score += 5

    return round(max(0, min(100, score)), 1)
