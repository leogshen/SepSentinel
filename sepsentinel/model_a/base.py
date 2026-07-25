# Abstract base for Model A calibration models.
#
# Model A converts raw electrochemical sensor signals into biomarker
# concentration estimates. Each analyte may use a different model
# (or a shared multi-output model).

from abc import ABC, abstractmethod

import numpy as np


class CalibrationModel(ABC):
    """Base class for electrochemical signal → concentration models.

    Unlike Model B (classification), Model A is a regression task:
    raw signal → continuous concentration value.
    """

    name: str = "base"
    analyte: str = ""  # "il6", "lactate", or "ph"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        """Train the calibration model.

        Args:
            X: Raw sensor signals. Shape depends on analyte:
               IL-6 SWV: (n_samples, n_voltages) or (n_samples, timesteps, n_voltages)
               Lactate amperometric: (n_samples, timesteps)
               pH potentiometric: (n_samples, timesteps)
            y: Known concentrations. Shape: (n_samples,)

        Returns:
            Metrics dict (e.g. MAE, R², calibration curve stats).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estimate concentration from raw signals. Shape: (n_samples,)"""

    @abstractmethod
    def save(self, directory: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> "CalibrationModel": ...
