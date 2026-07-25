# Preprocessing for variable-length patient sequences.
#
# - Forward-fill missing values within each patient, then fill remaining NaNs
#   with training-set feature means.
# - Z-score normalization fitted on training data only.
# - Outlier clipping to physiologically plausible ranges.
# - Extensible: adding features requires only updating the feature list and
#   clip ranges — the pipeline handles arbitrary feature counts.

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# Physiological clip ranges (min, max) to remove sensor artifacts.
CLIP_RANGES = {
    "heart_rate": (20, 250),
    "spo2": (50, 100),
    "temperature": (30, 43),
    "respiratory_rate": (2, 60),
    "lactate": (0, 30),
    "ph": (6.5, 7.8),
    "il6": (0, 1000),
}


class SequencePreprocessor:
    """Fits normalization on training episodes, transforms all splits.

    Handles variable-length sequences. Each episode's signals array
    is (seq_len, n_features).
    """

    def __init__(self, features, clip=True):
        """
        Args:
            features: List of feature names (e.g. ["heart_rate", "spo2", ...]).
            clip: Whether to clip values to physiological ranges.
        """
        self.features = features
        self.clip = clip
        self.mean = None
        self.std = None
        self.fill_values = None

    def fit(self, episodes):
        """Compute normalization stats from training episodes."""
        all_values = np.concatenate([e["signals"] for e in episodes], axis=0)

        # Compute fill values (per-feature mean of non-NaN values)
        self.fill_values = np.nanmean(all_values, axis=0)

        # Impute for stats computation
        filled = self._impute_array(all_values)
        if self.clip:
            filled = self._clip_array(filled)

        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, episodes):
        """Apply preprocessing to a list of episodes.

        Returns list of dicts with:
            signals: np.ndarray (seq_len, n_features) — preprocessed
            labels: np.ndarray (seq_len,) — per-timestep SepsisLabel
            length: int — original sequence length
            patient_id: str
            label: int — patient-level label
        """
        results = []
        for ep in episodes:
            signals = ep["signals"].copy()

            # Forward-fill within patient
            signals = self._forward_fill(signals)
            # Fill any remaining NaNs with training means
            signals = self._impute_array(signals)
            # Clip
            if self.clip:
                signals = self._clip_array(signals)
            # Normalize
            signals = (signals - self.mean) / self.std

            results.append({
                "signals": signals.astype(np.float32),
                "labels": ep["labels"].astype(np.float32),
                "length": len(signals),
                "patient_id": ep["patient_id"],
                "label": ep["label"],
            })
        return results

    def fit_transform(self, episodes):
        return self.fit(episodes).transform(episodes)

    def _forward_fill(self, arr):
        """Forward-fill NaNs along axis 0."""
        arr = arr.copy()
        for j in range(arr.shape[1]):
            for i in range(1, arr.shape[0]):
                if np.isnan(arr[i, j]):
                    arr[i, j] = arr[i - 1, j]
            # Back-fill if first value is NaN
            if np.isnan(arr[0, j]):
                for i in range(1, arr.shape[0]):
                    if not np.isnan(arr[i, j]):
                        arr[:i, j] = arr[i, j]
                        break
        return arr

    def _impute_array(self, arr):
        """Replace remaining NaNs with training-set feature means."""
        arr = arr.copy()
        if self.fill_values is not None:
            for j in range(arr.shape[1]):
                mask = np.isnan(arr[:, j])
                arr[mask, j] = self.fill_values[j]
        return arr

    def _clip_array(self, arr):
        """Clip values to physiological ranges."""
        arr = arr.copy()
        for j, feat in enumerate(self.features):
            if feat in CLIP_RANGES:
                lo, hi = CLIP_RANGES[feat]
                arr[:, j] = np.clip(arr[:, j], lo, hi)
        return arr


def collate_fn(batch):
    """Collate variable-length sequences into padded tensors.

    Args:
        batch: List of dicts from SequencePreprocessor.transform().

    Returns:
        signals: (batch, max_len, n_features) padded tensor
        labels: (batch, max_len) padded tensor
        lengths: (batch,) tensor of original lengths
        mask: (batch, max_len) boolean mask (True = valid, False = padding)
    """
    # Sort by length descending (required for pack_padded_sequence)
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)

    signals = [torch.from_numpy(b["signals"]) for b in batch]
    labels = [torch.from_numpy(b["labels"]) for b in batch]
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

    signals_padded = pad_sequence(signals, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1.0)

    max_len = signals_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return signals_padded, labels_padded, lengths, mask
