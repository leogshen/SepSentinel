# Causal missing-value representation preprocessor.
#
# Expands each raw feature into three channels:
#   value:  forward-filled + training-mean imputed, z-score normalized
#   mask:   1 = actually observed, 0 = imputed or carried forward
#   delta:  timesteps since last real observation (0 = observed now)
#
# No back-fill: leading NaNs are filled with the training-set feature mean.
#
# Delta convention for leading NaNs (before any observation):
#   delta = i + 1  (1-indexed elapsed time from sequence start)
#   This ensures delta > 0 for all imputed values and delta = 0 only for
#   actually observed values, removing ambiguity.
#
# Output channel order for N raw features:
#   [val_0, mask_0, delta_0, val_1, mask_1, delta_1, ..., val_N-1, mask_N-1, delta_N-1]
#   Total channels = 3 * N

import numpy as np
from sepsentinel.data.preprocessing import CLIP_RANGES


class MissingnessAwarePreprocessor:
    """Expands raw features into (value, mask, delta) triplets.

    Fit on training data only. All normalization statistics are frozen
    after fit() and applied identically to val/test.
    """

    def __init__(self, features, clip=True):
        self.features = features
        self.clip = clip
        self.n_raw = len(features)
        self.n_expanded = self.n_raw * 3

        # Fitted statistics (training set only)
        self.fill_values = None   # per-feature nanmean for imputation
        self.val_mean = None      # z-score mean for value channels
        self.val_std = None       # z-score std for value channels
        self.delta_mean = None    # z-score mean for delta channels
        self.delta_std = None     # z-score std for delta channels

    def fit(self, episodes):
        """Compute normalization statistics from training episodes."""
        # Training-set feature means for imputation
        all_raw = np.concatenate([e["signals"] for e in episodes], axis=0)
        self.fill_values = np.nanmean(all_raw, axis=0)

        # Process all training episodes to collect filled values and deltas
        all_values = []
        all_deltas = []
        for ep in episodes:
            values, _, deltas = self._extract_channels(ep["signals"])
            all_values.append(values)
            all_deltas.append(deltas)

        all_values = np.concatenate(all_values, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        self.val_mean = all_values.mean(axis=0)
        self.val_std = all_values.std(axis=0)
        self.val_std[self.val_std == 0] = 1.0

        self.delta_mean = all_deltas.mean(axis=0)
        self.delta_std = all_deltas.std(axis=0)
        self.delta_std[self.delta_std == 0] = 1.0

        return self

    def transform(self, episodes):
        """Apply preprocessing, returning expanded (value, mask, delta) channels."""
        results = []
        for ep in episodes:
            values, masks, deltas = self._extract_channels(ep["signals"])

            # Normalize values (z-score)
            values_norm = (values - self.val_mean) / self.val_std

            # Masks: keep as 0/1 (no normalization)

            # Normalize deltas (z-score)
            deltas_norm = (deltas - self.delta_mean) / self.delta_std

            # Interleave: [val_0, mask_0, delta_0, val_1, ...]
            n_steps = len(values)
            expanded = np.zeros((n_steps, self.n_expanded), dtype=np.float32)
            for j in range(self.n_raw):
                expanded[:, j * 3] = values_norm[:, j]
                expanded[:, j * 3 + 1] = masks[:, j]
                expanded[:, j * 3 + 2] = deltas_norm[:, j]

            results.append({
                "signals": expanded,
                "labels": ep["labels"].astype(np.float32),
                "length": n_steps,
                "patient_id": ep["patient_id"],
                "label": ep["label"],
            })
        return results

    def fit_transform(self, episodes):
        return self.fit(episodes).transform(episodes)

    def _extract_channels(self, raw_signals):
        """From raw signals (with NaNs), produce values, masks, deltas.

        Args:
            raw_signals: (n_steps, n_features) with NaNs.

        Returns:
            values: (n_steps, n_features) — forward-filled + mean-imputed, clipped
            masks:  (n_steps, n_features) — 1 = observed, 0 = imputed
            deltas: (n_steps, n_features) — timesteps since last observation
        """
        signals = raw_signals.copy()
        n_steps, n_feat = signals.shape

        # Step 3: observation mask BEFORE any imputation
        masks = (~np.isnan(signals)).astype(np.float32)

        # Step 1: forward-fill only (no back-fill)
        for j in range(n_feat):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]

        # Step 2: fill remaining leading NaNs with training-set mean
        if self.fill_values is not None:
            for j in range(n_feat):
                nan_mask = np.isnan(signals[:, j])
                signals[nan_mask, j] = self.fill_values[j]

        # Clip to physiological ranges
        if self.clip:
            for j, feat in enumerate(self.features):
                if feat in CLIP_RANGES:
                    lo, hi = CLIP_RANGES[feat]
                    signals[:, j] = np.clip(signals[:, j], lo, hi)

        # Step 4: time since last real observation
        deltas = np.zeros((n_steps, n_feat), dtype=np.float32)
        for j in range(n_feat):
            last_obs = -1
            for i in range(n_steps):
                if masks[i, j] == 1.0:
                    deltas[i, j] = 0.0
                    last_obs = i
                else:
                    if last_obs >= 0:
                        # Forward-filled: steps since last real observation
                        deltas[i, j] = float(i - last_obs)
                    else:
                        # Leading NaN: elapsed time from sequence start (1-indexed)
                        deltas[i, j] = float(i + 1)

        return signals, masks, deltas
