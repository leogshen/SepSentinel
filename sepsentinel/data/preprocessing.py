# Preprocessing utilities for signal data.
#
# Designed to work identically on synthetic and MIMIC data.
# Fit scalers on training data, then transform val/test.

import numpy as np


class StandardScaler:
    """Z-score normalization. Fits per-feature mean/std from training data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute mean and std from X. X shape: (n_samples, timesteps, features) or (n_samples, features)."""
        if X.ndim == 3:
            # Flatten samples and timesteps for per-feature stats
            flat = X.reshape(-1, X.shape[-1])
        else:
            flat = X
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X):
        """Apply normalization."""
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean


class MinMaxScaler:
    """Scale features to [0, 1] range."""

    def __init__(self):
        self.min = None
        self.range = None

    def fit(self, X):
        if X.ndim == 3:
            flat = X.reshape(-1, X.shape[-1])
        else:
            flat = X
        self.min = flat.min(axis=0)
        max_val = flat.max(axis=0)
        self.range = max_val - self.min
        self.range[self.range == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.min) / self.range

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.range + self.min


def get_scaler(method="zscore"):
    """Factory for scalers."""
    if method == "zscore":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaling method: {method}. Use 'zscore' or 'minmax'.")


def impute_missing(X, method="forward_fill"):
    """Fill NaN values in signal data.

    Args:
        X: np.ndarray, shape (n_samples, timesteps, features) or (timesteps, features)
        method: "forward_fill" or "mean"

    Returns:
        X with NaNs replaced.
    """
    X = X.copy()

    if method == "forward_fill":
        if X.ndim == 3:
            for i in range(X.shape[0]):
                _forward_fill_2d(X[i])
        else:
            _forward_fill_2d(X)

    elif method == "mean":
        if X.ndim == 3:
            flat = X.reshape(-1, X.shape[-1])
            col_means = np.nanmean(flat, axis=0)
            for j in range(X.shape[-1]):
                mask = np.isnan(X[:, :, j])
                X[:, :, j][mask] = col_means[j]
        else:
            col_means = np.nanmean(X, axis=0)
            for j in range(X.shape[-1]):
                mask = np.isnan(X[:, j])
                X[:, j][mask] = col_means[j]
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    return X


def _forward_fill_2d(arr):
    """Forward-fill NaNs along axis 0 of a 2D array in place."""
    for j in range(arr.shape[1]):
        for i in range(1, arr.shape[0]):
            if np.isnan(arr[i, j]):
                arr[i, j] = arr[i - 1, j]
    # If the first value is NaN, back-fill from the next valid value
    for j in range(arr.shape[1]):
        if np.isnan(arr[0, j]):
            for i in range(1, arr.shape[0]):
                if not np.isnan(arr[i, j]):
                    arr[0:i, j] = arr[i, j]
                    break


def preprocess_splits(splits, scaler_method="zscore", impute_method="forward_fill"):
    """Convenience: impute and normalize train/val/test splits.

    Fits scaler on training data only, then transforms all splits.

    Args:
        splits: dict from sequences.split_data()
        scaler_method: "zscore" or "minmax"
        impute_method: "forward_fill" or "mean"

    Returns:
        Preprocessed splits dict (same keys), plus "scaler" key.
    """
    result = {}

    for key in ["X_train", "X_val", "X_test"]:
        result[key] = impute_missing(splits[key], method=impute_method)

    scaler = get_scaler(scaler_method)
    result["X_train"] = scaler.fit_transform(result["X_train"])
    result["X_val"] = scaler.transform(result["X_val"])
    result["X_test"] = scaler.transform(result["X_test"])

    for key in ["y_train", "y_val", "y_test"]:
        result[key] = splits[key]

    result["scaler"] = scaler
    return result
