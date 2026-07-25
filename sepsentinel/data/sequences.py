# Sliding window extraction and tensor construction.
#
# Converts patient episodes into (X, y) arrays ready for model training.
# Window labeling strategy for early detection:
#   - Windows from healthy episodes -> label 0
#   - Windows from septic episodes where onset falls within
#     [window_end - prediction_horizon, window_end] -> label 1
#   - Windows from septic episodes before the prediction horizon -> label 0

import numpy as np
from sklearn.model_selection import train_test_split

from sepsentinel.config.signals import (
    DEFAULT_HISTORY_WINDOW_MIN, DEFAULT_SAMPLING_INTERVAL_MIN, NUM_FEATURES,
)


def episodes_to_windows(episodes, window_minutes=DEFAULT_HISTORY_WINDOW_MIN,
                        interval_minutes=DEFAULT_SAMPLING_INTERVAL_MIN,
                        stride_steps=1, prediction_horizon_minutes=60):
    """Extract labeled sliding windows from a list of episodes.

    Args:
        episodes: List of episode dicts from generate_episodes().
        window_minutes: Length of each input window in minutes.
        interval_minutes: Sampling interval (for computing steps).
        stride_steps: Step stride between consecutive windows.
        prediction_horizon_minutes: A window is labeled positive if sepsis
            onset falls within this many minutes after the window ends.

    Returns:
        X: np.ndarray (n_windows, window_steps, n_features)
        y: np.ndarray (n_windows,) binary labels
        meta: list of dicts with patient_id and window position info
    """
    window_steps = window_minutes // interval_minutes
    horizon_steps = prediction_horizon_minutes // interval_minutes

    X_list = []
    y_list = []
    meta = []

    for ep in episodes:
        signals = ep["signals"]  # (n_steps, 7)
        n_steps = signals.shape[0]
        onset = ep["onset_step"]

        if n_steps < window_steps:
            continue

        for start in range(0, n_steps - window_steps + 1, stride_steps):
            end = start + window_steps
            window = signals[start:end]

            if ep["label"] == 0:
                label = 0
            else:
                # Positive if onset is at or before (window_end + horizon)
                # AND onset is at or after window_start (don't label windows
                # long after onset — the patient is already septic, not early detection)
                if onset is not None and onset <= end + horizon_steps and onset >= start:
                    label = 1
                else:
                    label = 0

            X_list.append(window)
            y_list.append(label)
            meta.append({
                "patient_id": ep["patient_id"],
                "start_step": start,
                "end_step": end,
                "episode_label": ep["label"],
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, meta


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split into train/val/test sets.

    Returns:
        dict with keys "X_train", "y_train", "X_val", "y_val", "X_test", "y_test"
    """
    # First split off test
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split rest into train/val
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, random_state=random_state, stratify=y_rest
    )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def print_split_summary(splits):
    """Print label distribution for each split."""
    for name in ["train", "val", "test"]:
        y = splits[f"y_{name}"]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"  {name:5s}: {len(y):5d} windows  "
              f"({n_neg} healthy, {n_pos} septic, "
              f"{n_pos / len(y) * 100:.1f}% positive)")
