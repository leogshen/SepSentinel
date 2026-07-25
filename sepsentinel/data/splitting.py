# Patient-level stratified splitting for the PhysioNet Challenge dataset.
#
# All records from one patient stay in exactly one split.
# Chronological order within each patient is preserved.
# Stratified by patient-level sepsis label.

import numpy as np
from sklearn.model_selection import train_test_split


def patient_split(episodes, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                  random_state=42):
    """Split episodes into train/val/test at the patient level.

    Args:
        episodes: List of episode dicts with "patient_id" and "label" keys.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0.
        random_state: Random seed for reproducibility.

    Returns:
        dict with "train", "val", "test" keys, each a list of episodes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    patient_ids = np.array([e["patient_id"] for e in episodes])
    patient_labels = np.array([e["label"] for e in episodes])

    # First split: train vs (val + test)
    holdout_ratio = val_ratio + test_ratio
    train_idx, holdout_idx = train_test_split(
        np.arange(len(episodes)),
        test_size=holdout_ratio,
        random_state=random_state,
        stratify=patient_labels,
    )

    # Second split: val vs test (from holdout)
    holdout_labels = patient_labels[holdout_idx]
    val_fraction = val_ratio / holdout_ratio
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=1 - val_fraction,
        random_state=random_state,
        stratify=holdout_labels,
    )

    return {
        "train": [episodes[i] for i in train_idx],
        "val": [episodes[i] for i in val_idx],
        "test": [episodes[i] for i in test_idx],
    }


def print_split_summary(splits):
    """Print patient counts and label distribution per split."""
    for name in ["train", "val", "test"]:
        eps = splits[name]
        n = len(eps)
        n_sep = sum(1 for e in eps if e["label"] == 1)
        n_healthy = n - n_sep
        total_steps = sum(len(e["time"]) for e in eps)
        print(f"  {name:5s}: {n:5d} patients  "
              f"({n_healthy} healthy, {n_sep} sepsis, "
              f"{n_sep / n * 100:.1f}% sepsis)  "
              f"{total_steps:,} timesteps")
