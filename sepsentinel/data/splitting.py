# Patient-level stratified splitting for the PhysioNet Challenge dataset.
#
# All records from one patient stay in exactly one split.
# Chronological order within each patient is preserved.
# Stratified by patient-level sepsis label.
#
# For multi-stay datasets (MIMIC-IV, SICdb) use grouped_patient_split:
# it splits at the PERSON level (subject_id) so no person appears in two
# splits. The original patient_split is episode-level and only safe when
# one patient has exactly one episode (true for PhysioNet 2019).

from collections import defaultdict

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


def grouped_patient_split(episodes, train_ratio=0.70, val_ratio=0.15,
                          test_ratio=0.15, random_state=42,
                          group_key="subject_id"):
    """Split episodes at the person level for multi-stay datasets.

    All episodes sharing the same `group_key` value (default: subject_id)
    are assigned to the same split. Stratified by group-level label
    (1 if the person has ANY septic episode).

    Episodes missing `group_key` fall back to their patient_id as the group,
    which makes this function equivalent to patient_split() on single-stay
    datasets like PhysioNet 2019.

    Args:
        episodes: List of episode dicts with "label", "patient_id", and
            (for multi-stay data) `group_key` keys.
        train_ratio, val_ratio, test_ratio: Must sum to 1.0.
        random_state: Random seed.
        group_key: Episode dict key identifying the person.

    Returns:
        dict with "train", "val", "test" keys, each a list of episodes.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    group_to_indices = defaultdict(list)
    for i, ep in enumerate(episodes):
        gid = ep.get(group_key) or ep["patient_id"]
        group_to_indices[gid].append(i)

    group_ids = sorted(group_to_indices.keys())  # deterministic order
    group_labels = np.array([
        max(episodes[i]["label"] for i in group_to_indices[g])
        for g in group_ids
    ])

    idx = np.arange(len(group_ids))
    holdout_ratio = val_ratio + test_ratio
    train_g, holdout_g = train_test_split(
        idx, test_size=holdout_ratio, random_state=random_state,
        stratify=group_labels,
    )
    val_fraction = val_ratio / holdout_ratio
    val_g, test_g = train_test_split(
        holdout_g, test_size=1 - val_fraction, random_state=random_state,
        stratify=group_labels[holdout_g],
    )

    def expand(group_idx):
        out = []
        for gi in group_idx:
            out.extend(episodes[i] for i in group_to_indices[group_ids[gi]])
        return out

    return {
        "train": expand(train_g),
        "val": expand(val_g),
        "test": expand(test_g),
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
