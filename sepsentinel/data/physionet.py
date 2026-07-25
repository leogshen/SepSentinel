# PhysioNet/CinC 2019 Sepsis Challenge dataset loader.
#
# Loads the Challenge dataset and converts it into the same episode format
# as generate_episodes(), so the rest of the pipeline (windowing,
# preprocessing, model training) works identically.
#
# Dataset: hourly ICU time series with SepsisLabel per row.
# Available signals: HR, O2Sat, Temp, Resp, pH, Lactate (+ many more).
# No IL-6 in this dataset.

import os

import numpy as np
import pandas as pd

from sepsentinel.config.signals import PHYSIONET_COLUMN_MAP, STAGES


_DEFAULT_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
    "tea340yashjoshi", "sepsis-prediction-dataset", "versions", "1", "Dataset.csv"
)


def load_physionet(filepath=None, stage=1, min_length=6):
    """Load PhysioNet Sepsis Challenge data as episodes.

    Args:
        filepath: Path to Dataset.csv. If None, uses kagglehub cache location.
        stage: Which feature set to use (1, 2, or 3). See config.signals.STAGES.
        min_length: Minimum episode length in hours. Shorter episodes are dropped.

    Returns:
        List of episode dicts:
            patient_id: str
            time: np.ndarray (n_steps,) in hours
            signals: np.ndarray (n_steps, n_features)
            labels: np.ndarray (n_steps,) per-timestep SepsisLabel (0 or 1)
            label: int, patient-level label (1 if any timestep is septic)
            onset_step: int index of first SepsisLabel=1, or None
            features: list of feature names used
    """
    if filepath is None:
        filepath = _DEFAULT_CACHE
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download with: kagglehub.dataset_download('tea340yashjoshi/sepsis-prediction-dataset')"
        )

    features = STAGES[stage]

    # Map our feature names back to PhysioNet column names
    reverse_map = {v: k for k, v in PHYSIONET_COLUMN_MAP.items()}
    source_cols = [reverse_map[f] for f in features if f in reverse_map]
    needed_cols = source_cols + ["Patient_ID", "ICULOS", "SepsisLabel"]

    df = pd.read_csv(filepath, usecols=needed_cols)

    episodes = []
    for pid, group in df.groupby("Patient_ID"):
        group = group.sort_values("ICULOS")

        if len(group) < min_length:
            continue

        time_hours = group["ICULOS"].values.astype(np.float32)

        # Extract signals in STAGES order
        signal_data = np.zeros((len(group), len(features)), dtype=np.float32)
        for j, feat in enumerate(features):
            col = reverse_map.get(feat)
            if col and col in group.columns:
                signal_data[:, j] = group[col].values

        # Per-timestep and patient-level labels
        per_step_labels = group["SepsisLabel"].values.astype(np.float32)
        if per_step_labels.max() > 0:
            label = 1
            onset_step = int(np.argmax(per_step_labels > 0))
        else:
            label = 0
            onset_step = None

        episodes.append({
            "patient_id": f"PN-{int(pid):05d}",
            "time": time_hours,
            "signals": signal_data,
            "labels": per_step_labels,
            "label": label,
            "onset_step": onset_step,
            "features": features,
        })

    return episodes


def print_physionet_summary(episodes):
    """Print summary statistics for loaded episodes."""
    n_total = len(episodes)
    n_septic = sum(1 for e in episodes if e["label"] == 1)
    n_healthy = n_total - n_septic
    lengths = [len(e["time"]) for e in episodes]
    n_features = episodes[0]["signals"].shape[1] if episodes else 0

    print(f"  PhysioNet Challenge Dataset")
    print(f"    Episodes:  {n_total} ({n_healthy} healthy, {n_septic} septic)")
    print(f"    Features:  {n_features} ({episodes[0].get('features', [])})")
    print(f"    Length:    {min(lengths)}-{max(lengths)} hours "
          f"(mean {np.mean(lengths):.0f})")

    # NaN density per feature
    if episodes:
        all_signals = np.concatenate([e["signals"] for e in episodes], axis=0)
        nan_pcts = np.isnan(all_signals).mean(axis=0) * 100
        print(f"    NaN density per feature:")
        for feat, pct in zip(episodes[0]["features"], nan_pcts):
            print(f"      {feat:22s}: {pct:.1f}%")
