# Synthetic data generation for pipeline development and testing.
#
# Two modes:
#   1. generate_flat_dataset()  — static 7-feature records (bridge for RF/XGBoost)
#   2. generate_episodes()      — time-series patient episodes for sequential models
#
# Synthetic data is NOT intended to validate medical performance.
# It exists to develop the pipeline, test architectures, and debug dashboards.

import os
import random

import numpy as np
import pandas as pd

from sepsentinel.config.signals import FEATURE_ORDER, NUM_FEATURES

# Per-signal generation parameters.
# Baselines are randomized per patient around these centers.
_SIGNAL_PARAMS = {
    #                   baseline_mean, baseline_std, noise_std,  septic_delta, septic_exp
    "heart_rate":       (75,           8,            3.0,        50,           1.0),
    "respiratory_rate": (15,           2,            1.0,        15,           1.0),
    "temperature":      (36.8,         0.2,          0.15,       2.5,          1.2),
    "spo2":             (97,           1,            0.5,        -12,          1.0),
    "ph":               (7.40,         0.02,         0.01,       -0.20,        1.0),
    "lactate":          (1.2,          0.3,          0.15,       5.0,          1.5),
    "il6":              (3.0,          1.5,          1.0,        150,          2.0),
}

# Hard physiological bounds to clamp generated values.
_SIGNAL_BOUNDS = {
    "heart_rate":       (30, 200),
    "respiratory_rate": (4, 50),
    "temperature":      (34.0, 42.0),
    "spo2":             (70, 100),
    "ph":               (6.8, 7.6),
    "lactate":          (0.2, 15.0),
    "il6":              (0, 500),
}


def generate_episodes(num_episodes=200, duration_minutes=240,
                      interval_minutes=5, sepsis_ratio=0.5, seed=42):
    """Generate time-series patient episodes.

    Args:
        num_episodes: Total number of patient episodes.
        duration_minutes: Length of each episode in minutes.
        interval_minutes: Sampling interval.
        sepsis_ratio: Fraction of episodes that are septic.
        seed: Random seed.

    Returns:
        List of episode dicts:
            patient_id: str
            time: np.ndarray (n_steps,) in minutes
            signals: np.ndarray (n_steps, 7) in FEATURE_ORDER
            label: 0 (healthy) or 1 (septic)
            onset_step: int index where sepsis deterioration begins, or None
    """
    rng = np.random.RandomState(seed)
    num_septic = int(num_episodes * sepsis_ratio)
    num_healthy = num_episodes - num_septic

    labels = [0] * num_healthy + [1] * num_septic
    rng.shuffle(labels)

    time_points = np.arange(0, duration_minutes + interval_minutes, interval_minutes)
    n_steps = len(time_points)

    episodes = []
    for i, label in enumerate(labels):
        signals = np.zeros((n_steps, NUM_FEATURES))

        # Randomize per-patient baselines
        baselines = {}
        for j, key in enumerate(FEATURE_ORDER):
            mean, std, _, _, _ = _SIGNAL_PARAMS[key]
            baselines[key] = rng.normal(mean, std)

        if label == 1:
            # Onset between 20% and 70% through the episode
            onset_step = rng.randint(int(n_steps * 0.2), int(n_steps * 0.7))
            severity = rng.uniform(0.5, 1.0)
        else:
            onset_step = None
            severity = 0.0

        for j, key in enumerate(FEATURE_ORDER):
            base = baselines[key]
            _, _, noise_std, delta, exp = _SIGNAL_PARAMS[key]
            lo, hi = _SIGNAL_BOUNDS[key]

            for t in range(n_steps):
                value = base + rng.normal(0, noise_std)

                if label == 1 and t >= onset_step:
                    # Gradual deterioration after onset
                    steps_since = t - onset_step
                    steps_remaining = n_steps - onset_step
                    progress = steps_since / max(steps_remaining, 1)
                    value += delta * severity * (progress ** exp)

                signals[t, j] = np.clip(value, lo, hi)

        episodes.append({
            "patient_id": f"SYN-{i:04d}",
            "time": time_points.copy(),
            "signals": signals,
            "label": label,
            "onset_step": onset_step,
        })

    return episodes


def generate_flat_dataset(num_patients=500, seed=42):
    """Generate flat synthetic records with 7 features for baseline models."""
    rng = random.Random(seed)
    records = []
    num_healthy = num_patients // 2
    num_septic = num_patients - num_healthy

    for _ in range(num_healthy):
        records.append({
            "heart_rate": round(rng.uniform(60, 100)),
            "respiratory_rate": round(rng.uniform(12, 20)),
            "temperature": round(rng.uniform(36.1, 37.2), 1),
            "spo2": round(rng.uniform(95, 100), 1),
            "ph": round(rng.uniform(7.35, 7.45), 3),
            "lactate": round(rng.uniform(0.5, 2.0), 2),
            "il6": round(rng.uniform(0, 7), 1),
            "label": 0,
        })

    for _ in range(num_septic):
        severity = rng.uniform(0.3, 1.0)
        records.append({
            "heart_rate": round(rng.uniform(100, 100 + 40 * severity)),
            "respiratory_rate": round(rng.uniform(20, 20 + 15 * severity)),
            "temperature": round(rng.uniform(37.5, 37.5 + 2.5 * severity), 1),
            "spo2": round(rng.uniform(98 - 12 * severity, 98), 1),
            "ph": round(rng.uniform(7.45 - 0.25 * severity, 7.40), 3),
            "lactate": round(rng.uniform(2.0, 2.0 + 6.0 * severity), 2),
            "il6": round(rng.uniform(7, 7 + 200 * severity), 1),
            "label": 1,
        })

    rng.shuffle(records)
    return pd.DataFrame(records)


def save_dataset(df, filepath="data/synthetic_7feat.csv"):
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  Dataset saved to {filepath} ({len(df)} records)")


def load_dataset(filepath="data/synthetic_7feat.csv"):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)
