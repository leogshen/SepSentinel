# Synthetic data generation for pipeline development and testing.
#
# Two modes:
#   1. generate_flat_dataset()  — static 7-feature records (bridge for RF/XGBoost)
#   2. generate_episodes()      — time-series patient episodes (Module 6)
#
# Synthetic data is NOT intended to validate medical performance.
# It exists to develop the pipeline, test architectures, and debug dashboards.

import random
import os

import numpy as np
import pandas as pd

from sepsentinel.config.signals import FEATURE_ORDER


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


def generate_episodes(num_episodes=200, duration_minutes=240,
                      interval_minutes=5, seed=42):
    """Generate time-series patient episodes. (Module 6)"""
    raise NotImplementedError(
        "Time-series episode generation is planned for Module 6. "
        "Use generate_flat_dataset() for baseline model development."
    )
