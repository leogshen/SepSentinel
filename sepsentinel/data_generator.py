# Generates synthetic patient datasets for model training.

import random
import os
import pandas as pd


def generate_dataset(num_patients=500, seed=42):
    """Generate synthetic biomarker records (50% healthy, 50% septic)."""
    random.seed(seed)
    records = []
    num_healthy = num_patients // 2
    num_septic = num_patients - num_healthy

    for _ in range(num_healthy):
        records.append({
            "lactate": round(random.uniform(0.5, 2.5), 2),
            "il6": round(random.uniform(0, 15), 1),
            "ph": round(random.uniform(7.32, 7.48), 3),
            "label": 0,
        })

    for _ in range(num_septic):
        severity = random.uniform(0.3, 1.0)
        records.append({
            "lactate": round(random.uniform(2.0, 2.0 + 6.0 * severity), 2),
            "il6": round(random.uniform(7, 7 + 200 * severity), 1),
            "ph": round(random.uniform(7.45 - 0.30 * severity, 7.40), 3),
            "label": 1,
        })

    random.shuffle(records)
    return pd.DataFrame(records)


def save_dataset(df, filepath="data/synthetic_patients.csv"):
    """Save dataset to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  Dataset saved to {filepath} ({len(df)} records)")


def load_dataset(filepath="data/synthetic_patients.csv"):
    """Load dataset from CSV."""
    return pd.read_csv(filepath)
