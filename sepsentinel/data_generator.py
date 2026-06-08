# data_generator.py
# -------------------
# This file generates synthetic patient datasets for training the ML model.
#
# It creates two types of patients:
#   - Healthy: biomarkers stay within or near normal ranges
#   - Septic: biomarkers deviate into dangerous territory
#
# Each patient record is a single snapshot of (lactate, il6, ph, label)
# where label = 0 (healthy) or 1 (septic).
#
# In the future, this can be replaced with real clinical datasets.

import random
import pandas as pd


def generate_dataset(num_patients=500, seed=42):
    """
    Generate a synthetic dataset of patient biomarker readings.

    Args:
        num_patients: Total number of patient records to create.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame with columns: lactate, il6, ph, label
        where label is 0 (healthy) or 1 (septic).
    """
    random.seed(seed)

    records = []
    num_healthy = num_patients // 2
    num_septic = num_patients - num_healthy

    # --- Generate HEALTHY patients ---
    # Biomarkers stay within or near normal ranges
    for _ in range(num_healthy):
        lactate = random.uniform(0.5, 2.5)     # Normal: 0.5-2.0, slight overlap
        il6 = random.uniform(0, 15)             # Normal: 0-7, slight overlap
        ph = random.uniform(7.32, 7.48)         # Normal: 7.35-7.45, slight overlap
        records.append({
            "lactate": round(lactate, 2),
            "il6": round(il6, 1),
            "ph": round(ph, 3),
            "label": 0,
        })

    # --- Generate SEPTIC patients ---
    # Biomarkers are elevated/abnormal with varying severity
    for _ in range(num_septic):
        # Severity ranges from mild to severe sepsis
        severity = random.uniform(0.3, 1.0)

        lactate = random.uniform(2.0, 2.0 + 6.0 * severity)   # Up to ~8 mmol/L
        il6 = random.uniform(7, 7 + 200 * severity)           # Up to ~207 pg/mL
        ph = random.uniform(7.45 - 0.30 * severity, 7.40)     # Down to ~7.15
        records.append({
            "lactate": round(lactate, 2),
            "il6": round(il6, 1),
            "ph": round(ph, 3),
            "label": 1,
        })

    # Shuffle so healthy and septic are mixed together
    random.shuffle(records)

    return pd.DataFrame(records)


def save_dataset(df, filepath="data/synthetic_patients.csv"):
    """Save the dataset to a CSV file."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"  Dataset saved to {filepath} ({len(df)} records)")


def load_dataset(filepath="data/synthetic_patients.csv"):
    """Load a dataset from a CSV file."""
    return pd.read_csv(filepath)
