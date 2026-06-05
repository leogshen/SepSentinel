# visualization.py
# ------------------
# This file handles all plotting and visualization using matplotlib.
#
# It creates clear, labeled charts showing how each biomarker changes over time.
# The normal range for each biomarker is shown as a green shaded region,
# making it easy to see when values become abnormal.
#
# In future modules, this could be expanded into a real-time dashboard.

import matplotlib.pyplot as plt

from sepsentinel.biomarkers import BIOMARKERS


def plot_biomarker(time_points, values, biomarker_key):
    """
    Plot a single biomarker's values over time.

    Args:
        time_points: List of time values (minutes).
        values: List of biomarker measurements.
        biomarker_key: Key name in the BIOMARKERS dictionary (e.g., "lactate").
    """
    bio = BIOMARKERS[biomarker_key]
    normal_low, normal_high = bio["normal_range"]

    plt.figure(figsize=(8, 4))

    # Plot the biomarker values as a line with dots at each reading
    plt.plot(time_points, values, marker="o", color="#e74c3c", linewidth=2, label=bio["name"])

    # Shade the normal range in green so you can see when values go abnormal
    plt.axhspan(normal_low, normal_high, color="#2ecc71", alpha=0.15, label="Normal Range")

    # Labels and title
    plt.xlabel("Time (minutes)")
    plt.ylabel(f"{bio['name']} ({bio['unit']})")
    plt.title(f"SepSentinel - {bio['name']} Over Time")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_all_biomarkers(patient_data):
    """
    Plot all three biomarkers as separate charts.

    Args:
        patient_data: Dictionary returned by simulate_patient_data().
    """
    plot_biomarker(patient_data["time"], patient_data["lactate"], "lactate")
    plot_biomarker(patient_data["time"], patient_data["il6"], "il6")
    plot_biomarker(patient_data["time"], patient_data["ph"], "ph")

    # Show all plots at once
    plt.show()
