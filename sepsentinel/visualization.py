# visualization.py
# ------------------
# This file handles all plotting and visualization using matplotlib.
#
# It creates clear, labeled charts showing how each biomarker changes over time.
# The normal range for each biomarker is shown as a green shaded region,
# making it easy to see when values become abnormal.
#
# Module 2 addition: a risk score gauge display.

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


def plot_risk_gauge(risk_score):
    """
    Display the risk score as a simple horizontal bar gauge.

    Color changes based on severity:
        Green  (0-30%):  Low risk
        Orange (30-60%): Moderate risk
        Red    (60-100%): High risk

    Args:
        risk_score: A value from 0 to 100.
    """
    fig, ax = plt.subplots(figsize=(8, 2))

    # Choose color based on risk level
    if risk_score < 30:
        color = "#2ecc71"   # Green
        label = "LOW RISK"
    elif risk_score < 60:
        color = "#f39c12"   # Orange
        label = "MODERATE RISK"
    else:
        color = "#e74c3c"   # Red
        label = "HIGH RISK"

    # Draw the background bar (gray) and the filled bar (colored)
    ax.barh(0, 100, height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")
    ax.barh(0, risk_score, height=0.5, color=color, edgecolor="none")

    # Add the score text in the center
    ax.text(50, 0, f"{risk_score}% - {label}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="black")

    # Clean up the axes
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Sepsis Risk Score (%)")
    ax.set_title("SepSentinel - Risk Assessment")
    plt.tight_layout()
    plt.show()
