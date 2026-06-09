# Matplotlib plotting for biomarker trends and risk gauge.

import matplotlib.pyplot as plt
from sepsentinel.biomarkers import BIOMARKERS


def plot_biomarker(time_points, values, biomarker_key):
    """Plot a single biomarker over time with normal range shading."""
    bio = BIOMARKERS[biomarker_key]
    normal_low, normal_high = bio["normal_range"]

    plt.figure(figsize=(8, 4))
    plt.plot(time_points, values, marker="o", color="#e74c3c", linewidth=2, label=bio["name"])
    plt.axhspan(normal_low, normal_high, color="#2ecc71", alpha=0.15, label="Normal Range")
    plt.xlabel("Time (minutes)")
    plt.ylabel(f"{bio['name']} ({bio['unit']})")
    plt.title(f"SepSentinel - {bio['name']} Over Time")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_all_biomarkers(patient_data):
    """Plot all three biomarkers as separate charts."""
    plot_biomarker(patient_data["time"], patient_data["lactate"], "lactate")
    plot_biomarker(patient_data["time"], patient_data["il6"], "il6")
    plot_biomarker(patient_data["time"], patient_data["ph"], "ph")
    plt.show()


def plot_risk_gauge(risk_score):
    """Display risk score as a color-coded horizontal bar."""
    fig, ax = plt.subplots(figsize=(8, 2))

    if risk_score < 30:
        color, label = "#2ecc71", "LOW RISK"
    elif risk_score < 60:
        color, label = "#f39c12", "MODERATE RISK"
    else:
        color, label = "#e74c3c", "HIGH RISK"

    ax.barh(0, 100, height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")
    ax.barh(0, risk_score, height=0.5, color=color, edgecolor="none")
    ax.text(50, 0, f"{risk_score}% - {label}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="black")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Sepsis Risk Score (%)")
    ax.set_title("SepSentinel - Risk Assessment")
    plt.tight_layout()
    plt.show()
