# Matplotlib plotting for signal trends and risk gauge.

import matplotlib.pyplot as plt

from sepsentinel.config.signals import ALL_SIGNALS, PHYSIOLOGICAL_FEATURES, BIOMARKER_FEATURES

SIGNAL_COLORS = {
    "heart_rate": "#e74c3c",
    "respiratory_rate": "#e67e22",
    "temperature": "#f1c40f",
    "spo2": "#3498db",
    "ph": "#2ecc71",
    "lactate": "#9b59b6",
    "il6": "#1abc9c",
}


def plot_signal(time_points, values, signal_key, ax=None):
    """Plot a single signal over time with normal range shading."""
    sig = ALL_SIGNALS[signal_key]
    lo, hi = sig["normal_range"]
    color = SIGNAL_COLORS.get(signal_key, "#333333")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_points, values, marker="o", color=color, linewidth=2,
            markersize=3, label=sig["name"])
    ax.axhspan(lo, hi, color="#2ecc71", alpha=0.12, label="Normal")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{sig['name']} ({sig['unit']})")
    ax.set_title(sig["name"])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_all_signals(patient_data, show=True):
    """Plot all 7 signals in a 2-row grid."""
    keys = PHYSIOLOGICAL_FEATURES + BIOMARKER_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, key in enumerate(keys):
        plot_signal(patient_data["time"], patient_data[key], key, ax=axes[i])

    # Hide the 8th subplot (2x4 grid, 7 signals)
    axes[7].set_visible(False)

    fig.suptitle("SepSentinel - Patient Signals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_risk_gauge(risk_score, ax=None, show=True):
    """Display risk score as a color-coded horizontal bar."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 2))
    else:
        fig = ax.figure

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
    if show:
        plt.show()
    return fig
