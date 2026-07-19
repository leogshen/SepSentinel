# Reusable Streamlit UI components for the dashboard.

import streamlit as st
import matplotlib.pyplot as plt

from sepsentinel.config.signals import ALL_SIGNALS
from sepsentinel.alerts import check_signal_alerts, check_risk_alert


def render_risk_score(risk_score):
    """Display the risk score with color coding."""
    if risk_score < 30:
        st.markdown(f"## :green[{risk_score}%]")
        st.success("LOW RISK")
    elif risk_score < 60:
        st.markdown(f"## :orange[{risk_score}%]")
        st.warning("MODERATE RISK - Monitor closely")
    else:
        st.markdown(f"## :red[{risk_score}%]")
        st.error("HIGH RISK - Immediate attention needed")


def render_risk_gauge(risk_score):
    """Horizontal bar gauge for risk score."""
    fig, ax = plt.subplots(figsize=(10, 1.2))
    color = "#2ecc71" if risk_score < 30 else "#f39c12" if risk_score < 60 else "#e74c3c"
    ax.barh(0, 100, height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")
    ax.barh(0, risk_score, height=0.5, color=color, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Sepsis Risk Score (%)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_alerts(values: dict, risk_score: float):
    """Display signal and risk alerts."""
    alerts = check_signal_alerts(values)
    risk_alert = check_risk_alert(risk_score)

    if not alerts and risk_alert is None:
        st.success("All signals within normal range.")
        return

    for a in alerts:
        (st.error if a["level"] == "CRITICAL" else st.warning)(
            f"{a['level']}: {a['message']}")
    if risk_alert:
        (st.error if risk_alert["level"] == "CRITICAL" else st.warning)(
            f"{risk_alert['level']}: {risk_alert['message']}")


def render_signal_card(signal_key, value):
    """Display a single signal value with normal/abnormal indicator."""
    sig = ALL_SIGNALS[signal_key]
    lo, hi = sig["normal_range"]
    in_range = lo <= value <= hi

    st.metric(sig["name"], f"{value} {sig['unit']}")
    if in_range:
        st.success("Normal")
    else:
        st.warning(f"Outside normal ({lo}-{hi})")
