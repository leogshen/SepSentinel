# dashboard.py
# --------------
# SepSentinel real-time monitoring dashboard built with Streamlit.
#
# This provides a browser-based interface where you can:
#   - View live biomarker trends from a simulated patient
#   - Manually enter biomarker values using sliders
#   - See the sepsis risk score update in real time
#   - Get visual alerts when values become dangerous
#
# How to run:
#   From the terminal: streamlit run sepsentinel/dashboard.py
#   Or from main.py: select option [4] Launch Dashboard
#
# The dashboard opens in your web browser automatically.

import sys
import os

# Add the project root to the path so imports work when running with streamlit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sepsentinel.biomarkers import BIOMARKERS
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.risk_model import calculate_sepsis_risk, load_model
from sepsentinel.alerts import check_biomarker_alerts, check_risk_alert

# --- Page configuration ---
st.set_page_config(
    page_title="SepSentinel Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.title("SepSentinel - Sepsis Early Detection Dashboard")
st.caption("Non-invasive wearable biosensor prototype")

# --- Sidebar: Mode selection ---
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Select input mode:",
    ["Manual Input", "Patient Simulation"],
    help="Manual: enter values with sliders. Simulation: auto-generated worsening patient."
)

# Check if ML model is available
model = load_model()
if model is not None:
    st.sidebar.success("ML model loaded")
else:
    st.sidebar.warning("No ML model found. Using rule-based scoring. Run option [3] in main.py to train.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Biomarker Reference**")
for key, bio in BIOMARKERS.items():
    st.sidebar.markdown(
        f"**{bio['name']}**: {bio['normal_range'][0]}-{bio['normal_range'][1]} {bio['unit']}"
    )


# =====================================================================
# MODE 1: Manual Input
# =====================================================================
if mode == "Manual Input":
    st.header("Manual Biomarker Input")
    st.write("Use the sliders to enter patient biomarker values.")

    # Three columns for the three biomarkers
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Lactate")
        lactate = st.slider(
            "Lactate (mmol/L)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
            help="Normal: 0.5-2.0 mmol/L"
        )
        normal_low, normal_high = BIOMARKERS["lactate"]["normal_range"]
        if lactate < normal_low or lactate > normal_high:
            st.warning(f"Outside normal range ({normal_low}-{normal_high})")
        else:
            st.success("Within normal range")

    with col2:
        st.subheader("IL-6")
        il6 = st.slider(
            "IL-6 (pg/mL)",
            min_value=0.0, max_value=250.0, value=3.0, step=1.0,
            help="Normal: 0-7 pg/mL"
        )
        normal_low, normal_high = BIOMARKERS["il6"]["normal_range"]
        if il6 > normal_high:
            st.warning(f"Outside normal range ({normal_low}-{normal_high})")
        else:
            st.success("Within normal range")

    with col3:
        st.subheader("pH")
        ph = st.slider(
            "pH (units)",
            min_value=6.80, max_value=7.60, value=7.40, step=0.01,
            help="Normal: 7.35-7.45"
        )
        normal_low, normal_high = BIOMARKERS["ph"]["normal_range"]
        if ph < normal_low or ph > normal_high:
            st.warning(f"Outside normal range ({normal_low}-{normal_high})")
        else:
            st.success("Within normal range")

    # --- Risk Score ---
    st.markdown("---")
    risk_score = calculate_sepsis_risk(lactate, il6, ph)

    # Display the risk score with color-coded metric
    score_col, alert_col = st.columns([1, 2])

    with score_col:
        st.subheader("Sepsis Risk Score")
        if risk_score < 30:
            st.markdown(f"## :green[{risk_score}%]")
            st.success("LOW RISK")
        elif risk_score < 60:
            st.markdown(f"## :orange[{risk_score}%]")
            st.warning("MODERATE RISK - Monitor closely")
        else:
            st.markdown(f"## :red[{risk_score}%]")
            st.error("HIGH RISK - Immediate attention needed")

    with alert_col:
        st.subheader("Alerts")
        alerts = check_biomarker_alerts(lactate, il6, ph)
        risk_alert = check_risk_alert(risk_score)

        if not alerts and risk_alert is None:
            st.success("All biomarkers within normal range. No alerts.")
        else:
            for alert in alerts:
                if alert["level"] == "CRITICAL":
                    st.error(f"CRITICAL: {alert['message']}")
                else:
                    st.warning(f"WARNING: {alert['message']}")
            if risk_alert:
                if risk_alert["level"] == "CRITICAL":
                    st.error(f"CRITICAL: {risk_alert['message']}")
                else:
                    st.warning(f"WARNING: {risk_alert['message']}")

    # --- Risk gauge bar ---
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(10, 1.2))
    if risk_score < 30:
        color = "#2ecc71"
    elif risk_score < 60:
        color = "#f39c12"
    else:
        color = "#e74c3c"
    ax.barh(0, 100, height=0.5, color="#ecf0f1", edgecolor="#bdc3c7")
    ax.barh(0, risk_score, height=0.5, color=color, edgecolor="none")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Sepsis Risk Score (%)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# MODE 2: Patient Simulation
# =====================================================================
elif mode == "Patient Simulation":
    st.header("Simulated Patient Monitoring")
    st.write("Simulating a patient whose condition worsens over 60 minutes.")

    # Simulation controls
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        duration = st.slider("Duration (minutes)", 30, 120, 60, step=10)
    with sim_col2:
        interval = st.slider("Reading interval (minutes)", 1, 15, 5)

    # Generate data (use a button to resimulate with new random noise)
    if st.button("Simulate New Patient", type="primary"):
        st.session_state["sim_data"] = simulate_patient_data(duration, interval)

    # Use existing data or generate fresh
    if "sim_data" not in st.session_state:
        st.session_state["sim_data"] = simulate_patient_data(duration, interval)

    patient_data = st.session_state["sim_data"]

    # --- Biomarker trend charts ---
    st.subheader("Biomarker Trends")

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    for col, key, color in [
        (chart_col1, "lactate", "#e74c3c"),
        (chart_col2, "il6", "#e67e22"),
        (chart_col3, "ph", "#3498db"),
    ]:
        bio = BIOMARKERS[key]
        normal_low, normal_high = bio["normal_range"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(patient_data["time"], patient_data[key], marker="o",
                color=color, linewidth=2, markersize=4)
        ax.axhspan(normal_low, normal_high, color="#2ecc71", alpha=0.15, label="Normal")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{bio['name']} ({bio['unit']})")
        ax.set_title(bio["name"])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        with col:
            st.pyplot(fig)
            plt.close(fig)

    # --- Latest values and risk ---
    st.markdown("---")
    latest_lactate = patient_data["lactate"][-1]
    latest_il6 = patient_data["il6"][-1]
    latest_ph = patient_data["ph"][-1]

    val_col1, val_col2, val_col3, risk_col = st.columns(4)
    val_col1.metric("Lactate", f"{latest_lactate} mmol/L")
    val_col2.metric("IL-6", f"{latest_il6} pg/mL")
    val_col3.metric("pH", f"{latest_ph}")

    risk_score = calculate_sepsis_risk(latest_lactate, latest_il6, latest_ph)
    if risk_score < 30:
        risk_col.metric("Risk Score", f"{risk_score}%")
        risk_col.success("LOW RISK")
    elif risk_score < 60:
        risk_col.metric("Risk Score", f"{risk_score}%")
        risk_col.warning("MODERATE RISK")
    else:
        risk_col.metric("Risk Score", f"{risk_score}%")
        risk_col.error("HIGH RISK")

    # --- Alerts ---
    st.subheader("Alerts")
    alerts = check_biomarker_alerts(latest_lactate, latest_il6, latest_ph)
    risk_alert = check_risk_alert(risk_score)

    if not alerts and risk_alert is None:
        st.success("All biomarkers within normal range at final reading.")
    else:
        for alert in alerts:
            if alert["level"] == "CRITICAL":
                st.error(f"CRITICAL: {alert['message']}")
            else:
                st.warning(f"WARNING: {alert['message']}")
        if risk_alert:
            if risk_alert["level"] == "CRITICAL":
                st.error(f"CRITICAL: {risk_alert['message']}")
            else:
                st.warning(f"WARNING: {risk_alert['message']}")

    # --- Data table ---
    with st.expander("View Raw Data"):
        df = pd.DataFrame({
            "Time (min)": patient_data["time"],
            "Lactate (mmol/L)": patient_data["lactate"],
            "IL-6 (pg/mL)": patient_data["il6"],
            "pH": patient_data["ph"],
        })
        st.dataframe(df, use_container_width=True)
