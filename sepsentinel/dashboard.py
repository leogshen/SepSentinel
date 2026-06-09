# Streamlit dashboard (7-marker panel). Run: streamlit run sepsentinel/dashboard.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sepsentinel.biomarkers import BIOMARKERS
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.risk_model import calculate_sepsis_risk, load_model
from sepsentinel.alerts import check_biomarker_alerts, check_risk_alert

st.set_page_config(page_title="SepSentinel Dashboard", page_icon="🩺", layout="wide")
st.title("SepSentinel - Expanded Biomarker Panel")
st.caption("7-marker sepsis detection prototype")

st.sidebar.header("Mode")
mode = st.sidebar.radio("Select input mode:", ["Manual Input", "Patient Simulation"])

model = load_model()
st.sidebar.success("ML model loaded") if model else st.sidebar.warning("No ML model. Rule-based scoring.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Biomarker Reference**")
for key, bio in BIOMARKERS.items():
    st.sidebar.markdown(f"**{bio['name']}**: {bio['normal_range'][0]}-{bio['normal_range'][1]} {bio['unit']}")


def show_alerts(values, risk_score):
    alerts = check_biomarker_alerts(**values)
    risk_alert = check_risk_alert(risk_score)
    if not alerts and risk_alert is None:
        st.success("All biomarkers within normal range.")
        return
    for a in alerts:
        (st.error if a["level"] == "CRITICAL" else st.warning)(f"{a['level']}: {a['message']}")
    if risk_alert:
        (st.error if risk_alert["level"] == "CRITICAL" else st.warning)(f"{risk_alert['level']}: {risk_alert['message']}")


def show_risk_score(risk_score):
    if risk_score < 30:
        st.markdown(f"## :green[{risk_score}%]")
        st.success("LOW RISK")
    elif risk_score < 60:
        st.markdown(f"## :orange[{risk_score}%]")
        st.warning("MODERATE RISK")
    else:
        st.markdown(f"## :red[{risk_score}%]")
        st.error("HIGH RISK")


def show_risk_gauge(risk_score):
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


# Slider configs: (key, label, min, max, default, step)
SLIDER_CONFIG = [
    ("lactate", "Lactate (mmol/L)", 0.0, 10.0, 1.0, 0.1),
    ("il6", "IL-6 (pg/mL)", 0.0, 250.0, 3.0, 1.0),
    ("ph", "pH", 6.80, 7.60, 7.40, 0.01),
    ("presepsin", "Presepsin (pg/mL)", 0.0, 1500.0, 200.0, 10.0),
    ("strem1", "sTREM-1 (pg/mL)", 0.0, 600.0, 80.0, 5.0),
    ("il10", "IL-10 (pg/mL)", 0.0, 150.0, 3.0, 1.0),
    ("cxcl10", "CXCL10 (pg/mL)", 0.0, 1200.0, 150.0, 10.0),
]

# === Manual Input ===
if mode == "Manual Input":
    st.header("Manual Biomarker Input")

    # Row 1: Lactate, IL-6, pH (core markers)
    cols_row1 = st.columns(3)
    # Row 2: Presepsin, sTREM-1, IL-10, CXCL10
    cols_row2 = st.columns(4)
    all_cols = list(cols_row1) + list(cols_row2)

    values = {}
    for col, (key, label, mn, mx, default, step) in zip(all_cols, SLIDER_CONFIG):
        with col:
            st.subheader(BIOMARKERS[key]["name"].split(" (")[0])
            values[key] = st.slider(label, mn, mx, default, step)
            lo, hi = BIOMARKERS[key]["normal_range"]
            if key == "ph":
                st.warning(f"Outside normal") if values[key] < lo or values[key] > hi else st.success("Normal")
            else:
                st.warning(f"Outside normal") if values[key] > hi or values[key] < lo else st.success("Normal")

    st.markdown("---")
    risk_score = calculate_sepsis_risk(**values)

    score_col, alert_col = st.columns([1, 2])
    with score_col:
        st.subheader("Sepsis Risk Score")
        show_risk_score(risk_score)
    with alert_col:
        st.subheader("Alerts")
        show_alerts(values, risk_score)

    st.markdown("---")
    show_risk_gauge(risk_score)

# === Patient Simulation ===
elif mode == "Patient Simulation":
    st.header("Simulated Patient Monitoring")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        duration = st.slider("Duration (minutes)", 30, 120, 60, step=10)
    with sim_col2:
        interval = st.slider("Reading interval (minutes)", 1, 15, 5)

    if st.button("Simulate New Patient", type="primary"):
        st.session_state["sim_data"] = simulate_patient_data(duration, interval)
    if "sim_data" not in st.session_state:
        st.session_state["sim_data"] = simulate_patient_data(duration, interval)

    patient_data = st.session_state["sim_data"]

    # Biomarker charts — 2 rows
    st.subheader("Biomarker Trends")
    biomarker_keys = ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6", "#1abc9c", "#f1c40f", "#2ecc71"]

    row1_cols = st.columns(4)
    row2_cols = st.columns(3)
    all_chart_cols = list(row1_cols) + list(row2_cols)

    for col, key, color in zip(all_chart_cols, biomarker_keys, colors):
        bio = BIOMARKERS[key]
        lo, hi = bio["normal_range"]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(patient_data["time"], patient_data[key], marker="o", color=color, linewidth=2, markersize=3)
        ax.axhspan(lo, hi, color="#2ecc71", alpha=0.15, label="Normal")
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_ylabel(f"{bio['unit']}", fontsize=8)
        ax.set_title(bio["name"].split(" (")[0], fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        with col:
            st.pyplot(fig)
            plt.close(fig)

    # Latest values
    st.markdown("---")
    latest = {k: patient_data[k][-1] for k in biomarker_keys}

    metric_cols = st.columns(len(biomarker_keys) + 1)
    for col, key in zip(metric_cols, biomarker_keys):
        bio = BIOMARKERS[key]
        col.metric(bio["name"].split(" (")[0], f"{latest[key]} {bio['unit']}")

    risk_score = calculate_sepsis_risk(**latest)
    metric_cols[-1].metric("Risk Score", f"{risk_score}%")
    if risk_score < 30:
        metric_cols[-1].success("LOW")
    elif risk_score < 60:
        metric_cols[-1].warning("MODERATE")
    else:
        metric_cols[-1].error("HIGH")

    st.subheader("Alerts")
    show_alerts(latest, risk_score)

    with st.expander("View Raw Data"):
        df_data = {"Time (min)": patient_data["time"]}
        for key in biomarker_keys:
            bio = BIOMARKERS[key]
            df_data[f"{bio['name'].split(' (')[0]} ({bio['unit']})"] = patient_data[key]
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)
