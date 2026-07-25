# SepSentinel Streamlit Dashboard — 7-signal version.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sepsentinel.config.signals import (
    ALL_SIGNALS, PHYSIOLOGICAL_FEATURES, BIOMARKER_FEATURES, FEATURE_ORDER,
)
from sepsentinel.simulation import simulate_patient
from sepsentinel.model_b.base import rule_based_risk
from sepsentinel.dashboard.components import (
    render_risk_score, render_risk_gauge, render_alerts, render_signal_card,
)
from sepsentinel.visualization import SIGNAL_COLORS

st.set_page_config(page_title="SepSentinel Dashboard", page_icon="\U0001fa7a", layout="wide")
st.title("SepSentinel - Sepsis Early Detection Dashboard")
st.caption("Wearable multimodal biosensor prototype")

# --- Sidebar ---
st.sidebar.header("Mode")
mode = st.sidebar.radio("Select input mode:", ["Manual Input", "Patient Simulation"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Physiological Signals**")
for key in PHYSIOLOGICAL_FEATURES:
    sig = ALL_SIGNALS[key]
    lo, hi = sig["normal_range"]
    st.sidebar.markdown(f"{sig['name']}: {lo}-{hi} {sig['unit']}")

st.sidebar.markdown("**Electrochemical Biomarkers**")
for key in BIOMARKER_FEATURES:
    sig = ALL_SIGNALS[key]
    lo, hi = sig["normal_range"]
    st.sidebar.markdown(f"{sig['name']}: {lo}-{hi} {sig['unit']}")


# === Manual Input ===
if mode == "Manual Input":
    st.header("Manual Signal Input")

    st.subheader("Physiological Signals")
    p_cols = st.columns(4)
    values = {}

    with p_cols[0]:
        values["heart_rate"] = st.slider("Heart Rate (bpm)", 30, 180, 75)
        render_signal_card("heart_rate", values["heart_rate"])
    with p_cols[1]:
        values["respiratory_rate"] = st.slider("Respiratory Rate (br/min)", 4, 45, 15)
        render_signal_card("respiratory_rate", values["respiratory_rate"])
    with p_cols[2]:
        values["temperature"] = st.slider("Temperature (\u00b0C)", 34.0, 42.0, 36.8, 0.1)
        render_signal_card("temperature", values["temperature"])
    with p_cols[3]:
        values["spo2"] = st.slider("SpO2 (%)", 70.0, 100.0, 97.0, 0.5)
        render_signal_card("spo2", values["spo2"])

    st.subheader("Electrochemical Biomarkers")
    b_cols = st.columns(3)

    with b_cols[0]:
        values["lactate"] = st.slider("Lactate (mmol/L)", 0.0, 10.0, 1.0, 0.1)
        render_signal_card("lactate", values["lactate"])
    with b_cols[1]:
        values["il6"] = st.slider("IL-6 (pg/mL)", 0.0, 250.0, 3.0, 1.0)
        render_signal_card("il6", values["il6"])
    with b_cols[2]:
        values["ph"] = st.slider("pH", 6.80, 7.60, 7.40, 0.01)
        render_signal_card("ph", values["ph"])

    st.markdown("---")
    risk_score = rule_based_risk(values)

    score_col, alert_col = st.columns([1, 2])
    with score_col:
        st.subheader("Sepsis Risk Score")
        render_risk_score(risk_score)
    with alert_col:
        st.subheader("Alerts")
        render_alerts(values, risk_score)

    st.markdown("---")
    render_risk_gauge(risk_score)


# === Patient Simulation ===
elif mode == "Patient Simulation":
    st.header("Simulated Patient Monitoring")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        duration = st.slider("Duration (minutes)", 30, 240, 60, step=10)
    with sim_col2:
        interval = st.slider("Reading interval (minutes)", 1, 15, 5)
    with sim_col3:
        scenario = st.selectbox("Scenario", ["worsening", "stable"])

    if st.button("Simulate New Patient", type="primary"):
        st.session_state["sim_data"] = simulate_patient(duration, interval, scenario)
    if "sim_data" not in st.session_state:
        st.session_state["sim_data"] = simulate_patient(duration, interval, scenario)

    patient_data = st.session_state["sim_data"]

    # --- Physiological signal plots ---
    st.subheader("Physiological Signals")
    p_chart_cols = st.columns(4)
    for col, key in zip(p_chart_cols, PHYSIOLOGICAL_FEATURES):
        sig = ALL_SIGNALS[key]
        lo, hi = sig["normal_range"]
        color = SIGNAL_COLORS.get(key, "#333")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(patient_data["time"], patient_data[key], marker="o",
                color=color, linewidth=2, markersize=3)
        ax.axhspan(lo, hi, color="#2ecc71", alpha=0.12, label="Normal")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{sig['name']} ({sig['unit']})")
        ax.set_title(sig["name"])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        with col:
            st.pyplot(fig)
            plt.close(fig)

    # --- Biomarker plots ---
    st.subheader("Electrochemical Biomarkers")
    b_chart_cols = st.columns(3)
    for col, key in zip(b_chart_cols, BIOMARKER_FEATURES):
        sig = ALL_SIGNALS[key]
        lo, hi = sig["normal_range"]
        color = SIGNAL_COLORS.get(key, "#333")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(patient_data["time"], patient_data[key], marker="o",
                color=color, linewidth=2, markersize=3)
        ax.axhspan(lo, hi, color="#2ecc71", alpha=0.12, label="Normal")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{sig['name']} ({sig['unit']})")
        ax.set_title(sig["name"])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        with col:
            st.pyplot(fig)
            plt.close(fig)

    # --- Latest readings and risk ---
    st.markdown("---")
    latest = {key: patient_data[key][-1] for key in FEATURE_ORDER}

    metric_cols = st.columns(len(FEATURE_ORDER) + 1)
    for col, key in zip(metric_cols, FEATURE_ORDER):
        sig = ALL_SIGNALS[key]
        col.metric(sig["name"], f"{latest[key]} {sig['unit']}")

    risk_score = rule_based_risk(latest)
    metric_cols[-1].metric("Risk Score", f"{risk_score}%")
    if risk_score < 30:
        metric_cols[-1].success("LOW")
    elif risk_score < 60:
        metric_cols[-1].warning("MODERATE")
    else:
        metric_cols[-1].error("HIGH")

    st.subheader("Alerts")
    render_alerts(latest, risk_score)

    with st.expander("View Raw Data"):
        df = pd.DataFrame({"Time (min)": patient_data["time"]})
        for key in FEATURE_ORDER:
            sig = ALL_SIGNALS[key]
            df[f"{sig['name']} ({sig['unit']})"] = patient_data[key]
        st.dataframe(df, use_container_width=True)
