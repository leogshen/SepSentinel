# Streamlit Cloud entry point.
# This file lives at the repo root so imports work without path hacks.

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from sepsentinel.biomarkers import BIOMARKERS
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.risk_model import calculate_sepsis_risk, load_model
from sepsentinel.alerts import check_biomarker_alerts, check_risk_alert

st.set_page_config(page_title="SepSentinel Dashboard", page_icon="🩺", layout="wide")
st.title("SepSentinel - Sepsis Early Detection Dashboard")
st.caption("Non-invasive wearable biosensor prototype")

# Sidebar
st.sidebar.header("Mode")
mode = st.sidebar.radio("Select input mode:", ["Manual Input", "Patient Simulation"])

model = load_model()
st.sidebar.success("ML model loaded") if model else st.sidebar.warning("No ML model. Rule-based scoring.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Biomarker Reference**")
for key, bio in BIOMARKERS.items():
    st.sidebar.markdown(f"**{bio['name']}**: {bio['normal_range'][0]}-{bio['normal_range'][1]} {bio['unit']}")


def show_alerts(lactate, il6, ph, risk_score):
    alerts = check_biomarker_alerts(lactate, il6, ph)
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
        st.warning("MODERATE RISK - Monitor closely")
    else:
        st.markdown(f"## :red[{risk_score}%]")
        st.error("HIGH RISK - Immediate attention needed")


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


# === Manual Input ===
if mode == "Manual Input":
    st.header("Manual Biomarker Input")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Lactate")
        lactate = st.slider("Lactate (mmol/L)", 0.0, 10.0, 1.0, 0.1)
        lo, hi = BIOMARKERS["lactate"]["normal_range"]
        st.warning(f"Outside normal ({lo}-{hi})") if lactate < lo or lactate > hi else st.success("Normal")

    with col2:
        st.subheader("IL-6")
        il6 = st.slider("IL-6 (pg/mL)", 0.0, 250.0, 3.0, 1.0)
        lo, hi = BIOMARKERS["il6"]["normal_range"]
        st.warning(f"Outside normal ({lo}-{hi})") if il6 > hi else st.success("Normal")

    with col3:
        st.subheader("pH")
        ph = st.slider("pH (units)", 6.80, 7.60, 7.40, 0.01)
        lo, hi = BIOMARKERS["ph"]["normal_range"]
        st.warning(f"Outside normal ({lo}-{hi})") if ph < lo or ph > hi else st.success("Normal")

    st.markdown("---")
    risk_score = calculate_sepsis_risk(lactate, il6, ph)

    score_col, alert_col = st.columns([1, 2])
    with score_col:
        st.subheader("Sepsis Risk Score")
        show_risk_score(risk_score)
    with alert_col:
        st.subheader("Alerts")
        show_alerts(lactate, il6, ph, risk_score)

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

    st.subheader("Biomarker Trends")
    chart_cols = st.columns(3)
    for col, key, color in zip(chart_cols, ["lactate", "il6", "ph"], ["#e74c3c", "#e67e22", "#3498db"]):
        bio = BIOMARKERS[key]
        lo, hi = bio["normal_range"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(patient_data["time"], patient_data[key], marker="o", color=color, linewidth=2, markersize=4)
        ax.axhspan(lo, hi, color="#2ecc71", alpha=0.15, label="Normal")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{bio['name']} ({bio['unit']})")
        ax.set_title(bio["name"])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        with col:
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")
    latest_lactate = patient_data["lactate"][-1]
    latest_il6 = patient_data["il6"][-1]
    latest_ph = patient_data["ph"][-1]

    val_col1, val_col2, val_col3, risk_col = st.columns(4)
    val_col1.metric("Lactate", f"{latest_lactate} mmol/L")
    val_col2.metric("IL-6", f"{latest_il6} pg/mL")
    val_col3.metric("pH", f"{latest_ph}")

    risk_score = calculate_sepsis_risk(latest_lactate, latest_il6, latest_ph)
    risk_col.metric("Risk Score", f"{risk_score}%")
    if risk_score < 30:
        risk_col.success("LOW RISK")
    elif risk_score < 60:
        risk_col.warning("MODERATE RISK")
    else:
        risk_col.error("HIGH RISK")

    st.subheader("Alerts")
    show_alerts(latest_lactate, latest_il6, latest_ph, risk_score)

    with st.expander("View Raw Data"):
        df = pd.DataFrame({
            "Time (min)": patient_data["time"],
            "Lactate (mmol/L)": patient_data["lactate"],
            "IL-6 (pg/mL)": patient_data["il6"],
            "pH": patient_data["ph"],
        })
        st.dataframe(df, use_container_width=True)
