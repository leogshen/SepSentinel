# SepSentinel v2.0 — main entry point.
# Run: python main.py

import os
import subprocess
import sys

import numpy as np

from sepsentinel.config.signals import print_signal_info, FEATURE_ORDER
from sepsentinel.simulation import simulate_patient
from sepsentinel.visualization import plot_all_signals, plot_risk_gauge
from sepsentinel.models.base import rule_based_risk
from sepsentinel.alerts import check_signal_alerts, check_risk_alert, format_alerts_for_console
from sepsentinel.data.synthetic import generate_flat_dataset, save_dataset


def run_simulation():
    print("\n--- Simulating patient data (7 signals) ---\n")
    patient_data = simulate_patient(duration_minutes=60, interval_minutes=5)

    for key in FEATURE_ORDER:
        print(f"  {key:20s}: {patient_data[key]}")

    latest = {key: patient_data[key][-1] for key in FEATURE_ORDER}
    t = patient_data["time"][-1]

    print(f"\n  Latest readings (t={t} min):")
    for key, val in latest.items():
        print(f"    {key:20s}: {val}")

    risk = rule_based_risk(latest)
    print_risk_result(risk)
    print()
    format_alerts_for_console(check_signal_alerts(latest), check_risk_alert(risk))

    print("\nGenerating plots...")
    plot_all_signals(patient_data)
    plot_risk_gauge(risk)


def run_manual_input():
    print("\n--- Manual Signal Input (7 features) ---\n")
    values = {}
    prompts = {
        "heart_rate":        "  Heart Rate (bpm, normal: 60-100):        ",
        "respiratory_rate":  "  Respiratory Rate (br/min, normal: 12-20): ",
        "temperature":       "  Temperature (\u00b0C, normal: 36.1-37.2):     ",
        "spo2":              "  SpO2 (%, normal: 95-100):                ",
        "ph":                "  pH (normal: 7.35-7.45):                  ",
        "lactate":           "  Lactate (mmol/L, normal: 0.5-2.0):       ",
        "il6":               "  IL-6 (pg/mL, normal: 0-7):               ",
    }
    try:
        for key in FEATURE_ORDER:
            values[key] = float(input(prompts[key]))
    except ValueError:
        print("\n  Error: Please enter valid numbers.")
        return

    risk = rule_based_risk(values)
    print_risk_result(risk)
    print()
    format_alerts_for_console(check_signal_alerts(values), check_risk_alert(risk))
    plot_risk_gauge(risk)


def run_training():
    print("\n--- Training Random Forest (Synthetic 7-Feature Data) ---\n")
    from sepsentinel.models import get_model

    df = generate_flat_dataset(num_patients=500)
    save_dataset(df)

    healthy = len(df[df["label"] == 0])
    septic = len(df[df["label"] == 1])
    print(f"    {len(df)} records ({healthy} healthy, {septic} septic)")

    X = df[FEATURE_ORDER].values
    y = df["label"].values

    model = get_model("random_forest")
    metrics = model.fit(X, y)

    print(f"\n  Done!")
    print(f"    Train/Test: {metrics['train_size']} / {metrics['test_size']}")
    print(f"    Accuracy:   {metrics['accuracy'] * 100:.1f}%")
    print(f"    Cross-val:  {metrics['cv_accuracy_mean'] * 100:.1f}% "
          f"(+/- {metrics['cv_accuracy_std'] * 100:.1f}%)")
    print(f"\n{metrics['report']}")

    model.save("models")
    print("  Model saved to models/")


def run_dashboard():
    print("\n  Launching dashboard... (Ctrl+C to stop)\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "sepsentinel/dashboard/app.py"])


def print_risk_result(risk_score):
    print("\n" + "=" * 60)
    print(f"  SEPSIS RISK SCORE: {risk_score}%")
    if risk_score < 30:
        print("  Status: LOW RISK")
    elif risk_score < 60:
        print("  Status: MODERATE RISK - Monitor closely")
    else:
        print("  Status: HIGH RISK - Immediate attention needed")
    print("  (Rule-based scoring)")
    print("=" * 60)


def main():
    print_signal_info()

    print("\n  [1] Simulate a worsening patient")
    print("  [2] Enter signal values manually")
    print("  [3] Train Random Forest (synthetic data)")
    print("  [4] Launch Dashboard")
    print("  [5] Exit")

    choice = input("\n  Choice (1-5): ").strip()

    actions = {
        "1": run_simulation,
        "2": run_manual_input,
        "3": run_training,
        "4": run_dashboard,
        "5": lambda: print("\n  Goodbye!"),
    }

    actions.get(choice, lambda: print("\n  Invalid choice."))()


if __name__ == "__main__":
    main()
