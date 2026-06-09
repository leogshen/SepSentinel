# SepSentinel — main entry point.
# Run: python main.py

import os
import subprocess
import sys

from sepsentinel.biomarkers import print_biomarker_info
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.visualization import plot_all_biomarkers, plot_risk_gauge
from sepsentinel.risk_model import calculate_sepsis_risk, train_model, load_model
from sepsentinel.data_generator import generate_dataset, save_dataset
from sepsentinel.data_loader import load_real_dataset
from sepsentinel.alerts import check_biomarker_alerts, check_risk_alert, format_alerts_for_console


def run_simulation():
    print("\n--- Simulating patient data over 60 minutes ---\n")
    patient_data = simulate_patient_data(duration_minutes=60, interval_minutes=5)

    print(f"  Time points: {patient_data['time']}")
    print(f"  Lactate:     {patient_data['lactate']}")
    print(f"  IL-6:        {patient_data['il6']}")
    print(f"  pH:          {patient_data['ph']}")

    latest_lactate = patient_data["lactate"][-1]
    latest_il6 = patient_data["il6"][-1]
    latest_ph = patient_data["ph"][-1]

    print(f"\n  Latest readings (t={patient_data['time'][-1]} min):")
    print(f"    Lactate: {latest_lactate} mmol/L")
    print(f"    IL-6:    {latest_il6} pg/mL")
    print(f"    pH:      {latest_ph}")

    risk_score = calculate_sepsis_risk(latest_lactate, latest_il6, latest_ph)
    print_risk_result(risk_score)
    print()
    format_alerts_for_console(
        check_biomarker_alerts(latest_lactate, latest_il6, latest_ph),
        check_risk_alert(risk_score))

    print("\nGenerating plots...")
    plot_all_biomarkers(patient_data)
    plot_risk_gauge(risk_score)


def run_manual_input():
    print("\n--- Manual Biomarker Input ---\n")
    try:
        lactate = float(input("  Lactate (mmol/L, normal: 0.5-2.0):  "))
        il6 = float(input("  IL-6 (pg/mL, normal: 0-7):           "))
        ph = float(input("  pH (normal: 7.35-7.45):               "))
    except ValueError:
        print("\n  Error: Please enter valid numbers.")
        return

    risk_score = calculate_sepsis_risk(lactate, il6, ph)
    print_risk_result(risk_score)
    print()
    format_alerts_for_console(
        check_biomarker_alerts(lactate, il6, ph),
        check_risk_alert(risk_score))
    plot_risk_gauge(risk_score)


def run_training_synthetic():
    print("\n--- Training ML Model (Synthetic Data) ---\n")
    df = generate_dataset(num_patients=500)
    save_dataset(df)
    print(f"    {len(df)} records ({len(df[df['label'] == 0])} healthy, {len(df[df['label'] == 1])} septic)")

    model, metrics = train_model(df)
    _print_training_results(metrics, "synthetic")


def run_training_real():
    print("\n--- Training ML Model (Real Dataset) ---\n")

    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith((".csv", ".xlsx", ".xls"))]
        if files:
            print("  Files in data/:")
            for i, f in enumerate(files, 1):
                print(f"    [{i}] {f}")
            print()

    filepath = input("  Enter path to dataset (or filename in data/): ").strip()
    if os.sep not in filepath and not os.path.exists(filepath):
        filepath = os.path.join("data", filepath)

    if not os.path.exists(filepath):
        print(f"\n  Error: File not found: {filepath}")
        return

    try:
        df, report = load_real_dataset(filepath)
    except ValueError as e:
        print(f"\n  Error: {e}")
        return

    if len(df) < 20:
        print(f"\n  Error: Dataset too small ({len(df)} rows).")
        return

    print(f"\n  Training on: {report['biomarkers_available']}")
    model, metrics = train_model(df, feature_columns=report["biomarkers_available"])
    _print_training_results(metrics, os.path.basename(filepath))


def _print_training_results(metrics, data_source):
    print(f"\n  Done! (source: {data_source})")
    print(f"    Features:   {metrics['feature_columns']}")
    print(f"    Train/Test: {metrics['train_size']} / {metrics['test_size']}")
    print(f"    Accuracy:   {metrics['accuracy'] * 100:.1f}%")
    if "cv_accuracy_mean" in metrics:
        print(f"    Cross-val:  {metrics['cv_accuracy_mean'] * 100:.1f}% (+/- {metrics['cv_accuracy_std'] * 100:.1f}%)")
    print(f"\n{metrics['report']}")
    print("  Model saved.")


def run_dashboard():
    print("\n  Launching dashboard... (Ctrl+C to stop)\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "sepsentinel/dashboard.py"])


def print_risk_result(risk_score):
    print("\n" + "=" * 60)
    print(f"  SEPSIS RISK SCORE: {risk_score}%")
    if risk_score < 30:
        print("  Status: LOW RISK")
    elif risk_score < 60:
        print("  Status: MODERATE RISK - Monitor closely")
    else:
        print("  Status: HIGH RISK - Immediate attention needed")
    model = load_model()
    print("  (ML model)" if model else "  (Rule-based fallback)")
    print("=" * 60)


def main():
    print_biomarker_info()

    print("\n  [1] Simulate a worsening patient")
    print("  [2] Enter biomarker values manually")
    print("  [3] Train ML model (synthetic data)")
    print("  [4] Train ML model (real dataset)")
    print("  [5] Launch Dashboard")
    print("  [6] Exit")

    choice = input("\n  Choice (1-6): ").strip()

    actions = {"1": run_simulation, "2": run_manual_input, "3": run_training_synthetic,
               "4": run_training_real, "5": run_dashboard, "6": lambda: print("\n  Goodbye!")}

    actions.get(choice, lambda: print("\n  Invalid choice."))()


if __name__ == "__main__":
    main()
