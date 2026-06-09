# SepSentinel (Expanded Panel) — main entry point.
# 7 biomarkers: Presepsin, sTREM-1, IL-6, IL-10, CXCL10, Lactate, pH

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

    for key in ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]:
        print(f"  {key:>10}: {patient_data[key]}")

    latest = {k: patient_data[k][-1] for k in ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]}
    print(f"\n  Latest readings (t={patient_data['time'][-1]} min):")
    for k, v in latest.items():
        print(f"    {k}: {v}")

    risk_score = calculate_sepsis_risk(**latest)
    print_risk_result(risk_score)
    print()
    format_alerts_for_console(check_biomarker_alerts(**latest), check_risk_alert(risk_score))

    print("\nGenerating plots...")
    plot_all_biomarkers(patient_data)
    plot_risk_gauge(risk_score)


def run_manual_input():
    print("\n--- Manual Biomarker Input (7 markers) ---\n")
    prompts = [
        ("lactate", "Lactate (mmol/L, normal: 0.5-2.0)"),
        ("il6", "IL-6 (pg/mL, normal: 0-7)"),
        ("ph", "pH (normal: 7.35-7.45)"),
        ("presepsin", "Presepsin (pg/mL, normal: 60-365)"),
        ("strem1", "sTREM-1 (pg/mL, normal: 0-150)"),
        ("il10", "IL-10 (pg/mL, normal: 0-10)"),
        ("cxcl10", "CXCL10 (pg/mL, normal: 0-300)"),
    ]

    values = {}
    try:
        for key, prompt in prompts:
            values[key] = float(input(f"  {prompt}: "))
    except ValueError:
        print("\n  Error: Please enter valid numbers.")
        return

    risk_score = calculate_sepsis_risk(**values)
    print_risk_result(risk_score)
    print()
    format_alerts_for_console(check_biomarker_alerts(**values), check_risk_alert(risk_score))
    plot_risk_gauge(risk_score)


def run_training_synthetic():
    print("\n--- Training ML Model (Synthetic Data, 7 markers) ---\n")
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
    print("  (ML model)" if load_model() else "  (Rule-based fallback)")
    print("=" * 60)


def main():
    print_biomarker_info()

    print("\n  [1] Simulate a worsening patient")
    print("  [2] Enter biomarker values manually (7 markers)")
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
