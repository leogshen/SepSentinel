# main.py
# --------
# SepSentinel Prototype 2 - Module 4
#
# This is the entry point. It gives you a menu with options:
#   1. Simulate a worsening patient and see the results
#   2. Manually enter biomarker values to get a risk score
#   3. Train the ML model (synthetic data)
#   4. Train the ML model (real dataset from file)
#   5. Launch the Streamlit dashboard (opens in browser)
#   6. Exit
#
# How to run:
#   In PyCharm, right-click this file and select "Run 'main'"
#   Or from the terminal: python main.py

import os
import subprocess
import sys

from sepsentinel.biomarkers import print_biomarker_info
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.visualization import plot_all_biomarkers, plot_risk_gauge
from sepsentinel.risk_model import calculate_sepsis_risk, train_model, load_model
from sepsentinel.data_generator import generate_dataset, save_dataset
from sepsentinel.data_loader import load_real_dataset
from sepsentinel.alerts import (
    check_biomarker_alerts,
    check_risk_alert,
    format_alerts_for_console,
)


def run_simulation():
    """Option 1: Simulate a worsening patient over 60 minutes."""
    print("\n--- Simulating patient data over 60 minutes ---\n")
    patient_data = simulate_patient_data(duration_minutes=60, interval_minutes=5)

    # Show a preview of the generated data
    print(f"  Time points: {patient_data['time']}")
    print(f"  Lactate:     {patient_data['lactate']}")
    print(f"  IL-6:        {patient_data['il6']}")
    print(f"  pH:          {patient_data['ph']}")

    # Get the latest readings
    latest_lactate = patient_data["lactate"][-1]
    latest_il6 = patient_data["il6"][-1]
    latest_ph = patient_data["ph"][-1]

    print(f"\n  Latest readings (at t={patient_data['time'][-1]} min):")
    print(f"    Lactate: {latest_lactate} mmol/L")
    print(f"    IL-6:    {latest_il6} pg/mL")
    print(f"    pH:      {latest_ph}")

    # Calculate risk score
    risk_score = calculate_sepsis_risk(latest_lactate, latest_il6, latest_ph)
    print_risk_result(risk_score)

    # Check alerts
    alerts = check_biomarker_alerts(latest_lactate, latest_il6, latest_ph)
    risk_alert = check_risk_alert(risk_score)
    print()
    format_alerts_for_console(alerts, risk_alert)

    # Plot the biomarker trends
    print("\nGenerating biomarker trend plots...")
    plot_all_biomarkers(patient_data)

    # Show risk gauge
    plot_risk_gauge(risk_score)


def run_manual_input():
    """Option 2: Manually enter biomarker values."""
    print("\n--- Manual Biomarker Input ---")
    print("  Enter the three biomarker values below.\n")

    try:
        lactate = float(input("  Lactate (mmol/L, normal: 0.5-2.0):  "))
        il6 = float(input("  IL-6 (pg/mL, normal: 0-7):           "))
        ph = float(input("  pH (normal: 7.35-7.45):               "))
    except ValueError:
        print("\n  Error: Please enter valid numbers.")
        return

    print(f"\n  Your input:")
    print(f"    Lactate: {lactate} mmol/L")
    print(f"    IL-6:    {il6} pg/mL")
    print(f"    pH:      {ph}")

    # Calculate risk score
    risk_score = calculate_sepsis_risk(lactate, il6, ph)
    print_risk_result(risk_score)

    # Check alerts
    alerts = check_biomarker_alerts(lactate, il6, ph)
    risk_alert = check_risk_alert(risk_score)
    print()
    format_alerts_for_console(alerts, risk_alert)

    # Show risk gauge
    plot_risk_gauge(risk_score)


def run_training_synthetic():
    """Option 3: Generate synthetic data and train the ML model."""
    print("\n--- Training ML Model (Synthetic Data) ---\n")

    # Step 1: Generate synthetic dataset
    print("  Step 1: Generating synthetic patient dataset...")
    df = generate_dataset(num_patients=500)
    save_dataset(df)
    print(f"    - {len(df)} patient records created")
    print(f"    - Healthy: {len(df[df['label'] == 0])}, Septic: {len(df[df['label'] == 1])}")

    # Step 2: Train the model
    print("\n  Step 2: Training logistic regression model...")
    model, metrics = train_model(df)

    # Step 3: Show results
    _print_training_results(metrics, data_source="synthetic")


def run_training_real():
    """Option 4: Train the ML model from a real dataset file."""
    print("\n--- Training ML Model (Real Dataset) ---\n")

    # List available files in data/ folder
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir)
                 if f.endswith((".csv", ".xlsx", ".xls"))]
        if files:
            print("  Files found in data/ folder:")
            for i, f in enumerate(files, 1):
                print(f"    [{i}] {f}")
            print()

    filepath = input("  Enter path to dataset file (or filename in data/): ").strip()

    # If just a filename, look in data/ folder
    if not os.path.sep in filepath and not os.path.exists(filepath):
        filepath = os.path.join("data", filepath)

    if not os.path.exists(filepath):
        print(f"\n  Error: File not found: {filepath}")
        return

    # Load and auto-detect columns
    try:
        df, report = load_real_dataset(filepath)
    except ValueError as e:
        print(f"\n  Error: {e}")
        return

    if len(df) < 20:
        print(f"\n  Error: Dataset too small ({len(df)} rows). Need at least 20 rows.")
        return

    # Train the model using available biomarkers
    print(f"\n  Training on features: {report['biomarkers_available']}")
    model, metrics = train_model(df, feature_columns=report["biomarkers_available"])

    # Show results
    _print_training_results(metrics, data_source=os.path.basename(filepath))


def _print_training_results(metrics, data_source="unknown"):
    """Print training results in a consistent format."""
    print(f"\n  Training complete! (data source: {data_source})")
    print(f"    - Features used: {metrics['feature_columns']}")
    print(f"    - Training set:  {metrics['train_size']} patients")
    print(f"    - Test set:      {metrics['test_size']} patients")
    print(f"    - Accuracy:      {metrics['accuracy'] * 100:.1f}%")

    # Cross-validation results (more honest accuracy estimate)
    if "cv_accuracy_mean" in metrics:
        cv_mean = metrics["cv_accuracy_mean"] * 100
        cv_std = metrics["cv_accuracy_std"] * 100
        print(f"    - Cross-val:     {cv_mean:.1f}% (+/- {cv_std:.1f}%)")

    print(f"\n  Classification Report:")
    print(metrics["report"])
    print("  Model saved. It will be used automatically for future risk scoring.")


def run_dashboard():
    """Option 5: Launch the Streamlit dashboard in the browser."""
    print("\n--- Launching SepSentinel Dashboard ---")
    print("  Opening in your web browser...")
    print("  Press Ctrl+C in the terminal to stop the dashboard.\n")

    dashboard_path = "sepsentinel/dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def print_risk_result(risk_score):
    """Print the risk score with a status label."""
    print("\n" + "=" * 60)
    print(f"  SEPSIS RISK SCORE: {risk_score}%")

    if risk_score < 30:
        print("  Status: LOW RISK")
    elif risk_score < 60:
        print("  Status: MODERATE RISK - Monitor closely")
    else:
        print("  Status: HIGH RISK - Immediate attention needed")

    model = load_model()
    if model is not None:
        print("  (Scored using trained ML model)")
    else:
        print("  (Scored using rule-based fallback - train model for ML scoring)")

    print("=" * 60)


def main():
    print_biomarker_info()

    print("\n  What would you like to do?\n")
    print("  [1] Simulate a worsening patient (auto-generated data)")
    print("  [2] Enter biomarker values manually")
    print("  [3] Train the ML model (synthetic data)")
    print("  [4] Train the ML model (real dataset file)")
    print("  [5] Launch Dashboard (opens in browser)")
    print("  [6] Exit")

    choice = input("\n  Enter your choice (1-6): ").strip()

    if choice == "1":
        run_simulation()
    elif choice == "2":
        run_manual_input()
    elif choice == "3":
        run_training_synthetic()
    elif choice == "4":
        run_training_real()
    elif choice == "5":
        run_dashboard()
    elif choice == "6":
        print("\n  Goodbye!")
    else:
        print("\n  Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()
