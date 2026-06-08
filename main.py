# main.py
# --------
# SepSentinel Prototype 2 - Module 2
#
# This is the entry point. It gives you a menu with three options:
#   1. Simulate a worsening patient and see the results
#   2. Manually enter biomarker values to get a risk score
#   3. Train (or retrain) the ML model on synthetic data
#
# How to run:
#   In PyCharm, right-click this file and select "Run 'main'"
#   Or from the terminal: python main.py

from sepsentinel.biomarkers import print_biomarker_info
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.visualization import plot_all_biomarkers, plot_risk_gauge
from sepsentinel.risk_model import calculate_sepsis_risk, train_model, load_model
from sepsentinel.data_generator import generate_dataset, save_dataset


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

    # Show risk gauge
    plot_risk_gauge(risk_score)


def run_training():
    """Option 3: Generate synthetic data and train the ML model."""
    print("\n--- Training ML Model ---\n")

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
    print(f"\n  Training complete!")
    print(f"    - Training set: {metrics['train_size']} patients")
    print(f"    - Test set:     {metrics['test_size']} patients")
    print(f"    - Accuracy:     {metrics['accuracy'] * 100:.1f}%")
    print(f"\n  Classification Report:")
    print(metrics["report"])
    print("  Model saved. It will be used automatically for future risk scoring.")


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
    print("  [3] Train the ML model")
    print("  [4] Exit")

    choice = input("\n  Enter your choice (1-4): ").strip()

    if choice == "1":
        run_simulation()
    elif choice == "2":
        run_manual_input()
    elif choice == "3":
        run_training()
    elif choice == "4":
        print("\n  Goodbye!")
    else:
        print("\n  Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
