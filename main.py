# main.py
# --------
# This is the entry point for SepSentinel Prototype 2 - Module 1.
#
# What it does:
#   1. Prints biomarker definitions so you understand what we're measuring.
#   2. Simulates a patient whose condition is worsening over 60 minutes.
#   3. Plots all three biomarker trends (Lactate, IL-6, pH).
#   4. Computes a Sepsis Risk Score from the latest readings.
#   5. Prints the risk score to the console.
#
# How to run:
#   In PyCharm, right-click this file and select "Run 'main'"
#   Or from the terminal: python main.py

from sepsentinel.biomarkers import print_biomarker_info
from sepsentinel.sensor_simulation import simulate_patient_data
from sepsentinel.visualization import plot_all_biomarkers
from sepsentinel.risk_model import calculate_sepsis_risk


def main():
    # --- Step 1: Print biomarker information ---
    print_biomarker_info()

    # --- Step 2: Simulate patient data over 60 minutes ---
    print("\nSimulating patient data over 60 minutes...\n")
    patient_data = simulate_patient_data(duration_minutes=60, interval_minutes=5)

    # Show a preview of the generated data
    print(f"  Time points: {patient_data['time']}")
    print(f"  Lactate:     {patient_data['lactate']}")
    print(f"  IL-6:        {patient_data['il6']}")
    print(f"  pH:          {patient_data['ph']}")

    # --- Step 3: Get the latest (most recent) biomarker values ---
    latest_lactate = patient_data["lactate"][-1]
    latest_il6 = patient_data["il6"][-1]
    latest_ph = patient_data["ph"][-1]

    print(f"\n  Latest readings (at t={patient_data['time'][-1]} min):")
    print(f"    Lactate: {latest_lactate} mmol/L")
    print(f"    IL-6:    {latest_il6} pg/mL")
    print(f"    pH:      {latest_ph}")

    # --- Step 4: Calculate the Sepsis Risk Score ---
    risk_score = calculate_sepsis_risk(latest_lactate, latest_il6, latest_ph)

    print("\n" + "=" * 60)
    print(f"  SEPSIS RISK SCORE: {risk_score}%")

    if risk_score < 30:
        print("  Status: LOW RISK")
    elif risk_score < 60:
        print("  Status: MODERATE RISK - Monitor closely")
    else:
        print("  Status: HIGH RISK - Immediate attention needed")

    print("=" * 60)

    # --- Step 5: Plot the biomarker trends ---
    print("\nGenerating biomarker trend plots...")
    plot_all_biomarkers(patient_data)


if __name__ == "__main__":
    main()
