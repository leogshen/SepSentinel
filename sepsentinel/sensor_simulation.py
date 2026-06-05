# sensor_simulation.py
# ---------------------
# This file generates simulated biomarker data for a patient over 60 minutes.
#
# The simulation models a patient who is gradually worsening:
#   - Lactate rises (tissue is not getting enough oxygen)
#   - IL-6 spikes (immune system is responding to infection)
#   - pH drops (blood is becoming more acidic)
#
# In a real device, this data would come from biosensors on the wearable patch.
# For now, we use math to create realistic-looking trajectories.

import random


def simulate_patient_data(duration_minutes=60, interval_minutes=5):
    """
    Simulate biomarker readings for a patient over a period of time.

    Args:
        duration_minutes: How long to simulate (default: 60 minutes).
        interval_minutes: Time between each reading (default: every 5 minutes).

    Returns:
        A dictionary with:
            - "time": list of time points in minutes
            - "lactate": list of lactate values (mmol/L)
            - "il6": list of IL-6 values (pg/mL)
            - "ph": list of pH values (pH units)
    """
    # Create the list of time points: [0, 5, 10, 15, ..., 60]
    time_points = list(range(0, duration_minutes + 1, interval_minutes))
    num_readings = len(time_points)

    # Starting values (within normal range)
    lactate_start = 1.0   # Normal: 0.5-2.0 mmol/L
    il6_start = 5.0       # Normal: 0-7 pg/mL
    ph_start = 7.40       # Normal: 7.35-7.45

    # Generate worsening trajectories with some random noise
    lactate_values = []
    il6_values = []
    ph_values = []

    for i in range(num_readings):
        # progress goes from 0.0 (start) to 1.0 (end of simulation)
        progress = i / (num_readings - 1)

        # Lactate: rises from ~1.0 to ~5.0 mmol/L (above 4 is dangerous)
        noise = random.uniform(-0.2, 0.2)
        lactate = lactate_start + (4.0 * progress) + noise
        lactate_values.append(round(max(lactate, 0.5), 2))

        # IL-6: rises from ~5 to ~120 pg/mL (exponential-ish increase)
        noise = random.uniform(-5, 5)
        il6 = il6_start + (115 * (progress ** 2)) + noise
        il6_values.append(round(max(il6, 0), 1))

        # pH: drops from ~7.40 to ~7.20 (below 7.35 is acidotic)
        noise = random.uniform(-0.01, 0.01)
        ph = ph_start - (0.20 * progress) + noise
        ph_values.append(round(ph, 3))

    return {
        "time": time_points,
        "lactate": lactate_values,
        "il6": il6_values,
        "ph": ph_values,
    }
