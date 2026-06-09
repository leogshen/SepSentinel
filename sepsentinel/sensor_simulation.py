# Simulates worsening patient biomarker data over time.

import random


def simulate_patient_data(duration_minutes=60, interval_minutes=5):
    """Simulate biomarker readings for a worsening patient."""
    time_points = list(range(0, duration_minutes + 1, interval_minutes))
    num_readings = len(time_points)

    lactate_start = 1.0
    il6_start = 5.0
    ph_start = 7.40

    lactate_values = []
    il6_values = []
    ph_values = []

    for i in range(num_readings):
        progress = i / (num_readings - 1)

        # Lactate rises from ~1.0 to ~5.0 mmol/L
        lactate = lactate_start + (4.0 * progress) + random.uniform(-0.2, 0.2)
        lactate_values.append(round(max(lactate, 0.5), 2))

        # IL-6 rises from ~5 to ~120 pg/mL (exponential curve)
        il6 = il6_start + (115 * (progress ** 2)) + random.uniform(-5, 5)
        il6_values.append(round(max(il6, 0), 1))

        # pH drops from ~7.40 to ~7.20
        ph = ph_start - (0.20 * progress) + random.uniform(-0.01, 0.01)
        ph_values.append(round(ph, 3))

    return {
        "time": time_points,
        "lactate": lactate_values,
        "il6": il6_values,
        "ph": ph_values,
    }
