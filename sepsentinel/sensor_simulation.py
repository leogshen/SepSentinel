# Simulates worsening patient biomarker data over time (7-marker panel).

import random


def simulate_patient_data(duration_minutes=60, interval_minutes=5):
    """Simulate biomarker readings for a worsening patient."""
    time_points = list(range(0, duration_minutes + 1, interval_minutes))
    num_readings = len(time_points)

    data = {key: [] for key in ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]}

    for i in range(num_readings):
        progress = i / (num_readings - 1)

        # Lactate: 1.0 → ~5.0 mmol/L
        data["lactate"].append(round(max(1.0 + 4.0 * progress + random.uniform(-0.2, 0.2), 0.5), 2))

        # IL-6: 5 → ~120 pg/mL (exponential)
        data["il6"].append(round(max(5.0 + 115 * (progress ** 2) + random.uniform(-5, 5), 0), 1))

        # pH: 7.40 → ~7.20
        data["ph"].append(round(7.40 - 0.20 * progress + random.uniform(-0.01, 0.01), 3))

        # Presepsin: 200 → ~900 pg/mL
        data["presepsin"].append(round(max(200 + 700 * progress + random.uniform(-30, 30), 50), 0))

        # sTREM-1: 80 → ~450 pg/mL
        data["strem1"].append(round(max(80 + 370 * progress + random.uniform(-20, 20), 0), 0))

        # IL-10: 3 → ~80 pg/mL (delayed anti-inflammatory response)
        data["il10"].append(round(max(3.0 + 77 * (progress ** 1.5) + random.uniform(-4, 4), 0), 1))

        # CXCL10: 150 → ~800 pg/mL
        data["cxcl10"].append(round(max(150 + 650 * progress + random.uniform(-30, 30), 0), 0))

    return {"time": time_points, **data}
