# Real-time sensor simulation for dashboard and CLI demos.
# Generates worsening or stable patient trajectories across all 7 signals.

import random

from sepsentinel.config.signals import FEATURE_ORDER


def simulate_patient(duration_minutes=60, interval_minutes=5, scenario="worsening", seed=None):
    """Simulate 7-signal readings over time.

    Args:
        duration_minutes: Total simulation length.
        interval_minutes: Time between readings.
        scenario: "worsening" (healthy -> septic) or "stable" (stays healthy).
        seed: Random seed for reproducibility. None for random.

    Returns:
        dict with "time" (list of ints) and one list per signal name.
    """
    if seed is not None:
        random.seed(seed)

    time_points = list(range(0, duration_minutes + 1, interval_minutes))
    num_readings = len(time_points)

    signals = {name: [] for name in FEATURE_ORDER}

    for i in range(num_readings):
        progress = i / max(num_readings - 1, 1)

        if scenario == "worsening":
            signals["heart_rate"].append(
                round(75 + 45 * progress + random.uniform(-3, 3)))
            signals["respiratory_rate"].append(
                round(14 + 14 * progress + random.uniform(-1, 1)))
            signals["temperature"].append(
                round(36.8 + 2.2 * progress + random.uniform(-0.1, 0.1), 1))
            signals["spo2"].append(
                round(max(80, 98 - 10 * progress + random.uniform(-1, 1)), 1))
            signals["ph"].append(
                round(7.40 - 0.20 * progress + random.uniform(-0.01, 0.01), 3))
            signals["lactate"].append(
                round(max(0.5, 1.0 + 4.5 * progress + random.uniform(-0.2, 0.2)), 2))
            signals["il6"].append(
                round(max(0, 5.0 + 115 * (progress ** 2) + random.uniform(-5, 5)), 1))

        elif scenario == "stable":
            signals["heart_rate"].append(
                round(72 + random.uniform(-5, 5)))
            signals["respiratory_rate"].append(
                round(15 + random.uniform(-2, 2)))
            signals["temperature"].append(
                round(36.8 + random.uniform(-0.3, 0.3), 1))
            signals["spo2"].append(
                round(min(100, 97 + random.uniform(-1, 1)), 1))
            signals["ph"].append(
                round(7.40 + random.uniform(-0.02, 0.02), 3))
            signals["lactate"].append(
                round(max(0.5, 1.2 + random.uniform(-0.3, 0.3)), 2))
            signals["il6"].append(
                round(max(0, 3.0 + random.uniform(-1.5, 1.5)), 1))

    return {"time": time_points, **signals}
