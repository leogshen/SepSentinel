# Alert thresholds for all 7 signals and overall risk score.
# Each signal has WARNING and CRITICAL levels.
# "direction" indicates whether the alert triggers on high values, low values, or both.

ALERT_THRESHOLDS = {
    "heart_rate": {
        "direction": "both",
        "warning_low": 50, "critical_low": 40,
        "warning_high": 100, "critical_high": 120,
    },
    "respiratory_rate": {
        "direction": "both",
        "warning_low": 10, "critical_low": 8,
        "warning_high": 22, "critical_high": 30,
    },
    "temperature": {
        "direction": "both",
        "warning_low": 35.5, "critical_low": 35.0,
        "warning_high": 38.0, "critical_high": 39.0,
    },
    "spo2": {
        "direction": "low",
        "warning_low": 94, "critical_low": 90,
    },
    "ph": {
        "direction": "low",
        "warning_low": 7.35, "critical_low": 7.25,
    },
    "lactate": {
        "direction": "high",
        "warning_high": 2.0, "critical_high": 4.0,
    },
    "il6": {
        "direction": "high",
        "warning_high": 7, "critical_high": 50,
    },
    "risk_score": {
        "direction": "high",
        "warning_high": 30, "critical_high": 60,
    },
}
