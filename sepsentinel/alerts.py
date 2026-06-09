# Alert system — checks 7 biomarkers against WARNING/CRITICAL thresholds.

ALERT_THRESHOLDS = {
    "risk_warning": 30,
    "risk_critical": 60,
    "lactate_warning": 2.0,
    "lactate_critical": 4.0,
    "il6_warning": 7,
    "il6_critical": 50,
    "ph_warning": 7.35,
    "ph_critical": 7.25,
    "presepsin_warning": 365,
    "presepsin_critical": 600,
    "strem1_warning": 150,
    "strem1_critical": 300,
    "il10_warning": 10,
    "il10_critical": 50,
    "cxcl10_warning": 300,
    "cxcl10_critical": 500,
}

# Config: (key, display_name, unit, direction)
# direction: "high" = above threshold is bad, "low" = below threshold is bad
_BIOMARKER_CHECKS = [
    ("lactate", "Lactate", "mmol/L", "high"),
    ("il6", "IL-6", "pg/mL", "high"),
    ("ph", "pH", "", "low"),
    ("presepsin", "Presepsin", "pg/mL", "high"),
    ("strem1", "sTREM-1", "pg/mL", "high"),
    ("il10", "IL-10", "pg/mL", "high"),
    ("cxcl10", "CXCL10", "pg/mL", "high"),
]


def check_biomarker_alerts(lactate, il6, ph, presepsin=200, strem1=80, il10=3, cxcl10=150):
    """Return a list of alert dicts for abnormal biomarker values."""
    values = {"lactate": lactate, "il6": il6, "ph": ph,
              "presepsin": presepsin, "strem1": strem1, "il10": il10, "cxcl10": cxcl10}
    alerts = []

    for key, name, unit, direction in _BIOMARKER_CHECKS:
        val = values[key]
        warn = ALERT_THRESHOLDS[f"{key}_warning"]
        crit = ALERT_THRESHOLDS[f"{key}_critical"]
        unit_str = f" {unit}" if unit else ""

        if direction == "high":
            if val >= crit:
                alerts.append({"biomarker": name, "level": "CRITICAL",
                                "message": f"{name} is {val}{unit_str} (critical: >={crit})"})
            elif val >= warn:
                alerts.append({"biomarker": name, "level": "WARNING",
                                "message": f"{name} is {val}{unit_str} (elevated: >={warn})"})
        else:  # low
            if val <= crit:
                alerts.append({"biomarker": name, "level": "CRITICAL",
                                "message": f"{name} is {val}{unit_str} (critical: <={crit})"})
            elif val <= warn:
                alerts.append({"biomarker": name, "level": "WARNING",
                                "message": f"{name} is {val}{unit_str} (low: <={warn})"})

    return alerts


def check_risk_alert(risk_score):
    """Return an alert dict for the overall risk score, or None if normal."""
    if risk_score >= ALERT_THRESHOLDS["risk_critical"]:
        return {"level": "CRITICAL",
                "message": f"Sepsis risk is {risk_score}% - Immediate attention needed!"}
    elif risk_score >= ALERT_THRESHOLDS["risk_warning"]:
        return {"level": "WARNING",
                "message": f"Sepsis risk is {risk_score}% - Close monitoring recommended."}
    return None


def format_alerts_for_console(alerts, risk_alert=None):
    """Print alerts to the console."""
    if not alerts and risk_alert is None:
        print("  All biomarkers within normal range. No alerts.")
        return

    print("  ALERTS:")
    for alert in alerts:
        symbol = "!!!" if alert["level"] == "CRITICAL" else " ! "
        print(f"    [{symbol}] {alert['level']}: {alert['message']}")

    if risk_alert:
        symbol = "!!!" if risk_alert["level"] == "CRITICAL" else " ! "
        print(f"    [{symbol}] {risk_alert['level']}: {risk_alert['message']}")
