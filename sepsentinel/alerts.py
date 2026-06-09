# Alert system — checks biomarkers against WARNING/CRITICAL thresholds.

from sepsentinel.biomarkers import BIOMARKERS

ALERT_THRESHOLDS = {
    "risk_warning": 30,
    "risk_critical": 60,
    "lactate_warning": 2.0,
    "lactate_critical": 4.0,
    "il6_warning": 7,
    "il6_critical": 50,
    "ph_warning": 7.35,
    "ph_critical": 7.25,
}


def check_biomarker_alerts(lactate, il6, ph):
    """Return a list of alert dicts for abnormal biomarker values."""
    alerts = []

    if lactate >= ALERT_THRESHOLDS["lactate_critical"]:
        alerts.append({"biomarker": "Lactate", "level": "CRITICAL",
                        "message": f"Lactate is {lactate} mmol/L (critical: >={ALERT_THRESHOLDS['lactate_critical']})"})
    elif lactate >= ALERT_THRESHOLDS["lactate_warning"]:
        alerts.append({"biomarker": "Lactate", "level": "WARNING",
                        "message": f"Lactate is {lactate} mmol/L (elevated: >={ALERT_THRESHOLDS['lactate_warning']})"})

    if il6 >= ALERT_THRESHOLDS["il6_critical"]:
        alerts.append({"biomarker": "IL-6", "level": "CRITICAL",
                        "message": f"IL-6 is {il6} pg/mL (critical: >={ALERT_THRESHOLDS['il6_critical']})"})
    elif il6 >= ALERT_THRESHOLDS["il6_warning"]:
        alerts.append({"biomarker": "IL-6", "level": "WARNING",
                        "message": f"IL-6 is {il6} pg/mL (elevated: >={ALERT_THRESHOLDS['il6_warning']})"})

    if ph <= ALERT_THRESHOLDS["ph_critical"]:
        alerts.append({"biomarker": "pH", "level": "CRITICAL",
                        "message": f"pH is {ph} (critical: <={ALERT_THRESHOLDS['ph_critical']})"})
    elif ph <= ALERT_THRESHOLDS["ph_warning"]:
        alerts.append({"biomarker": "pH", "level": "WARNING",
                        "message": f"pH is {ph} (low: <={ALERT_THRESHOLDS['ph_warning']})"})

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
