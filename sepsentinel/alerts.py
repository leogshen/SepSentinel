# alerts.py
# ----------
# This file handles the alert system for SepSentinel.
#
# It checks biomarker values and risk scores against thresholds and
# generates warnings when values become concerning. In a real clinical
# setting, these alerts could trigger notifications to medical staff.
#
# Alert levels:
#   - NORMAL:   All values within safe range
#   - WARNING:  One or more values approaching dangerous levels
#   - CRITICAL: Values indicate high sepsis risk, immediate attention needed

from sepsentinel.biomarkers import BIOMARKERS


# Thresholds for triggering alerts
ALERT_THRESHOLDS = {
    "risk_warning": 30,     # Risk score above this = warning
    "risk_critical": 60,    # Risk score above this = critical
    "lactate_warning": 2.0,
    "lactate_critical": 4.0,
    "il6_warning": 7,
    "il6_critical": 50,
    "ph_warning": 7.35,     # Below this = warning
    "ph_critical": 7.25,    # Below this = critical
}


def check_biomarker_alerts(lactate, il6, ph):
    """
    Check individual biomarker values and return a list of alerts.

    Each alert is a dictionary with:
        - biomarker: which biomarker triggered it
        - level: "WARNING" or "CRITICAL"
        - message: human-readable description

    Args:
        lactate: Current lactate level (mmol/L).
        il6: Current IL-6 level (pg/mL).
        ph: Current pH level (pH units).

    Returns:
        A list of alert dictionaries (empty list = all normal).
    """
    alerts = []

    # --- Lactate alerts ---
    if lactate >= ALERT_THRESHOLDS["lactate_critical"]:
        alerts.append({
            "biomarker": "Lactate",
            "level": "CRITICAL",
            "message": f"Lactate is {lactate} mmol/L (critical: >={ALERT_THRESHOLDS['lactate_critical']})",
        })
    elif lactate >= ALERT_THRESHOLDS["lactate_warning"]:
        alerts.append({
            "biomarker": "Lactate",
            "level": "WARNING",
            "message": f"Lactate is {lactate} mmol/L (elevated: >={ALERT_THRESHOLDS['lactate_warning']})",
        })

    # --- IL-6 alerts ---
    if il6 >= ALERT_THRESHOLDS["il6_critical"]:
        alerts.append({
            "biomarker": "IL-6",
            "level": "CRITICAL",
            "message": f"IL-6 is {il6} pg/mL (critical: >={ALERT_THRESHOLDS['il6_critical']})",
        })
    elif il6 >= ALERT_THRESHOLDS["il6_warning"]:
        alerts.append({
            "biomarker": "IL-6",
            "level": "WARNING",
            "message": f"IL-6 is {il6} pg/mL (elevated: >={ALERT_THRESHOLDS['il6_warning']})",
        })

    # --- pH alerts (lower = worse) ---
    if ph <= ALERT_THRESHOLDS["ph_critical"]:
        alerts.append({
            "biomarker": "pH",
            "level": "CRITICAL",
            "message": f"pH is {ph} (critical: <={ALERT_THRESHOLDS['ph_critical']})",
        })
    elif ph <= ALERT_THRESHOLDS["ph_warning"]:
        alerts.append({
            "biomarker": "pH",
            "level": "WARNING",
            "message": f"pH is {ph} (low: <={ALERT_THRESHOLDS['ph_warning']})",
        })

    return alerts


def check_risk_alert(risk_score):
    """
    Check the overall risk score and return an alert level.

    Returns:
        A dictionary with "level" and "message", or None if normal.
    """
    if risk_score >= ALERT_THRESHOLDS["risk_critical"]:
        return {
            "level": "CRITICAL",
            "message": f"Sepsis risk is {risk_score}% - Immediate attention needed!",
        }
    elif risk_score >= ALERT_THRESHOLDS["risk_warning"]:
        return {
            "level": "WARNING",
            "message": f"Sepsis risk is {risk_score}% - Close monitoring recommended.",
        }
    return None


def format_alerts_for_console(alerts, risk_alert=None):
    """Print alerts to the console in a readable format."""
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
