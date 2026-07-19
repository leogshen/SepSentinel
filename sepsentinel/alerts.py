# Alert system — checks all 7 signals against WARNING/CRITICAL thresholds.

from sepsentinel.config.signals import ALL_SIGNALS
from sepsentinel.config.thresholds import ALERT_THRESHOLDS


def check_signal_alerts(values: dict) -> list[dict]:
    """Check all signal values against thresholds.

    Args:
        values: dict mapping signal names to current values.
                e.g. {"heart_rate": 115, "lactate": 3.2, ...}

    Returns:
        List of alert dicts: {"signal", "level", "message"}
    """
    alerts = []

    for signal_key, value in values.items():
        if signal_key not in ALERT_THRESHOLDS:
            continue
        thresh = ALERT_THRESHOLDS[signal_key]
        sig_info = ALL_SIGNALS.get(signal_key, {"name": signal_key, "unit": ""})
        name = sig_info["name"]
        unit = sig_info["unit"]

        direction = thresh["direction"]

        # Check high thresholds
        if direction in ("high", "both"):
            crit = thresh.get("critical_high")
            warn = thresh.get("warning_high")
            if crit is not None and value >= crit:
                alerts.append({
                    "signal": signal_key, "level": "CRITICAL",
                    "message": f"{name} is {value} {unit} (critical: >={crit})",
                })
                continue
            if warn is not None and value >= warn:
                alerts.append({
                    "signal": signal_key, "level": "WARNING",
                    "message": f"{name} is {value} {unit} (elevated: >={warn})",
                })
                continue

        # Check low thresholds
        if direction in ("low", "both"):
            crit = thresh.get("critical_low")
            warn = thresh.get("warning_low")
            if crit is not None and value <= crit:
                alerts.append({
                    "signal": signal_key, "level": "CRITICAL",
                    "message": f"{name} is {value} {unit} (critical: <={crit})",
                })
                continue
            if warn is not None and value <= warn:
                alerts.append({
                    "signal": signal_key, "level": "WARNING",
                    "message": f"{name} is {value} {unit} (low: <={warn})",
                })

    return alerts


def check_risk_alert(risk_score: float) -> dict | None:
    """Return an alert dict for the overall risk score, or None if normal."""
    thresh = ALERT_THRESHOLDS["risk_score"]
    if risk_score >= thresh["critical_high"]:
        return {"level": "CRITICAL",
                "message": f"Sepsis risk is {risk_score}% - Immediate attention needed!"}
    if risk_score >= thresh["warning_high"]:
        return {"level": "WARNING",
                "message": f"Sepsis risk is {risk_score}% - Close monitoring recommended."}
    return None


def format_alerts_for_console(alerts: list[dict], risk_alert: dict | None = None):
    """Print alerts to the console."""
    if not alerts and risk_alert is None:
        print("  All signals within normal range. No alerts.")
        return

    print("  ALERTS:")
    for alert in alerts:
        symbol = "!!!" if alert["level"] == "CRITICAL" else " ! "
        print(f"    [{symbol}] {alert['level']}: {alert['message']}")

    if risk_alert:
        symbol = "!!!" if risk_alert["level"] == "CRITICAL" else " ! "
        print(f"    [{symbol}] {risk_alert['level']}: {risk_alert['message']}")
