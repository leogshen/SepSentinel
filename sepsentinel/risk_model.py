# risk_model.py
# ---------------
# This file contains the Sepsis Risk Score function.
#
# RIGHT NOW: This is a simple rule-based placeholder. It takes in the three
# biomarker values and outputs a risk score from 0-100%.
#
# IN THE FUTURE: This function can be replaced with a trained machine learning
# model (e.g., logistic regression, random forest, or neural network) that
# learns from real patient data to make more accurate predictions.
#
# The key idea: biomarker values go IN, a risk score comes OUT.
# The internal logic can be swapped without changing the rest of the system.


def calculate_sepsis_risk(lactate, il6, ph):
    """
    Calculate a sepsis risk score based on three biomarker values.

    This is a DUMMY scoring function for demonstration purposes.
    It does NOT provide medical accuracy.

    Args:
        lactate: Current lactate level (mmol/L).
        il6: Current IL-6 level (pg/mL).
        ph: Current pH level (pH units).

    Returns:
        A risk score from 0 to 100 (percentage).

    How it works (simplified):
        - Each biomarker contributes a partial score based on how far
          it has deviated from the normal range.
        - The three partial scores are combined into a total risk score.
        - The score is clamped between 0 and 100.

    Future improvement:
        Replace this function body with a call to a trained ML model:
            model = load_model("sepsis_model.pkl")
            features = [lactate, il6, ph]
            risk = model.predict_proba(features)[0][1] * 100
            return risk
    """

    # --- Lactate score (0-40 points) ---
    # Normal: 0.5-2.0 mmol/L
    # Dangerous: above 4.0 mmol/L
    if lactate <= 2.0:
        lactate_score = 0
    elif lactate <= 4.0:
        # Linear scale: 2.0 -> 0 points, 4.0 -> 40 points
        lactate_score = (lactate - 2.0) / 2.0 * 40
    else:
        lactate_score = 40

    # --- IL-6 score (0-35 points) ---
    # Normal: 0-7 pg/mL
    # Dangerous: above 100 pg/mL
    if il6 <= 7:
        il6_score = 0
    elif il6 <= 100:
        # Linear scale: 7 -> 0 points, 100 -> 35 points
        il6_score = (il6 - 7) / 93 * 35
    else:
        il6_score = 35

    # --- pH score (0-25 points) ---
    # Normal: 7.35-7.45
    # Dangerous: below 7.25
    if ph >= 7.35:
        ph_score = 0
    elif ph >= 7.25:
        # Linear scale: 7.35 -> 0 points, 7.25 -> 25 points
        ph_score = (7.35 - ph) / 0.10 * 25
    else:
        ph_score = 25

    # Combine all scores (max possible = 40 + 35 + 25 = 100)
    total_risk = lactate_score + il6_score + ph_score

    # Clamp between 0 and 100
    total_risk = max(0, min(100, total_risk))

    return round(total_risk, 1)
