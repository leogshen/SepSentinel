# risk_model.py
# ---------------
# This file contains the Sepsis Risk Score system.
#
# It now has TWO modes:
#   1. ML mode (default): Uses a trained logistic regression model that learned
#      patterns from patient data. This is more accurate and scalable.
#   2. Rule-based fallback: The original dummy scoring function, used if no
#      trained model is available yet.
#
# The key idea stays the same: biomarker values go IN, a risk score comes OUT.
# The rest of the system doesn't need to know which mode is being used.

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Default path to save/load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sepsis_model.pkl")


def train_model(df, model_path=MODEL_PATH):
    """
    Train a logistic regression model on a labeled dataset.

    Args:
        df: A pandas DataFrame with columns: lactate, il6, ph, label
        model_path: Where to save the trained model file.

    Returns:
        The trained model and a dictionary of evaluation metrics.
    """
    # Separate features (X) from labels (y)
    X = df[["lactate", "il6", "ph"]].values
    y = df["label"].values

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a logistic regression classifier
    # Logistic regression is a good starting model because:
    #   - It's simple and interpretable
    #   - It outputs probabilities (not just yes/no)
    #   - It works well for binary classification (healthy vs septic)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Septic"])

    # Save the trained model to disk so we don't have to retrain every time
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "accuracy": accuracy,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, metrics


def load_model(model_path=MODEL_PATH):
    """
    Load a previously trained model from disk.

    Returns:
        The trained model, or None if no saved model exists.
    """
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def calculate_sepsis_risk(lactate, il6, ph):
    """
    Calculate a sepsis risk score based on three biomarker values.

    Uses the trained ML model if available, otherwise falls back
    to the rule-based scoring system.

    Args:
        lactate: Current lactate level (mmol/L).
        il6: Current IL-6 level (pg/mL).
        ph: Current pH level (pH units).

    Returns:
        A risk score from 0 to 100 (percentage).
    """
    # Try to use the trained ML model
    model = load_model()
    if model is not None:
        features = np.array([[lactate, il6, ph]])
        # predict_proba returns [prob_healthy, prob_septic]
        # We want the septic probability as our risk score
        risk = model.predict_proba(features)[0][1] * 100
        return round(risk, 1)

    # Fallback: rule-based scoring (from Module 1)
    return _rule_based_risk(lactate, il6, ph)


def _rule_based_risk(lactate, il6, ph):
    """
    Original rule-based dummy scoring function (Module 1 fallback).

    Used when no trained ML model is available.
    """
    # Lactate score (0-40 points)
    if lactate <= 2.0:
        lactate_score = 0
    elif lactate <= 4.0:
        lactate_score = (lactate - 2.0) / 2.0 * 40
    else:
        lactate_score = 40

    # IL-6 score (0-35 points)
    if il6 <= 7:
        il6_score = 0
    elif il6 <= 100:
        il6_score = (il6 - 7) / 93 * 35
    else:
        il6_score = 35

    # pH score (0-25 points)
    if ph >= 7.35:
        ph_score = 0
    elif ph >= 7.25:
        ph_score = (7.35 - ph) / 0.10 * 25
    else:
        ph_score = 25

    total_risk = lactate_score + il6_score + ph_score
    total_risk = max(0, min(100, total_risk))
    return round(total_risk, 1)
