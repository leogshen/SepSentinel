# Sepsis risk scoring — random forest with rule-based fallback (7-marker panel).

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sepsis_model.pkl")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

ALL_BIOMARKERS = ["lactate", "il6", "ph", "presepsin", "strem1", "il10", "cxcl10"]


def train_model(df, model_path=MODEL_PATH, feature_columns=None):
    """Train a random forest model on labeled biomarker data."""
    if feature_columns is None:
        feature_columns = [col for col in ALL_BIOMARKERS if col in df.columns]

    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in the dataset.")

    X = df[feature_columns].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Septic"])
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    config_path = os.path.join(os.path.dirname(model_path), "model_config.json")
    with open(config_path, "w") as f:
        json.dump({"feature_columns": feature_columns}, f)

    return model, {
        "accuracy": accuracy,
        "report": report,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "feature_columns": feature_columns,
    }


def load_model(model_path=MODEL_PATH):
    """Load a trained model from disk, or return None."""
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def load_model_config(config_path=MODEL_CONFIG_PATH):
    """Load model config to know which features it expects."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def calculate_sepsis_risk(lactate, il6, ph, presepsin=200, strem1=80, il10=3, cxcl10=150):
    """Calculate sepsis risk (0-100%) using ML model or rule-based fallback."""
    model = load_model()
    if model is not None:
        config = load_model_config()
        all_values = {
            "lactate": lactate, "il6": il6, "ph": ph,
            "presepsin": presepsin, "strem1": strem1,
            "il10": il10, "cxcl10": cxcl10,
        }
        feature_cols = config["feature_columns"] if config else ALL_BIOMARKERS
        features = np.array([[all_values[col] for col in feature_cols]])
        risk = model.predict_proba(features)[0][1] * 100
        return round(risk, 1)

    return _rule_based_risk(lactate, il6, ph, presepsin, strem1, il10, cxcl10)


def _rule_based_risk(lactate, il6, ph, presepsin, strem1, il10, cxcl10):
    """Rule-based fallback (7 markers, max 100 points)."""
    score = 0

    # Lactate (0-20)
    if lactate > 2.0:
        score += min(20, (lactate - 2.0) / 4.0 * 20)

    # IL-6 (0-15)
    if il6 > 7:
        score += min(15, (il6 - 7) / 93 * 15)

    # pH (0-15, inverted)
    if ph < 7.35:
        score += min(15, (7.35 - ph) / 0.10 * 15)

    # Presepsin (0-15)
    if presepsin > 365:
        score += min(15, (presepsin - 365) / 600 * 15)

    # sTREM-1 (0-12)
    if strem1 > 150:
        score += min(12, (strem1 - 150) / 300 * 12)

    # IL-10 (0-12)
    if il10 > 10:
        score += min(12, (il10 - 10) / 90 * 12)

    # CXCL10 (0-11)
    if cxcl10 > 300:
        score += min(11, (cxcl10 - 300) / 500 * 11)

    return round(max(0, min(100, score)), 1)
