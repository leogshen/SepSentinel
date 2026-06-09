# Sepsis risk scoring — ML model with rule-based fallback.
# Supports flexible feature sets (trains on whatever biomarkers are available).

import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sepsis_model.pkl")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


def train_model(df, model_path=MODEL_PATH, feature_columns=None):
    """Train a logistic regression model on labeled biomarker data."""
    all_biomarkers = ["lactate", "il6", "ph"]
    if feature_columns is None:
        feature_columns = [col for col in all_biomarkers if col in df.columns]

    if len(feature_columns) == 0:
        raise ValueError("No feature columns found in the dataset.")

    X = df[feature_columns].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Septic"])
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    # Save model and config
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


def calculate_sepsis_risk(lactate, il6, ph):
    """Calculate sepsis risk (0-100%) using ML model or rule-based fallback."""
    model = load_model()
    if model is not None:
        config = load_model_config()
        all_values = {"lactate": lactate, "il6": il6, "ph": ph}
        feature_cols = config["feature_columns"] if config else ["lactate", "il6", "ph"]
        features = np.array([[all_values[col] for col in feature_cols]])
        risk = model.predict_proba(features)[0][1] * 100
        return round(risk, 1)

    return _rule_based_risk(lactate, il6, ph)


def _rule_based_risk(lactate, il6, ph):
    """Rule-based fallback when no ML model is available."""
    if lactate <= 2.0:
        lactate_score = 0
    elif lactate <= 4.0:
        lactate_score = (lactate - 2.0) / 2.0 * 40
    else:
        lactate_score = 40

    if il6 <= 7:
        il6_score = 0
    elif il6 <= 100:
        il6_score = (il6 - 7) / 93 * 35
    else:
        il6_score = 35

    if ph >= 7.35:
        ph_score = 0
    elif ph >= 7.25:
        ph_score = (7.35 - ph) / 0.10 * 25
    else:
        ph_score = 25

    total_risk = max(0, min(100, lactate_score + il6_score + ph_score))
    return round(total_risk, 1)
