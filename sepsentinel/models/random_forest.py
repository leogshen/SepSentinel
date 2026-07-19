# Random Forest baseline — flattens temporal windows into feature vectors.

import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from sepsentinel.models.base import SepsisModel
from sepsentinel.models.registry import register


@register
class RandomForestModel(SepsisModel):
    name = "random_forest"
    requires_sequences = False

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.feature_count = None
        self._is_fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> dict:
        X = self._flatten(X)
        self.feature_count = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_fitted = True

        y_pred = self.model.predict(X_test)
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="accuracy")

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, target_names=["Healthy", "Septic"]),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._flatten(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "random_forest.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        with open(os.path.join(directory, "model_meta.json"), "w") as f:
            json.dump({"name": self.name, "feature_count": self.feature_count}, f)

    @classmethod
    def load(cls, directory: str) -> "RandomForestModel":
        with open(os.path.join(directory, "random_forest.pkl"), "rb") as f:
            sklearn_model = pickle.load(f)
        meta_path = os.path.join(directory, "model_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        instance = cls()
        instance.model = sklearn_model
        instance.feature_count = meta.get("feature_count")
        instance._is_fitted = True
        return instance
