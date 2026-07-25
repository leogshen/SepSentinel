# XGBoost baseline — flattens per-timestep features for gradient boosting.
#
# Like RandomForestModel, this is a flat model that ignores temporal structure.
# Each timestep is treated as an independent sample. For sequential approaches,
# see gru.py, tcn.py, and transformer.py.

import json
import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb

from sepsentinel.model_b.base import SepsisModel
from sepsentinel.model_b.registry import register


@register
class XGBoostModel(SepsisModel):
    name = "xgboost"
    requires_sequences = False

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 random_state=42, scale_pos_weight=None):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
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

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred,
                                            target_names=["Healthy", "Septic"]),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._flatten(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.model.save_model(os.path.join(directory, "xgboost.json"))
        with open(os.path.join(directory, "model_meta.json"), "w") as f:
            json.dump({"name": self.name, "feature_count": self.feature_count}, f)

    @classmethod
    def load(cls, directory: str) -> "XGBoostModel":
        instance = cls()
        instance.model.load_model(os.path.join(directory, "xgboost.json"))
        meta_path = os.path.join(directory, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            instance.feature_count = meta.get("feature_count")
        instance._is_fitted = True
        return instance
