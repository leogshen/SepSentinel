# Compare RF models trained on the 5000-patient ICU dataset vs 500-patient synthetic dataset.
# Generates confusion matrices, ROC curves, and feature importance for both.

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay,
)

from sepsentinel.data_generator import generate_dataset

MODEL_DIR = os.path.join("models")
RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_evaluate(X, y, feature_names, dataset_name):
    """Train RF, evaluate, return model + metrics + test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Septic"])

    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    print(f"  Samples: {len(X)} ({sum(y==0)} healthy, {sum(y==1)} septic)")
    print(f"  Features: {feature_names}")
    print(f"  Train/Test: {len(X_train)} / {len(X_test)}")
    print(f"  Accuracy: {acc*100:.1f}%")
    print(f"  Cross-val: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
    print(f"\n{report}")

    return {
        "model": model,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "y_proba": y_proba,
        "accuracy": acc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "feature_names": feature_names,
        "dataset_name": dataset_name,
        "report": report,
    }


def plot_confusion_matrix(result, ax):
    """Plot confusion matrix on given axes."""
    cm = confusion_matrix(result["y_test"], result["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Septic"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(result["dataset_name"])


def plot_roc_curve(result, ax):
    """Plot ROC curve on given axes."""
    fpr, tpr, _ = roc_curve(result["y_test"], result["y_proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f'{result["dataset_name"]} (AUC={roc_auc:.3f})')
    return roc_auc


def plot_feature_importance(result, ax):
    """Plot feature importance on given axes."""
    importances = result["model"].feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [result["feature_names"][i] for i in indices]
    ax.barh(range(len(importances)), importances[indices], color="#3498db")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title(result["dataset_name"])
    ax.invert_yaxis()


def main():
    # --- Dataset 1: 5000-patient ICU dataset (lactate + pH only) ---
    df_icu = pd.read_csv("data/sepsis_icu_synthetic.csv")
    df_icu = df_icu[["lactate_mmol", "ph_arterial", "sepsis_label"]].dropna()
    df_icu = df_icu.rename(columns={
        "lactate_mmol": "lactate", "ph_arterial": "ph", "sepsis_label": "label"
    })

    X_icu = df_icu[["lactate", "ph"]].values
    y_icu = df_icu["label"].values

    result_icu = train_and_evaluate(
        X_icu, y_icu, ["lactate", "ph"],
        "ICU Dataset (5000 patients, 2 features)"
    )

    # --- Dataset 2: 500-patient synthetic dataset (lactate + il6 + pH) ---
    df_syn = generate_dataset(num_patients=500, seed=42)

    X_syn = df_syn[["lactate", "il6", "ph"]].values
    y_syn = df_syn["label"].values

    result_syn = train_and_evaluate(
        X_syn, y_syn, ["lactate", "il6", "ph"],
        "Synthetic Dataset (500 patients, 3 features)"
    )

    # --- Generate comparison plots ---

    # 1. Confusion Matrices (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_confusion_matrix(result_icu, axes[0])
    plot_confusion_matrix(result_syn, axes[1])
    fig.suptitle("Confusion Matrix Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150)
    print(f"  Saved: {RESULTS_DIR}/confusion_matrices.png")

    # 2. ROC Curves (overlaid)
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_icu = plot_roc_curve(result_icu, ax)
    auc_syn = plot_roc_curve(result_syn, ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"), dpi=150)
    print(f"  Saved: {RESULTS_DIR}/roc_curves.png")

    # 3. Feature Importance (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_feature_importance(result_icu, axes[0])
    plot_feature_importance(result_syn, axes[1])
    fig.suptitle("Feature Importance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
    print(f"  Saved: {RESULTS_DIR}/feature_importance.png")

    # --- Pick the winner ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  ICU dataset:       Acc={result_icu['accuracy']*100:.1f}%  CV={result_icu['cv_mean']*100:.1f}%  AUC={auc_icu:.3f}")
    print(f"  Synthetic dataset: Acc={result_syn['accuracy']*100:.1f}%  CV={result_syn['cv_mean']*100:.1f}%  AUC={auc_syn:.3f}")

    # Prefer the ICU dataset if it's reasonably close — it's more realistic
    # Only pick synthetic if ICU AUC is really bad (< 0.65)
    if auc_icu >= 0.65:
        winner = result_icu
        reason = "More realistic data distribution (real ICU patterns)"
    else:
        winner = result_syn
        reason = "ICU dataset AUC too low — synthetic provides better separation"

    print(f"\n  Winner: {winner['dataset_name']}")
    print(f"  Reason: {reason}")

    # Save the winning model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "sepsis_model.pkl"), "wb") as f:
        pickle.dump(winner["model"], f)
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump({"feature_columns": winner["feature_names"]}, f)

    print(f"\n  Model saved to {MODEL_DIR}/sepsis_model.pkl")
    print(f"  Config: {winner['feature_names']}")

    plt.close("all")
    print(f"\n  All plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
