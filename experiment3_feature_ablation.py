#!/usr/bin/env python
"""Experiment 3: Feature Ablation Study

Determines which biochemical/clinical variables add predictive value
beyond wearable vitals, using controlled feature ablation.

Extends the experiment2_imputation framework:
- Reuses: compute_metrics, find_best_f1_threshold, collect_predictions,
  and all frozen hyperparameters from experiment2.
- Adds: AblationPreprocessor (generalises StrategyPreprocessor's Strategy B
  for arbitrary feature subsets), per-experiment artifact saving, overlay
  plots, and summary generation.

Experiments:
  A: vitals only (HR, SpO2, Resp, Temp)              [baseline]
  B: vitals + lactate                                  [add-one-in]
  C: vitals + pH                                       [add-one-in]
  D: vitals + creatinine                               [add-one-in]
  E: vitals + lactate + pH                             [add-two-in]
  F: vitals + all labs (full model)                    [full baseline]
  G: all minus lactate                                 [leave-one-out]
  H: all minus pH                                      [leave-one-out]
  I: all minus creatinine                              [leave-one-out]
  J: vitals + WBC                                      [add-one-in]
  K: vitals + platelets                                [add-one-in]
  L: vitals + bilirubin                                [add-one-in]
  M: all minus WBC                                     [leave-one-out]
  N: all minus platelets                               [leave-one-out]
  O: all minus bilirubin                               [leave-one-out]

Usage:
  python experiment3_feature_ablation.py
  python experiment3_feature_ablation.py --experiments A B F
  python experiment3_feature_ablation.py --seed 123
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.preprocessing import CLIP_RANGES
from sepsentinel.data.splitting import patient_split
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer

# Reuse experiment2's shared infrastructure
from experiment2_imputation import (
    compute_metrics as _compute_metrics_base,
    find_best_f1_threshold,
    collect_predictions,
    FEATURES as ALL_FEATURES,
    N_VITALS, N_LABS,
    SPLIT_SEED, EPOCHS, BATCH_SIZE, LR,
    PATIENCE, MIN_DELTA, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
)

# ============================================================
# Configuration
# ============================================================

VITALS = ALL_FEATURES[:N_VITALS]
LABS = ALL_FEATURES[N_VITALS:]
DEFAULT_SEED = 42
MIN_LENGTH = 6

EXPERIMENTS = OrderedDict([
    ("A", {"name": "vitals_only",          "features": list(VITALS),                                          "type": "baseline"}),
    ("B", {"name": "vitals_lactate",       "features": list(VITALS) + ["lactate"],                            "type": "add-one-in"}),
    ("C", {"name": "vitals_ph",            "features": list(VITALS) + ["ph"],                                 "type": "add-one-in"}),
    ("D", {"name": "vitals_creatinine",    "features": list(VITALS) + ["creatinine"],                         "type": "add-one-in"}),
    ("E", {"name": "vitals_lactate_ph",    "features": list(VITALS) + ["lactate", "ph"],                      "type": "add-two-in"}),
    ("F", {"name": "vitals_all_labs",      "features": list(ALL_FEATURES),                                    "type": "full-baseline"}),
    ("G", {"name": "all_minus_lactate",    "features": [f for f in ALL_FEATURES if f != "lactate"],           "type": "leave-one-out"}),
    ("H", {"name": "all_minus_ph",         "features": [f for f in ALL_FEATURES if f != "ph"],                "type": "leave-one-out"}),
    ("I", {"name": "all_minus_creatinine", "features": [f for f in ALL_FEATURES if f != "creatinine"],        "type": "leave-one-out"}),
    ("J", {"name": "vitals_wbc",           "features": list(VITALS) + ["wbc"],                                "type": "add-one-in"}),
    ("K", {"name": "vitals_platelets",     "features": list(VITALS) + ["platelets"],                          "type": "add-one-in"}),
    ("L", {"name": "vitals_bilirubin",     "features": list(VITALS) + ["bilirubin"],                          "type": "add-one-in"}),
    ("M", {"name": "all_minus_wbc",        "features": [f for f in ALL_FEATURES if f != "wbc"],               "type": "leave-one-out"}),
    ("N", {"name": "all_minus_platelets",  "features": [f for f in ALL_FEATURES if f != "platelets"],         "type": "leave-one-out"}),
    ("O", {"name": "all_minus_bilirubin",  "features": [f for f in ALL_FEATURES if f != "bilirubin"],         "type": "leave-one-out"}),
])


def compute_metrics(y_true, y_prob, threshold):
    """Extend experiment2's compute_metrics with confusion matrix components."""
    metrics = _compute_metrics_base(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    metrics["tp"] = int(((y_pred == 1) & (y_true == 1)).sum())
    metrics["fp"] = int(((y_pred == 1) & (y_true == 0)).sum())
    metrics["tn"] = int(((y_pred == 0) & (y_true == 0)).sum())
    metrics["fn"] = int(((y_pred == 0) & (y_true == 1)).sum())
    return metrics


# ============================================================
# Ablation Preprocessor
# ============================================================
# Generalises experiment2's StrategyPreprocessor (Strategy B only)
# to handle arbitrary feature subsets. The original hardcodes
# N_VITALS=4 and N_LABS=6; this version computes them dynamically
# from the selected feature list.

class AblationPreprocessor:
    """Strategy B preprocessor for arbitrary feature subsets.

    Vitals: causal forward-fill + training median (value channel only).
    Labs: causal forward-fill + training median + observation mask + time-since-last delta.

    Output layout: [vital_values, lab_values, lab_masks, lab_deltas]
    Total channels: n_vitals + n_labs * 3
    """

    def __init__(self, all_features, selected_features):
        self.all_features = all_features
        vitals_set = set(VITALS)

        # Enforce vitals-first, labs-second ordering
        self.sel_vitals = [f for f in selected_features if f in vitals_set]
        self.sel_labs = [f for f in selected_features if f not in vitals_set]
        self.selected_features = self.sel_vitals + self.sel_labs

        # Column indices in the full ALL_FEATURES array
        self.col_indices = [all_features.index(f) for f in self.selected_features]

        self.n_vitals = len(self.sel_vitals)
        self.n_labs = len(self.sel_labs)
        self.n_raw = self.n_vitals + self.n_labs

        if self.n_labs > 0:
            self.n_channels = self.n_vitals + self.n_labs * 3
        else:
            self.n_channels = self.n_vitals

        self.train_medians = None
        self.val_mean = None
        self.val_std = None
        self.delta_mean = None
        self.delta_std = None

    def fit(self, episodes):
        all_raw = np.concatenate(
            [e["signals"][:, self.col_indices] for e in episodes], axis=0
        )
        self.train_medians = np.nanmedian(all_raw, axis=0)

        all_values, all_deltas = [], []
        for ep in episodes:
            channels = self._fill_episode(ep["signals"])
            all_values.append(channels[:, : self.n_raw])
            if self.n_labs > 0:
                all_deltas.append(channels[:, self.n_raw + self.n_labs :])

        all_values = np.concatenate(all_values, axis=0)
        self.val_mean = all_values.mean(axis=0)
        self.val_std = all_values.std(axis=0)
        self.val_std[self.val_std == 0] = 1.0

        if self.n_labs > 0:
            all_deltas = np.concatenate(all_deltas, axis=0)
            self.delta_mean = all_deltas.mean(axis=0)
            self.delta_std = all_deltas.std(axis=0)
            self.delta_std[self.delta_std == 0] = 1.0

        return self

    def transform(self, episodes):
        results = []
        for ep in episodes:
            channels = self._fill_episode(ep["signals"])
            values = channels[:, : self.n_raw]
            values_norm = (values - self.val_mean) / self.val_std

            if self.n_labs > 0:
                masks = channels[:, self.n_raw : self.n_raw + self.n_labs]
                deltas = channels[:, self.n_raw + self.n_labs :]
                deltas_norm = (deltas - self.delta_mean) / self.delta_std
                expanded = np.concatenate(
                    [values_norm, masks, deltas_norm], axis=1
                ).astype(np.float32)
            else:
                expanded = values_norm.astype(np.float32)

            results.append({
                "signals": expanded,
                "labels": ep["labels"].astype(np.float32),
                "length": len(expanded),
                "patient_id": ep["patient_id"],
                "label": ep["label"],
            })
        return results

    def fit_transform(self, episodes):
        return self.fit(episodes).transform(episodes)

    def _fill_episode(self, raw_signals_full):
        signals = raw_signals_full[:, self.col_indices].copy()
        n_steps, n_feat = signals.shape

        # Record lab observation mask BEFORE any filling
        lab_observed = np.zeros((n_steps, self.n_labs), dtype=np.float32)
        for li in range(self.n_labs):
            col_idx = self.n_vitals + li
            lab_observed[:, li] = (~np.isnan(signals[:, col_idx])).astype(np.float32)

        # Causal forward-fill (no back-fill)
        for j in range(n_feat):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]

        # Fill remaining leading NaNs with training median
        for j in range(n_feat):
            nans = np.isnan(signals[:, j])
            if nans.any():
                signals[nans, j] = self.train_medians[j]

        # Clip to physiological ranges
        for j, feat in enumerate(self.selected_features):
            if feat in CLIP_RANGES:
                lo, hi = CLIP_RANGES[feat]
                signals[:, j] = np.clip(signals[:, j], lo, hi)

        if self.n_labs > 0:
            lab_deltas = np.zeros((n_steps, self.n_labs), dtype=np.float32)
            for li in range(self.n_labs):
                last_obs = -1
                for i in range(n_steps):
                    if lab_observed[i, li] == 1.0:
                        lab_deltas[i, li] = 0.0
                        last_obs = i
                    else:
                        if last_obs >= 0:
                            lab_deltas[i, li] = float(i - last_obs)
                        else:
                            lab_deltas[i, li] = float(i + 1)

            return np.concatenate([signals, lab_observed, lab_deltas], axis=1)
        else:
            return signals

    def get_channel_names(self):
        names = [f"val_{f}" for f in self.selected_features]
        names += [f"mask_{f}" for f in self.sel_labs]
        names += [f"delta_{f}" for f in self.sel_labs]
        return names


# ============================================================
# Plotting (new for experiment3)
# ============================================================

def save_roc_curve(y_true, y_prob, filepath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(filepath, dpi=150); plt.close(fig)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


def save_pr_curve(y_true, y_prob, filepath):
    from sklearn.metrics import average_precision_score
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr, lw=2, label=f"AUPRC = {auprc:.4f}")
    ax.axhline(prevalence, color="grey", ls="--", lw=1, label=f"Prevalence = {prevalence:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(filepath, dpi=150); plt.close(fig)
    return {"precision": prec_arr.tolist(), "recall": rec_arr.tolist()}


def save_overlay_roc(all_roc_data, filepath):
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, data in all_roc_data.items():
        ax.plot(data["fpr"], data["tpr"], lw=1.5, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Feature Ablation")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(filepath, dpi=150); plt.close(fig)


def save_overlay_pr(all_pr_data, prevalence, filepath):
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, data in all_pr_data.items():
        ax.plot(data["recall"], data["precision"], lw=1.5, label=label)
    ax.axhline(prevalence, color="grey", ls="--", lw=1, label=f"Prevalence = {prevalence:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves - Feature Ablation")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout(); fig.savefig(filepath, dpi=150); plt.close(fig)


def save_metrics_bar_chart(results_list, filepath):
    labels = [r["experiment_id"] for r in results_list]
    aurocs = [r["test"]["auroc"] for r in results_list]
    auprcs = [r["test"]["auprc"] for r in results_list]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color="#2196F3")
    ax.bar(x + width / 2, auprcs, width, label="AUPRC", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Feature Ablation: AUROC and AUPRC")
    ax.legend(); ax.set_ylim([0, 1.0])
    fig.tight_layout(); fig.savefig(filepath, dpi=150); plt.close(fig)


# ============================================================
# Single Experiment Runner
# ============================================================
# Extends experiment2's run_single pattern with per-experiment
# artifact saving (plots, JSON, confusion matrix, training history).

def run_experiment(exp_id, exp_config, splits_raw, seed, device, run_dir, logger):
    exp_name = exp_config["name"]
    features = exp_config["features"]
    exp_dir = os.path.join(run_dir, "experiments", f"{exp_id}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"{'='*70}")
    logger.info(f"  Experiment {exp_id}: {exp_name}")
    logger.info(f"  Raw features: {features}")
    logger.info(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Preprocess (Strategy B, variable feature subset)
    preprocessor = AblationPreprocessor(ALL_FEATURES, features)
    channel_names = preprocessor.get_channel_names()
    logger.info(f"  Input channels ({preprocessor.n_channels}): {channel_names}")

    preprocessor.fit(splits_raw["train"])
    train_data = preprocessor.transform(splits_raw["train"])
    val_data = preprocessor.transform(splits_raw["val"])
    test_data = preprocessor.transform(splits_raw["test"])

    # Split stats
    split_stats = {}
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        n_timesteps = sum(e["length"] for e in data)
        n_pos_steps = sum(int(e["labels"].sum()) for e in data)
        split_stats[split_name] = {
            "n_patients": len(splits_raw[split_name]),
            "n_septic": sum(1 for e in splits_raw[split_name] if e["label"] == 1),
            "n_timesteps": n_timesteps,
            "n_positive_steps": n_pos_steps,
            "prevalence": n_pos_steps / max(n_timesteps, 1),
        }

    # Pos weight (same pattern as experiment2's run_single)
    train_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = float(n_neg / max(n_pos, 1))

    # Model
    model = SepsisTransformer(input_dim=preprocessor.n_channels)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}  Pos weight: {pos_weight:.1f}")

    # Train (same Trainer + hyperparameters as experiment2)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    trainer = Trainer(model, device=device, checkpoint_dir=ckpt_dir, pos_weight=pos_weight)

    t0 = time.time()
    history = trainer.fit(
        train_data, val_data,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE, min_delta=MIN_DELTA,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
    )
    train_time = time.time() - t0
    best_epoch = max(history, key=lambda h: h["val_auroc"])["epoch"]

    # Threshold tuning on validation (reusing experiment2's functions)
    val_labels, val_probs = collect_predictions(model, val_data, device)
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    val_metrics = compute_metrics(val_labels, val_probs, best_threshold)

    # Test evaluation at validation threshold
    test_labels, test_probs = collect_predictions(model, test_data, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_threshold)

    logger.info(f"  Val  AUROC={val_metrics['auroc']:.4f}  AUPRC={val_metrics['auprc']:.4f}  F1={val_metrics['f1']:.4f}")
    logger.info(f"  Test AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  F1={test_metrics['f1']:.4f}")
    logger.info(f"  Threshold={best_threshold:.3f}  Best epoch={best_epoch}  Time={train_time:.0f}s")

    # Per-experiment artifacts
    roc_data = save_roc_curve(test_labels, test_probs, os.path.join(exp_dir, "roc_curve.png"))
    pr_data = save_pr_curve(test_labels, test_probs, os.path.join(exp_dir, "pr_curve.png"))

    history_out = []
    for h in history:
        history_out.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in h.items()})

    exp_metrics = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "experiment_type": exp_config["type"],
        "raw_features": features,
        "channel_names": channel_names,
        "n_channels": preprocessor.n_channels,
        "n_params": n_params,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "split_stats": split_stats,
        "val": val_metrics,
        "test": test_metrics,
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(exp_metrics, f, indent=2)
    with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
        json.dump(history_out, f, indent=2)
    with open(os.path.join(exp_dir, "roc_data.json"), "w") as f:
        json.dump(roc_data, f)
    with open(os.path.join(exp_dir, "pr_data.json"), "w") as f:
        json.dump(pr_data, f)
    with open(os.path.join(exp_dir, "confusion_matrix.json"), "w") as f:
        json.dump({"tp": test_metrics["tp"], "fp": test_metrics["fp"],
                    "tn": test_metrics["tn"], "fn": test_metrics["fn"]}, f, indent=2)

    exp_metrics["_roc_data"] = roc_data
    exp_metrics["_pr_data"] = pr_data
    exp_metrics["_history"] = history_out

    return exp_metrics


# ============================================================
# Summary Generation
# ============================================================

def generate_summary(all_results, run_dir, logger):
    sorted_results = sorted(all_results, key=lambda r: (r["test"]["auprc"], r["test"]["auroc"]), reverse=True)

    vitals_result = next((r for r in all_results if r["experiment_id"] == "A"), None)
    full_result = next((r for r in all_results if r["experiment_id"] == "F"), None)

    vitals_auroc = vitals_result["test"]["auroc"] if vitals_result else 0
    vitals_auprc = vitals_result["test"]["auprc"] if vitals_result else 0
    full_auroc = full_result["test"]["auroc"] if full_result else 0
    full_auprc = full_result["test"]["auprc"] if full_result else 0

    # --- CSV ---
    csv_path = os.path.join(run_dir, "ablation_results.csv")
    csv_fields = [
        "experiment_id", "experiment_name", "type", "raw_features",
        "n_channels", "n_params", "seed", "best_epoch", "train_time_sec",
        "auroc", "auprc", "precision", "recall", "specificity", "f1",
        "brier", "threshold",
        "delta_auroc_vs_vitals", "delta_auprc_vs_vitals",
        "delta_auroc_vs_full", "delta_auprc_vs_full",
        "n_train_patients", "n_val_patients", "n_test_patients",
        "train_prevalence", "test_prevalence",
    ]

    rows = []
    for r in sorted_results:
        row = {
            "experiment_id": r["experiment_id"],
            "experiment_name": r["experiment_name"],
            "type": r["experiment_type"],
            "raw_features": "; ".join(r["raw_features"]),
            "n_channels": r["n_channels"],
            "n_params": r["n_params"],
            "seed": r["seed"],
            "best_epoch": r["best_epoch"],
            "train_time_sec": r["train_time_sec"],
            "auroc": r["test"]["auroc"],
            "auprc": r["test"]["auprc"],
            "precision": r["test"]["precision"],
            "recall": r["test"]["recall"],
            "specificity": r["test"]["specificity"],
            "f1": r["test"]["f1"],
            "brier": r["test"]["brier"],
            "threshold": r["test"]["threshold"],
            "delta_auroc_vs_vitals": r["test"]["auroc"] - vitals_auroc,
            "delta_auprc_vs_vitals": r["test"]["auprc"] - vitals_auprc,
            "delta_auroc_vs_full": r["test"]["auroc"] - full_auroc,
            "delta_auprc_vs_full": r["test"]["auprc"] - full_auprc,
            "n_train_patients": r["split_stats"]["train"]["n_patients"],
            "n_val_patients": r["split_stats"]["val"]["n_patients"],
            "n_test_patients": r["split_stats"]["test"]["n_patients"],
            "train_prevalence": round(r["split_stats"]["train"]["prevalence"], 5),
            "test_prevalence": round(r["split_stats"]["test"]["prevalence"], 5),
        }
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    # --- JSON ---
    json_results = []
    for r in sorted_results:
        jr = dict(r)
        jr.pop("_roc_data", None)
        jr.pop("_pr_data", None)
        jr.pop("_history", None)
        json_results.append(jr)

    with open(os.path.join(run_dir, "ablation_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # --- Overlay plots ---
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    all_roc = OrderedDict()
    all_pr = OrderedDict()
    for r in sorted_results:
        lbl = f"{r['experiment_id']}: {r['experiment_name']} (AUROC={r['test']['auroc']:.4f})"
        if "_roc_data" in r:
            all_roc[lbl] = r["_roc_data"]
        lbl_pr = f"{r['experiment_id']}: {r['experiment_name']} (AUPRC={r['test']['auprc']:.4f})"
        if "_pr_data" in r:
            all_pr[lbl_pr] = r["_pr_data"]

    if all_roc:
        save_overlay_roc(all_roc, os.path.join(plots_dir, "roc_overlay.png"))
    if all_pr:
        prevalence = sorted_results[0]["test"]["prevalence"] if sorted_results else 0.02
        save_overlay_pr(all_pr, prevalence, os.path.join(plots_dir, "pr_overlay.png"))
    if sorted_results:
        save_metrics_bar_chart(sorted_results, os.path.join(plots_dir, "metrics_comparison.png"))

    # --- Human-readable summary ---
    lines = []
    lines.append("=" * 110)
    lines.append("  FEATURE ABLATION RESULTS (sorted by AUPRC)")
    lines.append("=" * 110)
    lines.append(
        f"  {'ID':>2s} {'Name':<24s} {'Type':<14s} "
        f"{'AUROC':>6s} {'dAUROC_v':>8s} {'dAUROC_f':>8s} "
        f"{'AUPRC':>6s} {'dAUPRC_v':>8s} {'dAUPRC_f':>8s} "
        f"{'Recall':>6s} {'Prec':>6s} {'F1':>6s} {'Ch':>3s} {'Time':>6s}"
    )
    lines.append(f"  {'-' * 106}")

    for r in sorted_results:
        t = r["test"]
        dt = r["train_time_sec"]
        time_str = f"{dt:.0f}s" if dt < 120 else f"{dt / 60:.1f}m"
        lines.append(
            f"  {r['experiment_id']:>2s} {r['experiment_name']:<24s} {r['experiment_type']:<14s} "
            f"{t['auroc']:>6.4f} {t['auroc'] - vitals_auroc:>+8.4f} {t['auroc'] - full_auroc:>+8.4f} "
            f"{t['auprc']:>6.4f} {t['auprc'] - vitals_auprc:>+8.4f} {t['auprc'] - full_auprc:>+8.4f} "
            f"{t['recall']:>6.4f} {t['precision']:>6.4f} {t['f1']:>6.4f} "
            f"{r['n_channels']:>3d} {time_str:>6s}"
        )

    lines.append("=" * 110)
    if vitals_result:
        lines.append(f"\n  Vitals-only baseline (A): AUROC={vitals_auroc:.4f}  AUPRC={vitals_auprc:.4f}")
    if full_result:
        lines.append(f"  Full model baseline  (F): AUROC={full_auroc:.4f}  AUPRC={full_auprc:.4f}")

    summary_text = "\n".join(lines)
    logger.info("\n" + summary_text)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    return sorted_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Feature Ablation")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment IDs to run (default: all A-O)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Model random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (default: timestamp)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing run directory")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Limit CPU threads to leave resources for desktop
    n_threads = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
    torch.set_num_threads(n_threads)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "feature_ablation", run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "experiments"), exist_ok=True)

    # Logging to file + console
    log_path = os.path.join(run_dir, "ablation.log")
    logger = logging.getLogger("ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    if args.experiments:
        exp_ids = args.experiments
        for eid in exp_ids:
            if eid not in EXPERIMENTS:
                logger.error(f"Unknown experiment ID: {eid}. Valid: {list(EXPERIMENTS.keys())}")
                sys.exit(1)
    else:
        exp_ids = list(EXPERIMENTS.keys())

    logger.info("=" * 70)
    logger.info("  EXPERIMENT 3: FEATURE ABLATION STUDY")
    logger.info(f"  Run ID:       {run_id}")
    logger.info(f"  Device:       {args.device}")
    logger.info(f"  Threads:      {n_threads}")
    logger.info(f"  Seed:         {args.seed}")
    logger.info(f"  Experiments:  {exp_ids}")
    logger.info(f"  Output:       {run_dir}")
    logger.info("=" * 70)

    # Load data once (all 10 features)
    logger.info("\nLoading PhysioNet data (all 10 features)...")
    episodes = load_physionet(features=ALL_FEATURES, min_length=MIN_LENGTH)
    logger.info(f"  {len(episodes)} episodes loaded")

    # Frozen split (same seed as experiment2)
    logger.info(f"\nSplitting (patient-level, seed={SPLIT_SEED})...")
    splits = patient_split(episodes, random_state=SPLIT_SEED)
    for name in ["train", "val", "test"]:
        n = len(splits[name])
        n_sep = sum(1 for e in splits[name] if e["label"] == 1)
        total = sum(e["signals"].shape[0] for e in splits[name])
        logger.info(f"  {name:5s}: {n} patients ({n_sep} septic), {total:,} timesteps")

    # Manifest
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "split_seed": SPLIT_SEED,
        "device": args.device,
        "n_threads": n_threads,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "min_episode_length": MIN_LENGTH,
        "model": "SepsisTransformer",
        "preprocessing": "Strategy B (causal forward-fill + training median + mask + delta for labs)",
        "all_features": list(ALL_FEATURES),
        "vitals": list(VITALS),
        "labs": list(LABS),
        "experiments": {eid: EXPERIMENTS[eid] for eid in exp_ids},
        "split_patients": {
            name: {"n_patients": len(splits[name]),
                   "n_septic": sum(1 for e in splits[name] if e["label"] == 1)}
            for name in ["train", "val", "test"]
        },
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Resume support (same incremental-save pattern as experiment2)
    completed_path = os.path.join(run_dir, "completed_results.json")
    all_results = []
    done_ids = set()
    if args.resume and os.path.exists(completed_path):
        with open(completed_path, "r") as f:
            all_results = json.load(f)
        done_ids = {r["experiment_id"] for r in all_results}
        logger.info(f"\n  Resuming: {len(done_ids)} experiments already completed: {done_ids}")

    # Sequential execution
    total_t0 = time.time()
    for exp_id in exp_ids:
        if exp_id in done_ids:
            logger.info(f"\n  Experiment {exp_id}: already completed, skipping")
            continue

        try:
            result = run_experiment(
                exp_id, EXPERIMENTS[exp_id], splits, args.seed, args.device,
                run_dir, logger,
            )
            if result is not None:
                all_results.append(result)

                # Incremental save (strip plot data)
                save_results = []
                for r in all_results:
                    sr = dict(r)
                    sr.pop("_roc_data", None)
                    sr.pop("_pr_data", None)
                    sr.pop("_history", None)
                    save_results.append(sr)
                with open(completed_path, "w") as f:
                    json.dump(save_results, f, indent=2)

        except Exception as e:
            logger.error(f"  FAILED: Experiment {exp_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    total_time = time.time() - total_t0
    logger.info(f"\n  All experiments completed in {total_time / 60:.1f} minutes")

    if all_results:
        generate_summary(all_results, run_dir, logger)
    else:
        logger.warning("  No results to summarize.")

    logger.info(f"\n  Results saved to: {run_dir}/")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
