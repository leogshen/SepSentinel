#!/usr/bin/env python
"""Experiment 5: Recall-Focused Loss Function Study

Scientific question: Can we materially improve patient-level early sepsis
recall without making false-alert burden unusable?

LABEL TIMING (PhysioNet/CinC 2019 Challenge)
=============================================
The SepsisLabel column is PRE-SHIFTED by the Challenge organizers.
SepsisLabel=1 begins 6 hours BEFORE clinical sepsis onset (t_sepsis).
SepsisLabel remains 1 from (t_sepsis - 6) through the rest of the stay.

Our pipeline uses these labels VERBATIM — zero additional shifting.

    onset_step = first timestep where SepsisLabel=1 = t_sepsis - 6
    Clinical onset (t_sepsis) = onset_step + 6

Lead times in this experiment are relative to CLINICAL onset (t_sepsis):
    "3h before onset"  → model alerts at or before step (onset_step + 3)
    "6h before onset"  → model alerts at or before step  onset_step
    "12h before onset" → model alerts at or before step (onset_step - 6)

A model that perfectly learns the training labels achieves:
    100% capture at ≤6h (labels turn on at t_sepsis - 6)
    0% capture at >6h   (no positive labels exist before onset_step)
Additional label spreading is justified ONLY for >6h lead times.

PHASES
======
Phase 0: Threshold sweep on current best model (Config I, no retraining).
         Evaluates timestep recall, patient recall, early-warning capture,
         and false-alert burden across a range of thresholds.

Phase 1: Controlled loss-function study. Same architecture, features, splits,
         preprocessing, and seeds — only the loss function changes.
         Configs: standard BCE, Weighted BCE (pw=2,5,10), Focal (gamma=2).

Usage:
    python experiment5_recall_study.py                    # full run
    python experiment5_recall_study.py --phase 0          # threshold sweep only
    python experiment5_recall_study.py --phase 1          # loss study only
    python experiment5_recall_study.py --seeds 42         # single seed
    python experiment5_recall_study.py --device cuda
"""

import argparse
import glob
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import collate_fn
from sepsentinel.model_b.training import Trainer, SequenceDataset
from sepsentinel.model_b.transformer import SepsisTransformer

from experiment2_imputation import (
    find_best_f1_threshold,
    FEATURES as ALL_FEATURES,
    N_VITALS,
    SPLIT_SEED, EPOCHS, BATCH_SIZE, LR,
    PATIENCE, MIN_DELTA, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
)
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS as EXP3_CONFIGS

# ============================================================
# Configuration
# ============================================================

CONFIG_I_FEATURES = EXP3_CONFIGS["I"]["features"]
SEEDS = [42, 123, 456]
MIN_LENGTH = 6
RESULTS_DIR = "results/experiment5_recall"

LEAD_HOURS = [3, 6, 12]
RECALL_TARGETS = [0.50, 0.70, 0.80, 0.85, 0.90]

LOSS_CONFIGS = OrderedDict([
    ("baseline",  {"type": "bce",   "pos_weight": None,  "description": "Standard BCE, no class weighting"}),
    ("wbce_2",    {"type": "bce",   "pos_weight": 2.0,   "description": "BCE with pos_weight=2"}),
    ("wbce_5",    {"type": "bce",   "pos_weight": 5.0,   "description": "BCE with pos_weight=5"}),
    ("wbce_10",   {"type": "bce",   "pos_weight": 10.0,  "description": "BCE with pos_weight=10"}),
    ("focal_g2",  {"type": "focal", "gamma": 2.0, "alpha": 0.25,
                   "description": "Focal loss (gamma=2, alpha=0.25)"}),
])


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma controls focusing: gamma=0 recovers BCE, gamma=2 (standard)
    strongly down-weights easy-to-classify examples so training focuses
    on the hard cases — which, in an imbalanced dataset, are
    disproportionately the rare positive class.

    alpha balances positive vs negative contribution:
        alpha_t = alpha   when target=1 (positive)
        alpha_t = 1-alpha when target=0 (negative)

    We use alpha=0.25 following Lin et al. (2017, RetinaNet).  With
    gamma=2, the focal term already suppresses the easy-negative
    majority, so a low alpha avoids over-correcting.  The original
    paper found alpha=0.25 + gamma=2 optimal across imbalance ratios
    from 1:10 to 1:100.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return focal_weight * bce


# ============================================================
# Modified Trainer with pluggable loss
# ============================================================

class RecallTrainer(Trainer):
    """Trainer that accepts a custom loss function."""

    def __init__(self, model, device="cpu", checkpoint_dir="models/checkpoints",
                 loss_fn=None, pos_weight=None):
        super().__init__(model, device=device, checkpoint_dir=checkpoint_dir,
                         pos_weight=pos_weight)
        if loss_fn is not None:
            self.criterion = loss_fn


# ============================================================
# Per-Patient Prediction Collection
# ============================================================

def collect_patient_predictions(model, preprocessed_data, raw_episodes_map, device):
    """Collect per-patient predictions with onset information.

    collate_fn sorts each batch by length (descending), so patient metadata
    must be paired via the same per-batch stable sort — pairing by sequential
    dataset order attaches probs to the wrong patients (fixed 2026-08-19).
    """
    loader = DataLoader(SequenceDataset(preprocessed_data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

    patient_results = []
    start = 0

    with torch.no_grad():
        model.eval()
        for signals, labels, lengths, mask in loader:
            batch_items = preprocessed_data[start:start + len(lengths)]
            start += len(lengths)
            sorted_items = sorted(batch_items, key=lambda x: x["length"], reverse=True)

            signals = signals.to(device)
            logits = model(signals, lengths)
            probs = torch.sigmoid(logits)

            for i in range(len(lengths)):
                sl = lengths[i].item()
                item = sorted_items[i]
                assert item["length"] == sl, "batch sort replication mismatch"
                pid = item["patient_id"]
                raw_ep = raw_episodes_map[pid]

                patient_results.append({
                    "patient_id": pid,
                    "label": raw_ep["label"],
                    "onset_step": raw_ep["onset_step"],
                    "probs": probs[i, :sl].cpu().numpy(),
                    "labels": labels[i, :sl].numpy(),
                    "length": sl,
                })

    return patient_results


# ============================================================
# Timestep-Level Metrics
# ============================================================

def compute_timestep_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "specificity": float(specificity),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ============================================================
# Patient-Level Early Warning Metrics
# ============================================================

def compute_early_warning_metrics(patient_results, threshold):
    """Patient-level early warning analysis.

    Lead time = t_sepsis - t_first_alarm (hours before clinical onset).
    Since data is hourly and t_sepsis = onset_step + 6, lead = onset_step + 6 - first_alarm.
    """
    septic = [p for p in patient_results if p["label"] == 1]
    healthy = [p for p in patient_results if p["label"] == 0]

    n_septic = len(septic)
    n_healthy = len(healthy)
    caught_any = 0
    lead_times = []
    caught_by_lead = {h: 0 for h in LEAD_HOURS}

    for pat in septic:
        preds = (pat["probs"] >= threshold).astype(int)
        onset = pat["onset_step"]
        t_sepsis = onset + 6

        pos_indices = np.where(preds == 1)[0]
        if len(pos_indices) > 0:
            first_alarm = int(pos_indices[0])
            caught_any += 1
            lead = t_sepsis - first_alarm
            lead_times.append(float(lead))

            for h in LEAD_HOURS:
                if first_alarm <= t_sepsis - h:
                    caught_by_lead[h] += 1

    patient_recall = caught_any / max(n_septic, 1)

    # Healthy patient false-alert burden
    alerts_per_patient = []
    patients_with_alert = 0
    total_healthy_hours = 0
    total_healthy_alerts = 0

    for pat in healthy:
        preds = (pat["probs"] >= threshold).astype(int)
        n_alerts = int(preds.sum())
        alerts_per_patient.append(n_alerts)
        if n_alerts > 0:
            patients_with_alert += 1
        total_healthy_alerts += n_alerts
        total_healthy_hours += pat["length"]

    alerts_per_day = (total_healthy_alerts / max(total_healthy_hours, 1)) * 24

    return {
        "threshold": float(threshold),
        "n_septic": n_septic,
        "n_healthy": n_healthy,
        "patient_recall": float(patient_recall),
        "caught_any": caught_any,
        "median_lead_time_h": float(np.median(lead_times)) if lead_times else None,
        "mean_lead_time_h": float(np.mean(lead_times)) if lead_times else None,
        "std_lead_time_h": float(np.std(lead_times)) if lead_times else None,
        "min_lead_time_h": float(np.min(lead_times)) if lead_times else None,
        "max_lead_time_h": float(np.max(lead_times)) if lead_times else None,
        "lead_time_distribution": lead_times,
        "capture_rate_by_lead_hour": {
            str(h): float(caught_by_lead[h] / max(n_septic, 1)) for h in LEAD_HOURS
        },
        "healthy_patients_with_alerts": patients_with_alert,
        "healthy_false_alert_rate": float(patients_with_alert / max(n_healthy, 1)),
        "mean_alerts_per_healthy_patient": float(np.mean(alerts_per_patient)) if alerts_per_patient else 0.0,
        "median_alerts_per_healthy_patient": float(np.median(alerts_per_patient)) if alerts_per_patient else 0.0,
        "alerts_per_patient_day": float(alerts_per_day),
        "total_healthy_alerts": total_healthy_alerts,
        "total_healthy_hours": total_healthy_hours,
    }


# ============================================================
# Threshold Sweep Utilities
# ============================================================

def find_threshold_for_recall(y_true, y_prob, target_recall):
    """Find lowest threshold achieving at least target_recall."""
    best_t = 0.005
    best_diff = float("inf")

    for t in np.arange(0.005, 0.995, 0.005):
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        rec = tp / max(tp + fn, 1)
        diff = abs(rec - target_recall)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)

    return best_t


def threshold_sweep(patient_results, y_true, y_prob, logger):
    """Comprehensive threshold sweep with timestep and patient-level metrics."""
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    logger.info(f"\n{'='*140}")
    logger.info(f"  THRESHOLD SWEEP  (reference: AUROC={auroc:.4f}, AUPRC={auprc:.4f})")
    logger.info(f"{'='*140}")

    best_f1_thresh = find_best_f1_threshold(y_true, y_prob)

    sweep_points = [("best_F1", best_f1_thresh)]
    for target in RECALL_TARGETS:
        t = find_threshold_for_recall(y_true, y_prob, target)
        sweep_points.append((f"recall_{target:.2f}", t))

    header = (
        f"  {'Label':>12s} | {'Thresh':>6s} | "
        f"{'TS_Rec':>6s} | {'Pt_Rec':>6s} | {'Prec':>6s} | "
        f"{'Spec':>6s} | {'F1':>6s} | {'FPR':>6s} | "
        f"{'Cap3h':>5s} | {'Cap6h':>5s} | {'Cap12h':>6s} | "
        f"{'FAlrt/d':>7s} | {'MedLead':>7s}"
    )
    logger.info(header)
    logger.info(f"  {'-'*135}")

    results = []
    for label, thresh in sweep_points:
        ts = compute_timestep_metrics(y_true, y_prob, thresh)
        ew = compute_early_warning_metrics(patient_results, thresh)

        combined = {
            "label": label,
            "auroc": auroc,
            "auprc": auprc,
            **ts,
            **{f"ew_{k}": v for k, v in ew.items() if k != "lead_time_distribution"},
        }
        results.append(combined)

        med_str = f"{ew['median_lead_time_h']:.1f}h" if ew["median_lead_time_h"] is not None else "N/A"
        cap = ew["capture_rate_by_lead_hour"]

        logger.info(
            f"  {label:>12s} | {thresh:>6.3f} | "
            f"{ts['recall']:>6.3f} | {ew['patient_recall']:>6.3f} | "
            f"{ts['precision']:>6.3f} | {ts['specificity']:>6.3f} | "
            f"{ts['f1']:>6.3f} | {ts['fpr']:>6.3f} | "
            f"{cap['3']:>5.3f} | {cap['6']:>5.3f} | {cap['12']:>6.3f} | "
            f"{ew['alerts_per_patient_day']:>7.2f} | {med_str:>7s}"
        )

    return results


# ============================================================
# Phase 0: Threshold Sweep on Existing Model
# ============================================================

def find_existing_checkpoint():
    """Find most recent Config I checkpoint from experiment 3."""
    pattern = os.path.join(
        "results", "feature_ablation", "*", "experiments",
        "I_all_minus_creatinine", "checkpoints", "best_model.pt",
    )
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def build_raw_map(splits):
    raw_map = {}
    for split in ["train", "val", "test"]:
        for ep in splits[split]:
            raw_map[ep["patient_id"]] = ep
    return raw_map


def phase0_threshold_sweep(splits_raw, device, run_dir, logger):
    logger.info(f"\n{'='*140}")
    logger.info("  PHASE 0: THRESHOLD SWEEP ON CURRENT BEST MODEL (Config I, auto pos_weight)")
    logger.info(f"{'='*140}")

    phase_dir = os.path.join(run_dir, "phase0_sweep")
    os.makedirs(phase_dir, exist_ok=True)

    preprocessor = AblationPreprocessor(list(ALL_FEATURES), CONFIG_I_FEATURES)
    preprocessor.fit(splits_raw["train"])

    train_data = preprocessor.transform(splits_raw["train"])
    val_data = preprocessor.transform(splits_raw["val"])
    test_data = preprocessor.transform(splits_raw["test"])

    # Auto pos_weight (same as experiments 2-4)
    train_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    auto_pw = float(n_neg / max(n_pos, 1))
    logger.info(f"  Auto pos_weight (class ratio): {auto_pw:.1f}")

    # Try existing checkpoint
    ckpt_path = find_existing_checkpoint()

    if ckpt_path:
        logger.info(f"  Loading existing checkpoint: {ckpt_path}")
        model = SepsisTransformer(input_dim=preprocessor.n_channels)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model = model.to(device)
    else:
        logger.info("  No existing checkpoint found — training baseline (seed=42, auto pos_weight)...")
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = SepsisTransformer(input_dim=preprocessor.n_channels)
        ckpt_dir = os.path.join(phase_dir, "baseline_checkpoint")
        trainer = Trainer(model, device=device, checkpoint_dir=ckpt_dir, pos_weight=auto_pw)
        trainer.fit(train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    patience=PATIENCE, min_delta=MIN_DELTA,
                    scheduler_factor=SCHEDULER_FACTOR, scheduler_patience=SCHEDULER_PATIENCE)

    raw_map = build_raw_map(splits_raw)

    logger.info("  Collecting test-set predictions...")
    patient_preds = collect_patient_predictions(model, test_data, raw_map, device)

    y_true = np.concatenate([p["labels"] for p in patient_preds])
    y_prob = np.concatenate([p["probs"] for p in patient_preds])

    # --- Sweep ---
    sweep_results = threshold_sweep(patient_preds, y_true, y_prob, logger)

    with open(os.path.join(phase_dir, "threshold_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    # --- Lead-time distribution plot ---
    best_f1_thresh = find_best_f1_threshold(y_true, y_prob)
    ew_at_f1 = compute_early_warning_metrics(patient_preds, best_f1_thresh)

    if ew_at_f1["lead_time_distribution"]:
        leads = ew_at_f1["lead_time_distribution"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(leads, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(np.median(leads), color="red", ls="--", lw=2,
                   label=f"Median = {np.median(leads):.1f}h")
        ax.axvline(6, color="orange", ls=":", lw=2,
                   label="6h (label onset = clinical − 6h)")
        ax.set_xlabel("Lead Time Before Clinical Onset (hours)")
        ax.set_ylabel("Number of Septic Patients")
        ax.set_title(f"First-Alert Lead Time (threshold={best_f1_thresh:.3f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(phase_dir, "lead_time_distribution.png"), dpi=150)
        plt.close(fig)

    # --- Tradeoff plots ---
    thresholds_dense = np.arange(0.01, 0.99, 0.01)
    ts_recs, pt_recs, precs, f1s, alert_rates = [], [], [], [], []

    for t in thresholds_dense:
        tsm = compute_timestep_metrics(y_true, y_prob, t)
        ewm = compute_early_warning_metrics(patient_preds, t)
        ts_recs.append(tsm["recall"])
        pt_recs.append(ewm["patient_recall"])
        precs.append(tsm["precision"])
        f1s.append(tsm["f1"])
        alert_rates.append(ewm["alerts_per_patient_day"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(thresholds_dense, ts_recs, label="Timestep Recall", lw=2)
    ax.plot(thresholds_dense, pt_recs, label="Patient Recall", lw=2)
    ax.plot(thresholds_dense, precs, label="Precision", lw=2)
    ax.plot(thresholds_dense, f1s, label="F1", lw=2)
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Metrics vs Threshold"); ax.legend()
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    ax = axes[1]
    ax.plot(ts_recs, precs, lw=2, color="tab:blue")
    ax.set_xlabel("Timestep Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Tradeoff")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    ax = axes[2]
    ax.plot(pt_recs, alert_rates, lw=2, color="tab:red")
    ax.set_xlabel("Patient Recall"); ax.set_ylabel("False Alerts / Healthy Patient-Day")
    ax.set_title("Recall vs False-Alert Burden")
    ax.set_xlim([0, 1.02])

    fig.tight_layout()
    fig.savefig(os.path.join(phase_dir, "tradeoff_plots.png"), dpi=150)
    plt.close(fig)

    logger.info(f"\n  Phase 0 artifacts saved to {phase_dir}/")
    return sweep_results, preprocessor


# ============================================================
# Phase 1: Loss Function Study
# ============================================================

def phase1_loss_study(splits_raw, preprocessor, sweep_results, device,
                      run_dir, seeds, logger):
    logger.info(f"\n{'='*140}")
    logger.info("  PHASE 1: LOSS FUNCTION STUDY")
    logger.info(f"{'='*140}")

    phase_dir = os.path.join(run_dir, "phase1_loss")
    os.makedirs(phase_dir, exist_ok=True)

    train_data = preprocessor.transform(splits_raw["train"])
    val_data = preprocessor.transform(splits_raw["val"])
    test_data = preprocessor.transform(splits_raw["test"])

    train_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    auto_pw = float(n_neg / max(n_pos, 1))
    logger.info(f"  Auto pos_weight (class ratio): {auto_pw:.1f}")
    logger.info(f"  Note: Phase 0 model used pos_weight={auto_pw:.1f}. Phase 1 'baseline' uses pos_weight=None.")

    raw_map = build_raw_map(splits_raw)

    # Phase 0 recall-targeted thresholds (for cross-model comparison)
    phase0_thresholds = {}
    if sweep_results:
        for r in sweep_results:
            if r["label"].startswith("recall_"):
                phase0_thresholds[r["label"]] = r["threshold"]

    all_results = []
    results_path = os.path.join(phase_dir, "all_results.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
        done = {(r["loss_config"], r["seed"]) for r in all_results}
        logger.info(f"  Resuming: {len(done)} runs already completed")
    else:
        done = set()

    for loss_name, loss_cfg in LOSS_CONFIGS.items():
        logger.info(f"\n  {'='*110}")
        logger.info(f"  Loss: {loss_name} — {loss_cfg['description']}")
        logger.info(f"  {'='*110}")

        for seed in seeds:
            if (loss_name, seed) in done:
                logger.info(f"    Seed {seed}: already completed, skipping")
                continue

            logger.info(f"\n    Seed {seed}:")

            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = SepsisTransformer(input_dim=preprocessor.n_channels)
            ckpt_dir = os.path.join(phase_dir, loss_name, f"seed_{seed}")

            if loss_cfg["type"] == "focal":
                loss_fn = FocalLoss(gamma=loss_cfg["gamma"], alpha=loss_cfg["alpha"])
                trainer = RecallTrainer(model, device=device, checkpoint_dir=ckpt_dir,
                                       loss_fn=loss_fn)
                effective_pw = None
            else:
                pw = loss_cfg["pos_weight"]
                trainer = Trainer(model, device=device, checkpoint_dir=ckpt_dir,
                                 pos_weight=pw)
                effective_pw = pw

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

            # Per-patient predictions on test
            patient_preds = collect_patient_predictions(model, test_data, raw_map, device)
            y_true = np.concatenate([p["labels"] for p in patient_preds])
            y_prob = np.concatenate([p["probs"] for p in patient_preds])

            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))

            # --- A. Evaluate at best-F1 threshold ---
            best_f1_thresh = find_best_f1_threshold(y_true, y_prob)
            ts_f1 = compute_timestep_metrics(y_true, y_prob, best_f1_thresh)
            ew_f1 = compute_early_warning_metrics(patient_preds, best_f1_thresh)
            ew_f1_clean = {k: v for k, v in ew_f1.items() if k != "lead_time_distribution"}

            # --- B. Evaluate at Phase 0 recall-targeted thresholds ---
            phase0_evals = {}
            for lbl, thr in phase0_thresholds.items():
                ts_ = compute_timestep_metrics(y_true, y_prob, thr)
                ew_ = compute_early_warning_metrics(patient_preds, thr)
                ew_.pop("lead_time_distribution", None)
                phase0_evals[lbl] = {"timestep": ts_, "early_warning": ew_}

            # --- B2. Evaluate at this model's own recall thresholds ---
            own_recall_evals = {}
            for target in RECALL_TARGETS:
                t = find_threshold_for_recall(y_true, y_prob, target)
                ts_ = compute_timestep_metrics(y_true, y_prob, t)
                ew_ = compute_early_warning_metrics(patient_preds, t)
                ew_.pop("lead_time_distribution", None)
                own_recall_evals[f"recall_{target:.2f}"] = {
                    "threshold": float(t),
                    "timestep": ts_,
                    "early_warning": ew_,
                }

            result = {
                "loss_config": loss_name,
                "loss_params": {k: v for k, v in loss_cfg.items() if k != "description"},
                "seed": seed,
                "effective_pos_weight": effective_pw,
                "auto_pos_weight": auto_pw,
                "train_time_sec": round(train_time, 1),
                "best_epoch": best_epoch,
                "auroc": auroc,
                "auprc": auprc,
                "best_f1_threshold": best_f1_thresh,
                "at_best_f1": {"timestep": ts_f1, "early_warning": ew_f1_clean},
                "at_phase0_thresholds": phase0_evals,
                "at_own_recall_thresholds": own_recall_evals,
            }

            all_results.append(result)

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

            logger.info(f"      AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                        f"F1_thresh={best_f1_thresh:.3f}  epoch={best_epoch}")
            logger.info(f"      @best_F1: ts_rec={ts_f1['recall']:.3f}  "
                        f"pt_rec={ew_f1['patient_recall']:.3f}  "
                        f"prec={ts_f1['precision']:.3f}  "
                        f"F1={ts_f1['f1']:.3f}  "
                        f"alerts/d={ew_f1['alerts_per_patient_day']:.2f}  "
                        f"med_lead={ew_f1['median_lead_time_h']}")

    return all_results


# ============================================================
# Summary Generation
# ============================================================

def generate_summary(all_results, sweep_results, run_dir, logger):
    logger.info(f"\n\n{'='*140}")
    logger.info("  EXPERIMENT 5: FINAL SUMMARY")
    logger.info(f"{'='*140}")

    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    configs = list(LOSS_CONFIGS.keys())

    # === Table 1: At best-F1 threshold ===
    logger.info(f"\n  TABLE 1: Loss Function Comparison at Best-F1 Threshold (mean ± std across seeds)")
    logger.info(f"  {'-'*145}")
    logger.info(
        f"  {'Loss':>12s} | {'AUROC':>14s} | {'AUPRC':>14s} | "
        f"{'F1':>14s} | {'TS_Recall':>14s} | {'Pt_Recall':>14s} | "
        f"{'Prec':>14s} | {'Alerts/d':>14s} | {'MedLead':>9s}"
    )
    logger.info(f"  {'-'*145}")

    summary_table = []

    for cfg in configs:
        runs = [r for r in all_results if r["loss_config"] == cfg]
        if not runs:
            continue

        aurocs = [r["auroc"] for r in runs]
        auprcs = [r["auprc"] for r in runs]
        f1s = [r["at_best_f1"]["timestep"]["f1"] for r in runs]
        ts_recs = [r["at_best_f1"]["timestep"]["recall"] for r in runs]
        pt_recs = [r["at_best_f1"]["early_warning"]["patient_recall"] for r in runs]
        precs = [r["at_best_f1"]["timestep"]["precision"] for r in runs]
        alerts = [r["at_best_f1"]["early_warning"]["alerts_per_patient_day"] for r in runs]
        leads = [r["at_best_f1"]["early_warning"]["median_lead_time_h"] for r in runs
                 if r["at_best_f1"]["early_warning"]["median_lead_time_h"] is not None]

        def fmt(vals):
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"

        lead_str = f"{np.mean(leads):.1f}h" if leads else "N/A"

        logger.info(
            f"  {cfg:>12s} | {fmt(aurocs):>14s} | {fmt(auprcs):>14s} | "
            f"{fmt(f1s):>14s} | {fmt(ts_recs):>14s} | {fmt(pt_recs):>14s} | "
            f"{fmt(precs):>14s} | {fmt(alerts):>14s} | {lead_str:>9s}"
        )

        summary_table.append({
            "loss_config": cfg,
            "auroc_mean": float(np.mean(aurocs)), "auroc_std": float(np.std(aurocs)),
            "auprc_mean": float(np.mean(auprcs)), "auprc_std": float(np.std(auprcs)),
            "f1_mean": float(np.mean(f1s)),
            "ts_recall_mean": float(np.mean(ts_recs)),
            "pt_recall_mean": float(np.mean(pt_recs)),
            "precision_mean": float(np.mean(precs)),
            "alerts_per_day_mean": float(np.mean(alerts)),
            "median_lead_mean": float(np.mean(leads)) if leads else None,
            "n_seeds": len(runs),
        })

    # === Table 2: Recall-targeted comparison ===
    logger.info(f"\n\n  TABLE 2: Recall-Targeted Comparison (each model's own threshold, mean across seeds)")
    logger.info(f"  {'-'*145}")

    for target in RECALL_TARGETS:
        target_key = f"recall_{target:.2f}"
        logger.info(f"\n  Target timestep recall ≥ {target:.0%}:")
        logger.info(
            f"  {'Loss':>12s} | {'Thresh':>6s} | {'TS_Rec':>6s} | "
            f"{'Pt_Rec':>6s} | {'Prec':>6s} | {'Spec':>6s} | "
            f"{'F1':>6s} | {'Cap3h':>5s} | {'Cap6h':>5s} | {'Cap12h':>6s} | "
            f"{'FAlrt/d':>7s} | {'MedLead':>7s}"
        )

        for cfg in configs:
            runs = [r for r in all_results if r["loss_config"] == cfg]
            if not runs:
                continue

            vals = {"thresh": [], "ts_rec": [], "pt_rec": [], "prec": [],
                    "spec": [], "f1_": [], "cap3": [], "cap6": [], "cap12": [],
                    "alerts": [], "med_lead": []}

            for r in runs:
                rr = r["at_own_recall_thresholds"].get(target_key)
                if rr is None:
                    continue
                vals["thresh"].append(rr["threshold"])
                vals["ts_rec"].append(rr["timestep"]["recall"])
                vals["pt_rec"].append(rr["early_warning"]["patient_recall"])
                vals["prec"].append(rr["timestep"]["precision"])
                vals["spec"].append(rr["timestep"]["specificity"])
                vals["f1_"].append(rr["timestep"]["f1"])
                vals["cap3"].append(rr["early_warning"]["capture_rate_by_lead_hour"]["3"])
                vals["cap6"].append(rr["early_warning"]["capture_rate_by_lead_hour"]["6"])
                vals["cap12"].append(rr["early_warning"]["capture_rate_by_lead_hour"]["12"])
                vals["alerts"].append(rr["early_warning"]["alerts_per_patient_day"])
                ml = rr["early_warning"]["median_lead_time_h"]
                if ml is not None:
                    vals["med_lead"].append(ml)

            if not vals["thresh"]:
                continue

            def mn(lst):
                return np.mean(lst) if lst else 0.0

            ml_str = f"{mn(vals['med_lead']):.1f}h" if vals["med_lead"] else "N/A"

            logger.info(
                f"  {cfg:>12s} | {mn(vals['thresh']):>6.3f} | "
                f"{mn(vals['ts_rec']):>6.3f} | {mn(vals['pt_rec']):>6.3f} | "
                f"{mn(vals['prec']):>6.3f} | {mn(vals['spec']):>6.3f} | "
                f"{mn(vals['f1_']):>6.3f} | {mn(vals['cap3']):>5.3f} | "
                f"{mn(vals['cap6']):>5.3f} | {mn(vals['cap12']):>6.3f} | "
                f"{mn(vals['alerts']):>7.2f} | {ml_str:>7s}"
            )

    # === Answers to the 5 summary questions ===
    logger.info(f"\n\n{'='*140}")
    logger.info("  ANSWERS TO PRIMARY QUESTIONS")
    logger.info(f"{'='*140}")

    # Q1
    max_rec = max(summary_table, key=lambda x: x["pt_recall_mean"])
    logger.info(f"""
  1. BEST MODEL FOR MAXIMUM RECALL
     {max_rec['loss_config']}: patient recall = {max_rec['pt_recall_mean']:.3f} at best-F1 threshold
     (Higher recall achievable by lowering threshold — see Table 2)
""")

    # Q2
    logger.info("  2. BEST MODEL FOR CLINICALLY REASONABLE RECALL/PRECISION TRADEOFF")
    for row in sorted(summary_table, key=lambda x: -x["pt_recall_mean"]):
        logger.info(f"     {row['loss_config']:>12s}:  pt_recall={row['pt_recall_mean']:.3f}  "
                    f"precision={row['precision_mean']:.3f}  "
                    f"alerts/day={row['alerts_per_day_mean']:.2f}  "
                    f"AUROC={row['auroc_mean']:.4f}")
    logger.info("")

    # Q3
    baseline_auroc = next((r["auroc_mean"] for r in summary_table if r["loss_config"] == "baseline"), 0)
    baseline_auprc = next((r["auprc_mean"] for r in summary_table if r["loss_config"] == "baseline"), 0)
    logger.info("  3. DID LOSS REWEIGHTING IMPROVE DISCRIMINATION OR MERELY CHANGE CALIBRATION?")
    logger.info(f"     AUROC and AUPRC measure discrimination (threshold-independent).")
    logger.info(f"     If Δ AUROC ≈ 0, the loss only changed calibration (which threshold is optimal).")
    for row in summary_table:
        da = row["auroc_mean"] - baseline_auroc
        dp = row["auprc_mean"] - baseline_auprc
        logger.info(f"     {row['loss_config']:>12s}:  AUROC={row['auroc_mean']:.4f} (d={da:+.4f})  "
                    f"AUPRC={row['auprc_mean']:.4f} (d={dp:+.4f})")
    logger.info("")

    # Q4
    logger.info("  4. IS PATIENT-LEVEL RECALL SUBSTANTIALLY BETTER THAN TIMESTEP-LEVEL RECALL?")
    for row in summary_table:
        gap = row["pt_recall_mean"] - row["ts_recall_mean"]
        logger.info(f"     {row['loss_config']:>12s}:  ts_recall={row['ts_recall_mean']:.3f}  "
                    f"pt_recall={row['pt_recall_mean']:.3f}  gap={gap:+.3f}")
    logger.info("")

    # Q5
    logger.info("  5. IS LABEL SPREADING STILL SCIENTIFICALLY JUSTIFIED?")
    logger.info("     The CinC 2019 labels already include a 6h pre-onset window.")
    logger.info("     SepsisLabel=1 starts at t_sepsis - 6 and continues to discharge.")
    logger.info("")
    if sweep_results:
        best_f1_r = next((r for r in sweep_results if r["label"] == "best_F1"), None)
        if best_f1_r:
            cap6 = best_f1_r.get("ew_capture_rate_by_lead_hour", {}).get("6", 0)
            cap12 = best_f1_r.get("ew_capture_rate_by_lead_hour", {}).get("12", 0)
            logger.info(f"     Phase 0 model at best-F1: 6h capture={cap6:.1%}, 12h capture={cap12:.1%}")
            if cap6 > 0.90:
                logger.info("     -> 6h capture is saturated. Label spreading to 12h+ may be justified")
                logger.info("       if we want the model to alert even earlier than the 6h window.")
            elif cap6 > 0.70:
                logger.info("     -> 6h capture is moderate. Loss reweighting / threshold tuning should")
                logger.info("       be explored first; label spreading is a second-order improvement.")
            else:
                logger.info("     -> 6h capture is low. Focus on improving recall within the existing")
                logger.info("       6h window before extending it. Label spreading is not yet justified.")
    logger.info("")

    with open(os.path.join(summary_dir, "summary_table.json"), "w") as f:
        json.dump(summary_table, f, indent=2)

    logger.info(f"  Summary saved to {summary_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Recall-Focused Loss Study")
    parser.add_argument("--phase", type=int, default=None, choices=[0, 1],
                        help="Run only phase 0 or 1")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing run directory")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    n_threads = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
    torch.set_num_threads(n_threads)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "experiment5.log")
    logger = logging.getLogger("exp5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 140)
    logger.info("  EXPERIMENT 5: RECALL-FOCUSED LOSS FUNCTION STUDY")
    logger.info(f"  Run ID:          {run_id}")
    logger.info(f"  Device:          {args.device}")
    logger.info(f"  Seeds:           {args.seeds}")
    logger.info(f"  Features:        Config I (9 features, exclude creatinine)")
    logger.info(f"  Preprocessing:   Strategy B (causal ffill + median + mask + delta)")
    logger.info(f"  Architecture:    SepsisTransformer (d_model=64, nhead=4, layers=2)")
    logger.info(f"  Loss configs:    {list(LOSS_CONFIGS.keys())}")
    logger.info(f"  Recall targets:  {RECALL_TARGETS}")
    logger.info(f"  Lead-time hrs:   {LEAD_HOURS}")
    logger.info("=" * 140)

    # Load data
    logger.info("\nLoading PhysioNet data...")
    episodes = load_physionet(features=list(ALL_FEATURES), min_length=MIN_LENGTH)
    logger.info(f"  {len(episodes)} episodes loaded")

    # Split
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
        "seeds": args.seeds,
        "split_seed": SPLIT_SEED,
        "device": args.device,
        "features": CONFIG_I_FEATURES,
        "n_features": len(CONFIG_I_FEATURES),
        "preprocessing": "Strategy B (causal forward-fill + training median + mask + delta for labs)",
        "model": "SepsisTransformer (d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.2)",
        "loss_configs": {k: {kk: vv for kk, vv in v.items()}
                         for k, v in LOSS_CONFIGS.items()},
        "recall_targets": RECALL_TARGETS,
        "lead_hours": LEAD_HOURS,
        "label_timing": {
            "source": "PhysioNet/CinC 2019 Challenge",
            "pre_shift": "SepsisLabel=1 starts 6h before clinical onset (t_sepsis)",
            "our_modification": "None — labels used verbatim from the dataset",
            "onset_step_meaning": "First timestep where SepsisLabel=1 = t_sepsis - 6",
        },
        "hyperparameters": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "patience": PATIENCE, "min_delta": MIN_DELTA,
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
        },
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Phase 0
    sweep_results = None
    preprocessor = None

    if args.phase is None or args.phase == 0:
        sweep_results, preprocessor = phase0_threshold_sweep(
            splits, args.device, run_dir, logger)

    # Phase 1
    if args.phase is None or args.phase == 1:
        if preprocessor is None:
            preprocessor = AblationPreprocessor(list(ALL_FEATURES), CONFIG_I_FEATURES)
            preprocessor.fit(splits["train"])

        if sweep_results is None:
            sweep_path = os.path.join(run_dir, "phase0_sweep", "threshold_sweep.json")
            if os.path.exists(sweep_path):
                with open(sweep_path, "r") as f:
                    sweep_results = json.load(f)
            else:
                sweep_results = []

        all_results = phase1_loss_study(
            splits, preprocessor, sweep_results, args.device,
            run_dir, args.seeds, logger,
        )

        generate_summary(all_results, sweep_results, run_dir, logger)

    logger.info(f"\n  All results saved to {run_dir}/")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
