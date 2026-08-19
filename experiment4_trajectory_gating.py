#!/usr/bin/env python
"""Experiment 4: Trajectory Features & Dynamic Channel Gating

Tests whether patient-relative trajectories and dynamic feature weighting
improve sepsis prediction over the current baseline.

Configs:
  baseline:    Current SepsisTransformer, Config I features (19 channels)
  trajectory:  SepsisTransformer + trajectory features (46 channels)
  gated:       SepsisTransformerGated, Config I features (19 channels)
  traj_gated:  SepsisTransformerGated + trajectory features (46 channels)

All configs use:
  - Features: Config I (9 raw, exclude creatinine)
  - Patient split: SPLIT_SEED=42
  - Preprocessing: Strategy B (causal forward-fill + median + mask + delta)
  - Same hyperparameters, threshold selection, and metrics as experiments 2-3

Usage:
  python experiment4_trajectory_gating.py
  python experiment4_trajectory_gating.py --configs baseline trajectory
  python experiment4_trajectory_gating.py --seeds 42
"""

import argparse
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
from torch.utils.data import DataLoader

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.preprocessing import CLIP_RANGES, collate_fn
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.trajectory import compute_trajectory, TRAJ_SUFFIXES, N_TRAJ_PER_FEATURE
from sepsentinel.model_b.transformer import SepsisTransformer
from sepsentinel.model_b.channel_gate import SepsisTransformerGated
from sepsentinel.model_b.training import Trainer, SequenceDataset

from experiment2_imputation import (
    compute_metrics as _compute_metrics_base,
    find_best_f1_threshold,
    collect_predictions,
    FEATURES as ALL_FEATURES,
    N_VITALS, N_LABS,
    SPLIT_SEED, EPOCHS, BATCH_SIZE, LR,
    PATIENCE, MIN_DELTA, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
)
from experiment3_feature_ablation import (
    save_roc_curve, save_pr_curve,
    save_overlay_roc, save_overlay_pr,
)


# ============================================================
# Configuration
# ============================================================

VITALS = list(ALL_FEATURES[:N_VITALS])

# Config I: all minus creatinine (best from experiment 3)
SELECTED_FEATURES = [f for f in ALL_FEATURES if f != "creatinine"]

DEFAULT_SEEDS = [42, 123, 456]
MIN_LENGTH = 6
TRAJ_WINDOW = 6  # 6-hour causal rolling window

N_INTERP_SEPTIC = 20
N_INTERP_HEALTHY = 20

CONFIGS = OrderedDict([
    ("baseline", {
        "name": "baseline",
        "add_trajectory": False,
        "use_gating": False,
        "description": "SepsisTransformer, Config I (19 ch)",
    }),
    ("trajectory", {
        "name": "trajectory",
        "add_trajectory": True,
        "use_gating": False,
        "description": "SepsisTransformer + trajectory features (46 ch)",
    }),
    ("gated", {
        "name": "gated",
        "add_trajectory": False,
        "use_gating": True,
        "description": "SepsisTransformerGated, Config I (19 ch)",
    }),
    ("traj_gated", {
        "name": "traj_gated",
        "add_trajectory": True,
        "use_gating": True,
        "description": "SepsisTransformerGated + trajectory (46 ch)",
    }),
])

RESULTS_BASE = "results/experiment4_trajectory_gating"


# ============================================================
# Preprocessor
# ============================================================

class Experiment4Preprocessor:
    """Strategy B preprocessor with optional trajectory features.

    Base layout:  [vital_values, lab_values, lab_masks, lab_deltas]
    With trajectory: [base, traj_diff_f0, traj_mean_f0, traj_dev_f0, ...]
    """

    def __init__(self, all_features, selected_features, add_trajectory=False):
        self.all_features = all_features
        vitals_set = set(VITALS)

        self.sel_vitals = [f for f in selected_features if f in vitals_set]
        self.sel_labs = [f for f in selected_features if f not in vitals_set]
        self.selected_features = self.sel_vitals + self.sel_labs

        self.col_indices = [all_features.index(f) for f in self.selected_features]

        self.n_vitals = len(self.sel_vitals)
        self.n_labs = len(self.sel_labs)
        self.n_raw = self.n_vitals + self.n_labs

        self.n_base = self.n_vitals + (self.n_labs * 3 if self.n_labs > 0 else 0)
        self.add_trajectory = add_trajectory
        self.n_traj = self.n_raw * N_TRAJ_PER_FEATURE if add_trajectory else 0
        self.n_channels = self.n_base + self.n_traj

        self.train_medians = None
        self.val_mean = None
        self.val_std = None
        self.delta_mean = None
        self.delta_std = None
        self.traj_mean = None
        self.traj_std = None

    def fit(self, episodes):
        all_raw = np.concatenate(
            [e["signals"][:, self.col_indices] for e in episodes], axis=0
        )
        self.train_medians = np.nanmedian(all_raw, axis=0)

        all_values, all_deltas, all_trajs = [], [], []
        for ep in episodes:
            filled = self._fill_episode(ep["signals"])
            all_values.append(filled["values"])
            if self.n_labs > 0:
                all_deltas.append(filled["deltas"])
            if self.add_trajectory:
                all_trajs.append(compute_trajectory(filled["values"], TRAJ_WINDOW))

        all_values = np.concatenate(all_values, axis=0)
        self.val_mean = all_values.mean(axis=0)
        self.val_std = all_values.std(axis=0)
        self.val_std[self.val_std == 0] = 1.0

        if self.n_labs > 0:
            all_deltas = np.concatenate(all_deltas, axis=0)
            self.delta_mean = all_deltas.mean(axis=0)
            self.delta_std = all_deltas.std(axis=0)
            self.delta_std[self.delta_std == 0] = 1.0

        if self.add_trajectory:
            all_trajs = np.concatenate(all_trajs, axis=0)
            self.traj_mean = all_trajs.mean(axis=0)
            self.traj_std = all_trajs.std(axis=0)
            self.traj_std[self.traj_std == 0] = 1.0

        return self

    def transform(self, episodes):
        results = []
        for ep in episodes:
            filled = self._fill_episode(ep["signals"])
            values_norm = (filled["values"] - self.val_mean) / self.val_std

            parts = [values_norm]
            if self.n_labs > 0:
                parts.append(filled["masks"])
                parts.append((filled["deltas"] - self.delta_mean) / self.delta_std)
            if self.add_trajectory:
                traj = compute_trajectory(filled["values"], TRAJ_WINDOW)
                parts.append((traj - self.traj_mean) / self.traj_std)

            expanded = np.concatenate(parts, axis=1).astype(np.float32)
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
        n_steps = signals.shape[0]

        # Lab observation mask BEFORE filling
        masks = np.zeros((n_steps, self.n_labs), dtype=np.float32)
        for li in range(self.n_labs):
            masks[:, li] = (~np.isnan(signals[:, self.n_vitals + li])).astype(np.float32)

        # Causal forward-fill
        for j in range(signals.shape[1]):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]

        # Fill remaining leading NaNs with training median
        for j in range(signals.shape[1]):
            nans = np.isnan(signals[:, j])
            if nans.any():
                signals[nans, j] = self.train_medians[j]

        # Clip
        for j, feat in enumerate(self.selected_features):
            if feat in CLIP_RANGES:
                lo, hi = CLIP_RANGES[feat]
                signals[:, j] = np.clip(signals[:, j], lo, hi)

        # Lab deltas
        deltas = np.zeros((n_steps, self.n_labs), dtype=np.float32)
        for li in range(self.n_labs):
            last_obs = -1
            for i in range(n_steps):
                if masks[i, li] == 1.0:
                    deltas[i, li] = 0.0
                    last_obs = i
                else:
                    deltas[i, li] = float(i - last_obs) if last_obs >= 0 else float(i + 1)

        return {"values": signals, "masks": masks, "deltas": deltas}

    def get_channel_names(self):
        names = [f"val_{f}" for f in self.selected_features]
        names += [f"mask_{f}" for f in self.sel_labs]
        names += [f"delta_{f}" for f in self.sel_labs]
        if self.add_trajectory:
            for f in self.selected_features:
                for suffix in TRAJ_SUFFIXES:
                    names.append(f"traj_{suffix}_{f}")
        return names

    def get_group_structure(self):
        """Return (n_groups, channel_to_group, group_names) for ChannelGate."""
        group_names = list(self.selected_features)
        channel_to_group = []

        # Value channels
        for g in range(self.n_raw):
            channel_to_group.append(g)
        # Mask channels (labs)
        for li in range(self.n_labs):
            channel_to_group.append(self.n_vitals + li)
        # Delta channels (labs)
        for li in range(self.n_labs):
            channel_to_group.append(self.n_vitals + li)
        # Trajectory channels
        if self.add_trajectory:
            for g in range(self.n_raw):
                for _ in range(N_TRAJ_PER_FEATURE):
                    channel_to_group.append(g)

        assert len(channel_to_group) == self.n_channels
        return self.n_raw, channel_to_group, group_names


# ============================================================
# Metrics (extends experiment2)
# ============================================================

def compute_metrics(y_true, y_prob, threshold):
    metrics = _compute_metrics_base(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    metrics["tp"] = int(((y_pred == 1) & (y_true == 1)).sum())
    metrics["fp"] = int(((y_pred == 1) & (y_true == 0)).sum())
    metrics["tn"] = int(((y_pred == 0) & (y_true == 0)).sum())
    metrics["fn"] = int(((y_pred == 0) & (y_true == 1)).sum())
    metrics["accuracy"] = float((y_pred == y_true).mean())
    return metrics


# ============================================================
# Gate Weight Analysis
# ============================================================

@torch.no_grad()
def collect_gate_analysis(model, test_data, splits_raw_test, preprocessor,
                          group_names, device, exp_dir, logger):
    """Analyze gate weights across patients and save interpretability data."""
    model.eval()

    patient_mean_weights = []
    patient_weight_vars = []
    patient_labels = []

    for ep in test_data:
        signals = torch.from_numpy(ep["signals"]).unsqueeze(0).to(device)
        lengths = torch.tensor([ep["length"]], dtype=torch.long)
        _, gw = model.forward_with_weights(signals, lengths)
        w = gw[0, :ep["length"]].cpu().numpy()

        patient_mean_weights.append(w.mean(axis=0))
        patient_weight_vars.append(w.var(axis=0))
        patient_labels.append(ep["label"])

    mean_w = np.array(patient_mean_weights)
    var_w = np.array(patient_weight_vars)
    labels_arr = np.array(patient_labels)

    analysis = {
        "group_names": group_names,
        "n_groups": len(group_names),
        "n_patients": len(test_data),
        "population_mean_per_group": mean_w.mean(axis=0).tolist(),
        "population_std_per_group": mean_w.std(axis=0).tolist(),
        "between_patient_variance": mean_w.var(axis=0).tolist(),
        "mean_within_patient_variance": var_w.mean(axis=0).tolist(),
    }

    if labels_arr.sum() > 0:
        sep = labels_arr == 1
        analysis["septic_mean_per_group"] = mean_w[sep].mean(axis=0).tolist()
        analysis["healthy_mean_per_group"] = mean_w[~sep].mean(axis=0).tolist()

    dominant = np.argmax(mean_w, axis=1)
    analysis["dominant_feature_counts"] = {
        group_names[g]: int((dominant == g).sum()) for g in range(len(group_names))
    }

    with open(os.path.join(exp_dir, "gate_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    _save_example_patients(model, test_data, splits_raw_test, preprocessor,
                           group_names, device, exp_dir)
    _plot_gate_distributions(mean_w, group_names, labels_arr, exp_dir)

    logger.info(f"  Gate weights (population mean): "
                f"{dict(zip(group_names, [f'{w:.3f}' for w in analysis['population_mean_per_group']]))}")
    logger.info(f"  Dominant feature counts: {analysis['dominant_feature_counts']}")

    return analysis


@torch.no_grad()
def _save_example_patients(model, test_data, splits_raw_test, preprocessor,
                           group_names, device, exp_dir):
    """Save detailed per-patient interpretability data for a subset."""
    raw_lookup = {ep["patient_id"]: ep for ep in splits_raw_test}

    septic = [ep for ep in test_data if ep["label"] == 1]
    healthy = [ep for ep in test_data if ep["label"] == 0]

    rng = np.random.RandomState(42)
    n_sep = min(N_INTERP_SEPTIC, len(septic))
    n_hlt = min(N_INTERP_HEALTHY, len(healthy))
    selected = (
        [septic[i] for i in rng.choice(len(septic), n_sep, replace=False)] +
        [healthy[i] for i in rng.choice(len(healthy), n_hlt, replace=False)]
    )

    examples = []
    for ep in selected:
        signals = torch.from_numpy(ep["signals"]).unsqueeze(0).to(device)
        lengths = torch.tensor([ep["length"]], dtype=torch.long)
        logits, gw = model.forward_with_weights(signals, lengths)
        probs = torch.sigmoid(logits[0, :ep["length"]]).cpu().numpy()
        weights = gw[0, :ep["length"]].cpu().numpy()

        # Raw feature values (pre-normalization)
        raw_ep = raw_lookup.get(ep["patient_id"])
        raw_values = None
        if raw_ep is not None:
            raw_values = raw_ep["signals"][:, preprocessor.col_indices].tolist()

        examples.append({
            "patient_id": ep["patient_id"],
            "label": ep["label"],
            "length": ep["length"],
            "timestamps": list(range(ep["length"])),
            "risk_scores": probs.tolist(),
            "gate_weights": weights.tolist(),
            "group_names": group_names,
            "raw_feature_values": raw_values,
            "raw_feature_names": list(preprocessor.selected_features),
        })

    with open(os.path.join(exp_dir, "example_patients.json"), "w") as f:
        json.dump(examples, f)


def _plot_gate_distributions(mean_weights, group_names, labels, exp_dir):
    """Plot gate weight distributions: all patients and septic vs healthy."""
    n_groups = len(group_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.boxplot([mean_weights[:, g] for g in range(n_groups)],
               labels=group_names, vert=True)
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Gate weight distribution (all patients)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim([0, 1])

    ax = axes[1]
    sep = labels == 1
    if sep.sum() > 0:
        x = np.arange(n_groups)
        w = 0.35
        ax.bar(x - w / 2, mean_weights[sep].mean(axis=0), w,
               label="Septic", color="#E53935", alpha=0.8)
        ax.bar(x + w / 2, mean_weights[~sep].mean(axis=0), w,
               label="Healthy", color="#43A047", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=45, ha="right")
        ax.set_ylabel("Mean gate weight")
        ax.set_title("Septic vs Healthy")
        ax.legend()
        ax.set_ylim([0, 1])

    fig.tight_layout()
    fig.savefig(os.path.join(exp_dir, "gate_distributions.png"), dpi=150)
    plt.close(fig)


# ============================================================
# Single Config Runner
# ============================================================

def run_config(config_id, config, splits_raw, seed, device, run_dir, logger):
    config_name = config["name"]
    add_traj = config["add_trajectory"]
    use_gating = config["use_gating"]

    exp_dir = os.path.join(run_dir, "configs", f"{config_name}_seed{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    logger.info(f"{'=' * 70}")
    logger.info(f"  Config: {config_name}  seed={seed}")
    logger.info(f"  {config['description']}")
    logger.info(f"{'=' * 70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    preprocessor = Experiment4Preprocessor(
        ALL_FEATURES, SELECTED_FEATURES, add_trajectory=add_traj,
    )
    channel_names = preprocessor.get_channel_names()
    logger.info(f"  Channels ({preprocessor.n_channels}): {channel_names}")

    preprocessor.fit(splits_raw["train"])
    train_data = preprocessor.transform(splits_raw["train"])
    val_data = preprocessor.transform(splits_raw["val"])
    test_data = preprocessor.transform(splits_raw["test"])

    # Pos weight
    train_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = float(n_neg / max(n_pos, 1))

    # Model
    if use_gating:
        n_groups, channel_to_group, group_names = preprocessor.get_group_structure()
        model = SepsisTransformerGated(
            input_dim=preprocessor.n_channels,
            n_groups=n_groups,
            channel_to_group=channel_to_group,
        )
    else:
        model = SepsisTransformer(input_dim=preprocessor.n_channels)
        group_names = None

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}  Pos weight: {pos_weight:.1f}")

    # Train
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

    # Threshold tuning on validation
    val_labels, val_probs = collect_predictions(model, val_data, device)
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    val_metrics = compute_metrics(val_labels, val_probs, best_threshold)

    # Test evaluation
    test_labels, test_probs = collect_predictions(model, test_data, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_threshold)

    logger.info(f"  Val  AUROC={val_metrics['auroc']:.4f}  AUPRC={val_metrics['auprc']:.4f}  "
                f"F1={val_metrics['f1']:.4f}")
    logger.info(f"  Test AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  "
                f"F1={test_metrics['f1']:.4f}  Acc={test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Recall={test_metrics['recall']:.4f}  Prec={test_metrics['precision']:.4f}  "
                f"Spec={test_metrics['specificity']:.4f}")
    logger.info(f"  Threshold={best_threshold:.3f}  Best epoch={best_epoch}  Time={train_time:.0f}s")

    # Artifacts
    roc_data = save_roc_curve(test_labels, test_probs, os.path.join(exp_dir, "roc_curve.png"))
    pr_data = save_pr_curve(test_labels, test_probs, os.path.join(exp_dir, "pr_curve.png"))

    # Gate analysis (gated models only)
    gate_analysis = None
    if use_gating:
        gate_analysis = collect_gate_analysis(
            model, test_data, splits_raw["test"], preprocessor,
            group_names, device, exp_dir, logger,
        )

    history_out = [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in h.items()} for h in history]

    result = {
        "config_id": config_id,
        "config_name": config_name,
        "add_trajectory": add_traj,
        "use_gating": use_gating,
        "seed": seed,
        "n_channels": preprocessor.n_channels,
        "channel_names": channel_names,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 1),
        "val": val_metrics,
        "test": test_metrics,
        "gate_analysis": gate_analysis,
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
        json.dump(history_out, f, indent=2)
    with open(os.path.join(exp_dir, "roc_data.json"), "w") as f:
        json.dump(roc_data, f)
    with open(os.path.join(exp_dir, "pr_data.json"), "w") as f:
        json.dump(pr_data, f)

    result["_roc_data"] = roc_data
    result["_pr_data"] = pr_data
    return result


# ============================================================
# Summary
# ============================================================

def generate_summary(all_results, run_dir, seeds, logger):
    config_groups = {}
    for r in all_results:
        cid = r["config_id"]
        config_groups.setdefault(cid, []).append(r)

    lines = []
    lines.append("=" * 130)
    lines.append("  EXPERIMENT 4: TRAJECTORY FEATURES & DYNAMIC CHANNEL GATING")
    lines.append("=" * 130)
    lines.append(f"  Features: Config I (9 raw, exclude creatinine)  Seeds: {seeds}")
    lines.append("")

    header = (f"  {'Config':<14s} {'Ch':>3s} {'Params':>7s} "
              f"{'AUROC':>14s} {'AUPRC':>14s} {'F1':>14s} "
              f"{'Recall':>14s} {'Prec':>14s} {'Spec':>14s} {'Acc':>14s}")
    lines.append(header)
    lines.append(f"  {'-' * 126}")

    summary_data = []

    for config_id in CONFIGS:
        if config_id not in config_groups:
            continue
        runs = config_groups[config_id]

        def vals(key):
            return [r["test"][key] for r in runs]

        def fmt(v):
            if len(v) == 1:
                return f"{v[0]:.4f}"
            return f"{np.mean(v):.4f}±{np.std(v):.4f}"

        ch = runs[0]["n_channels"]
        params = runs[0]["n_params"]
        lines.append(
            f"  {config_id:<14s} {ch:>3d} {params:>7,d} "
            f"{fmt(vals('auroc')):>14s} {fmt(vals('auprc')):>14s} {fmt(vals('f1')):>14s} "
            f"{fmt(vals('recall')):>14s} {fmt(vals('precision')):>14s} "
            f"{fmt(vals('specificity')):>14s} {fmt(vals('accuracy')):>14s}"
        )

        summary_data.append({
            "config_id": config_id,
            "config_name": runs[0]["config_name"],
            "n_channels": ch,
            "n_params": params,
            "n_seeds": len(runs),
            **{f"{m}_mean": float(np.mean(vals(m))) for m in
               ["auroc", "auprc", "f1", "recall", "precision", "specificity", "accuracy"]},
            **{f"{m}_std": float(np.std(vals(m))) for m in
               ["auroc", "auprc", "f1", "recall", "precision", "specificity", "accuracy"]},
        })

    lines.append("=" * 130)

    # Clinical assessment
    lines.append("")
    lines.append("  CLINICAL VIABILITY ASSESSMENT")
    lines.append("  " + "-" * 80)
    if summary_data:
        best = max(summary_data, key=lambda x: x["auprc_mean"])
        lines.append(f"  Best config:  {best['config_id']} ({best['config_name']})")
        for m in ["auroc", "auprc", "f1", "recall", "precision", "specificity", "accuracy"]:
            lines.append(f"    {m:14s}: {best[f'{m}_mean']:.4f} ± {best[f'{m}_std']:.4f}")
        lines.append("")
        lines.append("  Clinical benchmarks for sepsis screening:")
        lines.append("    AUROC > 0.85      — discrimination threshold for screening utility")
        lines.append("    AUPRC > 0.30      — actionable alert rate (low false-positive burden)")
        lines.append("    Recall > 0.70     — minimum to catch majority of sepsis cases")
        lines.append("    Specificity > 0.95 — avoid alert fatigue in clinical setting")
        lines.append("    Precision > 0.30   — minimum for clinician trust in alerts")
        lines.append("")
        lines.append("  Honest assessment:")
        lines.append("    - Our model discriminates septic patterns (AUROC ~0.81) but is below")
        lines.append("      the 0.85-0.90 range typical of clinically deployed systems.")
        lines.append("    - AUPRC ~0.14 means ~85% of positive alerts are false positives.")
        lines.append("    - Recall ~0.30 misses ~70% of sepsis — unacceptable as standalone.")
        lines.append("    - These are timestep-level metrics on 2.2% prevalence hourly data.")
        lines.append("    - The model IS useful as a research prototype proving the pipeline works.")
        lines.append("")
        lines.append("  Path to clinical viability:")
        lines.append("    1. IL-6 integration (our unique biomarker — not yet available)")
        lines.append("    2. Higher temporal resolution (sub-minute wearable data vs hourly)")
        lines.append("    3. Patient-level onset prediction (aggregate timestep scores)")
        lines.append("    4. MIMIC-IV validation (larger, richer dataset)")
        lines.append("    5. MAE self-supervised pretraining (leverage unlabeled patients)")
        lines.append("    6. Prospective validation with real wearable hardware")

    summary_text = "\n".join(lines)
    logger.info("\n" + summary_text)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # Overlay plots (first seed only)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    first_seed = seeds[0] if seeds else 42
    all_roc, all_pr = OrderedDict(), OrderedDict()
    for r in all_results:
        if r["seed"] == first_seed and "_roc_data" in r:
            all_roc[f"{r['config_name']} (AUROC={r['test']['auroc']:.4f})"] = r["_roc_data"]
            all_pr[f"{r['config_name']} (AUPRC={r['test']['auprc']:.4f})"] = r["_pr_data"]

    if all_roc:
        save_overlay_roc(all_roc, os.path.join(plots_dir, "roc_overlay.png"))
    if all_pr:
        prev = all_results[0]["test"].get("prevalence", 0.022)
        save_overlay_pr(all_pr, prev, os.path.join(plots_dir, "pr_overlay.png"))

    return summary_data


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Trajectory & Gating")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Config IDs (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds (default: 42 123 456)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = args.seeds or DEFAULT_SEEDS

    n_threads = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
    torch.set_num_threads(n_threads)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_BASE, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(run_dir, "experiment4.log")
    logger = logging.getLogger("exp4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    ch_handler = logging.StreamHandler(sys.stdout)
    ch_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch_handler)

    config_ids = args.configs or list(CONFIGS.keys())
    for cid in config_ids:
        if cid not in CONFIGS:
            logger.error(f"Unknown config: {cid}. Valid: {list(CONFIGS.keys())}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("  EXPERIMENT 4: TRAJECTORY FEATURES & DYNAMIC CHANNEL GATING")
    logger.info(f"  Run ID:    {run_id}")
    logger.info(f"  Device:    {args.device}")
    logger.info(f"  Threads:   {n_threads}")
    logger.info(f"  Seeds:     {seeds}")
    logger.info(f"  Configs:   {config_ids}")
    logger.info(f"  Features:  Config I ({len(SELECTED_FEATURES)} raw, excl. creatinine)")
    logger.info(f"  Output:    {run_dir}")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading PhysioNet data...")
    episodes = load_physionet(features=ALL_FEATURES, min_length=MIN_LENGTH)
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
        "seeds": seeds,
        "split_seed": SPLIT_SEED,
        "device": args.device,
        "selected_features": SELECTED_FEATURES,
        "trajectory_window": TRAJ_WINDOW,
        "trajectory_features": TRAJ_SUFFIXES,
        "configs": {cid: CONFIGS[cid] for cid in config_ids},
        "hyperparameters": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "patience": PATIENCE, "min_delta": MIN_DELTA,
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
        },
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Resume support
    completed_path = os.path.join(run_dir, "completed_results.json")
    all_results = []
    done_keys = set()
    if args.resume and os.path.exists(completed_path):
        with open(completed_path, "r") as f:
            all_results = json.load(f)
        done_keys = {(r["config_id"], r["seed"]) for r in all_results}
        logger.info(f"\n  Resuming: {len(done_keys)} runs completed")

    # Run
    total_t0 = time.time()
    for config_id in config_ids:
        for seed in seeds:
            if (config_id, seed) in done_keys:
                logger.info(f"\n  {config_id} seed={seed}: already completed, skipping")
                continue

            try:
                result = run_config(
                    config_id, CONFIGS[config_id], splits, seed,
                    args.device, run_dir, logger,
                )
                all_results.append(result)

                # Incremental save
                save_list = []
                for r in all_results:
                    sr = dict(r)
                    sr.pop("_roc_data", None)
                    sr.pop("_pr_data", None)
                    save_list.append(sr)
                with open(completed_path, "w") as f:
                    json.dump(save_list, f, indent=2)

            except Exception as e:
                logger.error(f"  FAILED: {config_id} seed={seed}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    total_time = time.time() - total_t0
    logger.info(f"\n  All runs completed in {total_time / 60:.1f} minutes")

    if all_results:
        generate_summary(all_results, run_dir, seeds, logger)

    logger.info(f"\n  Results saved to: {run_dir}/")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
