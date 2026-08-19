#!/usr/bin/env python
"""Experiment 6: MAE Self-Supervised Pretraining for Improved Recall

Scientific question: Does Masked Autoencoder pretraining on unlabeled
clinical time series improve downstream sepsis prediction, specifically
patient-level recall at a clinically usable false-alert rate?

APPROACH
========
Phase 0: MAE Pretraining (self-supervised, no labels)
  - Randomly mask 40% of available (non-NaN) feature values per timestep
  - Train the TransformerEncoder to reconstruct masked values
  - Lightweight decoder (1-layer transformer, d_model=32), discarded after
  - Loss: MSE on masked positions only (9 value channels, not mask/delta)
  - Data: train set only

Phase 1: Fine-tuning Comparison (3 seeds each)
  Config A: From-scratch SepsisTransformer (baseline)
  Config B: MAE-pretrained encoder -> fine-tune full model
  Config C: MAE-pretrained encoder (frozen) -> train only classification head

All evaluated at 0.70 patient recall threshold.

RECALL TARGET
=============
Primary success metric: at 0.70 patient recall, beat current baselines:
  - Precision > 5.3% (Experiment 5 baseline at threshold=0.30)
  - Alerts/patient/day < 8
  - Median lead time >= 24h

Usage:
    python experiment6_mae_pretraining.py                    # full run
    python experiment6_mae_pretraining.py --phase 0          # MAE pretraining only
    python experiment6_mae_pretraining.py --phase 1          # fine-tuning only
    python experiment6_mae_pretraining.py --seeds 42         # single seed
    python experiment6_mae_pretraining.py --device cuda
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
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import collate_fn
from sepsentinel.model_b.training import Trainer, SequenceDataset
from sepsentinel.model_b.transformer import (
    SepsisTransformer, TransformerEncoder, PositionalEncoding,
)

from experiment2_imputation import (
    find_best_f1_threshold,
    FEATURES as ALL_FEATURES,
    N_VITALS,
    SPLIT_SEED, EPOCHS, BATCH_SIZE, LR,
    PATIENCE, MIN_DELTA, SCHEDULER_FACTOR, SCHEDULER_PATIENCE,
)
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS as EXP3_CONFIGS
from experiment5_recall_study import (
    collect_patient_predictions,
    compute_timestep_metrics,
    compute_early_warning_metrics,
    threshold_sweep,
    find_threshold_for_recall,
    LEAD_HOURS,
)

# ============================================================
# Configuration
# ============================================================

CONFIG_I_FEATURES = EXP3_CONFIGS["I"]["features"]
SEEDS = [42, 123, 456]
MIN_LENGTH = 6
RESULTS_DIR = "results/experiment6_mae"
PATIENT_RECALL_TARGET = 0.70

# MAE pretraining config
MAE_MASK_RATIO = 0.40
MAE_EPOCHS = 100
MAE_PATIENCE = 10
MAE_LR = 1e-4
MAE_BATCH_SIZE = 32
MAE_DECODER_D_MODEL = 32
MAE_DECODER_LAYERS = 1

# Model architecture (same as production)
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.2

FINETUNE_CONFIGS = OrderedDict([
    ("A_scratch", {
        "pretrained": False,
        "freeze_encoder": False,
        "description": "From-scratch SepsisTransformer (baseline)",
    }),
    ("B_mae_finetune", {
        "pretrained": True,
        "freeze_encoder": False,
        "description": "MAE-pretrained encoder, full fine-tuning",
    }),
    ("C_mae_frozen", {
        "pretrained": True,
        "freeze_encoder": True,
        "description": "MAE-pretrained encoder (frozen), train head only",
    }),
])


# ============================================================
# MAE Model
# ============================================================

class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""

    def __init__(self, d_model_encoder, d_model_decoder, n_value_channels,
                 nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_model_encoder, d_model_decoder)
        self.pos_encoding = PositionalEncoding(d_model_decoder, dropout=dropout)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_decoder,
            nhead=nhead,
            dim_feedforward=d_model_decoder * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model_decoder, n_value_channels)

    def forward(self, encoder_output, lengths=None):
        x = self.proj_in(encoder_output)
        x = self.pos_encoding(x)

        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        pad_mask = None
        if lengths is not None:
            pad_mask = (
                torch.arange(seq_len, device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        return self.output_proj(x)


class MAEModel(nn.Module):
    """Masked Autoencoder for clinical time series.

    Encoder: same TransformerEncoder as SepsisTransformer
    Decoder: lightweight 1-layer transformer
    Task: reconstruct masked value channels from corrupted input
    """

    def __init__(self, input_dim, n_value_channels, n_vitals):
        super().__init__()
        self.input_dim = input_dim
        self.n_value_channels = n_value_channels
        self.n_vitals = n_vitals
        self.n_labs = n_value_channels - n_vitals

        self.encoder = TransformerEncoder(
            input_dim, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT,
        )
        self.decoder = MAEDecoder(
            D_MODEL, MAE_DECODER_D_MODEL, n_value_channels,
            nhead=4, num_layers=MAE_DECODER_LAYERS, dropout=0.1,
        )

    def forward(self, x, lengths=None):
        h = self.encoder(x, lengths)
        recon = self.decoder(h, lengths)
        return recon


def apply_mae_masking(signals, lengths, n_vitals, n_labs, mask_ratio=0.40):
    """Apply random masking to preprocessed signals for MAE pretraining.

    Input layout: [vital_values(nv), lab_values(nl), lab_masks(nl), lab_deltas(nl)]
    Total channels: nv + nl*3

    Masking strategy:
    - For each valid (non-padded) timestep, randomly mask `mask_ratio` of the
      n_value = nv + nl value channels
    - When a value channel is masked: set value=0
    - When a lab value is masked: also set its mask channel=0, delta unchanged
    - Record which positions were masked for loss computation

    Returns:
        masked_signals: (B, T, C) with masked values zeroed
        recon_target: (B, T, n_values) original values before masking
        recon_mask: (B, T, n_values) binary, 1 = masked position (compute loss here)
    """
    B, T, C = signals.shape
    n_values = n_vitals + n_labs
    n_to_mask = max(1, int(n_values * mask_ratio))

    masked = signals.clone()
    recon_target = signals[:, :, :n_values].clone()

    rand_scores = torch.rand(B, T, n_values, device=signals.device)
    _, top_indices = rand_scores.topk(n_to_mask, dim=2)
    recon_mask = torch.zeros(B, T, n_values, device=signals.device)
    recon_mask.scatter_(2, top_indices, 1.0)

    pad_mask = torch.arange(T, device=signals.device).unsqueeze(0) >= lengths.unsqueeze(1)
    recon_mask[pad_mask] = 0.0

    masked[:, :, :n_values] *= (1.0 - recon_mask)

    if n_labs > 0:
        lab_recon_mask = recon_mask[:, :, n_vitals:]
        masked[:, :, n_values:n_values + n_labs] *= (1.0 - lab_recon_mask)

    return masked, recon_target, recon_mask


# ============================================================
# MAE Pretraining Loop
# ============================================================

def pretrain_mae(mae_model, train_data, val_data, device, logger, checkpoint_dir):
    """Self-supervised MAE pretraining."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader = DataLoader(
        SequenceDataset(train_data), batch_size=MAE_BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SequenceDataset(val_data), batch_size=MAE_BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=MAE_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAE_EPOCHS, eta_min=1e-6,
    )

    n_vitals = mae_model.n_vitals
    n_labs = mae_model.n_labs

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    logger.info(f"\n  MAE Pretraining: mask_ratio={MAE_MASK_RATIO}, "
                f"lr={MAE_LR}, epochs={MAE_EPOCHS}, patience={MAE_PATIENCE}")
    logger.info(f"  Encoder: d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NHEAD}")
    logger.info(f"  Decoder: d_model={MAE_DECODER_D_MODEL}, layers={MAE_DECODER_LAYERS}")

    for epoch in range(1, MAE_EPOCHS + 1):
        t0 = time.time()

        # Train
        mae_model.train()
        train_loss_sum = 0.0
        train_n = 0

        for signals, labels, lengths, mask in train_loader:
            signals = signals.to(device)
            lengths_dev = lengths.to(device)

            masked_signals, recon_target, recon_mask = apply_mae_masking(
                signals, lengths, n_vitals, n_labs, MAE_MASK_RATIO,
            )

            optimizer.zero_grad()
            recon = mae_model(masked_signals, lengths_dev)

            loss_per_elem = (recon - recon_target) ** 2
            loss = (loss_per_elem * recon_mask).sum() / recon_mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae_model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * recon_mask.sum().item()
            train_n += recon_mask.sum().item()

        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        mae_model.eval()
        val_loss_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for signals, labels, lengths, mask in val_loader:
                signals = signals.to(device)
                lengths_dev = lengths.to(device)

                masked_signals, recon_target, recon_mask = apply_mae_masking(
                    signals, lengths, n_vitals, n_labs, MAE_MASK_RATIO,
                )

                recon = mae_model(masked_signals, lengths_dev)
                loss_per_elem = (recon - recon_target) ** 2
                val_loss = (loss_per_elem * recon_mask).sum() / recon_mask.sum().clamp(min=1)

                val_loss_sum += val_loss.item() * recon_mask.sum().item()
                val_n += recon_mask.sum().item()

        val_loss = val_loss_sum / max(val_n, 1)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
        })

        logger.info(f"  MAE Epoch {epoch:3d}/{MAE_EPOCHS}  "
                     f"train_mse={train_loss:.6f}  val_mse={val_loss:.6f}  "
                     f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(mae_model.state_dict(), os.path.join(checkpoint_dir, "mae_best.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= MAE_PATIENCE:
            logger.info(f"  MAE early stopping at epoch {epoch} "
                        f"(no improvement for {MAE_PATIENCE} epochs)")
            break

    mae_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "mae_best.pt"), weights_only=True)
    )
    logger.info(f"  MAE pretraining complete. Best val MSE: {best_val_loss:.6f}")

    return history


# ============================================================
# Fine-tuning with Patient-Recall Target
# ============================================================

def find_threshold_for_patient_recall(patient_results, target_recall):
    """Find highest threshold achieving at least target patient recall."""
    best_t = 0.005
    for t in np.arange(0.995, 0.004, -0.005):
        septic = [p for p in patient_results if p["label"] == 1]
        caught = 0
        for pat in septic:
            if (pat["probs"] >= t).any():
                caught += 1
        pr = caught / max(len(septic), 1)
        if pr >= target_recall:
            best_t = float(t)
            break
    return best_t


def evaluate_config(model, test_data, raw_episodes_map, device, logger, config_name):
    """Full evaluation of a fine-tuned model at the 0.70 patient recall target."""
    patient_results = collect_patient_predictions(
        model, test_data, raw_episodes_map, device,
    )

    y_true = np.concatenate([p["labels"] for p in patient_results])
    y_prob = np.concatenate([p["probs"] for p in patient_results])

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    best_f1_thresh = find_best_f1_threshold(y_true, y_prob)

    # Metrics at best-F1 threshold
    ts_f1 = compute_timestep_metrics(y_true, y_prob, best_f1_thresh)
    ew_f1 = compute_early_warning_metrics(patient_results, best_f1_thresh)

    # Find threshold for 0.70 patient recall
    target_thresh = find_threshold_for_patient_recall(
        patient_results, PATIENT_RECALL_TARGET,
    )
    ts_target = compute_timestep_metrics(y_true, y_prob, target_thresh)
    ew_target = compute_early_warning_metrics(patient_results, target_thresh)

    logger.info(f"\n  {config_name} Results:")
    logger.info(f"    AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    logger.info(f"    Best-F1 threshold={best_f1_thresh:.3f}: "
                f"TS_recall={ts_f1['recall']:.3f}, precision={ts_f1['precision']:.3f}, "
                f"patient_recall={ew_f1['patient_recall']:.3f}")
    logger.info(f"    Target threshold={target_thresh:.3f} (patient_recall>=0.70): "
                f"patient_recall={ew_target['patient_recall']:.3f}, "
                f"precision={ts_target['precision']:.4f}, "
                f"alerts/day={ew_target['alerts_per_patient_day']:.1f}, "
                f"median_lead={ew_target.get('median_lead_time_h', 'N/A')}h")

    return {
        "config": config_name,
        "auroc": auroc,
        "auprc": auprc,
        "best_f1_threshold": best_f1_thresh,
        "best_f1_ts_recall": ts_f1["recall"],
        "best_f1_precision": ts_f1["precision"],
        "best_f1_f1": ts_f1["f1"],
        "best_f1_patient_recall": ew_f1["patient_recall"],
        "best_f1_alerts_per_day": ew_f1["alerts_per_patient_day"],
        "best_f1_median_lead_h": ew_f1.get("median_lead_time_h"),
        "target_threshold": target_thresh,
        "target_patient_recall": ew_target["patient_recall"],
        "target_ts_recall": ts_target["recall"],
        "target_precision": ts_target["precision"],
        "target_f1": ts_target["f1"],
        "target_specificity": ts_target["specificity"],
        "target_alerts_per_day": ew_target["alerts_per_patient_day"],
        "target_median_lead_h": ew_target.get("median_lead_time_h"),
        "target_mean_lead_h": ew_target.get("mean_lead_time_h"),
        "target_capture_3h": ew_target["capture_rate_by_lead_hour"].get("3"),
        "target_capture_6h": ew_target["capture_rate_by_lead_hour"].get("6"),
        "target_capture_12h": ew_target["capture_rate_by_lead_hour"].get("12"),
        "target_false_alert_rate": ew_target["healthy_false_alert_rate"],
        "target_mean_alerts_per_healthy": ew_target["mean_alerts_per_healthy_patient"],
    }


# ============================================================
# Plotting
# ============================================================

def plot_mae_loss(history, save_path):
    """Plot MAE pretraining loss curves."""
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train MSE", linewidth=2)
    ax.plot(epochs, val, label="Val MSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MAE Pretraining Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison(all_results, save_path):
    """Bar chart comparing configs at 0.70 patient recall."""
    configs = list(all_results.keys())
    metrics = ["target_precision", "target_alerts_per_day", "target_median_lead_h", "auroc"]
    labels = ["Precision\n@ 0.70 Pt Recall", "Alerts/Day\n@ 0.70 Pt Recall",
              "Median Lead (h)\n@ 0.70 Pt Recall", "AUROC"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric, label in zip(axes, metrics, labels):
        means = []
        stds = []
        for cfg in configs:
            vals = [r[metric] for r in all_results[cfg] if r[metric] is not None]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)

        x = np.arange(len(configs))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                      color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=9)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{m:.3f}" if m < 1 else f"{m:.1f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Experiment 6: MAE Pretraining vs Baseline @ 0.70 Patient Recall",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def setup_logging(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger("experiment6")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(run_dir, "experiment6.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: MAE Pretraining")
    parser.add_argument("--phase", type=int, default=None,
                        help="0=MAE pretraining, 1=fine-tuning. Default: both")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mae-checkpoint", default=None,
                        help="Path to existing MAE checkpoint (skip phase 0)")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    logger = setup_logging(run_dir)
    device = torch.device(args.device)

    logger.info(f"Experiment 6: MAE Pretraining for Recall")
    logger.info(f"  Device: {device}")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Patient recall target: {PATIENT_RECALL_TARGET}")
    logger.info(f"  Run dir: {run_dir}")

    # ---- Load data ----
    logger.info("\n  Loading PhysioNet data...")
    raw_episodes = load_physionet(features=ALL_FEATURES, min_length=MIN_LENGTH)
    logger.info(f"  Loaded {len(raw_episodes)} episodes "
                f"({sum(1 for e in raw_episodes if e['label']==1)} septic)")

    splits = patient_split(raw_episodes, random_state=SPLIT_SEED)
    for name in ["train", "val", "test"]:
        n = len(splits[name])
        ns = sum(1 for e in splits[name] if e["label"] == 1)
        logger.info(f"    {name}: {n} episodes ({ns} septic)")

    # Preprocess with Strategy B (Config I)
    preprocessor = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    train_data = preprocessor.fit_transform(splits["train"])
    val_data = preprocessor.transform(splits["val"])
    test_data = preprocessor.transform(splits["test"])

    n_channels = preprocessor.n_channels
    n_vitals_sel = preprocessor.n_vitals
    n_labs_sel = preprocessor.n_labs
    n_values = n_vitals_sel + n_labs_sel

    logger.info(f"  Preprocessing: {n_channels} channels "
                f"({n_vitals_sel} vitals + {n_labs_sel} labs x3)")

    raw_map = {e["patient_id"]: e for e in raw_episodes}

    # Compute pos_weight for fine-tuning
    all_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    pos_weight = float(n_neg / max(n_pos, 1))
    logger.info(f"  pos_weight: {pos_weight:.1f} ({n_neg} neg / {n_pos} pos)")

    # ---- Phase 0: MAE Pretraining ----
    mae_checkpoint_path = args.mae_checkpoint
    run_phase_0 = args.phase is None or args.phase == 0

    if run_phase_0 and mae_checkpoint_path is None:
        logger.info("\n" + "=" * 80)
        logger.info("  PHASE 0: MAE PRETRAINING")
        logger.info("=" * 80)

        mae_dir = os.path.join(run_dir, "mae_pretrain")
        mae_model = MAEModel(n_channels, n_values, n_vitals_sel).to(device)

        total_params = sum(p.numel() for p in mae_model.parameters())
        enc_params = sum(p.numel() for p in mae_model.encoder.parameters())
        dec_params = sum(p.numel() for p in mae_model.decoder.parameters())
        logger.info(f"  MAE model: {total_params:,} params "
                    f"(encoder: {enc_params:,}, decoder: {dec_params:,})")

        mae_history = pretrain_mae(
            mae_model, train_data, val_data, device, logger, mae_dir,
        )

        with open(os.path.join(mae_dir, "mae_history.json"), "w") as f:
            json.dump(mae_history, f, indent=2)

        plot_mae_loss(mae_history, os.path.join(mae_dir, "mae_loss.png"))
        mae_checkpoint_path = os.path.join(mae_dir, "mae_best.pt")
        logger.info(f"  MAE checkpoint saved: {mae_checkpoint_path}")

    elif mae_checkpoint_path is not None:
        logger.info(f"\n  Using existing MAE checkpoint: {mae_checkpoint_path}")
    else:
        mae_checkpoint_path = os.path.join(run_dir, "mae_pretrain", "mae_best.pt")
        if not os.path.exists(mae_checkpoint_path):
            logger.error("  No MAE checkpoint found. Run phase 0 first.")
            return

    # ---- Phase 1: Fine-tuning Comparison ----
    run_phase_1 = args.phase is None or args.phase == 1

    if not run_phase_1:
        logger.info("  Phase 1 skipped.")
        return

    logger.info("\n" + "=" * 80)
    logger.info("  PHASE 1: FINE-TUNING COMPARISON")
    logger.info("=" * 80)

    all_results = {cfg: [] for cfg in FINETUNE_CONFIGS}
    finetune_dir = os.path.join(run_dir, "finetune")
    os.makedirs(finetune_dir, exist_ok=True)

    for seed in args.seeds:
        logger.info(f"\n  --- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for cfg_name, cfg in FINETUNE_CONFIGS.items():
            logger.info(f"\n  Config {cfg_name}: {cfg['description']}")

            # Build model
            model = SepsisTransformer(
                n_channels, D_MODEL, NHEAD, NUM_LAYERS,
                DIM_FEEDFORWARD, DROPOUT,
            ).to(device)

            # Load pretrained encoder weights if applicable
            if cfg["pretrained"]:
                mae_state = torch.load(mae_checkpoint_path, weights_only=True)
                encoder_state = {
                    k.replace("encoder.", "", 1): v
                    for k, v in mae_state.items()
                    if k.startswith("encoder.")
                }
                model.encoder.load_state_dict(encoder_state)
                logger.info(f"    Loaded MAE encoder weights ({len(encoder_state)} params)")

                if cfg["freeze_encoder"]:
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    trainable = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                    logger.info(f"    Encoder frozen. Trainable params: {trainable:,}")

            # Train
            ckpt_dir = os.path.join(
                finetune_dir, f"{cfg_name}_seed{seed}", "checkpoints",
            )
            trainer = Trainer(
                model, device=str(device), checkpoint_dir=ckpt_dir,
                pos_weight=pos_weight,
            )

            trainer.fit(
                train_data, val_data,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                patience=PATIENCE, min_delta=MIN_DELTA,
                scheduler_factor=SCHEDULER_FACTOR,
                scheduler_patience=SCHEDULER_PATIENCE,
            )

            # Evaluate
            result = evaluate_config(
                model, test_data, raw_map, device, logger, cfg_name,
            )
            result["seed"] = seed
            all_results[cfg_name].append(result)

            result_path = os.path.join(
                finetune_dir, f"{cfg_name}_seed{seed}", "results.json",
            )
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

    # ---- Summary ----
    logger.info("\n" + "=" * 80)
    logger.info("  SUMMARY: All Configs @ 0.70 Patient Recall Target")
    logger.info("=" * 80)

    summary = []
    header = (
        f"  {'Config':<20s} | {'AUROC':>12s} | {'AUPRC':>12s} | "
        f"{'Pt Recall':>12s} | {'Precision':>12s} | "
        f"{'Alerts/Day':>12s} | {'Med Lead':>12s} | "
        f"{'Cap 6h':>8s}"
    )
    logger.info(header)
    logger.info("  " + "-" * 120)

    for cfg_name in FINETUNE_CONFIGS:
        results = all_results[cfg_name]
        auroc_m = np.mean([r["auroc"] for r in results])
        auroc_s = np.std([r["auroc"] for r in results])
        auprc_m = np.mean([r["auprc"] for r in results])
        auprc_s = np.std([r["auprc"] for r in results])
        pr_m = np.mean([r["target_patient_recall"] for r in results])
        prec_m = np.mean([r["target_precision"] for r in results])
        prec_s = np.std([r["target_precision"] for r in results])
        alerts_m = np.mean([r["target_alerts_per_day"] for r in results])
        leads = [r["target_median_lead_h"] for r in results if r["target_median_lead_h"] is not None]
        lead_m = np.mean(leads) if leads else float("nan")
        cap6 = np.mean([r["target_capture_6h"] for r in results
                        if r["target_capture_6h"] is not None])

        logger.info(
            f"  {cfg_name:<20s} | "
            f"{auroc_m:.4f}+/-{auroc_s:.4f} | "
            f"{auprc_m:.4f}+/-{auprc_s:.4f} | "
            f"{pr_m:.3f}         | "
            f"{prec_m:.4f}+/-{prec_s:.4f} | "
            f"{alerts_m:>8.1f}     | "
            f"{lead_m:>8.1f}     | "
            f"{cap6:.3f}"
        )

        summary.append({
            "config": cfg_name,
            "description": FINETUNE_CONFIGS[cfg_name]["description"],
            "n_seeds": len(results),
            "auroc_mean": auroc_m,
            "auroc_std": auroc_s,
            "auprc_mean": auprc_m,
            "auprc_std": auprc_s,
            "target_patient_recall_mean": pr_m,
            "target_precision_mean": prec_m,
            "target_precision_std": prec_s,
            "target_alerts_per_day_mean": alerts_m,
            "target_median_lead_h_mean": lead_m,
            "target_capture_6h_mean": cap6,
        })

    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    with open(os.path.join(summary_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(summary_dir, "all_results.json"), "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)

    plot_comparison(all_results, os.path.join(summary_dir, "comparison.png"))

    # Final verdict
    logger.info("\n  EXPERIMENT 5 BASELINE (from-scratch, threshold=0.30):")
    logger.info("    Patient recall=0.731, Precision=0.053, Alerts/day=7.9, Median lead=27h")
    logger.info("\n  SUCCESS CRITERIA:")
    logger.info("    At 0.70 patient recall: precision > 0.053, alerts/day < 8, lead >= 24h")

    baseline_results = all_results.get("A_scratch", [])
    mae_results = all_results.get("B_mae_finetune", [])
    if baseline_results and mae_results:
        b_prec = np.mean([r["target_precision"] for r in baseline_results])
        m_prec = np.mean([r["target_precision"] for r in mae_results])
        b_alerts = np.mean([r["target_alerts_per_day"] for r in baseline_results])
        m_alerts = np.mean([r["target_alerts_per_day"] for r in mae_results])
        b_auroc = np.mean([r["auroc"] for r in baseline_results])
        m_auroc = np.mean([r["auroc"] for r in mae_results])

        prec_delta = m_prec - b_prec
        alerts_delta = m_alerts - b_alerts
        auroc_delta = m_auroc - b_auroc

        logger.info(f"\n  MAE vs BASELINE DELTA:")
        logger.info(f"    AUROC:      {auroc_delta:+.4f} ({b_auroc:.4f} -> {m_auroc:.4f})")
        logger.info(f"    Precision:  {prec_delta:+.4f} ({b_prec:.4f} -> {m_prec:.4f})")
        logger.info(f"    Alerts/day: {alerts_delta:+.1f} ({b_alerts:.1f} -> {m_alerts:.1f})")

        if prec_delta > 0 and alerts_delta < 0:
            logger.info("    --> MAE pretraining IMPROVES recall-precision tradeoff")
        elif auroc_delta > 0.005:
            logger.info("    --> MAE pretraining improves discrimination (AUROC) but "
                        "may not fully translate to recall target")
        else:
            logger.info("    --> MAE pretraining shows no significant improvement")

    logger.info(f"\n  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
