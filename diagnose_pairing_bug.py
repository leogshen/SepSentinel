#!/usr/bin/env python
"""Diagnostic: verify the batch-sorting / patient-metadata misalignment bug
in collect_patient_predictions (experiments 5 & 6), quantify its impact on
patient-level metrics, and compute per-row feature-availability classes.

The suspected bug:
  collate_fn sorts each batch by sequence length (descending), but
  collect_patient_predictions assigns patient_id/onset_step/label by
  sequential dataset order (ep_idx). Within any batch whose episodes are
  not already length-sorted, probs are paired with the WRONG patient's
  metadata. Timestep metrics (AUROC etc.) are unaffected because probs and
  timestep labels both come from the sorted batch.
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import collate_fn
from sepsentinel.model_b.training import SequenceDataset
from sepsentinel.model_b.transformer import SepsisTransformer

from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED, BATCH_SIZE
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS as EXP3_CONFIGS
from experiment5_recall_study import (
    compute_early_warning_metrics, compute_timestep_metrics,
)

CONFIG_I_FEATURES = EXP3_CONFIGS["I"]["features"]
CKPT = "results/experiment6_mae/20260817_143526/finetune/A_scratch_seed42/checkpoints/best_model.pt"
RECORDED_TARGET_THRESH = 0.480   # from exp6 A_scratch seed42 results.json


def collect_buggy(model, preprocessed_data, raw_map, device):
    """Replicates the ORIGINAL (buggy) pairing exactly."""
    loader = DataLoader(SequenceDataset(preprocessed_data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
    out, ep_idx = [], 0
    with torch.no_grad():
        model.eval()
        for signals, labels, lengths, mask in loader:
            signals = signals.to(device)
            probs = torch.sigmoid(model(signals, lengths))
            for i in range(len(lengths)):
                sl = lengths[i].item()
                pid = preprocessed_data[ep_idx]["patient_id"]
                raw_ep = raw_map[pid]
                out.append({
                    "patient_id": pid,
                    "label": raw_ep["label"],
                    "onset_step": raw_ep["onset_step"],
                    "probs": probs[i, :sl].cpu().numpy(),
                    "labels": labels[i, :sl].numpy(),
                    "length": sl,
                })
                ep_idx += 1
    return out


def collect_fixed(model, preprocessed_data, raw_map, device):
    """Corrected pairing: replicate collate's per-batch stable sort."""
    loader = DataLoader(SequenceDataset(preprocessed_data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
    out, start = [], 0
    with torch.no_grad():
        model.eval()
        for signals, labels, lengths, mask in loader:
            batch_items = preprocessed_data[start:start + len(lengths)]
            start += len(lengths)
            sorted_items = sorted(batch_items, key=lambda x: x["length"], reverse=True)
            signals = signals.to(device)
            probs = torch.sigmoid(model(signals, lengths))
            for i in range(len(lengths)):
                sl = lengths[i].item()
                item = sorted_items[i]
                assert item["length"] == sl, "sort replication mismatch"
                raw_ep = raw_map[item["patient_id"]]
                out.append({
                    "patient_id": item["patient_id"],
                    "label": raw_ep["label"],
                    "onset_step": raw_ep["onset_step"],
                    "probs": probs[i, :sl].cpu().numpy(),
                    "labels": labels[i, :sl].numpy(),
                    "length": sl,
                })
    return out


def find_thresh_patient_recall(patient_results, target):
    for t in np.arange(0.995, 0.004, -0.005):
        septic = [p for p in patient_results if p["label"] == 1]
        caught = sum(1 for p in septic if (p["probs"] >= t).any())
        if caught / max(len(septic), 1) >= target:
            return float(t)
    return 0.005


def main():
    device = torch.device("cpu")

    print("Loading data...")
    raw_episodes = load_physionet(features=ALL_FEATURES, min_length=6)
    raw_map = {e["patient_id"]: e for e in raw_episodes}
    splits = patient_split(raw_episodes, random_state=SPLIT_SEED)

    pp = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    pp.fit(splits["train"])
    test_data = pp.transform(splits["test"])

    # ---------- A) Structural pairing check (no model needed) ----------
    print("\n=== A) STRUCTURAL PAIRING CHECK (test split, batch=32) ===")
    n_total, n_misaligned, n_label_corrupt = 0, 0, 0
    n_septic_meta, n_septic_meta_probs_septic = 0, 0
    for start in range(0, len(test_data), BATCH_SIZE):
        batch = test_data[start:start + BATCH_SIZE]
        sorted_batch = sorted(batch, key=lambda x: x["length"], reverse=True)
        for i in range(len(batch)):
            seq_item = batch[i]          # metadata source under the bug
            sorted_item = sorted_batch[i]  # probs/labels source under the bug
            n_total += 1
            if seq_item["patient_id"] != sorted_item["patient_id"]:
                n_misaligned += 1
            if seq_item["label"] != sorted_item["label"]:
                n_label_corrupt += 1
            if seq_item["label"] == 1:
                n_septic_meta += 1
                if sorted_item["label"] == 1:
                    n_septic_meta_probs_septic += 1

    print(f"  Episodes:                      {n_total}")
    print(f"  Wrong patient attached:        {n_misaligned} ({n_misaligned/n_total*100:.1f}%)")
    print(f"  Patient-label corrupted:       {n_label_corrupt} ({n_label_corrupt/n_total*100:.1f}%)")
    print(f"  'Septic' rows w/ septic probs: {n_septic_meta_probs_septic}/{n_septic_meta}")

    # ---------- B) Impact on reported metrics (A_scratch seed42) ----------
    print("\n=== B) METRIC IMPACT: exp6 A_scratch seed42 checkpoint ===")
    model = SepsisTransformer(pp.n_channels).to(device)
    model.load_state_dict(torch.load(CKPT, weights_only=True))

    buggy = collect_buggy(model, test_data, raw_map, device)
    fixed = collect_fixed(model, test_data, raw_map, device)

    # sanity: timestep metrics identical
    yt_b = np.concatenate([p["labels"] for p in buggy])
    yp_b = np.concatenate([p["probs"] for p in buggy])
    yt_f = np.concatenate([p["labels"] for p in fixed])
    yp_f = np.concatenate([p["probs"] for p in fixed])
    ts_b = compute_timestep_metrics(yt_b, yp_b, RECORDED_TARGET_THRESH)
    ts_f = compute_timestep_metrics(yt_f, yp_f, RECORDED_TARGET_THRESH)
    print(f"  Timestep recall  buggy={ts_b['recall']:.4f}  fixed={ts_f['recall']:.4f} (should match)")
    print(f"  Timestep prec    buggy={ts_b['precision']:.4f}  fixed={ts_f['precision']:.4f} (should match)")

    for name, res in [("BUGGY", buggy), ("FIXED", fixed)]:
        ew = compute_early_warning_metrics(res, RECORDED_TARGET_THRESH)
        print(f"\n  [{name}] at recorded threshold {RECORDED_TARGET_THRESH}:")
        print(f"    patient_recall = {ew['patient_recall']:.4f}  "
              f"({ew['caught_any']}/{ew['n_septic']})")
        print(f"    median_lead    = {ew['median_lead_time_h']}h")
        print(f"    alerts/day     = {ew['alerts_per_patient_day']:.2f}")
        print(f"    capture@6h     = {ew['capture_rate_by_lead_hour']['6']:.3f}")

    t_fixed = find_thresh_patient_recall(fixed, 0.70)
    ew_ff = compute_early_warning_metrics(fixed, t_fixed)
    ts_ff = compute_timestep_metrics(yt_f, yp_f, t_fixed)
    print(f"\n  [FIXED] re-derived threshold for 0.70 patient recall: {t_fixed:.3f}")
    print(f"    patient_recall = {ew_ff['patient_recall']:.4f}")
    print(f"    precision      = {ts_ff['precision']:.4f}")
    print(f"    alerts/day     = {ew_ff['alerts_per_patient_day']:.2f}")
    print(f"    median_lead    = {ew_ff['median_lead_time_h']}h")
    print(f"    capture@6h     = {ew_ff['capture_rate_by_lead_hour']['6']:.3f}")

    # ---------- C) Availability classes ----------
    print("\n=== C) AVAILABILITY CLASSES (9 Config-I features, all 546K rows) ===")
    col_idx = [ALL_FEATURES.index(f) for f in CONFIG_I_FEATURES]
    all_sig = np.concatenate([e["signals"][:, col_idx] for e in raw_episodes], axis=0)
    all_lbl = np.concatenate([e["labels"] for e in raw_episodes], axis=0)
    observed = (~np.isnan(all_sig)).astype(np.int64)
    powers = 2 ** np.arange(len(col_idx))
    codes = observed @ powers
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)

    print(f"  Rows: {len(codes):,}   Distinct patterns: {len(uniq)}")
    cum = 0
    print(f"  {'pattern (1=observed)':>28s} | {'rows':>8s} | {'%':>6s} | {'sepsis%':>7s}")
    feat_names = [f[:4] for f in CONFIG_I_FEATURES]
    print(f"  {'[' + ' '.join(feat_names) + ']':>28s}")
    for k in order[:12]:
        code = uniq[k]
        bits = [(code >> b) & 1 for b in range(len(col_idx))]
        pat = "".join(str(b) for b in bits)
        n = counts[k]
        cum += n
        sep_rate = all_lbl[codes == code].mean() * 100
        print(f"  {pat:>28s} | {n:8,d} | {n/len(codes)*100:5.1f}% | {sep_rate:6.2f}%")
    print(f"  Top 12 cover {cum/len(codes)*100:.1f}% of rows")

    n_all_vitals = observed[:, :4].all(axis=1).mean() * 100
    n_any_lab = observed[:, 4:].any(axis=1).mean() * 100
    n_nothing = (observed.sum(axis=1) == 0).mean() * 100
    print(f"\n  Rows with all 4 vitals observed: {n_all_vitals:.1f}%")
    print(f"  Rows with >=1 lab observed:      {n_any_lab:.1f}%")
    print(f"  Rows with nothing observed:      {n_nothing:.1f}%")


if __name__ == "__main__":
    main()
