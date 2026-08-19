#!/usr/bin/env python
"""Re-evaluate patient-level early-warning metrics with the corrected
patient/prediction pairing (bug fixed 2026-08-19 in collect_patient_predictions).

Evaluation only — no retraining. Covers:
  - All 9 Experiment 6 fine-tuned checkpoints (3 configs x 3 seeds)
  - The Experiment 5 Phase 0 model (Config I from experiment 3)

For each model, reports at:
  - best-F1 threshold
  - threshold achieving >= 0.70 patient recall (re-derived, corrected pairing)
"""

import json
import os

import numpy as np
import torch

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.model_b.transformer import SepsisTransformer

from experiment2_imputation import (
    FEATURES as ALL_FEATURES, SPLIT_SEED, find_best_f1_threshold,
)
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS as EXP3_CONFIGS
from experiment5_recall_study import (
    collect_patient_predictions,
    compute_timestep_metrics,
    compute_early_warning_metrics,
)

CONFIG_I_FEATURES = EXP3_CONFIGS["I"]["features"]
EXP6_DIR = "results/experiment6_mae/20260817_143526"
EXP5_PHASE0_CKPT = ("results/feature_ablation/20260808_135533/experiments/"
                    "I_all_minus_creatinine/checkpoints/best_model.pt")
OUT_PATH = os.path.join(EXP6_DIR, "summary", "corrected_patient_metrics.json")

CHECKPOINTS = []
for cfg in ["A_scratch", "B_mae_finetune", "C_mae_frozen"]:
    for seed in [42, 123, 456]:
        CHECKPOINTS.append((
            f"exp6_{cfg}_seed{seed}",
            os.path.join(EXP6_DIR, "finetune", f"{cfg}_seed{seed}",
                         "checkpoints", "best_model.pt"),
        ))
CHECKPOINTS.append(("exp5_phase0_configI", EXP5_PHASE0_CKPT))


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
    print(f"Test: {len(test_data)} episodes, {pp.n_channels} channels")

    results = {}
    for name, ckpt in CHECKPOINTS:
        if not os.path.exists(ckpt):
            print(f"\n[{name}] MISSING checkpoint: {ckpt}")
            continue

        print(f"\n[{name}]")
        model = SepsisTransformer(pp.n_channels).to(device)
        model.load_state_dict(torch.load(ckpt, weights_only=True))

        pr = collect_patient_predictions(model, test_data, raw_map, device)
        y_true = np.concatenate([p["labels"] for p in pr])
        y_prob = np.concatenate([p["probs"] for p in pr])

        best_f1_t = find_best_f1_threshold(y_true, y_prob)
        t70 = find_thresh_patient_recall(pr, 0.70)

        entry = {"checkpoint": ckpt}
        for label, t in [("best_f1", best_f1_t), ("recall70", t70)]:
            ts = compute_timestep_metrics(y_true, y_prob, t)
            ew = compute_early_warning_metrics(pr, t)
            ew.pop("lead_time_distribution", None)
            entry[label] = {"timestep": ts, "early_warning": ew}
            print(f"  {label}: t={t:.3f}  pt_recall={ew['patient_recall']:.3f}  "
                  f"precision={ts['precision']:.4f}  "
                  f"alerts/day={ew['alerts_per_patient_day']:.2f}  "
                  f"med_lead={ew['median_lead_time_h']}h  "
                  f"cap6h={ew['capture_rate_by_lead_hour']['6']:.3f}")

        results[name] = entry

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

    # Aggregate per config
    print("\n=== CORRECTED SUMMARY @ 0.70 patient recall (3-seed mean) ===")
    for cfg in ["A_scratch", "B_mae_finetune", "C_mae_frozen"]:
        rows = [results[f"exp6_{cfg}_seed{s}"] for s in [42, 123, 456]
                if f"exp6_{cfg}_seed{s}" in results]
        if not rows:
            continue
        prec = np.mean([r["recall70"]["timestep"]["precision"] for r in rows])
        alerts = np.mean([r["recall70"]["early_warning"]["alerts_per_patient_day"] for r in rows])
        lead = np.mean([r["recall70"]["early_warning"]["median_lead_time_h"] for r in rows])
        cap6 = np.mean([float(r["recall70"]["early_warning"]["capture_rate_by_lead_hour"]["6"]) for r in rows])
        thr = np.mean([r["recall70"]["early_warning"]["threshold"] for r in rows])
        print(f"  {cfg:16s}: thresh={thr:.3f}  precision={prec:.4f}  "
              f"alerts/day={alerts:.2f}  med_lead={lead:.1f}h  cap6h={cap6:.3f}")


if __name__ == "__main__":
    main()
