#!/usr/bin/env python
"""Why does performance plateau? Diagnostics, not a model.

Three questions, each answered with a measurement rather than an argument:

1. WHEN is sepsis detectable? Discrimination is recomputed in bins of
   hours-before-onset (septic hours in that bin vs all control hours). If the
   signal only appears close to onset, the ceiling is in the data/labels, not
   the architecture -- no sequence model can find information that is not
   there yet.

2. WHAT is the model actually looking at in the pre-onset window? Measurement
   availability per feature by hours-before-onset. Labs are ~93% missing
   overall; if they are absent in the hours that matter, the effective input
   is four vital signs.

3. Does HISTORY help? Same features, current hour only vs the full causal
   past, is already answered by logreg-vs-Transformer. Here we add the
   cleanest version: how much does discrimination change if the model sees
   only the current hour's raw values with no forward-fill, masks or deltas?

Also emits the full operating curve (patient recall x alert burden x lead
time across all thresholds), so nothing depends on a single chosen
operating point.

Usage:
    python scripts/diagnose_ceiling.py --episodes results/mimic31_full_trunc3.pkl \
        --out-dir results/ceiling_diagnosis
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data.splitting import grouped_patient_split
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from experiment5_recall_study import compute_early_warning_metrics
from scripts.run_baselines import patient_results_from_probs

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]
BINS = [(-6, 0), (-12, -6), (-24, -12), (-48, -24), (-1000, -48)]
BIN_LABELS = ["0-6h before", "6-12h before", "12-24h before",
              "24-48h before", ">48h before"]


def fit_xgb(X, y, seed=42):
    import xgboost as xgb
    spw = float((1 - y.mean()) / max(y.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=seed, scale_pos_weight=spw,
                            eval_metric="logloss", tree_method="hist")
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}

    pre = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    tr = pre.fit_transform(splits["train"])
    te = pre.transform(splits["test"])
    X_tr = np.concatenate([d["signals"] for d in tr])
    y_tr = np.concatenate([d["labels"] for d in tr])
    X_te = np.concatenate([d["signals"] for d in te])
    y_te = np.concatenate([d["labels"] for d in te])

    clf = fit_xgb(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    overall = float(roc_auc_score(y_te, prob))
    print("Overall test AUROC (XGBoost, Strategy B): %.3f" % overall)

    # --- 1. detectability vs hours before onset ---------------------------
    # Positives: septic hours falling in the bin. Negatives held fixed at all
    # hours from nonseptic patients, so bins are directly comparable.
    hours_to_onset, is_septic_pat = [], []
    for d in te:
        raw = raw_map[d["patient_id"]]
        n = len(d["labels"])
        if raw["label"] == 1:
            t = raw["t_sepsis_hour"]
            hours_to_onset.append(np.arange(n, dtype=float) - t)
            is_septic_pat.append(np.ones(n, dtype=bool))
        else:
            hours_to_onset.append(np.full(n, np.nan))
            is_septic_pat.append(np.zeros(n, dtype=bool))
    hto = np.concatenate(hours_to_onset)
    septic_pat = np.concatenate(is_septic_pat)
    control_mask = ~septic_pat
    ctrl_prob = prob[control_mask]

    rows = []
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        m = septic_pat & (hto >= lo) & (hto < hi)
        if m.sum() < 30:
            rows.append((label, int(m.sum()), float("nan")))
            continue
        yy = np.concatenate([np.ones(m.sum()), np.zeros(control_mask.sum())])
        pp = np.concatenate([prob[m], ctrl_prob])
        rows.append((label, int(m.sum()), float(roc_auc_score(yy, pp))))
    print("\nDetectability by time before onset (vs all control hours):")
    for label, n, auc in rows:
        print("  %-14s n=%7d  AUROC %s" % (label, n, "%.3f" % auc if auc == auc
                                           else "n/a"))

    # --- 2. measurement availability before onset -------------------------
    # Raw (pre-imputation) NaN pattern, so this is what was actually measured.
    avail = {}
    for (lo, hi), label in zip(BINS, BIN_LABELS):
        counts = np.zeros(len(CONFIG_I_FEATURES))
        total = 0
        for ep in splits["test"]:
            if ep["label"] != 1:
                continue
            n = len(ep["labels"])
            rel = np.arange(n, dtype=float) - ep["t_sepsis_hour"]
            m = (rel >= lo) & (rel < hi)
            if not m.any():
                continue
            sig = ep["signals"][:, [ALL_FEATURES.index(f)
                                    for f in CONFIG_I_FEATURES]]
            counts += (~np.isnan(sig[m])).sum(axis=0)
            total += int(m.sum())
        avail[label] = {f: (100.0 * counts[j] / total if total else float("nan"))
                        for j, f in enumerate(CONFIG_I_FEATURES)}
        avail[label]["_hours"] = total

    print("\nMeasured-hour percentage in septic episodes (raw, pre-imputation):")
    hdr = "  %-14s " % "window" + " ".join("%7s" % f[:7]
                                           for f in CONFIG_I_FEATURES)
    print(hdr)
    for label in BIN_LABELS:
        if not avail[label]["_hours"]:
            continue
        print("  %-14s " % label + " ".join("%6.1f%%" % avail[label][f]
                                            for f in CONFIG_I_FEATURES))

    # --- 3. does the engineering/history matter? --------------------------
    # Raw current-hour values only: no forward-fill, no masks, no deltas.
    # NaN goes in as NaN (XGBoost handles missing natively).
    idx = [ALL_FEATURES.index(f) for f in CONFIG_I_FEATURES]
    Xr_tr = np.concatenate([ep["signals"][:, idx] for ep in splits["train"]])
    yr_tr = np.concatenate([ep["labels"] for ep in splits["train"]])
    Xr_te = np.concatenate([ep["signals"][:, idx] for ep in splits["test"]])
    yr_te = np.concatenate([ep["labels"] for ep in splits["test"]])
    raw_auc = float(roc_auc_score(yr_te, fit_xgb(Xr_tr, yr_tr)
                                 .predict_proba(Xr_te)[:, 1]))
    print("\nAUROC, raw current-hour values only (no ffill/mask/delta): %.3f"
          % raw_auc)
    print("AUROC, Strategy B 19 channels:                             %.3f"
          % overall)

    # --- 4. full operating curve, no fixed recall point -------------------
    pr = patient_results_from_probs(te, raw_map, prob)
    curve = []
    for t in np.arange(0.02, 1.0, 0.02):
        ew = compute_early_warning_metrics(pr, float(t))
        curve.append({"threshold": float(t),
                      "patient_recall": ew["patient_recall"],
                      "alerts_per_patient_day": ew["alerts_per_patient_day"],
                      "median_lead_time_h": ew["median_lead_time_h"],
                      "capture_6h": ew["capture_rate_by_lead_hour"]["6"],
                      "capture_12h": ew["capture_rate_by_lead_hour"]["12"]})
    print("\nOperating curve (XGBoost), no fixed recall point:")
    print("  recall  alerts/pt-day  median lead  cap>=6h  cap>=12h")
    for c in curve:
        if c["patient_recall"] < 0.05:
            continue
        print("   %.2f       %6.2f        %6s     %.2f     %.2f"
              % (c["patient_recall"], c["alerts_per_patient_day"],
                 "%.1f" % c["median_lead_time_h"]
                 if c["median_lead_time_h"] is not None else "n/a",
                 c["capture_6h"], c["capture_12h"]))

    with open(os.path.join(args.out_dir, "diagnosis.json"), "w") as fh:
        json.dump({"overall_auroc": overall, "raw_current_hour_auroc": raw_auc,
                   "detectability_by_window": [
                       {"window": l, "n_hours": n, "auroc": a}
                       for l, n, a in rows],
                   "availability_pct": avail, "operating_curve": curve},
                  fh, indent=2)
    print("\nSaved -> %s" % os.path.join(args.out_dir, "diagnosis.json"))


if __name__ == "__main__":
    main()
