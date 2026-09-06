#!/usr/bin/env python
"""Flat (non-sequential) baselines on the same episodes, split and metrics.

Answers one question before any architecture work: how much of the
performance comes from the DATA, and how much from the sequence model?
Every timestep is an independent sample here -- no temporal model at all --
but the features are the identical Strategy B 19 channels (which already
carry causal forward-fill, observation masks and time-since-last deltas), the
split is the same grouped subject-level split with the same seed, and the
metrics are the same corrected patient-level ones.

Read the results as a floor: whatever the Transformer scores above these
numbers is what the sequence model is buying.

No recall target is imposed. Results are reported threshold-free (AUROC,
AUPRC) and at equal ALERT BURDEN, which is the constraint a unit actually
imposes; see scripts/operating_curves.py for the full curves.

Usage:
    python scripts/run_baselines.py --episodes results/mimic31_full_trunc3.pkl \
        --out-dir results/baselines_full_trunc3
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    patient_results_from_probs, build_preprocessor, curve_for, at_burden,
    BURDEN_POINTS,
)

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]


def flatten(data):
    X = np.concatenate([d["signals"] for d in data], axis=0)
    y = np.concatenate([d["labels"] for d in data], axis=0)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="logreg,xgboost")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="all", choices=["all", "config_i"],
                    help="which extracted features to feed the model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    print("Episodes %d (%d septic)"
          % (len(episodes), sum(e["label"] for e in episodes)))

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}

    pre = build_preprocessor(episodes, args.features)
    data = {"train": pre.fit_transform(splits["train"])}
    data["test"] = pre.transform(splits["test"])
    X_tr, y_tr = flatten(data["train"])
    X_te, y_te = flatten(data["test"])
    print("Train rows %s, test rows %s, positive rate %.3f"
          % (X_tr.shape, X_te.shape, y_tr.mean()))

    scale_pos = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    rows, metrics = [], {}
    for name in args.models.split(","):
        name = name.strip()
        t1 = time.time()
        if name == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=args.seed)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
        elif name == "xgboost":
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=args.seed, scale_pos_weight=scale_pos,
                eval_metric="logloss", tree_method="hist")
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
        else:
            raise SystemExit("unknown model: %s" % name)

        auroc = float(roc_auc_score(y_te, prob))
        auprc = float(average_precision_score(y_te, prob))
        curve = curve_for(patient_results_from_probs(data["test"], raw_map, prob))
        print("  %-8s AUROC %.3f AUPRC %.3f  (%.1f min)"
              % (name, auroc, auprc, (time.time() - t1) / 60.0))
        for b in BURDEN_POINTS:
            c = at_burden(curve, b)
            if c is None:
                continue
            print("      at <=%.1f alerts/pt-day: recall %.2f, median lead %s h,"
                  " capture>=6h %.2f"
                  % (b, c["patient_recall"],
                     "%.1f" % c["median_lead_time_h"]
                     if c["median_lead_time_h"] is not None else "n/a",
                     c["capture_6h"]))
        rows.append((name, auroc, auprc, curve))
        metrics[name] = {"auroc": auroc, "auprc": auprc, "curve": curve}

    lines = ["# Flat baselines (no sequence model)", "",
             "Episodes: `%s`" % args.episodes,
             "Same grouped split (seed %d), same Strategy B 19 channels, same "
             "metrics as the Transformer runs. Every timestep is an "
             "independent sample." % SPLIT_SEED,
             "Built in %.1f min." % ((time.time() - t0) / 60.0), "",
             "## Threshold-free discrimination", "",
             "| Model | AUROC | AUPRC |", "|---|---|---|"]
    for name, auroc, auprc, _ in rows:
        lines.append("| %s | %.3f | %.3f |" % (name, auroc, auprc))

    lines += ["", "## At equal alert burden", "",
              "No recall target is imposed: each row is the best patient "
              "recall reachable inside the stated false-alert budget.", ""]
    for b in BURDEN_POINTS:
        lines += ["**Budget: %.1f false alerts per nonseptic patient-day**" % b,
                  "", "| Model | Patient recall | Median lead (h) | "
                  "Capture >=6h | Capture >=12h |", "|---|---|---|---|---|"]
        for name, _, _, curve in rows:
            c = at_burden(curve, b)
            if c is None:
                lines.append("| %s | (unreachable) | | | |" % name)
                continue
            lines.append("| %s | %.2f | %s | %.2f | %.2f |"
                         % (name, c["patient_recall"],
                            "%.1f" % c["median_lead_time_h"]
                            if c["median_lead_time_h"] is not None else "n/a",
                            c["capture_6h"], c["capture_12h"]))
        lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "BASELINES.md"), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print()
    print(report)


if __name__ == "__main__":
    main()
