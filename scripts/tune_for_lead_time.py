#!/usr/bin/env python
"""Hyperparameter search that selects on EARLY WARNING, not AUROC.

Every tuning loop in this project has implicitly optimised discrimination,
because that is what early stopping and model selection watched. This project
has now observed five times that AUROC is anti-correlated with lead time --
including in a synthetic control where a 16x stronger injected signal gave
higher AUROC and shorter warning. Adding static attributes did it again:
+0.014 AUROC, -0.024 capture at 6h.

So the obvious question is whether tuning ON the deployment objective
recovers what tuning on AUROC gives away. This searches XGBoost
hyperparameters and selects on:

    capture >= 6h before onset, at a false-alert burden <= 1.0 per
    nonseptic patient-day

computed on the VALIDATION split, with the test split touched once at the
end for the winner only.

XGBoost rather than the Transformer because a search needs many fits and
XGBoost trains in ~30s against ~14 min; if selecting on the deployment
objective works here, it is worth repeating for the sequence model.

Usage:
    python scripts/tune_for_lead_time.py --episodes results/...pkl \
        --out-dir results/tune_lead --n-trials 24
"""

import argparse
import itertools
import json
import os
import pickle
import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, curve_for, at_burden,
)

GRID = {
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.1, 0.3],
    "n_estimators": [200, 400, 800],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "min_child_weight": [1, 10, 50],
    "reg_lambda": [1.0, 10.0],
}
BUDGET = 1.0          # false alerts per nonseptic patient-day


def score(curve, budget=BUDGET):
    """Selection objective: capture >=6h inside the alert budget."""
    c = at_burden(curve, budget)
    return (c["capture_6h"] if c else 0.0), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    import xgboost as xgb
    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}
    pre = build_preprocessor(episodes, "all")
    tr = pre.fit_transform(splits["train"])
    va = pre.transform(splits["val"])
    te = pre.transform(splits["test"])

    def design(d):
        return (np.concatenate([x["signals"] for x in d]),
                np.concatenate([x["labels"] for x in d]))

    X_tr, y_tr = design(tr)
    X_va, y_va = design(va)
    X_te, y_te = design(te)
    spw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    print("train %s val %s test %s" % (X_tr.shape, X_va.shape, X_te.shape))

    keys = sorted(GRID)
    all_combos = list(itertools.product(*(GRID[k] for k in keys)))
    random.shuffle(all_combos)
    trials = all_combos[:args.n_trials]
    print("searching %d of %d configurations, selecting on capture>=6h at "
          "<=%.1f alerts/patient-day (validation)"
          % (len(trials), len(all_combos), BUDGET))

    rows = []
    for i, combo in enumerate(trials, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        clf = xgb.XGBClassifier(random_state=args.seed, scale_pos_weight=spw,
                                eval_metric="logloss", tree_method="hist",
                                **params).fit(X_tr, y_tr)
        prob = clf.predict_proba(X_va)[:, 1]
        curve = curve_for(patient_results_from_probs(va, raw_map, prob))
        cap, c = score(curve)
        auroc = float(roc_auc_score(y_va, prob))
        rows.append({"params": params, "val_capture_6h": cap,
                     "val_auroc": auroc,
                     "val_recall": c["patient_recall"] if c else None,
                     "val_lead": c["median_lead_time_h"] if c else None,
                     "minutes": (time.time() - t0) / 60.0})
        print("  [%2d/%d] capture6h %.3f  AUROC %.3f  %s  (%.1f min)"
              % (i, len(trials), cap, auroc,
                 " ".join("%s=%s" % (k, params[k]) for k in keys), rows[-1]["minutes"]))

    by_capture = max(rows, key=lambda r: r["val_capture_6h"])
    by_auroc = max(rows, key=lambda r: r["val_auroc"])
    print("\nbest by CAPTURE>=6h : capture %.3f, AUROC %.3f"
          % (by_capture["val_capture_6h"], by_capture["val_auroc"]))
    print("best by AUROC       : capture %.3f, AUROC %.3f"
          % (by_auroc["val_capture_6h"], by_auroc["val_auroc"]))
    print("same configuration? %s" % (by_capture["params"] == by_auroc["params"]))

    # Test set touched once, for both winners, to quantify what the choice costs.
    final = {}
    for tag, row in [("selected_by_capture", by_capture),
                     ("selected_by_auroc", by_auroc)]:
        clf = xgb.XGBClassifier(random_state=args.seed, scale_pos_weight=spw,
                                eval_metric="logloss", tree_method="hist",
                                **row["params"]).fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        curve = curve_for(patient_results_from_probs(te, raw_map, prob))
        cap, c = score(curve)
        final[tag] = {"params": row["params"],
                      "test_auroc": float(roc_auc_score(y_te, prob)),
                      "test_auprc": float(average_precision_score(y_te, prob)),
                      "test_at_1_per_day": c}
        print("\n%s -> test AUROC %.4f, capture>=6h %.3f, recall %.2f, lead %s h"
              % (tag, final[tag]["test_auroc"], cap,
                 c["patient_recall"],
                 "%.1f" % c["median_lead_time_h"]
                 if c["median_lead_time_h"] is not None else "n/a"))

    with open(os.path.join(args.out_dir, "tuning.json"), "w") as fh:
        json.dump({"trials": rows, "final": final}, fh, indent=2)
    print("\nSaved -> %s" % os.path.join(args.out_dir, "tuning.json"))


if __name__ == "__main__":
    main()
