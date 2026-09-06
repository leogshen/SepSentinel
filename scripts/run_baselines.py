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
from sepsentinel.data.splitting import grouped_patient_split
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from experiment5_recall_study import (
    compute_timestep_metrics, compute_early_warning_metrics,
)
from scripts.run_mve import threshold_at_patient_recall  # noqa: E402

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]
PRIMARY_RECALL = 0.70
REF = {"auroc": "0.814 +/- 0.004", "auprc": "0.144", "precision": "0.093",
       "alerts": "1.7", "lead": "23.5"}


def flatten(data):
    X = np.concatenate([d["signals"] for d in data], axis=0)
    y = np.concatenate([d["labels"] for d in data], axis=0)
    return X, y


def patient_results_from_probs(data, raw_map, probs):
    """Rebuild the per-patient structure the early-warning metrics expect.

    Flat models predict row by row, so probabilities are sliced back out in
    episode order -- no collate_fn sort to replicate here (that hazard is
    specific to the batched sequence path).
    """
    out, start = [], 0
    for d in data:
        n = len(d["labels"])
        raw = raw_map[d["patient_id"]]
        out.append({
            "patient_id": d["patient_id"],
            "label": raw["label"],
            "onset_step": raw["onset_step"],
            "t_sepsis_hour": raw.get("t_sepsis_hour"),
            "probs": probs[start:start + n],
            "labels": np.asarray(d["labels"]),
            "length": n,
        })
        start += n
    assert start == len(probs), "probability vector length mismatch"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="logreg,xgboost")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    print("Episodes %d (%d septic)"
          % (len(episodes), sum(e["label"] for e in episodes)))

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}

    pre = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    data = {"train": pre.fit_transform(splits["train"])}
    data["test"] = pre.transform(splits["test"])
    X_tr, y_tr = flatten(data["train"])
    X_te, y_te = flatten(data["test"])
    print("Train rows %s, test rows %s, positive rate %.3f"
          % (X_tr.shape, X_te.shape, y_tr.mean()))

    scale_pos = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    rows = []
    metrics = {}
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
        pr = patient_results_from_probs(data["test"], raw_map, prob)
        thr, ew = threshold_at_patient_recall(pr, PRIMARY_RECALL)
        ts = compute_timestep_metrics(y_te, prob, thr)
        print("  %-8s AUROC %.3f AUPRC %.3f | at %.0f%% patient recall: "
              "precision %.3f, %.2f alerts/patient-day, median lead %s h  (%.1f min)"
              % (name, auroc, auprc, 100 * ew["patient_recall"],
                 ts.get("precision", float("nan")),
                 ew["alerts_per_patient_day"],
                 ew["median_lead_time_h"], (time.time() - t1) / 60.0))
        rows.append((name, auroc, auprc, ts.get("precision", float("nan")),
                     ew["alerts_per_patient_day"], ew["median_lead_time_h"],
                     ew["capture_rate_by_lead_hour"]))
        metrics[name] = {"auroc": auroc, "auprc": auprc, "threshold": thr,
                         "timestep_precision": ts.get("precision"),
                         "alerts_per_patient_day": ew["alerts_per_patient_day"],
                         "median_lead_time_h": ew["median_lead_time_h"],
                         "patient_recall": ew["patient_recall"],
                         "capture_rate_by_lead_hour":
                             ew["capture_rate_by_lead_hour"]}

    lines = ["# Flat baselines (no sequence model)", "",
             "Episodes: `%s`" % args.episodes,
             "Same grouped split (seed %d), same Strategy B 19 channels, same "
             "metrics as the Transformer runs. Every timestep is an "
             "independent sample." % SPLIT_SEED,
             "Built in %.1f min." % ((time.time() - t0) / 60.0), "",
             "| Model | AUROC | AUPRC | Precision @70% pt recall | "
             "Alerts/patient-day | Median lead (h) |", "|---|---|---|---|---|---|"]
    for name, auroc, auprc, prec, alerts, lead, _ in rows:
        lines.append("| %s | %.3f | %.3f | %.3f | %.2f | %s |"
                     % (name, auroc, auprc, prec, alerts,
                        "%.1f" % lead if lead is not None else "n/a"))
    lines.append("| PhysioNet Transformer (Config I) | %s | %s | %s | %s | %s |"
                 % (REF["auroc"], REF["auprc"], REF["precision"], REF["alerts"],
                    REF["lead"]))
    lines += ["", "Capture rates:", "",
              "| Model | >=3h | >=6h | >=12h |", "|---|---|---|---|"]
    for name, _, _, _, _, _, cap in rows:
        lines.append("| %s | %.2f | %.2f | %.2f |"
                     % (name, cap["3"], cap["6"], cap["12"]))
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
