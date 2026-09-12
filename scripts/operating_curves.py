#!/usr/bin/env python
"""Model comparison across the WHOLE operating range -- no fixed recall point.

A single chosen operating point (we had been quoting 70% patient recall,
inherited from the PhysioNet baseline) hides more than it shows: models that
look close at one threshold can separate elsewhere, and forcing equal recall
silently trades away the metric you care about. This reports the full curve
instead: for every threshold, patient recall against alert burden, lead time
and capture.

Reuses trained Transformer checkpoints if they exist (inference only, no
retraining) and refits the flat baselines, which take seconds.

Usage:
    python scripts/operating_curves.py \
        --episodes results/mimic31_full_trunc3.pkl \
        --checkpoints results/full_trunc3_transformer \
        --out-dir results/operating_curves
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.transformer import SepsisTransformer
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from experiment5_recall_study import (
    collect_patient_predictions, compute_early_warning_metrics,
)

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]
# Report at fixed ALERT BURDENS rather than fixed recall: the burden is the
# constraint a clinical deployment actually imposes ("how many false alarms
# per patient-day will the unit tolerate"), and unlike recall it is
# comparable across models without forcing anything.
BURDEN_POINTS = [0.5, 1.0, 2.0, 4.0]


def build_preprocessor(episodes, selection="all"):
    """AblationPreprocessor matched to whatever feature set the episodes carry.

    Episodes record their own `features` and `vitals`, so a Config-I pickle
    and an extended one both work without the caller knowing which is which.
    `selection` is "all" (every extracted feature) or "config_i" (the nine
    PhysioNet-comparable ones, for like-for-like comparisons).
    """
    all_features = list(episodes[0]["features"])
    vitals = list(episodes[0].get("vitals", all_features[:4]))
    if selection == "config_i":
        selected = [f for f in CONFIG_I_FEATURES if f in all_features]
        missing = [f for f in CONFIG_I_FEATURES if f not in all_features]
        if missing:
            raise SystemExit("episodes lack Config I features: %s" % missing)
    else:
        selected = list(all_features)
    return AblationPreprocessor(all_features, selected, vitals=vitals)


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


def curve_for(patient_results):
    """Operating curve over thresholds.

    Carries both levels of precision, which answer different questions:
      timestep_precision  of all alarm-hours raised, the fraction that were
                          labelled positive -- the alarm-level burden a nurse
                          experiences hour by hour.
      patient_precision   of all patients ever alarmed, the fraction who were
                          genuinely septic -- the number a clinician means by
                          "when it fires, how often is it right".
    Timestep recall and F1 accompany timestep precision so the raw
    confusion-matrix view is available without recomputing: AUROC is
    optimistic at this prevalence and should never be quoted alone.
    Patient precision is the higher and more favourable of the two; both are
    reported so neither can be quoted selectively.
    """
    y_true = np.concatenate([np.asarray(p["labels"]) for p in patient_results])
    y_prob = np.concatenate([np.asarray(p["probs"]) for p in patient_results])
    is_septic = np.array([p["label"] == 1 for p in patient_results])

    out = []
    for t in np.arange(0.01, 1.0, 0.01):
        ew = compute_early_warning_metrics(patient_results, float(t))
        pred = y_prob >= t
        tp = float((pred & (y_true == 1)).sum())
        fp = float((pred & (y_true == 0)).sum())
        fn = float(((~pred) & (y_true == 1)).sum())
        ts_prec = tp / (tp + fp) if (tp + fp) else float("nan")
        ts_rec = tp / (tp + fn) if (tp + fn) else float("nan")
        alarmed = np.array([bool((np.asarray(p["probs"]) >= t).any())
                            for p in patient_results])
        n_alarmed = int(alarmed.sum())
        out.append({
            "threshold": float(t),
            "patient_recall": ew["patient_recall"],
            "alerts_per_patient_day": ew["alerts_per_patient_day"],
            "median_lead_time_h": ew["median_lead_time_h"],
            "timestep_precision": ts_prec,
            "timestep_recall": ts_rec,
            "timestep_f1": (2 * ts_prec * ts_rec / (ts_prec + ts_rec)
                            if (tp + fp) and (tp + fn) and (ts_prec + ts_rec)
                            else float("nan")),
            "patient_precision": (float((alarmed & is_septic).sum()) / n_alarmed
                                  if n_alarmed else float("nan")),
            "patients_alarmed": n_alarmed,
            "capture_3h": ew["capture_rate_by_lead_hour"]["3"],
            "capture_6h": ew["capture_rate_by_lead_hour"]["6"],
            "capture_12h": ew["capture_rate_by_lead_hour"]["12"],
        })
    return out


def at_burden(curve, budget):
    """Best-recall point whose alert burden stays within budget."""
    ok = [c for c in curve if c["alerts_per_patient_day"] <= budget]
    return max(ok, key=lambda c: c["patient_recall"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--checkpoints", default=None,
                    help="directory holding checkpoints_seed*/best_model.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--features", default="all", choices=["all", "config_i"],
                    help="which extracted features to feed the models; must "
                         "match what any supplied checkpoints were trained on")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}

    pre = build_preprocessor(episodes, args.features)
    tr = pre.fit_transform(splits["train"])
    te = pre.transform(splits["test"])
    X_tr = np.concatenate([d["signals"] for d in tr])
    y_tr = np.concatenate([d["labels"] for d in tr])
    X_te = np.concatenate([d["signals"] for d in te])
    y_te = np.concatenate([d["labels"] for d in te])
    print("train %s  test %s  positive rate %.3f"
          % (X_tr.shape, X_te.shape, y_tr.mean()))

    models = {}

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=42).fit(X_tr, y_tr)
    models["logreg"] = clf.predict_proba(X_te)[:, 1]

    import xgboost as xgb
    spw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    xclf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=42, scale_pos_weight=spw,
                             eval_metric="logloss", tree_method="hist")
    xclf.fit(X_tr, y_tr)
    models["xgboost"] = xclf.predict_proba(X_te)[:, 1]

    # Transformer: reuse checkpoints, average probabilities over seeds.
    tf_results = None
    if args.checkpoints:
        ckpts = sorted(glob.glob(os.path.join(args.checkpoints,
                                              "checkpoints_seed*",
                                              "best_model.pt")))
        per_seed = []
        for ck in ckpts:
            model = SepsisTransformer(input_dim=pre.n_channels)
            model.load_state_dict(torch.load(ck, weights_only=True))
            model.to(args.device)
            res = collect_patient_predictions(model, te, raw_map, args.device)
            per_seed.append(res)
            print("loaded %s" % ck)
        if per_seed:
            # collect_patient_predictions returns patients in a fixed order,
            # so seeds align patient-for-patient.
            tf_results = []
            for i, base in enumerate(per_seed[0]):
                probs = np.mean([s[i]["probs"] for s in per_seed], axis=0)
                assert all(s[i]["patient_id"] == base["patient_id"]
                           for s in per_seed), "seed ordering mismatch"
                tf_results.append(dict(base, probs=probs))

    report = {}
    lines = ["# Operating curves -- no fixed recall point", "",
             "Episodes: `%s`" % args.episodes,
             "Test set: %d episodes, %d septic."
             % (len(splits["test"]), sum(e["label"] for e in splits["test"])),
             "",
             "Transformer probabilities are averaged over the seed "
             "checkpoints. Flat baselines are single-fit.", "",
             "## Threshold-free discrimination", "",
             "| Model | AUROC | AUPRC |", "|---|---|---|"]

    curves = {}
    for name, prob in models.items():
        pr = patient_results_from_probs(te, raw_map, prob)
        curves[name] = curve_for(pr)
        lines.append("| %s | %.3f | %.3f |"
                     % (name, roc_auc_score(y_te, prob),
                        average_precision_score(y_te, prob)))
        report[name] = {"auroc": float(roc_auc_score(y_te, prob)),
                        "auprc": float(average_precision_score(y_te, prob)),
                        "curve": curves[name]}
    if tf_results is not None:
        yt = np.concatenate([p["labels"] for p in tf_results])
        yp = np.concatenate([p["probs"] for p in tf_results])
        curves["transformer"] = curve_for(tf_results)
        lines.append("| transformer (seed-avg) | %.3f | %.3f |"
                     % (roc_auc_score(yt, yp), average_precision_score(yt, yp)))
        report["transformer"] = {"auroc": float(roc_auc_score(yt, yp)),
                                 "auprc": float(average_precision_score(yt, yp)),
                                 "curve": curves["transformer"]}

    lines += ["", "## At equal alert burden", "",
              "Alert burden is the constraint a unit actually imposes, and "
              "comparing at equal burden forces nothing about recall.", ""]
    for budget in BURDEN_POINTS:
        lines += ["**Budget: %.1f false alerts per nonseptic patient-day**"
                  % budget, "",
                  "| Model | Patient recall | Timestep prec. | Patient prec. | "
                  "Median lead (h) | Capture >=6h | Capture >=12h |",
                  "|---|---|---|---|---|---|---|"]
        for name in curves:
            c = at_burden(curves[name], budget)
            if c is None:
                lines.append("| %s | (unreachable) | | | |" % name)
                continue
            lines.append("| %s | %.2f | %.3f | %.3f | %s | %.2f | %.2f |"
                         % (name, c["patient_recall"], c["timestep_precision"],
                            c["patient_precision"],
                            "%.1f" % c["median_lead_time_h"]
                            if c["median_lead_time_h"] is not None else "n/a",
                            c["capture_6h"], c["capture_12h"]))
        lines.append("")

    lines += ["## Full curves", ""]
    for name in curves:
        lines += ["### %s" % name, "",
                  "| Threshold | Patient recall | Alerts/pt-day | "
                  "Median lead (h) | Cap >=6h | Cap >=12h |",
                  "|---|---|---|---|---|---|"]
        for c in curves[name][::5]:
            if c["patient_recall"] < 0.02:
                continue
            lines.append("| %.2f | %.2f | %.2f | %s | %.2f | %.2f |"
                         % (c["threshold"], c["patient_recall"],
                            c["alerts_per_patient_day"],
                            "%.1f" % c["median_lead_time_h"]
                            if c["median_lead_time_h"] is not None else "n/a",
                            c["capture_6h"], c["capture_12h"]))
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "OPERATING_CURVES.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "curves.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print()
    print("\n".join(lines[:60]))
    print("...\nSaved -> %s" % os.path.join(args.out_dir, "OPERATING_CURVES.md"))


if __name__ == "__main__":
    main()
