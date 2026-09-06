#!/usr/bin/env python
"""Synthetic positive control: how big must a pre-onset signal be to be seen?

The real data shows no detectability cliff before onset (AUROC 0.72 at 0-6h,
0.61 beyond 48h). Two explanations fit that: (a) no strong pre-onset signal
exists in the measured variables, or (b) the pipeline cannot recover one. This
distinguishes them by MANUFACTURING a cliff and checking whether we find it.

Into septic episodes only, a prodrome is injected starting `--window` hours
before onset: a linear ramp reaching `effect` standard deviations by onset.
Controls are untouched. Sweeping effect size gives the pipeline's sensitivity
floor -- the smallest pre-onset shift it can turn into an alarm.

Crucially the same effect is injected two ways:
  * into a DENSE channel (heart rate, charted ~95% of hours), and
  * into a SPARSE channel (lactate, measured in ~4% of pre-onset hours),
    perturbing only the hours where a measurement actually exists.
The gap between those two curves is the value of measurement FREQUENCY at
matched biological effect size -- i.e. the quantitative case for continuous
sensing over intermittent labs, which is Model A's design question.

Effect size 0.0 is the control condition and must reproduce the untouched
baseline; the script asserts nothing about it but it should be checked.

Usage:
    python scripts/positive_control.py --episodes results/mve_c_5k_trunc3.pkl \
        --out-dir results/positive_control
"""

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from sepsentinel.data.preprocessing import CLIP_RANGES
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    patient_results_from_probs, curve_for, at_burden,
)

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]
DENSE_CHANNEL = "heart_rate"     # charted ~95% of hours
SPARSE_CHANNEL = "lactate"       # measured ~4% of pre-onset hours
EFFECTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
BUDGET = 1.0                     # alerts per nonseptic patient-day


def inject(episodes, feature, effect_sd, window_h, sd):
    """Copy of `episodes` with a pre-onset ramp added to one channel.

    The ramp is added ONLY where a real measurement exists, so a sparse
    channel stays sparse: injecting into lactate perturbs the handful of
    hours lactate was actually drawn, exactly as a real biomarker shift
    would only be visible when someone measured it.
    """
    j = ALL_FEATURES.index(feature)
    out = []
    for ep in episodes:
        if ep["label"] != 1 or effect_sd <= 0:
            out.append(ep)          # untouched: nothing downstream mutates it
            continue
        t = ep["t_sepsis_hour"]
        n = ep["signals"].shape[0]
        hours = np.arange(n, dtype=float)
        # 0 at window start, full effect at onset, held after
        ramp = np.clip((hours - (t - window_h)) / max(window_h, 1e-9), 0, 1)
        sig = ep["signals"].copy()
        col = sig[:, j]
        observed = ~np.isnan(col)
        col[observed] = col[observed] + effect_sd * sd * ramp[observed]
        sig[:, j] = col
        out.append(dict(ep, signals=sig))
    return out


def evaluate(episodes, seed=42):
    """Grouped split -> Strategy B -> XGBoost -> AUROC and lead at BUDGET."""
    import xgboost as xgb
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}
    pre = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    tr = pre.fit_transform(splits["train"])
    te = pre.transform(splits["test"])
    X_tr = np.concatenate([d["signals"] for d in tr])
    y_tr = np.concatenate([d["labels"] for d in tr])
    X_te = np.concatenate([d["signals"] for d in te])
    y_te = np.concatenate([d["labels"] for d in te])
    spw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=seed, scale_pos_weight=spw,
                            eval_metric="logloss", tree_method="hist")
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    curve = curve_for(patient_results_from_probs(te, raw_map, prob))
    c = at_burden(curve, BUDGET)
    return {"auroc": float(roc_auc_score(y_te, prob)),
            "recall_at_budget": c["patient_recall"] if c else None,
            "lead_at_budget": c["median_lead_time_h"] if c else None,
            "capture_6h": c["capture_6h"] if c else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=float, default=12.0,
                    help="hours before onset over which the ramp builds")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    print("Episodes %d (%d septic); prodrome window %.0fh"
          % (len(episodes), sum(e["label"] for e in episodes), args.window))

    # Per-channel SD from observed values only, so "1 SD" means one SD of the
    # real distribution of that measurement.
    # SD must be measured on the distribution the MODEL sees: raw MIMIC
    # contains impossible values (heart rate up to 11,337 bpm) whose SD is
    # meaningless, and the preprocessor clips to CLIP_RANGES anyway. Using the
    # raw SD silently gives a dense channel a far larger perturbation per
    # "SD" than a sparse one, which would rig this comparison.
    sds = {}
    for f in (DENSE_CHANNEL, SPARSE_CHANNEL):
        j = ALL_FEATURES.index(f)
        vals = np.concatenate([ep["signals"][:, j] for ep in episodes])
        if f in CLIP_RANGES:
            lo, hi = CLIP_RANGES[f]
            vals = np.where((vals < lo) | (vals > hi), np.nan, vals)
        sds[f] = float(np.nanstd(vals))
        obs = float(np.mean(~np.isnan(vals)) * 100)
        print("  %-12s SD %.2f, measured in %.1f%% of all hours"
              % (f, sds[f], obs))

    results = {}
    for feature in (DENSE_CHANNEL, SPARSE_CHANNEL):
        results[feature] = []
        for eff in EFFECTS:
            eps = inject(episodes, feature, eff, args.window, sds[feature])
            m = evaluate(eps)
            m["effect_sd"] = eff
            results[feature].append(m)
            print("  %-12s effect %.2f SD -> AUROC %.3f | at <=%.1f alerts/day:"
                  " recall %.2f, lead %s h, capture>=6h %s"
                  % (feature, eff, m["auroc"], BUDGET,
                     m["recall_at_budget"] or float("nan"),
                     "%.1f" % m["lead_at_budget"]
                     if m["lead_at_budget"] is not None else "n/a",
                     "%.2f" % m["capture_6h"]
                     if m["capture_6h"] is not None else "n/a"))

    lines = ["# Synthetic positive control", "",
             "Episodes: `%s`" % args.episodes,
             "A prodrome is injected into septic episodes only, ramping "
             "linearly to `effect` SD over the %.0f h before onset. Controls "
             "are untouched. The injection is applied only where a real "
             "measurement exists, so a sparse channel stays sparse." % args.window,
             "",
             "| Channel | Measured in | SD |", "|---|---|---|"]
    for f in (DENSE_CHANNEL, SPARSE_CHANNEL):
        j = ALL_FEATURES.index(f)
        vals = np.concatenate([ep["signals"][:, j] for ep in episodes])
        lines.append("| %s | %.1f%% of hours | %.2f |"
                     % (f, 100 * np.mean(~np.isnan(vals)), sds[f]))
    lines.append("")

    for feature in (DENSE_CHANNEL, SPARSE_CHANNEL):
        lines += ["## Injected into `%s`" % feature, "",
                  "| Effect (SD) | AUROC | Recall @<=%.1f alerts/day | "
                  "Median lead (h) | Capture >=6h |" % BUDGET,
                  "|---|---|---|---|---|"]
        for m in results[feature]:
            lines.append("| %.2f | %.3f | %s | %s | %s |"
                         % (m["effect_sd"], m["auroc"],
                            "%.2f" % m["recall_at_budget"]
                            if m["recall_at_budget"] is not None else "n/a",
                            "%.1f" % m["lead_at_budget"]
                            if m["lead_at_budget"] is not None else "n/a",
                            "%.2f" % m["capture_6h"]
                            if m["capture_6h"] is not None else "n/a"))
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "POSITIVE_CONTROL.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump({"window_h": args.window, "sds": sds, "results": results},
                  fh, indent=2)
    print()
    print(text)


if __name__ == "__main__":
    main()
