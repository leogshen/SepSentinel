#!/usr/bin/env python
"""Did the PhysioNet baseline suffer the same post-onset problem?

The pre-onset target bought +7h of lead time on MIMIC-IV. The obvious next
question is whether the PhysioNet Config I baseline -- the number this whole
project has been quoting -- was leaving the same lead time on the table.

PhysioNet is partly protected: the Challenge truncated records at about
t_sepsis + 4h, so its septic episodes carry far fewer established-sepsis
hours than our +24h MIMIC variant did. But 37% of its positive hours are
still post-onset, and its labels still turn on at t_sepsis - 6 rather than
being confined to a prodrome window. So the flaw is present, just milder.

This runs both targets on identical PhysioNet episodes, identical split,
identical metrics:
  standard  labels as shipped (1 from t_sepsis - 6 onward)
  prodrome  episode truncated at t_sepsis, positives only in
            [t_sepsis - W, t_sepsis)

t_sepsis = onset_step + 6, since CinC 2019 labels are pre-shifted by 6h.

Usage:
    python scripts/physionet_prodrome_check.py --out-dir results/physionet_prodrome
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import FEATURES as ALL_FEATURES, SPLIT_SEED
from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, curve_for, at_burden,
    BURDEN_POINTS,
)

DEFAULT_CACHE = ("C:/Users/Chaopeng Shen/.cache/kagglehub/datasets/"
                 "tea340yashjoshi/sepsis-prediction-dataset/versions/1/"
                 "Dataset.csv")


def make_variants(episodes, window_h):
    """(standard, prodrome) copies with t_sepsis_hour set on both.

    Setting t_sepsis_hour explicitly on BOTH matters: the lead-time metric
    otherwise falls back to onset_step + 6 for one variant and reads the
    stored value for the other, and the two must be identical for the
    comparison to mean anything.
    """
    std, pro = [], []
    for ep in episodes:
        if ep["label"] != 1:
            base = dict(ep, t_sepsis_hour=None)
            std.append(base)
            pro.append(base)
            continue
        t = float(ep["onset_step"] + 6)          # CinC labels are pre-shifted
        std.append(dict(ep, t_sepsis_hour=t))

        n = int(min(np.ceil(t), len(ep["labels"])))
        if n < 6:                                 # too short once truncated
            continue
        lab = np.zeros(n, dtype=np.float32)
        start = max(0, int(np.ceil(t - window_h)))
        if start < n:
            lab[start:] = 1.0
        if lab.max() == 0:
            continue
        pro.append(dict(ep, signals=ep["signals"][:n], labels=lab,
                        time=ep["time"][:n], t_sepsis_hour=t,
                        onset_step=int(np.argmax(lab > 0)), label=1))
    return std, pro


def evaluate(episodes, name, seed=42):
    import xgboost as xgb
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}
    pre = build_preprocessor(episodes, "all")
    tr = pre.fit_transform(splits["train"])
    te = pre.transform(splits["test"])
    X_tr = np.concatenate([d["signals"] for d in tr])
    y_tr = np.concatenate([d["labels"] for d in tr])
    X_te = np.concatenate([d["signals"] for d in te])
    y_te = np.concatenate([d["labels"] for d in te])
    spw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))

    out = {}
    for model in ("logreg", "xgboost"):
        if model == "logreg":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=seed).fit(X_tr, y_tr)
        else:
            clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                    learning_rate=0.1, random_state=seed,
                                    scale_pos_weight=spw, eval_metric="logloss",
                                    tree_method="hist").fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        curve = curve_for(patient_results_from_probs(te, raw_map, prob))
        auroc = float(roc_auc_score(y_te, prob))
        out[model] = {"auroc": auroc,
                      "auprc": float(average_precision_score(y_te, prob)),
                      "curve": curve}
        print("  %-9s %-8s AUROC %.3f" % (name, model, auroc))
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
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window", type=float, default=12.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Explicit canonical feature list: STAGES[3] contains il6, which
    # PhysioNet does not measure.
    episodes = load_physionet(args.data, features=ALL_FEATURES)
    print("PhysioNet: %d episodes, %d septic, features %s"
          % (len(episodes), sum(e["label"] for e in episodes),
             episodes[0]["features"]))

    std, pro = make_variants(episodes, args.window)
    print("standard: %d episodes (%d septic) | prodrome: %d (%d septic)"
          % (len(std), sum(e["label"] for e in std),
             len(pro), sum(e["label"] for e in pro)))

    results = {"standard": evaluate(std, "standard"),
               "prodrome": evaluate(pro, "prodrome")}

    lines = ["# PhysioNet 2019: standard vs pre-onset target", "",
             "Same episodes, same split (seed %d), same metrics. `standard` is "
             "the label as shipped by the Challenge (1 from t_sepsis-6 "
             "onward); `prodrome` truncates each septic episode at t_sepsis "
             "and keeps positives only in the %.0f h before it."
             % (SPLIT_SEED, args.window), "",
             "Lead time and capture are anchored to the same unshifted "
             "t_sepsis = onset_step + 6 in both, so they are comparable; "
             "AUROC is not (different labels).", ""]
    for b in BURDEN_POINTS:
        lines += ["**Budget: %.1f false alerts per nonseptic patient-day**" % b,
                  "", "| Target | Model | Patient recall | Median lead (h) | "
                  "Capture >=6h | Capture >=12h |", "|---|---|---|---|---|---|"]
        for target in ("standard", "prodrome"):
            for model in ("logreg", "xgboost"):
                c = at_burden(results[target][model]["curve"], b)
                if c is None:
                    lines.append("| %s | %s | (unreachable) | | | |"
                                 % (target, model))
                    continue
                lines.append("| %s | %s | %.2f | %s | %.2f | %.2f |"
                             % (target, model, c["patient_recall"],
                                "%.1f" % c["median_lead_time_h"]
                                if c["median_lead_time_h"] is not None else "n/a",
                                c["capture_6h"], c["capture_12h"]))
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "PHYSIONET_PRODROME.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print()
    print(text)


if __name__ == "__main__":
    main()
