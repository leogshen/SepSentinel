#!/usr/bin/env python
"""Train toward the metric we actually report: capture at fixed alert burden.

The project's recurring finding is that gains come from aligning the
optimisation target with the reporting target. The product decision now fixes
that target: the alert goes to a nurse or clinical response team, whose
action (cultures, fluids, escalation) needs hours to matter, so the governing
metric is CAPTURE >=6h at constrained burden.

The current target does not encode that at all. Every hour in the pre-onset
window carries label 1, so an alarm 1 h before onset scores exactly like an
alarm 11 h before -- even though only one of them gives the team time to act.

The CinC 2019 winner's largest single lever was to stop optimising the binary
label and regress on the per-timestep utility differential instead. This is
that, with our cost structure:

    B(l) = 0                              l <= L_MIN    (no time to act)
         = (l - L_MIN)/(L_SAT - L_MIN)    between
         = 1                              l >= L_SAT    (full credit)

for l = hours until onset. With L_SAT = 6 this is capture >=6h written as a
training weight.

A NOTE ON TLS, which is the same machinery pointed the other way. Temporal
Label Smoothing sets q = 1 AT onset and decays backwards, so it weights the
hours immediately before onset most heavily -- the hours that are worth least
for early warning. Under the product decision TLS's ramp is BACKWARDS. That
is a concrete, testable prediction rather than a criticism: if the ramp
direction is what matters, the utility arm should beat both hard labels and
TLS on capture, and TLS should look better on plain recall (which credits any
lead > 0) than on capture. Both arms are run here so the prediction can fail.

Weighting uses the same exact weighted-CE reduction as xgb_tls_experiment:
    w = spw*q + (1 - q),  y = spw*q / w
which equals scale_pos_weight semantics at q in {0,1}. A mass-matched utility
arm is included because any ramp carries less total positive mass than hard
labels, and that confound produced a phantom result once already.

Evaluation: exact equal burden across arms, paired patient bootstrap.

Usage:
    python scripts/utility_surrogate.py \
        --episodes results/mimic31_pooled17.pkl \
        --out-dir results/utility_surrogate
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, threshold_for_exact_burden,
)
from scripts.xgb_tls_experiment import (
    per_patient_stats, reduce_stats, paired_bootstrap, weighted_targets,
    tls_from_onset, fmt,
)

L_MIN = 1.0    # below this, the alarm arrives too late to change anything
L_SAT = 6.0    # at/above this, full credit -- matches capture >=6h
KEYS = ("capture_6h", "capture_12h", "capture_3h", "patient_recall",
        "patient_precision", "timestep_precision", "timestep_recall",
        "timestep_f1", "median_lead_time_h", "alerts_per_patient_day")
LABEL = {"capture_6h": "**Capture >=6h**", "capture_12h": "Capture >=12h",
         "capture_3h": "Capture >=3h", "patient_recall": "Patient recall",
         "patient_precision": "Patient precision",
         "timestep_precision": "Timestep precision",
         "timestep_recall": "Timestep recall", "timestep_f1": "Timestep F1",
         "median_lead_time_h": "Median lead (h)",
         "alerts_per_patient_day": "Realised burden (alerts/pt-day)"}


def utility_ramp(n_hours, t_sepsis_hour, window_h):
    """B(l): value of alarming at each hour, rising with lead time.

    Zero outside the prodrome window, zero within L_MIN of onset, saturating
    at L_SAT. The mirror image of the TLS ramp.
    """
    q = np.zeros(n_hours, dtype=np.float32)
    if t_sepsis_hour is None:
        return q
    l = t_sepsis_hour - np.arange(n_hours, dtype=np.float32)
    inside = (l > 0) & (l <= window_h)
    q[inside] = np.clip((l[inside] - L_MIN) / (L_SAT - L_MIN), 0.0, 1.0)
    return q


def fit_predict(X_tr, y, w, seed, X_eval):
    import xgboost as xgb
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
              "tree_method": "hist", "eval_metric": "logloss", "seed": seed}
    bst = xgb.train(params, xgb.DMatrix(X_tr, label=y, weight=w),
                    num_boost_round=200)
    return {k: bst.predict(xgb.DMatrix(v)) for k, v in X_eval.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-h", type=float, default=12.0)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    splits = grouped_patient_split(eps, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    pre = build_preprocessor(eps, "all")
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    y_hard = np.concatenate([d["labels"] for d in data["train"]])
    spw = float((1 - y_hard.mean()) / max(y_hard.mean(), 1e-9))
    print("%d episodes | train positive rate %.4f | spw %.1f"
          % (len(eps), y_hard.mean(), spw))

    q_tls = np.concatenate([
        tls_from_onset(len(d["labels"]),
                       raw_map[d["patient_id"]]["t_sepsis_hour"],
                       args.window_h, args.gamma) for d in data["train"]])
    q_util = np.concatenate([
        utility_ramp(len(d["labels"]),
                     raw_map[d["patient_id"]]["t_sepsis_hour"],
                     args.window_h) for d in data["train"]])
    mass = {"hard": float(y_hard.sum()), "tls": float(q_tls.sum()),
            "utility": float(q_util.sum())}
    ratio = mass["hard"] / max(mass["utility"], 1e-9)
    print("positive mass: hard %.0f | tls %.0f (%.0f%%) | utility %.0f (%.0f%%)"
          % (mass["hard"], mass["tls"], 100 * mass["tls"] / mass["hard"],
             mass["utility"], 100 * mass["utility"] / mass["hard"]))

    configs = [("hard", y_hard, spw), ("tls", q_tls, spw),
               ("utility", q_util, spw),
               ("utility_massmatched", q_util, spw * ratio)]

    stats, results = {}, {}
    for name, q, arm_spw in configs:
        t1 = time.time()
        yy, ww = weighted_targets(q.astype(np.float64), arm_spw)
        probs = fit_predict(X["train"], yy, ww, args.seed,
                            {"test": X["test"]})
        pr = patient_results_from_probs(data["test"], raw_map, probs["test"])
        st = per_patient_stats(pr, threshold_for_exact_burden(pr, args.burden))
        stats[name] = st
        yte = np.concatenate([d["labels"] for d in data["test"]])
        results[name] = dict(reduce_stats(st), scale_pos_weight=arm_spw,
                             auroc=float(roc_auc_score(yte, probs["test"])),
                             auprc=float(average_precision_score(
                                 yte, probs["test"])),
                             minutes=(time.time() - t1) / 60.0)
        m = results[name]
        print("  %-20s cap6h %.3f  cap12h %.3f  recall %.3f  lead %s  "
              "burden %.2f  AUROC %.4f  (%.1f min)"
              % (name, m["capture_6h"], m["capture_12h"], m["patient_recall"],
                 fmt(m["median_lead_time_h"], "%.1f"),
                 m["alerts_per_patient_day"], m["auroc"], m["minutes"]))

    contrasts = {n: paired_bootstrap(stats["hard"], stats[n], args.n_boot,
                                     args.seed, KEYS)
                 for n in stats if n != "hard"}
    contrasts["utility_vs_tls"] = paired_bootstrap(
        stats["tls"], stats["utility"], args.n_boot, args.seed, KEYS)

    out = {"config": {"episodes": args.episodes, "window_h": args.window_h,
                      "l_min": L_MIN, "l_sat": L_SAT, "burden": args.burden,
                      "positive_mass": mass, "massmatch_ratio": ratio},
           "arms": results, "paired_bootstrap": contrasts,
           "minutes": (time.time() - t0) / 60.0}
    with open(os.path.join(args.out_dir, "utility.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    order = ["hard", "tls", "utility", "utility_massmatched"]
    lines = [
        "# Training toward capture, not toward the label", "",
        "The target the project reports is capture >=6h at fixed alert "
        "burden, but the target it TRAINS on gives every pre-onset hour the "
        "same label -- an alarm 1 h before onset scores like one 11 h before. "
        "The utility arms weight each hour by the warning it actually "
        "delivers: B(l) = 0 below %.0f h, rising to full credit at %.0f h."
        % (L_MIN, L_SAT), "",
        "TLS is the same machinery pointed the other way: it peaks AT onset "
        "and decays backwards, weighting the least useful hours most. If ramp "
        "direction is what matters, utility should beat TLS on capture while "
        "TLS looks relatively better on plain recall, which credits any lead "
        "above zero.", "",
        "Positive mass: hard %.0f, tls %.0f (%.0f%%), utility %.0f (%.0f%%). "
        "The mass-matched arm rescales spw by %.2fx so the ramp's SHAPE is "
        "isolated from simply carrying less positive weight."
        % (mass["hard"], mass["tls"], 100 * mass["tls"] / mass["hard"],
           mass["utility"], 100 * mass["utility"] / mass["hard"], ratio), "",
        "Flat XGBoost, thresholds solved for exact equal burden, paired "
        "patient bootstrap (%d resamples)." % args.n_boot, "",
        "| | " + " | ".join(order) + " |", "|" + "---|" * (len(order) + 1)]
    for k in KEYS:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        lines.append("| %s | %s |" % (LABEL[k], " | ".join(
            fmt(results[a][k], spec) for a in order)))
    lines.append("| AUROC | %s |" % " | ".join("%.4f" % results[a]["auroc"]
                                               for a in order))
    lines += ["", "## Paired differences (95% CI)", "",
              "| Contrast | Capture >=6h | Capture >=12h | Patient recall | "
              "Median lead (h) |", "|---|---|---|---|---|"]
    for name, b in contrasts.items():
        row = []
        for k in ("capture_6h", "capture_12h", "patient_recall",
                  "median_lead_time_h"):
            d = b[k]
            star = "" if (d["lo95"] <= 0 <= d["hi95"]) else " **"
            row.append("%+.3f [%+.3f, %+.3f]%s"
                       % (d["mean_diff"], d["lo95"], d["hi95"], star))
        lines.append("| %s | %s |" % (name, " | ".join(row)))
    lines += ["", "`**` marks a paired 95% interval excluding zero.", "",
              "Built in %.1f min." % ((time.time() - t0) / 60), ""]

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "UTILITY.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
