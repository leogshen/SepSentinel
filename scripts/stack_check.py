#!/usr/bin/env python
"""Do the two wins stack? Utility weighting x dropping the labs, 2x2.

Two improvements were measured separately against the same baseline and have
never been run together:

  utility ramp   weight each pre-onset hour by the warning it delivers  +0.042
  drop labs      delete all 9 routine labs, keep the 8 vitals           +0.041

They attack different things -- one changes WHICH HOURS the model weights, the
other WHICH CHANNELS it sees -- so they might add. But they might not, and the
reason matters: if the labs hurt BECAUSE they teach the model to fire late,
and the utility ramp already penalises firing late, then both are fixes for
one defect and the second buys nothing on top of the first. A 2x2 separates
those cases; two separate runs cannot.

Everything held fixed: same episodes, same grouped split, same XGBoost
settings, thresholds solved for exact equal burden, paired patient bootstrap
so all four arms are compared on the same resampled patients.

Usage:
    python scripts/stack_check.py \
        --episodes results/mimic31_pooled17.pkl \
        --out-dir results/stack_check
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
    per_patient_stats, reduce_stats, paired_bootstrap, weighted_targets, fmt,
)
from scripts.utility_surrogate import utility_ramp, fit_predict
from scripts.density_ablation import thin

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-h", type=float, default=12.0)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    feats = list(eps[0]["features"])
    vitals = list(eps[0]["vitals"])
    lab_cols = [i for i, f in enumerate(feats) if f not in vitals]
    splits = grouped_patient_split(eps, random_state=SPLIT_SEED)
    print("%d episodes | %d features | dropping %d labs in the no-labs arms"
          % (len(eps), len(feats), len(lab_cols)))

    stats, results = {}, {}
    for drop_labs in (False, True):
        src = {k: (thin(v, lab_cols, 0.0, args.seed) if drop_labs else v)
               for k, v in splits.items()}
        raw_map = {e["patient_id"]: e for s in src.values() for e in s}
        pre = build_preprocessor(src["train"], "all")
        data = {"train": pre.fit_transform(src["train"]),
                "val": pre.transform(src["val"]),
                "test": pre.transform(src["test"])}
        X = {k: np.concatenate([d["signals"] for d in v])
             for k, v in data.items()}
        y_hard = np.concatenate([d["labels"] for d in data["train"]])
        spw = float((1 - y_hard.mean()) / max(y_hard.mean(), 1e-9))
        q_util = np.concatenate([
            utility_ramp(len(d["labels"]),
                         raw_map[d["patient_id"]]["t_sepsis_hour"],
                         args.window_h) for d in data["train"]])
        ratio = float(y_hard.sum() / max(q_util.sum(), 1e-9))

        for target in ("hard", "utility"):
            q, arm_spw = ((y_hard, spw) if target == "hard"
                          else (q_util, spw * ratio))
            name = "%s%s" % (target, "_nolabs" if drop_labs else "")
            t1 = time.time()
            yy, ww = weighted_targets(q.astype(np.float64), arm_spw)
            probs = fit_predict(X["train"], yy, ww, args.seed,
                                {"test": X["test"]})
            pr = patient_results_from_probs(data["test"], raw_map,
                                            probs["test"])
            st = per_patient_stats(pr,
                                   threshold_for_exact_burden(pr, args.burden))
            stats[name] = st
            yte = np.concatenate([d["labels"] for d in data["test"]])
            results[name] = dict(
                reduce_stats(st), channels=len(feats) - (len(lab_cols)
                                                         if drop_labs else 0),
                auroc=float(roc_auc_score(yte, probs["test"])),
                auprc=float(average_precision_score(yte, probs["test"])),
                minutes=(time.time() - t1) / 60.0)
            m = results[name]
            print("  %-16s cap6h %.3f  cap12h %.3f  recall %.3f  lead %s  "
                  "burden %.2f  AUROC %.4f"
                  % (name, m["capture_6h"], m["capture_12h"],
                     m["patient_recall"], fmt(m["median_lead_time_h"], "%.1f"),
                     m["alerts_per_patient_day"], m["auroc"]))

    base = "hard"
    contrasts = {n: paired_bootstrap(stats[base], stats[n], args.n_boot,
                                     args.seed, KEYS)
                 for n in stats if n != base}
    # The interaction: does dropping labs still help ONCE the utility ramp is
    # already in? If this is null while utility_nolabs beats hard, the two are
    # fixes for the same defect rather than independent gains.
    contrasts["interaction_nolabs_given_utility"] = paired_bootstrap(
        stats["utility"], stats["utility_nolabs"], args.n_boot, args.seed, KEYS)
    contrasts["interaction_utility_given_nolabs"] = paired_bootstrap(
        stats["hard_nolabs"], stats["utility_nolabs"], args.n_boot, args.seed,
        KEYS)

    with open(os.path.join(args.out_dir, "stack.json"), "w") as fh:
        json.dump({"arms": results, "paired_bootstrap": contrasts,
                   "minutes": (time.time() - t0) / 60.0}, fh, indent=2)

    order = ["hard", "utility", "hard_nolabs", "utility_nolabs"]
    lines = [
        "# Do the two wins stack?", "",
        "Utility weighting (+0.042 capture >=6h) and dropping the nine routine "
        "labs (+0.041) were measured separately against the same baseline and "
        "never run together. They attack different things -- which HOURS are "
        "weighted versus which CHANNELS are seen -- so they may add. Or they "
        "may be two fixes for one defect: if the labs hurt BECAUSE they teach "
        "late firing, and the utility ramp already penalises late firing, the "
        "second buys nothing on top of the first.", "",
        "Same split, same XGBoost settings, thresholds solved for exact equal "
        "burden, paired bootstrap across all four arms.", "",
        "| | " + " | ".join(order) + " |", "|" + "---|" * (len(order) + 1)]
    for k in KEYS:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        lines.append("| %s | %s |" % (LABEL[k], " | ".join(
            fmt(results[a][k], spec) for a in order)))
    lines.append("| Channels | %s |"
                 % " | ".join(str(results[a]["channels"]) for a in order))
    lines.append("| AUROC | %s |"
                 % " | ".join("%.4f" % results[a]["auroc"] for a in order))
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
    lines += ["", "`**` marks a paired 95% interval excluding zero.",
              "", "Built in %.1f min." % ((time.time() - t0) / 60), ""]

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "STACK.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
