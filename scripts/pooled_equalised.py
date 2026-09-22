#!/usr/bin/env python
"""Re-score the four arms at EQUAL realised test burden.

pooled_pretrain.py freezes each arm's threshold on validation, which is the
unbiased rule but lets realised test burden drift. It drifted: the pretrained
arms land 0.099 (mae_mimic) and 0.182 (mae_pooled) alerts/patient-day BELOW
scratch, both intervals excluding zero. An arm that alarms less catches less,
so the capture deficits in that table are partly an artifact of where the
threshold landed rather than of the model.

This re-pins each arm's threshold so every arm spends the same TEST burden,
then recomputes the paired contrasts. It touches test data, but only the
control patients' alarm rate -- never the septic outcomes being compared --
which is far weaker than selecting a threshold to maximise test capture.

Nothing is retrained: the checkpoints from pooled_pretrain.py are re-scored,
and the XGBoost arm is a deterministic refit.

Usage:
    python scripts/pooled_equalised.py \
        --mimic results/mimic31_pooled17.pkl \
        --run-dir results/pooled_pretrain \
        --out-dir results/pooled_pretrain
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs,
    threshold_for_exact_burden,
)
from scripts.xgb_tls_experiment import (
    per_patient_stats, reduce_stats, freeze_threshold, paired_bootstrap, fmt,
)
from experiment5_recall_study import collect_patient_predictions

ARMS = ["scratch", "mae_mimic", "mae_pooled"]
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
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.mimic, "rb") as fh:
        mim = pickle.load(fh)
    splits = grouped_patient_split(mim, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    pre = build_preprocessor(mim, "all")
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    in_dim = data["train"][0]["signals"].shape[1]

    import xgboost as xgb
    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    y = np.concatenate([d["labels"] for d in data["train"]])
    pw = float((1 - y.mean()) / max(y.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=42, scale_pos_weight=pw,
                            eval_metric="logloss",
                            tree_method="hist").fit(X["train"], y)
    pr_test = patient_results_from_probs(data["test"], raw_map,
                                         clf.predict_proba(X["test"])[:, 1])
    stats = {"xgboost": [per_patient_stats(
        pr_test, threshold_for_exact_burden(pr_test, args.burden))]}
    print("xgboost re-pinned: burden %.3f"
          % reduce_stats(stats["xgboost"][0])["alerts_per_patient_day"])

    for arm in ARMS:
        arm_stats = []
        for ck in sorted(glob.glob(os.path.join(args.run_dir, "%s_seed*" % arm,
                                                "best_model.pt"))):
            model = SepsisTransformer(input_dim=in_dim)
            model.load_state_dict(torch.load(ck, weights_only=True))
            model.to(args.device)
            res = collect_patient_predictions(model, data["test"], raw_map,
                                              args.device)
            thr = threshold_for_exact_burden(res, args.burden)
            arm_stats.append(per_patient_stats(res, thr))
        stats[arm] = arm_stats
        m = reduce_stats(arm_stats[0])
        print("%-11s re-pinned: burden %.3f  cap6h %.3f"
              % (arm, m["alerts_per_patient_day"], m["capture_6h"]))

    means = {}
    for arm, sl in stats.items():
        means[arm] = {k: float(np.nanmean([reduce_stats(s)[k] for s in sl]))
                      for k in KEYS}

    contrasts = {}
    pairs = [("mae_mimic", "scratch"), ("mae_pooled", "scratch"),
             ("mae_pooled", "mae_mimic"), ("scratch", "xgboost")]
    for b, a in pairs:
        contrasts["%s_vs_%s" % (b, a)] = paired_bootstrap(
            stats[a][0], stats[b][0], args.n_boot, 42, KEYS)

    out = {"burden_target": args.burden, "means": means,
           "paired_bootstrap": contrasts,
           "minutes": (time.time() - t0) / 60.0}
    with open(os.path.join(args.out_dir, "pooled_equalised.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    order = ["xgboost"] + ARMS
    lines = ["# Four arms, re-scored at EQUAL test burden", "",
             "pooled_pretrain.py froze thresholds on validation, which is "
             "unbiased but let realised burden drift: the pretrained arms "
             "landed 0.099 and 0.182 alerts/patient-day BELOW scratch, both "
             "intervals excluding zero. An arm that alarms less catches less, "
             "so those capture deficits were partly an artifact of the "
             "operating point. Here every arm is re-pinned to spend the same "
             "test burden. Nothing retrained.", "",
             "| | " + " | ".join(order) + " |", "|" + "---|" * (len(order) + 1)]
    for k in KEYS:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        lines.append("| %s | %s |" % (LABEL[k], " | ".join(
            fmt(means[a][k], spec) for a in order)))
    lines += ["", "## Paired differences at equal burden (95% CI)", "",
              "| Contrast | Capture >=6h | Patient recall | Median lead (h) | "
              "Realised burden |", "|---|---|---|---|---|"]
    for name, b in contrasts.items():
        row = []
        for k in ("capture_6h", "patient_recall", "median_lead_time_h",
                  "alerts_per_patient_day"):
            d = b[k]
            star = "" if (d["lo95"] <= 0 <= d["hi95"]) else " **"
            row.append("%+.3f [%+.3f, %+.3f]%s"
                       % (d["mean_diff"], d["lo95"], d["hi95"], star))
        lines.append("| %s | %s |" % (name, " | ".join(row)))
    lines += ["", "`**` marks a paired 95% interval excluding zero.",
              "", "Built in %.1f min." % ((time.time() - t0) / 60), ""]

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "POOLED_EQUALISED.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
