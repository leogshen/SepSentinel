#!/usr/bin/env python
"""Is measurement density causally the binding constraint?

The project's standing claim is that the ceiling is measurement sparsity, not
model capacity: in the pre-onset window there are effectively three live
signals (HR, SpO2, respiratory rate at ~95% of hours) while every lab sits at
2-8%. That claim has only ever been supported correlationally, plus one
synthetic positive control.

The obvious test -- stratify patients by how much they were measured -- does
NOT work. Sicker patients get more labs, so observed density and severity are
nearly inseparable, and "the model does better on densely-measured patients"
is expected whether or not measurement is the bottleneck.

This is the interventional version. Take the SAME patients and DELETE
observations at random, then retrain and re-evaluate. Severity is held fixed
by construction because it is the same cohort in every arm; the only thing
that changes is how much of it the model gets to see. If performance falls as
observations are removed, density is causally binding. If it is flat, the
model was not using them and the ceiling is elsewhere.

Thinned separately for three targets, because they answer different questions:
  labs    the sparse channels (2-8% observed). Does more lab measurement help?
  vitals  the dense channels (85-95%). Does continuous vital-sign sensing help?
  all     both.

Only genuinely OBSERVED cells are eligible for deletion; NaN stays NaN.
Thinning is applied to train, val and test alike, which is the deployment
question ("a site that measures less"), not the transfer question ("trained
rich, deployed sparse").

Evaluation: flat XGBoost, thresholds solved for EXACT equal burden so no arm
can buy capture by alarming more, paired patient-bootstrap intervals against
the full-density arm, capture >=6h governing.

Usage:
    python scripts/density_ablation.py \
        --episodes results/mimic31_pooled17.pkl \
        --out-dir results/density_ablation
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
    per_patient_stats, reduce_stats, paired_bootstrap, fmt,
)

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


def thin(episodes, cols, keep, seed):
    """Delete a fraction of observed cells in `cols`, same patients throughout.

    keep=1.0 returns the cohort untouched; keep=0.0 removes the channels
    entirely. Deletion is per-cell and independent, so a channel thins toward
    its own sparser version rather than losing whole patients.
    """
    if keep >= 1.0:
        return episodes
    rng = np.random.default_rng(seed)
    out = []
    for e in episodes:
        s = e["signals"].copy()
        sub = s[:, cols]
        obs = ~np.isnan(sub)
        if keep <= 0.0:
            sub[obs] = np.nan
        else:
            drop = obs & (rng.random(sub.shape) > keep)
            sub[drop] = np.nan
        s[:, cols] = sub
        out.append(dict(e, signals=s))
    return out


def observed_fraction(episodes, cols):
    tot = obs = 0
    for e in episodes:
        sub = e["signals"][:, cols]
        tot += sub.size
        obs += int(np.sum(~np.isnan(sub)))
    return obs / max(tot, 1)


def run_config(splits, raw_map, feats, cols, keep, seed, burden):
    data_raw = {k: thin(v, cols, keep, seed) for k, v in splits.items()}
    raw_thinned = {e["patient_id"]: e for s in data_raw.values() for e in s}
    pre = build_preprocessor(data_raw["train"], "all")
    data = {"train": pre.fit_transform(data_raw["train"]),
            "val": pre.transform(data_raw["val"]),
            "test": pre.transform(data_raw["test"])}

    import xgboost as xgb
    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    y = np.concatenate([d["labels"] for d in data["train"]])
    spw = float((1 - y.mean()) / max(y.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=42, scale_pos_weight=spw,
                            eval_metric="logloss",
                            tree_method="hist").fit(X["train"], y)
    p = clf.predict_proba(X["test"])[:, 1]
    # raw_map must be the THINNED episodes: patient_results_from_probs reads
    # label/onset_step/t_sepsis_hour from it, and those are unchanged by
    # thinning, but using one consistent source avoids any drift.
    pr = patient_results_from_probs(data["test"], raw_thinned, p)
    st = per_patient_stats(pr, threshold_for_exact_burden(pr, burden))
    yte = np.concatenate([d["labels"] for d in data["test"]])
    return st, {"auroc": float(roc_auc_score(yte, p)),
                "auprc": float(average_precision_score(yte, p)),
                "observed_fraction_target": observed_fraction(
                    data_raw["train"], cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keeps", default="0.5,0.25,0.0")
    ap.add_argument("--targets", default="labs,vitals,all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "density.json")
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    feats = list(eps[0]["features"])
    vitals = list(eps[0].get("vitals", feats[:4]))
    idx_vitals = [feats.index(f) for f in vitals]
    idx_labs = [i for i, f in enumerate(feats) if f not in vitals]
    targets = {"labs": idx_labs, "vitals": idx_vitals,
               "all": list(range(len(feats)))}
    splits = grouped_patient_split(eps, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    print("%d episodes, %d features (%d vitals, %d labs)"
          % (len(eps), len(feats), len(idx_vitals), len(idx_labs)))

    keeps = [float(k) for k in args.keeps.split(",")]
    tnames = [t.strip() for t in args.targets.split(",")]

    # Observed fraction of each target's channels BEFORE thinning, so the
    # table shows 0.048 -> 0.012 rather than comparing a labs-only fraction
    # against an all-channel one.
    base_obs = {t: observed_fraction(splits["train"], targets[t])
                for t in tnames}
    print("baseline observed fraction: %s"
          % {k: round(v, 3) for k, v in base_obs.items()})

    results, stats = {}, {}
    t1 = time.time()
    st, meta = run_config(splits, raw_map, feats, targets["all"], 1.0,
                          args.seed, args.burden)
    stats["full"] = st
    results["full"] = dict(reduce_stats(st), **meta,
                           baseline_observed=base_obs,
                           minutes=(time.time() - t1) / 60.0)
    m = results["full"]
    print("full density      cap6h %.3f  recall %.3f  AUROC %.4f  (%.1f min)"
          % (m["capture_6h"], m["patient_recall"], m["auroc"], m["minutes"]))

    for tname in tnames:
        for keep in keeps:
            t1 = time.time()
            key = "%s_keep%g" % (tname, keep)
            st, meta = run_config(splits, raw_map, feats, targets[tname],
                                  keep, args.seed, args.burden)
            stats[key] = st
            results[key] = dict(reduce_stats(st), **meta, target=tname,
                                keep=keep, baseline_observed=base_obs[tname],
                                minutes=(time.time() - t1) / 60.0)
            r = results[key]
            print("%-8s keep %-5g  observed %.3f  cap6h %.3f  recall %.3f  "
                  "AUROC %.4f  (%.1f min)"
                  % (tname, keep, r["observed_fraction_target"],
                     r["capture_6h"], r["patient_recall"], r["auroc"],
                     r["minutes"]))
            with open(json_path, "w") as fh:
                json.dump(results, fh, indent=2)

    contrasts = {k: paired_bootstrap(stats["full"], stats[k], args.n_boot,
                                     args.seed, KEYS)
                 for k in stats if k != "full"}
    with open(json_path, "w") as fh:
        json.dump({"arms": results, "paired_vs_full": contrasts}, fh, indent=2)

    lines = [
        "# Density ablation: is measurement the binding constraint?", "",
        "Same patients throughout, observations DELETED at random. Severity "
        "cannot confound this the way stratifying by observed density would, "
        "because the cohort is identical in every arm and only what the model "
        "sees changes. Flat XGBoost; thresholds solved for exact equal burden "
        "(%.2f alerts/nonseptic patient-day) so no arm can buy capture by "
        "alarming more; paired patient-bootstrap intervals against full "
        "density." % args.burden, "",
        "Episodes: `%s`, %d episodes, %d features."
        % (args.episodes, len(eps), len(feats)), "",
        "Baseline observed fraction of each target's own channels: %s."
        % ", ".join("%s %.3f" % (k, v) for k, v in base_obs.items()), "",
        "| Arm | Observed frac. of thinned channels | Capture >=6h | "
        "Capture >=12h | Patient recall | Median lead | AUROC | "
        "vs full: capture >=6h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    order = ["full"] + ["%s_keep%g" % (t, k) for t in tnames for k in keeps]
    for key in order:
        r = results[key]
        if key == "full":
            diff = "-"
            r = dict(r, observed_fraction_target=float("nan"))
        elif abs(r["alerts_per_patient_day"] - args.burden) > 0.02:
            # Degenerate arm: with every channel removed the model emits one
            # constant probability, so the burden quantile is a wall of ties
            # and `>= threshold` fires on everybody. Realised burden overshoots
            # and recall goes to 1.0 by construction. Not a result.
            diff = "INVALID -- burden %.2f, not %.2f" % (
                r["alerts_per_patient_day"], args.burden)
        else:
            b = contrasts[key]["capture_6h"]
            star = "" if (b["lo95"] <= 0 <= b["hi95"]) else " **"
            diff = "%+.3f [%+.3f, %+.3f]%s" % (b["mean_diff"], b["lo95"],
                                               b["hi95"], star)
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %s | %.4f | %s |"
                     % (key, r["observed_fraction_target"], r["capture_6h"],
                        r["capture_12h"], r["patient_recall"],
                        fmt(r["median_lead_time_h"], "%.1f"), r["auroc"],
                        diff))
    lines += ["", "`**` marks a paired 95% interval excluding zero.", "",
              "Built in %.1f min." % ((time.time() - t0) / 60), ""]
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "DENSITY.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
