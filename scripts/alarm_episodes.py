#!/usr/bin/env python
"""Alarm-episode semantics with a refractory period (TODO item 4, spec s10).

Everything reported so far counts raw alarm HOURS. A clinician does not
experience 6 consecutive alarming hours as 6 events, so per-hour counting
overstates burden by an unknown factor and the cost side of any utility
calculation is mis-specified.

Under episode semantics a threshold crossing raises ONE alarm, and further
crossings inside a refractory window of R hours raise nothing. The clinical
reading of R is "having been told, how long before I want to be told again".

Why this is not a discount applied to existing numbers
------------------------------------------------------
Merging hours into episodes lowers the burden at a fixed threshold, which
means the SAME alert budget now buys a LOWER threshold. Lower threshold ->
earlier first crossing -> different recall, lead and capture. So capture and
lead must be recomputed under the same policy that defines the budget, which
is what this module does. Reporting episode-burden against hour-selected
thresholds would be the mistake.

One subtlety worth stating: with a first-crossing alarm rule, a refractory
period cannot suppress the FIRST alarm on a septic patient, so at a FIXED
threshold lead and capture are unchanged. The whole effect comes through
re-tuning the threshold to the new budget. If a later policy adds
auto-resolution or requires re-triggering, that stops being true and this
module has to grow a real state machine.

Definitions
-----------
episodes(pred, R)   indices where an alarm fires: the first True, then any
                    True at least R hours after the previous alarm.
burden              alarm EPISODES per non-septic patient-day.
first alarm         first episode index (== first crossing).

Usage:
    python scripts/alarm_episodes.py \
        --episodes results/mimic31_full_ext_prodrome.pkl \
        --out-dir results/alarm_episodes --refractory 2,6,12
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs,
)
from scripts.xgb_tls_experiment import (
    LEAD_POINTS, reduce_stats, paired_bootstrap, fmt,
)

KEYS = ("patient_recall", "patient_precision", "timestep_precision",
        "timestep_recall", "timestep_f1", "median_lead_time_h",
        "capture_3h", "capture_6h", "capture_12h", "alerts_per_patient_day",
        "healthy_alarmed_frac")


def episode_starts(pred, refractory_h):
    """Indices at which an alarm episode begins.

    `refractory_h = 0` reproduces per-hour counting exactly, which is what
    makes the two regimes comparable in one code path.
    """
    idx = np.flatnonzero(pred)
    if idx.size == 0:
        return idx
    if refractory_h <= 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= refractory_h:
            keep.append(i)
    return np.asarray(keep)


def per_patient_stats_episodes(patient_results, thr, refractory_h):
    """per_patient_stats, but alarm COUNTS are episodes rather than hours.

    Timestep precision/recall/F1 stay hour-based: they describe the labelling
    of hours and are unaffected by how alarms are merged for the human. Only
    the burden term and the alarm counts change.
    """
    n = len(patient_results)
    septic = np.zeros(n, bool)
    alarmed = np.zeros(n, bool)
    lead = np.full(n, np.nan)
    caught = {h: np.zeros(n, bool) for h in LEAD_POINTS}
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    n_alerts = np.zeros(n); length = np.zeros(n)

    for i, p in enumerate(patient_results):
        probs = np.asarray(p["probs"])
        labels = np.asarray(p["labels"])
        pred = probs >= thr
        starts = episode_starts(pred, refractory_h)
        septic[i] = (p["label"] == 1)
        length[i] = p["length"]
        n_alerts[i] = float(starts.size)
        tp[i] = float((pred & (labels == 1)).sum())
        fp[i] = float((pred & (labels == 0)).sum())
        fn[i] = float((~pred & (labels == 1)).sum())
        if not septic[i] or starts.size == 0:
            continue
        alarmed[i] = True
        first = int(starts[0])
        t_sepsis = p.get("t_sepsis_hour")
        if t_sepsis is None:
            t_sepsis = p["onset_step"] + 6
        lead[i] = t_sepsis - first
        for h in LEAD_POINTS:
            caught[h][i] = first <= t_sepsis - h
    return {"septic": septic, "alarmed": alarmed, "lead": lead,
            "caught": caught, "tp": tp, "fp": fp, "fn": fn,
            "n_alerts": n_alerts, "length": length}


def curve_episodes(patient_results, refractory_h, step=0.01):
    """Burden/recall curve under episode semantics, for threshold selection."""
    out = []
    for t in np.arange(step, 1.0, step):
        st = per_patient_stats_episodes(patient_results, float(t), refractory_h)
        m = reduce_stats(st)
        m["threshold"] = float(t)
        out.append(m)
    return out


def spend_burden_episodes(curve, budget):
    ok = [c for c in curve if c["alerts_per_patient_day"] <= budget]
    return max(ok, key=lambda c: c["alerts_per_patient_day"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--refractory", default="0,2,6,12",
                    help="refractory periods in hours; 0 = per-hour counting")
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="all", choices=["all", "config_i"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    Rs = [float(r) for r in args.refractory.split(",")]

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    splits = grouped_patient_split(eps, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    pre = build_preprocessor(eps, args.features)
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    y = np.concatenate([d["labels"] for d in data["train"]])
    print("Episodes %d, %d channels, train positive rate %.4f"
          % (len(eps), X["train"].shape[1], y.mean()))

    import xgboost as xgb
    spw = float((1 - y.mean()) / max(y.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=args.seed, scale_pos_weight=spw,
                            eval_metric="logloss",
                            tree_method="hist").fit(X["train"], y)
    res = {k: patient_results_from_probs(data[k], raw_map,
                                         clf.predict_proba(X[k])[:, 1])
           for k in ("val", "test")}

    stats, results = {}, {}
    for R in Rs:
        t1 = time.time()
        # Threshold pinned on VALIDATION under the SAME episode policy that
        # defines the budget -- the point of the exercise.
        c = spend_burden_episodes(curve_episodes(res["val"], R), args.burden)
        thr = float(c["threshold"])
        st = per_patient_stats_episodes(res["test"], thr, R)
        stats[R] = st
        m = reduce_stats(st)
        results["R%g" % R] = dict(m, threshold_from_val=thr,
                                  refractory_h=R,
                                  minutes=(time.time() - t1) / 60.0)
        print("  R=%-4g thr(val)=%.2f  test burden %.2f  recall %.3f  "
              "lead %s  cap6h %.3f  (%.1f min)"
              % (R, thr, m["alerts_per_patient_day"], m["patient_recall"],
                 fmt(m["median_lead_time_h"], "%.1f"), m["capture_6h"],
                 results["R%g" % R]["minutes"]))
        with open(os.path.join(args.out_dir, "episodes.json"), "w") as fh:
            json.dump(results, fh, indent=2)

    ref = Rs[0]
    contrasts = {}
    for R in Rs[1:]:
        contrasts["R=%g h" % R] = paired_bootstrap(
            stats[ref], stats[R], args.n_boot, args.seed, KEYS)

    label = {"patient_recall": "**Patient recall**",
             "patient_precision": "Patient precision",
             "timestep_precision": "Timestep precision",
             "timestep_recall": "Timestep recall",
             "timestep_f1": "Timestep F1",
             "median_lead_time_h": "Median lead (h)",
             "capture_3h": "Capture >=3h", "capture_6h": "Capture >=6h",
             "capture_12h": "Capture >=12h",
             "alerts_per_patient_day": "Realised burden (episodes/pt-day)",
             "healthy_alarmed_frac": "**Nonseptic patients ever alarmed**"}

    lines = [
        "# Alarm episodes with a refractory period", "",
        "Episodes: `%s`. Flat XGBoost, grouped subject split (seed %d), "
        "threshold pinned on VALIDATION under the same episode policy that "
        "defines the budget (<=%.1f per nonseptic patient-day), applied "
        "unchanged to test." % (args.episodes, SPLIT_SEED, args.burden),
        "Built in %.1f min." % ((time.time() - t0) / 60.0), "",
        "`R=0` is per-hour counting, i.e. every number the project has "
        "reported so far. Merging hours into episodes lowers burden at a "
        "fixed threshold, so the same budget buys a lower threshold, an "
        "earlier first crossing, and therefore different recall, lead and "
        "capture. That is why capture and lead are recomputed here rather "
        "than carried over.", "",
        "Timestep precision/recall/F1 stay hour-based: they describe how "
        "hours are labelled and are not affected by how alarms are merged "
        "for the human reader.", "",
        "| | " + " | ".join("R=%g h" % R for R in Rs) + " | " +
        " | ".join("%s - R=%g h (95%% CI)" % (n, ref) for n in contrasts)
        + " |",
        "|" + "---|" * (1 + len(Rs) + len(contrasts)),
    ]
    for k in KEYS:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        cells = [fmt(results["R%g" % R][k], spec) for R in Rs]
        for nm in contrasts:
            b = contrasts[nm][k]
            star = "" if (b["lo95"] <= 0 <= b["hi95"]) else " **"
            cells.append("%+.3f [%+.3f, %+.3f]%s"
                         % (b["mean_diff"], b["lo95"], b["hi95"], star))
        lines.append("| %s | %s |" % (label[k], " | ".join(cells)))

    lines += ["", "| Refractory | Threshold (from val) |", "|---|---|"]
    for R in Rs:
        lines.append("| R=%g h | %.2f |" % (R, results["R%g" % R]["threshold_from_val"]))
    lines += ["", "`**` marks a paired 95%% interval excluding zero, from %d "
              "patient-level resamples." % args.n_boot, ""]

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "EPISODES.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
