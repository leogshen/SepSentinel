#!/usr/bin/env python
"""Bedside clinical scores as comparators, on our cohort and our metrics.

The honest question for any ML sepsis model is not "what AUROC did we get"
but "does it beat what the ward already does". These are the scores a nurse
computes from the chart:

  SIRS   temp <36 or >38 C, HR >90, RR >20, WBC <4 or >12 -- 1 point each.
         Sepsis-2 criteria. NOT part of our Sepsis-3 label, so it is a clean
         comparator.
  qSOFA  RR >=22, SBP <=100, GCS <15 -- 1 point each. Partially overlaps the
         label (hypotension and GCS feed the cardiovascular and CNS SOFA
         components), so read it with that caveat.
  NEWS2  the deployed UK deterioration standard: RR, SpO2, supplemental O2,
         temperature, SBP, HR and consciousness, each banded 0-3.

DELIBERATELY ABSENT: SOFA itself. Our label is defined as a >=2-point SOFA
rise, so scoring SOFA against it would be near-tautological -- it would look
excellent for a reason that says nothing about early warning. Anyone
comparing against SOFA on Sepsis-3 labels is measuring their own label.

Scores are computed causally: each hour uses the most recent value charted at
or before that hour (forward-fill), which is what a clinician at the bedside
has. Hours with no value yet for a component score 0 for it.

Usage:
    python scripts/clinical_scores.py --episodes results/mve_ext_5k.pkl \
        --out-dir results/clinical_scores
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import curve_for, at_burden, BURDEN_POINTS

REQUIRED = {
    "sirs": ["temperature", "heart_rate", "respiratory_rate", "wbc"],
    "qsofa": ["respiratory_rate", "sbp", "gcs"],
    "news2": ["respiratory_rate", "spo2", "temperature", "sbp", "heart_rate",
              "gcs", "fio2"],
}


def ffill(col):
    """Causal forward-fill; leading gap stays NaN (nothing charted yet)."""
    out = col.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def band(x, edges, points):
    """NEWS2-style banding: first matching upper edge wins."""
    out = np.zeros_like(x, dtype=float)
    assigned = np.zeros_like(x, dtype=bool)
    for edge, p in zip(edges, points):
        m = (~assigned) & (x <= edge)
        out[m] = p
        assigned |= m
    out[~assigned] = points[-1]
    out[np.isnan(x)] = 0.0
    return out


def score_episode(ep, which, idx):
    cols = {f: ffill(ep["signals"][:, j]) for f, j in idx.items()}
    n = ep["signals"].shape[0]

    if which == "sirs":
        s = np.zeros(n)
        t, hr = cols["temperature"], cols["heart_rate"]
        rr, wbc = cols["respiratory_rate"], cols["wbc"]
        s += np.where(~np.isnan(t) & ((t < 36) | (t > 38)), 1, 0)
        s += np.where(~np.isnan(hr) & (hr > 90), 1, 0)
        s += np.where(~np.isnan(rr) & (rr > 20), 1, 0)
        s += np.where(~np.isnan(wbc) & ((wbc < 4) | (wbc > 12)), 1, 0)
        return s / 4.0

    if which == "qsofa":
        s = np.zeros(n)
        rr, sbp, gcs = cols["respiratory_rate"], cols["sbp"], cols["gcs"]
        s += np.where(~np.isnan(rr) & (rr >= 22), 1, 0)
        s += np.where(~np.isnan(sbp) & (sbp <= 100), 1, 0)
        s += np.where(~np.isnan(gcs) & (gcs < 15), 1, 0)
        return s / 3.0

    # NEWS2
    rr = cols["respiratory_rate"]
    spo2 = cols["spo2"]
    temp = cols["temperature"]
    sbp = cols["sbp"]
    hr = cols["heart_rate"]
    gcs = cols["gcs"]
    fio2 = cols["fio2"]
    s = np.zeros(n)
    s += band(rr, [8, 11, 20, 24], [3, 1, 0, 2, 3])
    s += band(spo2, [91, 93, 95], [3, 2, 1, 0])
    s += np.where(~np.isnan(fio2) & (fio2 > 21), 2, 0)      # supplemental O2
    s += band(temp, [35.0, 36.0, 38.0, 39.0], [3, 1, 0, 1, 2])
    s += band(sbp, [90, 100, 110, 219], [3, 2, 1, 0, 3])
    s += band(hr, [40, 50, 90, 110, 130], [3, 1, 0, 1, 2, 3])
    s += np.where(~np.isnan(gcs) & (gcs < 15), 3, 0)        # not alert
    return s / 20.0            # scale into [0,1] so thresholds sweep cleanly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    features = list(episodes[0]["features"])
    print("Episodes %d (%d septic); %d features available"
          % (len(episodes), sum(e["label"] for e in episodes), len(features)))

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    test = splits["test"]          # rule-based: no fitting, evaluate on test

    rows, metrics = [], {}
    for which, needed in REQUIRED.items():
        missing = [f for f in needed if f not in features]
        if missing:
            print("  %-6s SKIPPED, episodes lack %s" % (which, missing))
            continue
        idx = {f: features.index(f) for f in needed}
        results, y_true, y_prob = [], [], []
        for ep in test:
            s = score_episode(ep, which, idx)
            results.append({
                "patient_id": ep["patient_id"], "label": ep["label"],
                "onset_step": ep["onset_step"],
                "t_sepsis_hour": ep.get("t_sepsis_hour"),
                "probs": s, "labels": ep["labels"], "length": len(s),
            })
            y_true.append(ep["labels"])
            y_prob.append(s)
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        curve = curve_for(results)
        print("  %-6s AUROC %.3f AUPRC %.3f" % (which, auroc, auprc))
        for b in BURDEN_POINTS:
            c = at_burden(curve, b)
            if c is None:
                print("      at <=%.1f alerts/pt-day: unreachable "
                      "(score is coarse-grained)" % b)
                continue
            print("      at <=%.1f alerts/pt-day: recall %.2f, ts-prec %.3f, "
                  "pt-prec %.3f, median lead %s h, capture>=6h %.2f"
                  % (b, c["patient_recall"], c["timestep_precision"],
                     c["patient_precision"],
                     "%.1f" % c["median_lead_time_h"]
                     if c["median_lead_time_h"] is not None else "n/a",
                     c["capture_6h"]))
        rows.append((which, auroc, auprc, curve))
        metrics[which] = {"auroc": auroc, "auprc": auprc, "curve": curve}

    lines = ["# Bedside clinical scores on our cohort", "",
             "Episodes: `%s`; test split only (these are rule-based, nothing "
             "is fitted)." % args.episodes,
             "",
             "SOFA is deliberately excluded: our label IS a >=2-point SOFA "
             "rise, so scoring SOFA against it would be near-tautological.",
             "", "## Threshold-free discrimination", "",
             "| Score | AUROC | AUPRC |", "|---|---|---|"]
    for which, auroc, auprc, _ in rows:
        lines.append("| %s | %.3f | %.3f |" % (which, auroc, auprc))
    lines += ["", "## At equal alert burden", ""]
    for b in BURDEN_POINTS:
        lines += ["**Budget: %.1f false alerts per nonseptic patient-day**" % b,
                  "", "| Score | Patient recall | Timestep prec. | Patient prec. | "
                  "Median lead (h) | Capture >=6h | Capture >=12h |",
                  "|---|---|---|---|---|---|---|"]
        for which, _, _, curve in rows:
            c = at_burden(curve, b)
            if c is None:
                lines.append("| %s | (unreachable) | | | |" % which)
                continue
            lines.append("| %s | %.2f | %.3f | %.3f | %s | %.2f | %.2f |"
                         % (which, c["patient_recall"], c["timestep_precision"],
                            c["patient_precision"],
                            "%.1f" % c["median_lead_time_h"]
                            if c["median_lead_time_h"] is not None else "n/a",
                            c["capture_6h"], c["capture_12h"]))
        lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "CLINICAL_SCORES.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\nSaved -> %s" % os.path.join(args.out_dir, "CLINICAL_SCORES.md"))


if __name__ == "__main__":
    main()
