#!/usr/bin/env python
"""SICdb 1.0.8 -> the same episode schema as extract_mimic.py.

Produces episodes on the POOLED 17-feature set (sicdb.POOLED_FEATURES), which
is the MIMIC extended set minus GCS. See sicdb.py: GCS is unusable in SICdb,
and carrying it as an all-missing channel would let a pooled encoder read the
hospital off the mask channel.

Labels come from scripts/build_sicdb_labels.py and are RUNG 1 -- measured to
over-call by roughly 3-7x. They are attached for stratification and transfer
evaluation only; the pretraining arms this feeds do not use them.

Usage:
    python scripts/extract_sicdb.py \
        --data-root F:/Claude/Sepsentinel/data_local/sicdb-1.0.8 \
        --labels results/sicdb_labels.csv \
        --out results/sicdb_episodes.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sicdb
from sepsentinel.data.gridding import make_labels
from scripts.build_sicdb_labels import build_hourly
from scripts.extract_mimic import PLAUSIBLE

MAX_HOURS = 336
POST_ONSET_TRUNCATE_H = 0.0   # match the MIMIC pre-onset extract
MIN_LENGTH = 6
MAX_EMPTY_HOUR_FRACTION = 0.5
HR_REQUIRED_WITHIN_H = 6
EARLY_ONSET_EXCLUSION_H = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prodrome-window-h", type=float, default=12.0)
    ap.add_argument("--label-shift-hours", type=float, default=6.0)
    ap.add_argument("--sample-cases", type=int, default=None,
                    help="random subsample of cases, for a quick MVE run")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--temp-dir",
                    default="F:/Claude/Sepsentinel/data_local/duckdb_tmp")
    args = ap.parse_args()
    t0 = time.time()

    import duckdb
    con = duckdb.connect()
    con.execute("SET temp_directory='%s'" % args.temp_dir)
    sicdb.register_sources(con, args.data_root)

    feats = list(sicdb.POOLED_FEATURES)
    vitals = list(sicdb.POOLED_VITALS)
    print("Pooled feature set: %d (%d vitals, %d labs); GCS excluded: %s"
          % (len(feats), len(vitals), len(feats) - len(vitals),
             sicdb.EXCLUDED_FROM_POOL))

    print("Building hourly grid...")
    hourly = build_hourly(con, MAX_HOURS)
    print("  %d case-hours, %d cases (%.1f min)"
          % (len(hourly), hourly.CaseID.nunique(), (time.time() - t0) / 60))

    # Unit harmonisation, then plausibility clipping -- same order as the
    # MIMIC extract, where the filter is the first line of defence and
    # CLIP_RANGES the second.
    for f, scale in sicdb.UNIT_SCALE.items():
        if f in hourly.columns:
            hourly[f] = hourly[f] * scale
    dropped = {}
    for f in feats:
        if f not in hourly.columns or f not in PLAUSIBLE:
            continue
        lo, hi = PLAUSIBLE[f]
        bad = hourly[f].notna() & ((hourly[f] < lo) | (hourly[f] > hi))
        dropped[f] = int(bad.sum())
        hourly.loc[bad, f] = np.nan

    lab = pd.read_csv(args.labels).set_index("CaseID")
    t_sep = lab["t_sepsis_hour"].to_dict()

    # ICU length of stay, in hours from ICU admission. data_float_h spans the
    # whole PDMS (hospital) stay, not the ICU stay: max(hr) runs to 144 for
    # essentially every case, and the post-ICU ward hours carry no monitoring.
    # Gridding to max(hr) therefore manufactured empty tails that tripped the
    # sparsity filter and discarded 63% of the cohort. Bound by LOS instead,
    # exactly as gridding.build_episode does.
    cases = con.execute(
        "SELECT CaseID, PatientID, TimeOfStay, ICUOffset, "
        "(TimeOfStay - ICUOffset) / 3600.0 AS los_hours, "
        "SurgicalAdmissionType, DischargeState, saps3 "
        "FROM cases").df().set_index("CaseID")

    ids = hourly.CaseID.unique()
    if args.sample_cases:
        rng = np.random.default_rng(args.sample_seed)
        ids = rng.choice(ids, size=min(args.sample_cases, len(ids)),
                         replace=False)
        hourly = hourly[hourly.CaseID.isin(ids)]

    attrition = {"cases_gridded": int(len(ids)), "early_onset": 0,
                 "no_hr_in_first_6h": 0, "too_sparse": 0,
                 "empty_or_too_short": 0, "kept": 0}
    episodes = []
    for cid, g in hourly.groupby("CaseID"):
        g = g.sort_values("hr")
        ts = t_sep.get(cid)
        ts = None if ts is None or (isinstance(ts, float) and np.isnan(ts)) \
            else float(ts)

        los = (float(cases.loc[cid, "los_hours"])
               if cid in cases.index else np.nan)
        n_hours = int(g.hr.max()) + 1
        if np.isfinite(los):
            n_hours = min(n_hours, int(np.floor(los)))
        if ts is not None and POST_ONSET_TRUNCATE_H is not None:
            n_hours = min(n_hours, int(np.ceil(ts + POST_ONSET_TRUNCATE_H)))
        n_hours = min(n_hours, MAX_HOURS)
        if n_hours < MIN_LENGTH:
            attrition["empty_or_too_short"] += 1
            continue
        if ts is not None and ts <= EARLY_ONSET_EXCLUSION_H:
            attrition["early_onset"] += 1
            continue

        sig = np.full((n_hours, len(feats)), np.nan, dtype=np.float32)
        gg = g[g.hr < n_hours]
        idx = gg.hr.to_numpy().astype(int)
        for j, f in enumerate(feats):
            if f in gg.columns:
                sig[idx, j] = gg[f].to_numpy(dtype=np.float32)

        if np.all(np.isnan(sig)):
            attrition["empty_or_too_short"] += 1
            continue
        hr_col = feats.index("heart_rate")
        if not np.any(~np.isnan(sig[:min(HR_REQUIRED_WITHIN_H, n_hours),
                                    hr_col])):
            attrition["no_hr_in_first_6h"] += 1
            continue
        empty = np.mean(np.all(np.isnan(sig), axis=1))
        if empty > MAX_EMPTY_HOUR_FRACTION:
            attrition["too_sparse"] += 1
            continue

        labels = make_labels(n_hours, ts, args.label_shift_hours,
                             args.prodrome_window_h)
        pos = np.nansum(labels) > 0
        meta = cases.loc[cid] if cid in cases.index else None
        episodes.append({
            "patient_id": "sicdb_%d" % cid,
            "subject_id": ("sicdb_p%d" % int(meta.PatientID)) if meta is not None
                          else "sicdb_%d" % cid,
            "stay_id": str(cid),
            "time": np.arange(n_hours, dtype=np.float32),
            "signals": sig,
            "labels": labels,
            "label": 1 if pos else 0,
            "onset_step": int(np.argmax(labels > 0)) if pos else None,
            "t_sepsis_hour": ts,
            "dataset": "sicdb",
            "features": list(feats),
            "vitals": list(vitals),
            # Kept for the surgical-inflammation stratification and for the
            # hospital-shortcut probe.
            "surgical_admission_type": (int(meta.SurgicalAdmissionType)
                                        if meta is not None else None),
            "saps3": float(meta.saps3) if meta is not None
                     and not pd.isna(meta.saps3) else None,
        })
        attrition["kept"] += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(episodes, fh)

    n_sep = sum(e["label"] for e in episodes)
    dens = float(np.mean([np.mean(~np.isnan(e["signals"])) for e in episodes]))
    manifest = {
        "data_root": args.data_root, "labels": args.labels,
        "label_name": sicdb.LABEL_NAME, "label_rung": sicdb.LABEL_RUNG,
        "label_warning": sicdb.LABEL_WARNING,
        "features": feats, "vitals": vitals,
        "excluded_from_pool": sicdb.EXCLUDED_FROM_POOL,
        "prodrome_window_h": args.prodrome_window_h,
        "post_onset_truncate_h": POST_ONSET_TRUNCATE_H,
        "implausible_values_dropped": dropped,
        "attrition": attrition,
        "episodes": len(episodes), "septic_episodes": int(n_sep),
        "mean_observed_fraction": dens,
        "minutes": (time.time() - t0) / 60.0,
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print("attrition: %s" % attrition)
    print("implausible dropped: %s"
          % {k: v for k, v in dropped.items() if v})
    print("episodes %d (%d rung-1 septic, %.1f%%), mean observed fraction %.3f"
          % (len(episodes), n_sep, 100 * n_sep / max(len(episodes), 1), dens))
    print("Saved -> %s (%.1f min)" % (args.out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
