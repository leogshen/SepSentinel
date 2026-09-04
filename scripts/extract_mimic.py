#!/usr/bin/env python
"""Extract SepSentinel episodes from MIMIC-IV (full or demo) via DuckDB.

Produces episodes in the physionet.py schema (+ subject_id/stay_id/
t_sepsis_hour/dataset) as a pickle, per DATA_ACCESS_SPEC.md sections 0/5,
with the section-2 cohort rules applied and a CONSORT-style attrition log
(checklist step 5).

Labels: Sepsis-3 with the Challenge timing rules, from
sepsentinel/data/sepsis3.py. Computed inline by default; pass
--sepsis3-labels to reuse a CSV from scripts/build_sepsis3_labels.py, or
--no-sepsis3 to extract everything as controls (pipeline smoke tests).

Usage:
    python scripts/extract_mimic.py --data-root .../mimic-iv-clinical-database-demo-2.2 --out results/mimic_demo_episodes.pkl
    python scripts/extract_mimic.py --data-root C:/data/mimic-iv-3.1 --limit-stays 1000 --out results/mimic_mve_episodes.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import gridding
from sepsentinel.data.gridding import build_episode
from sepsentinel.data import sepsis3

# P0 variables (DATA_ACCESS_SPEC section 5), itemids verified vs 3.1
# dictionaries on 2026-09-03.
CHART_ITEMS = {
    220045: "heart_rate",
    220210: "respiratory_rate",
    220277: "spo2",
    223762: "temperature",        # Celsius
    223761: "temperature_f",      # Fahrenheit -> converted to C below
}
LAB_ITEMS = {
    50813: "lactate",
    50820: "ph",
    50912: "creatinine",
    51301: "wbc",
    51265: "platelets",
    50885: "bilirubin",
}
# The canonical 10-feature list from experiment2_imputation.py, in that exact
# order: AblationPreprocessor indexes into it POSITIONALLY, so a MIMIC episode
# must carry the same layout (Config I then selects the 9 non-creatinine
# features downstream, as it does for PhysioNet).
FEATURES = ["heart_rate", "spo2", "respiratory_rate", "temperature",
            "lactate", "ph", "creatinine", "wbc", "platelets", "bilirubin"]
VITALS = FEATURES[:4]
MIN_AGE = sepsis3.MIN_AGE
MIN_LENGTH = sepsis3.MIN_LOS_HOURS

# Section 2 cohort rules.
EARLY_ONSET_EXCLUSION_H = 4   # onset at/before this ICU hour -> drop the stay
HR_REQUIRED_WITHIN_H = 6      # >=1 heart-rate measurement in the first 6h
MAX_EMPTY_HOUR_FRACTION = 0.5 # >50% of hours with no observation -> drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-stays", type=int, default=None,
                    help="take the first N stays by stay_id (deterministic)")
    ap.add_argument("--sample-stays", type=int, default=None,
                    help="take a reproducible RANDOM draw of N stays "
                         "(spec section 11 wants the MVE cohort random, not "
                         "the head of the stay_id ordering)")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--sepsis3-labels", default=None,
                    help="labels CSV from scripts/build_sepsis3_labels.py; "
                         "default is to compute them inline")
    ap.add_argument("--no-sepsis3", action="store_true",
                    help="skip labelling entirely (every stay a control)")
    ap.add_argument("--label-shift-hours", type=float, default=6.0)
    ap.add_argument("--post-onset-truncate-h", type=float,
                    default=gridding.POST_ONSET_TRUNCATE_H,
                    help="drop septic-episode hours beyond t_sepsis + this "
                         "(spec section 2 wants both variants recorded); "
                         "pass -1 to keep the whole stay")
    ap.add_argument("--keep-early-onset", action="store_true",
                    help="keep stays with onset at/before hour %d "
                         "(sensitivity analysis)" % EARLY_ONSET_EXCLUSION_H)
    ap.add_argument("--memory-limit", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--temp-dir", default=None,
                    help="DuckDB spill directory (keep it off the system "
                         "disk for full-database runs)")
    args = ap.parse_args()

    root = args.data_root.rstrip("/\\").replace("\\", "/")
    con = duckdb.connect()
    if args.memory_limit:
        con.execute("SET memory_limit='%s'" % args.memory_limit)
    if args.threads:
        con.execute("SET threads=%d" % args.threads)
    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)
        con.execute("SET temp_directory='%s'" % args.temp_dir.replace("\\", "/"))
    t0 = time.time()

    if args.sample_stays:
        # hash-ordered draw: reproducible for a given seed, and independent of
        # any correlation between stay_id and admission time.
        order_limit = ("ORDER BY hash(i.stay_id + %d) LIMIT %d"
                       % (args.sample_seed, args.sample_stays))
    elif args.limit_stays:
        order_limit = "ORDER BY i.stay_id LIMIT %d" % args.limit_stays
    else:
        order_limit = "ORDER BY i.stay_id"

    stay_sql = """
        SELECT i.stay_id, i.subject_id, i.intime,
               DATEDIFF('second', i.intime, i.outtime) / 3600.0 AS los_hours,
               p.anchor_age
        FROM read_csv_auto('%s/icu/icustays.csv.gz') i
        JOIN read_csv_auto('%s/hosp/patients.csv.gz') p USING (subject_id)
        WHERE p.anchor_age >= %d
          AND DATEDIFF('second', i.intime, i.outtime) / 3600.0 >= %d
        %s
    """ % (root, root, MIN_AGE, MIN_LENGTH, order_limit)
    stays = con.execute(stay_sql).fetchall()
    print("[%6.1fs] %d qualifying stays" % (time.time() - t0, len(stays)))
    stay_ids = ",".join(str(s[0]) for s in stays)
    stay_meta = {s[0]: s for s in stays}

    # --- labels ------------------------------------------------------------
    # sepsis3.run applies the identical age/LOS/ordering filters, so the
    # cohorts match stay-for-stay.
    labels = {}
    if args.no_sepsis3:
        print("[%6.1fs] labelling skipped (--no-sepsis3)" % (time.time() - t0))
    elif args.sepsis3_labels:
        labels = sepsis3.load_label_map(args.sepsis3_labels)
        print("[%6.1fs] %d septic stays from %s"
              % (time.time() - t0, len(labels), args.sepsis3_labels))
    else:
        print("[%6.1fs] building Sepsis-3 labels (Challenge rules)"
              % (time.time() - t0))
        lab_con = duckdb.connect()
        if args.memory_limit:
            lab_con.execute("SET memory_limit='%s'" % args.memory_limit)
        if args.threads:
            lab_con.execute("SET threads=%d" % args.threads)
        # Pass the cohort explicitly when it is a subset, so the labeller
        # cannot drift from the extraction cohort.
        subset = ([int(s[0]) for s in stays]
                  if (args.sample_stays or args.limit_stays) else None)
        df = sepsis3.run(lab_con, root, stay_ids=subset)
        for stay_id, t_sepsis in zip(df["stay_id"], df["t_sepsis_hour"]):
            if t_sepsis is not None and np.isfinite(t_sepsis):
                labels[int(stay_id)] = float(t_sepsis)
        lab_con.close()

    # --- events ------------------------------------------------------------
    chart_sql = """
        SELECT c.stay_id, c.itemid,
               DATEDIFF('second', i.intime, c.charttime) / 3600.0 AS hours,
               c.valuenum
        FROM read_csv_auto('%s/icu/chartevents.csv.gz') c
        JOIN read_csv_auto('%s/icu/icustays.csv.gz') i USING (stay_id)
        WHERE c.itemid IN (%s)
          AND c.valuenum IS NOT NULL
          AND c.stay_id IN (%s)
    """ % (root, root, ",".join(map(str, CHART_ITEMS)), stay_ids)
    chart = con.execute(chart_sql).fetchall()
    print("[%6.1fs] %d chart events" % (time.time() - t0, len(chart)))

    # labevents has no stay_id: join by subject + charttime within stay window
    lab_sql = """
        SELECT i.stay_id, l.itemid,
               DATEDIFF('second', i.intime, l.charttime) / 3600.0 AS hours,
               l.valuenum
        FROM read_csv_auto('%s/hosp/labevents.csv.gz') l
        JOIN read_csv_auto('%s/icu/icustays.csv.gz') i
          ON l.subject_id = i.subject_id
         AND l.charttime >= i.intime AND l.charttime <= i.outtime
        WHERE l.itemid IN (%s)
          AND l.valuenum IS NOT NULL
          AND i.stay_id IN (%s)
    """ % (root, root, ",".join(map(str, LAB_ITEMS)), stay_ids)
    labs = con.execute(lab_sql).fetchall()
    print("[%6.1fs] %d lab events" % (time.time() - t0, len(labs)))

    per_stay = {}
    for sid, itemid, hours, val in chart:
        feat = CHART_ITEMS[itemid]
        if feat == "temperature_f":
            feat, val = "temperature", (val - 32.0) * 5.0 / 9.0
        per_stay.setdefault(sid, []).append((feat, hours, val))
    for sid, itemid, hours, val in labs:
        per_stay.setdefault(sid, []).append((LAB_ITEMS[itemid], hours, val))

    # --- episodes + section-2 exclusions -----------------------------------
    attrition = {"qualifying_stays": len(stays), "early_onset": 0,
                 "no_hr_in_first_%dh" % HR_REQUIRED_WITHIN_H: 0,
                 "too_sparse": 0, "empty_or_too_short": 0, "kept": 0}
    episodes = []
    for sid, (stay_id, subject_id, intime, los_hours, age) in stay_meta.items():
        t_sepsis = labels.get(int(sid))
        if (t_sepsis is not None and not args.keep_early_onset
                and t_sepsis <= EARLY_ONSET_EXCLUSION_H):
            attrition["early_onset"] += 1
            continue

        events = per_stay.get(sid, [])
        if not any(f == "heart_rate" and 0 <= h < HR_REQUIRED_WITHIN_H
                   for f, h, _ in events):
            attrition["no_hr_in_first_%dh" % HR_REQUIRED_WITHIN_H] += 1
            continue

        ep = build_episode(
            stay_id, subject_id, events, los_hours, FEATURES, VITALS,
            t_sepsis_hour=t_sepsis, dataset="mimic4",
            label_shift_hours=args.label_shift_hours, min_length=MIN_LENGTH,
            post_onset_truncate_h=(None if args.post_onset_truncate_h < 0
                                   else args.post_onset_truncate_h),
        )
        if ep is None:
            attrition["empty_or_too_short"] += 1
            continue

        empty_frac = float(np.isnan(ep["signals"]).all(axis=1).mean())
        if empty_frac > MAX_EMPTY_HOUR_FRACTION:
            attrition["too_sparse"] += 1
            continue

        episodes.append(ep)
        attrition["kept"] += 1

    print("[%6.1fs] %d episodes built" % (time.time() - t0, len(episodes)))
    print("\nCohort attrition (spec section 2):")
    for k, v in attrition.items():
        print("  %-22s %d" % (k, v))

    n_pos = sum(e["label"] for e in episodes)
    all_sig = np.concatenate([e["signals"] for e in episodes], axis=0)
    all_lab = np.concatenate([e["labels"] for e in episodes], axis=0)
    print("\nGrid: %d patient-hours x %d features"
          % (all_sig.shape[0], all_sig.shape[1]))
    print("Label prevalence (PhysioNet reference in parens):")
    print("  patient-level : %5.1f%%   (8.8%%)"
          % (100.0 * n_pos / max(len(episodes), 1)))
    print("  timestep-level: %5.1f%%   (2.2%%)" % (100.0 * all_lab.mean()))
    print("NaN density (PhysioNet reference in parens):")
    ref = {"heart_rate": 7.7, "spo2": 12.0, "respiratory_rate": 9.8,
           "temperature": 66.0, "lactate": 97.3, "ph": 89.3,
           "wbc": 93.6, "platelets": 94.0, "bilirubin": 98.5}
    nan_density = {}
    for j, f in enumerate(FEATURES):
        pct = float(np.isnan(all_sig[:, j]).mean() * 100)
        nan_density[f] = round(pct, 2)
        r = ref.get(f)
        print("  %-18s: %5.1f%%   (%s)"
              % (f, pct, "%.1f%%" % r if r is not None else "n/a"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(episodes, fh)

    manifest = {
        "data_root": root,
        "limit_stays": args.limit_stays,
        "sample_stays": args.sample_stays,
        "sample_seed": args.sample_seed if args.sample_stays else None,
        "post_onset_truncate_h": args.post_onset_truncate_h,
        "label_shift_hours": args.label_shift_hours,
        "label_source": ("none" if args.no_sepsis3
                         else args.sepsis3_labels or "inline sepsis3.run"),
        "features": FEATURES,
        "attrition": attrition,
        "episodes": len(episodes),
        "septic_episodes": int(n_pos),
        "patient_prevalence_pct": round(100.0 * n_pos / max(len(episodes), 1), 2),
        "timestep_prevalence_pct": round(100.0 * float(all_lab.mean()), 3),
        "patient_hours": int(all_sig.shape[0]),
        "nan_density_pct": nan_density,
        "built_seconds": round(time.time() - t0, 1),
    }
    manifest_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nSaved %d episodes -> %s" % (len(episodes), args.out))
    print("Saved manifest    -> %s" % manifest_path)


if __name__ == "__main__":
    main()
