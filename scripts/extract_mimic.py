#!/usr/bin/env python
"""Extract SepSentinel episodes from MIMIC-IV (full or demo) via DuckDB.

Produces episodes in the physionet.py schema (+ subject_id/stay_id/
t_sepsis_hour/dataset) as a pickle, per DATA_ACCESS_SPEC.md sections 0/5.

Labels: t_sepsis is NOT yet computed here (Challenge-rule sepsis3 SQL is a
separate step); all episodes come out as controls until that lands. The
pipeline (gridding, Strategy B, model forward) is fully testable regardless.

Usage:
    python scripts/extract_mimic.py --data-root C:/Users/openq/data/mimic-iv-clinical-database-demo-2.2 --out results/mimic_demo_episodes.pkl
    python scripts/extract_mimic.py --data-root C:/Users/openq/data/mimic-iv-3.1 --limit-stays 1000 --out results/mimic_mve_episodes.pkl
"""

import argparse
import os
import pickle
import sys
import time

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data.gridding import build_episode

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
FEATURES = ["heart_rate", "spo2", "respiratory_rate", "temperature",
            "lactate", "ph", "wbc", "platelets", "bilirubin"]  # Config I order
VITALS = FEATURES[:4]
MIN_AGE = 18
MIN_LENGTH = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-stays", type=int, default=None)
    args = ap.parse_args()

    root = args.data_root.rstrip("/\\")
    con = duckdb.connect()
    t0 = time.time()

    stay_sql = f"""
        SELECT i.stay_id, i.subject_id, i.intime,
               DATEDIFF('minute', i.intime, i.outtime) / 60.0 AS los_hours,
               p.anchor_age
        FROM read_csv_auto('{root}/icu/icustays.csv.gz') i
        JOIN read_csv_auto('{root}/hosp/patients.csv.gz') p USING (subject_id)
        WHERE p.anchor_age >= {MIN_AGE}
          AND DATEDIFF('minute', i.intime, i.outtime) / 60.0 >= {MIN_LENGTH}
        ORDER BY i.stay_id
        {f'LIMIT {args.limit_stays}' if args.limit_stays else ''}
    """
    stays = con.execute(stay_sql).fetchall()
    print(f"[{time.time()-t0:6.1f}s] {len(stays)} qualifying stays")
    stay_ids = ",".join(str(s[0]) for s in stays)
    stay_meta = {s[0]: s for s in stays}

    chart_sql = f"""
        SELECT c.stay_id, c.itemid,
               DATEDIFF('second', i.intime, c.charttime) / 3600.0 AS hours,
               c.valuenum
        FROM read_csv_auto('{root}/icu/chartevents.csv.gz') c
        JOIN read_csv_auto('{root}/icu/icustays.csv.gz') i USING (stay_id)
        WHERE c.itemid IN ({",".join(map(str, CHART_ITEMS))})
          AND c.valuenum IS NOT NULL
          AND c.stay_id IN ({stay_ids})
    """
    chart = con.execute(chart_sql).fetchall()
    print(f"[{time.time()-t0:6.1f}s] {len(chart):,} chart events")

    # labevents has no stay_id: join by subject + charttime within stay window
    lab_sql = f"""
        SELECT i.stay_id, l.itemid,
               DATEDIFF('second', i.intime, l.charttime) / 3600.0 AS hours,
               l.valuenum
        FROM read_csv_auto('{root}/hosp/labevents.csv.gz') l
        JOIN read_csv_auto('{root}/icu/icustays.csv.gz') i
          ON l.subject_id = i.subject_id
         AND l.charttime >= i.intime AND l.charttime <= i.outtime
        WHERE l.itemid IN ({",".join(map(str, LAB_ITEMS))})
          AND l.valuenum IS NOT NULL
          AND i.stay_id IN ({stay_ids})
    """
    labs = con.execute(lab_sql).fetchall()
    print(f"[{time.time()-t0:6.1f}s] {len(labs):,} lab events")

    per_stay = {}
    for sid, itemid, hours, val in chart:
        feat = CHART_ITEMS[itemid]
        if feat == "temperature_f":
            feat, val = "temperature", (val - 32.0) * 5.0 / 9.0
        per_stay.setdefault(sid, []).append((feat, hours, val))
    for sid, itemid, hours, val in labs:
        per_stay.setdefault(sid, []).append((LAB_ITEMS[itemid], hours, val))

    episodes = []
    for sid, (stay_id, subject_id, intime, los_hours, age) in stay_meta.items():
        ep = build_episode(
            stay_id, subject_id, per_stay.get(sid, []), los_hours,
            FEATURES, VITALS, t_sepsis_hour=None, dataset="mimic4",
            min_length=MIN_LENGTH,
        )
        if ep is not None:
            episodes.append(ep)

    print(f"[{time.time()-t0:6.1f}s] {len(episodes)} episodes built")

    all_sig = np.concatenate([e["signals"] for e in episodes], axis=0)
    print(f"\nGrid: {all_sig.shape[0]:,} patient-hours x {all_sig.shape[1]} features")
    print("NaN density (PhysioNet reference in parens):")
    ref = {"heart_rate": 7.7, "spo2": 12.0, "respiratory_rate": 9.8,
           "temperature": 66.0, "lactate": 97.3, "ph": 89.3,
           "wbc": 93.6, "platelets": 94.0, "bilirubin": 98.5}
    for j, f in enumerate(FEATURES):
        pct = np.isnan(all_sig[:, j]).mean() * 100
        print(f"  {f:18s}: {pct:5.1f}%   ({ref.get(f, float('nan')):.1f}%)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(episodes, fh)
    print(f"\nSaved {len(episodes)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
