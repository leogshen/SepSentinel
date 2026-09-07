#!/usr/bin/env python
"""Do static attributes earn a place in the input?

Age, sex, weight, first care unit and admission type are ~100% complete in
MIMIC-IV and no model in this project uses any of them. Before building a
static pathway into the sequence model, this checks whether they carry signal
at all, using the flat models where adding them is just extra columns.

Each static is broadcast to every hour of its stay (that is what a "static
token" would amount to for a flat model). Categoricals are one-hot encoded.

Usage:
    python scripts/static_ablation.py --episodes results/...pkl \
        --data-root <mimic-iv-3.1> --out-dir results/static_ablation
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, curve_for, at_burden,
    BURDEN_POINTS,
)


def load_statics(root, stay_ids):
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    root = root.rstrip("/\\").replace("\\", "/")
    ids = ",".join(str(int(s)) for s in stay_ids)
    df = con.execute("""
        SELECT i.stay_id, p.anchor_age AS age,
               CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS male,
               i.first_careunit, a.admission_type
        FROM read_csv_auto('%s/icu/icustays.csv.gz') i
        JOIN read_csv_auto('%s/hosp/patients.csv.gz') p USING (subject_id)
        LEFT JOIN read_csv_auto('%s/hosp/admissions.csv.gz') a USING (hadm_id)
        WHERE i.stay_id IN (%s)
    """ % (root, root, root, ids)).df()
    wt = con.execute("""
        SELECT stay_id, MEDIAN(valuenum) AS weight
        FROM read_csv_auto('%s/icu/chartevents.csv.gz',
             types={'valuenum':'DOUBLE','itemid':'BIGINT','stay_id':'BIGINT'})
        WHERE itemid IN (226512, 224639) AND valuenum BETWEEN 20 AND 400
          AND stay_id IN (%s)
        GROUP BY stay_id
    """ % (root, ids)).df()
    df = df.merge(wt, on="stay_id", how="left")
    return df.set_index("stay_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    stay_ids = [int(e["stay_id"]) for e in episodes]
    stat = load_statics(args.data_root, stay_ids)
    print("statics loaded for %d/%d stays" % (len(stat), len(stay_ids)))

    units = sorted(stat["first_careunit"].dropna().unique())
    adm = sorted(stat["admission_type"].dropna().unique())
    print("%d care units, %d admission types" % (len(units), len(adm)))

    med_age = float(stat["age"].median())
    med_wt = float(stat["weight"].median())

    def static_vec(sid):
        try:
            r = stat.loc[sid]
        except KeyError:
            return np.zeros(4 + len(units) + len(adm), dtype=np.float32)
        v = [float(r["age"]) if r["age"] == r["age"] else med_age,
             float(r["male"]),
             float(r["weight"]) if r["weight"] == r["weight"] else med_wt,
             1.0 if r["weight"] != r["weight"] else 0.0]      # weight-missing flag
        v += [1.0 if r["first_careunit"] == u else 0.0 for u in units]
        v += [1.0 if r["admission_type"] == a else 0.0 for a in adm]
        return np.asarray(v, dtype=np.float32)

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}
    pre = build_preprocessor(episodes, "all")
    tr = pre.fit_transform(splits["train"])
    te = pre.transform(splits["test"])

    def design(data, with_static):
        X = np.concatenate([d["signals"] for d in data])
        y = np.concatenate([d["labels"] for d in data])
        if not with_static:
            return X, y
        blocks = [np.tile(static_vec(int(d["patient_id"])), (len(d["labels"]), 1))
                  for d in data]
        return np.concatenate([X, np.concatenate(blocks)], axis=1), y

    results = {}
    for with_static in (False, True):
        tag = "with_statics" if with_static else "timeseries_only"
        X_tr, y_tr = design(tr, with_static)
        X_te, y_te = design(te, with_static)
        spw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
        print("\n%s: %d features" % (tag, X_tr.shape[1]))
        results[tag] = {}
        for name in ("logreg", "xgboost"):
            if name == "logreg":
                clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                         random_state=42).fit(X_tr, y_tr)
            else:
                import xgboost as xgb
                clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                        learning_rate=0.1, random_state=42,
                                        scale_pos_weight=spw,
                                        eval_metric="logloss",
                                        tree_method="hist").fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]
            curve = curve_for(patient_results_from_probs(te, raw_map, prob))
            c = at_burden(curve, 1.0)
            auroc = float(roc_auc_score(y_te, prob))
            results[tag][name] = {
                "auroc": auroc,
                "auprc": float(average_precision_score(y_te, prob)),
                "at_1_per_day": c}
            print("  %-8s AUROC %.4f AUPRC %.4f | at <=1.0/day: recall %.2f, "
                  "lead %s h, capture>=6h %.2f"
                  % (name, auroc, results[tag][name]["auprc"],
                     c["patient_recall"],
                     "%.1f" % c["median_lead_time_h"]
                     if c["median_lead_time_h"] is not None else "n/a",
                     c["capture_6h"]))

    print("\nDelta from adding statics:")
    for name in ("logreg", "xgboost"):
        a = results["timeseries_only"][name]
        b = results["with_statics"][name]
        print("  %-8s AUROC %+.4f  AUPRC %+.4f  capture>=6h %+.3f"
              % (name, b["auroc"] - a["auroc"], b["auprc"] - a["auprc"],
                 b["at_1_per_day"]["capture_6h"] - a["at_1_per_day"]["capture_6h"]))

    with open(os.path.join(args.out_dir, "static_ablation.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
