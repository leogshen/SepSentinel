#!/usr/bin/env python
"""Dataset characterisation for the advisor report.

Reports, for each episode pickle given: patients vs stays, record lengths,
positive/negative fractions at both patient and timestep level, and
per-channel missingness. Also queries MIMIC-IV directly for the STATIC
attributes we do not currently use, with their completeness, so the question
"what else could go in, and how much of it is there" has a number attached.

Usage:
    python scripts/dataset_report.py --episodes a.pkl b.pkl \
        --data-root <mimic-iv-3.1> --out results/DATASET_REPORT.md
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def describe(path):
    with open(path, "rb") as fh:
        eps = pickle.load(fh)
    feats = eps[0]["features"]
    lens = np.array([len(e["labels"]) for e in eps])
    lab = np.array([e["label"] for e in eps])
    subj = set(e.get("subject_id", e["patient_id"]) for e in eps)
    y = np.concatenate([e["labels"] for e in eps])
    sig = np.concatenate([e["signals"] for e in eps], axis=0)

    sep_lens = lens[lab == 1]
    ctl_lens = lens[lab == 0]
    d = {
        "path": path, "episodes": len(eps), "subjects": len(subj),
        "stays_per_subject": len(eps) / max(len(subj), 1),
        "rows": int(lens.sum()),
        "len_mean": float(lens.mean()), "len_median": float(np.median(lens)),
        "len_p25": float(np.percentile(lens, 25)),
        "len_p75": float(np.percentile(lens, 75)),
        "len_min": int(lens.min()), "len_max": int(lens.max()),
        "sep_len_median": float(np.median(sep_lens)) if len(sep_lens) else None,
        "ctl_len_median": float(np.median(ctl_lens)) if len(ctl_lens) else None,
        "positives": int((lab == 1).sum()), "negatives": int((lab == 0).sum()),
        "patient_pos_pct": 100.0 * float((lab == 1).mean()),
        "timestep_pos_pct": 100.0 * float(np.nanmean(y)),
        "pos_rows": int(np.nansum(y)), "features": feats,
        "nan_pct": {f: 100.0 * float(np.isnan(sig[:, j]).mean())
                    for j, f in enumerate(feats)},
    }
    return d


def static_attributes(root):
    """Availability of static/admission attributes we do NOT currently use."""
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    root = root.rstrip("/\\").replace("\\", "/")
    con.execute("""
        CREATE TEMP TABLE co AS
        SELECT i.stay_id, i.subject_id, i.hadm_id, i.first_careunit,
               DATEDIFF('second', i.intime, i.outtime) / 3600.0 AS los_hours,
               p.anchor_age, p.gender
        FROM read_csv_auto('%s/icu/icustays.csv.gz') i
        JOIN read_csv_auto('%s/hosp/patients.csv.gz') p USING (subject_id)
        WHERE p.anchor_age >= 18
          AND DATEDIFF('second', i.intime, i.outtime) / 3600.0 >= 6
    """ % (root, root))
    n = con.execute("SELECT COUNT(*) FROM co").fetchone()[0]

    rows = []
    for label, expr in [
            ("age (patients.anchor_age)", "anchor_age"),
            ("sex (patients.gender)", "gender"),
            ("first ICU care unit", "first_careunit"),
    ]:
        k = con.execute("SELECT COUNT(*) FROM co WHERE %s IS NOT NULL"
                        % expr).fetchone()[0]
        rows.append((label, 100.0 * k / n, "static"))

    adm = con.execute("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN a.admission_type IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN a.insurance IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN a.race IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN a.marital_status IS NOT NULL THEN 1 ELSE 0 END)
        FROM co JOIN read_csv_auto('%s/hosp/admissions.csv.gz') a
          USING (hadm_id)
    """ % root).fetchone()
    for i, label in enumerate(["admission type", "insurance", "race",
                               "marital status"], start=1):
        rows.append((label + " (admissions)", 100.0 * adm[i] / max(adm[0], 1),
                     "static"))

    # Weight and height are charted, not static columns.
    for label, items in [("weight (chartevents 226512/224639)",
                          (226512, 224639)),
                         ("height (chartevents 226730)", (226730,))]:
        k = con.execute("""
            SELECT COUNT(DISTINCT c.stay_id)
            FROM read_csv_auto('%s/icu/chartevents.csv.gz',
                 types={'valuenum':'DOUBLE','itemid':'BIGINT','stay_id':'BIGINT'}) c
            JOIN co USING (stay_id)
            WHERE c.itemid IN (%s) AND c.valuenum IS NOT NULL
        """ % (root, ",".join(map(str, items)))).fetchone()[0]
        rows.append((label, 100.0 * k / n, "static-ish (charted once)"))
    return n, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    described = [describe(p) for p in args.episodes]
    lines = ["# Dataset characterisation", ""]
    lines += ["## Cohort size and composition", "",
              "| Dataset | Episodes (ICU stays) | Unique patients | Stays/patient | "
              "Total patient-hours | Septic | Control | Patient-level positive | "
              "Timestep-level positive |", "|---|---|---|---|---|---|---|---|---|"]
    for d in described:
        lines.append("| `%s` | %d | %d | %.2f | %d | %d | %d | %.1f%% | %.2f%% |"
                     % (os.path.basename(d["path"]), d["episodes"],
                        d["subjects"], d["stays_per_subject"], d["rows"],
                        d["positives"], d["negatives"], d["patient_pos_pct"],
                        d["timestep_pos_pct"]))
    lines += ["", "## Record length (hours per ICU stay)", "",
              "| Dataset | Mean | Median | p25 | p75 | Min | Max | "
              "Median septic | Median control |", "|---|---|---|---|---|---|---|---|---|"]
    for d in described:
        lines.append("| `%s` | %.1f | %.0f | %.0f | %.0f | %d | %d | %s | %s |"
                     % (os.path.basename(d["path"]), d["len_mean"],
                        d["len_median"], d["len_p25"], d["len_p75"],
                        d["len_min"], d["len_max"],
                        "%.0f" % d["sep_len_median"] if d["sep_len_median"] else "-",
                        "%.0f" % d["ctl_len_median"] if d["ctl_len_median"] else "-"))

    lines += ["", "## Missingness per input channel (% of patient-hours with "
              "no measurement)", ""]
    allf = sorted({f for d in described for f in d["features"]})
    lines.append("| Channel | " + " | ".join(os.path.basename(d["path"])
                                             for d in described) + " |")
    lines.append("|---" * (len(described) + 1) + "|")
    for f in allf:
        cells = ["%.1f%%" % d["nan_pct"][f] if f in d["nan_pct"] else "n/a"
                 for d in described]
        lines.append("| %s | %s |" % (f, " | ".join(cells)))

    if args.data_root:
        n, rows = static_attributes(args.data_root)
        lines += ["", "## Static / admission attributes NOT currently used as "
                  "model inputs", "",
                  "Completeness over the %d qualifying ICU stays." % n, "",
                  "| Attribute | Available | Kind |", "|---|---|---|"]
        for label, pct, kind in rows:
            lines.append("| %s | %.1f%% | %s |" % (label, pct, kind))

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
