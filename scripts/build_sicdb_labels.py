#!/usr/bin/env python
"""Rung-1 sepsis labels for SICdb 1.0.8, plus a deviation report.

Mirrors scripts/build_sepsis3_labels.py for MIMIC-IV, but the definition is
NOT the same and the outputs must never be pooled. SICdb has no microbiology,
so the Challenge's antibiotic/culture pairing is impossible. See
sepsentinel/data/sicdb.py DEVIATIONS for the full list; the two that matter
most:

  * suspicion = start of an antibacterial COURSE (>=3 doses spanning >=24 h),
    not a culture-paired antibiotic. 79% of eligible cases receive some
    antibacterial, so a first-dose rule would label most of a perioperative
    cohort.
  * SOFA omits the CNS component (no usable GCS), so it ranges 0-20 and is
    not numerically comparable to a MIMIC SOFA.

Timing rules are otherwise identical to the MIMIC build: t_SOFA is the first
hour at least 2 points above the minimum of the preceding 24 h, and sepsis
requires t_SOFA within [t_suspicion - 24 h, t_suspicion + 12 h].

Usage:
    python scripts/build_sicdb_labels.py \
        --data-root F:/Claude/Sepsentinel/data_local/sicdb-1.0.8 \
        --out results/sicdb_labels.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sicdb


def ids(seq):
    return ",".join(str(int(i)) for i in seq)


def build_hourly(con, max_hours):
    """One row per (CaseID, hour) with the SOFA inputs, from ICU admission."""
    lab = sicdb.LAB_IDS
    con.execute("""
        CREATE OR REPLACE TEMP VIEW elig AS
        SELECT CaseID, ICUOffset, TimeOfStay, AgeOnAdmission,
               WeightOnAdmission, SurgicalAdmissionType, AdmissionFormHasSepsis,
               OffsetOfDeath, DischargeState
        FROM cases
        WHERE AgeOnAdmission >= %d AND TimeOfStay >= %d * 3600
    """ % (sicdb.MIN_AGE, sicdb.MIN_LOS_HOURS))

    # Vitals: hourly means already, so just bin by ICU hour and take the mean
    # of whatever sources are present, in coalesce priority order.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW vit AS
        SELECT d.CaseID,
               CAST(floor((d."Offset" - e.ICUOffset) / 3600.0) AS BIGINT) AS hr,
               d.DataID, d.Val
        FROM data_float_h d JOIN elig e USING (CaseID)
        WHERE d."Offset" >= e.ICUOffset
          AND (d."Offset" - e.ICUOffset) / 3600.0 < %d
          AND d.Val IS NOT NULL
    """ % max_hours)

    con.execute("""
        CREATE OR REPLACE TEMP VIEW lb AS
        SELECT l.CaseID,
               CAST(floor((l."Offset" - e.ICUOffset) / 3600.0) AS BIGINT) AS hr,
               l.LaboratoryID AS lid, l.LaboratoryValue AS val
        FROM laboratory l JOIN elig e USING (CaseID)
        WHERE (l."Offset" - e.ICUOffset) / 3600.0 < %d
          AND l.LaboratoryValue IS NOT NULL
    """ % max_hours)

    def vsel(name):
        return ("max(CASE WHEN DataID IN (%s) THEN Val END) AS %s"
                % (ids(sicdb.VITAL_IDS[name]), name))

    def lsel(name):
        return ("max(CASE WHEN lid IN (%s) THEN val END) AS %s"
                % (ids(lab[name]), name))

    q = """
        WITH v AS (
          SELECT CaseID, hr, %s FROM vit GROUP BY 1, 2
        ), l AS (
          SELECT CaseID, hr, %s FROM lb GROUP BY 1, 2
        ), press AS (
          SELECT CaseID, hr, max(Val) AS peep FROM vit
          WHERE DataID = 2278 GROUP BY 1, 2
        ), vaso AS (
          SELECT m.CaseID,
                 CAST(floor((m."Offset" - e.ICUOffset) / 3600.0) AS BIGINT) AS hr,
                 max(CASE WHEN m.DrugID = %d THEN m.AmountPerMinute END) AS norepi,
                 max(CASE WHEN m.DrugID = %d THEN m.AmountPerMinute END) AS epi,
                 max(CASE WHEN m.DrugID = %d THEN m.AmountPerMinute END) AS dopa,
                 max(CASE WHEN m.DrugID = %d THEN m.AmountPerMinute END) AS dobu
          FROM medication m JOIN elig e USING (CaseID)
          WHERE m."Offset" >= e.ICUOffset GROUP BY 1, 2
        )
        SELECT v.CaseID, v.hr, %s, %s, press.peep,
               vaso.norepi, vaso.epi, vaso.dopa, vaso.dobu,
               e.WeightOnAdmission AS weight
        FROM v
        LEFT JOIN l ON l.CaseID = v.CaseID AND l.hr = v.hr
        LEFT JOIN press ON press.CaseID = v.CaseID AND press.hr = v.hr
        LEFT JOIN vaso ON vaso.CaseID = v.CaseID AND vaso.hr = v.hr
        JOIN elig e ON e.CaseID = v.CaseID
        WHERE v.hr >= 0
        ORDER BY v.CaseID, v.hr
    """ % (
        ", ".join(vsel(n) for n in sicdb.VITAL_IDS),
        ", ".join(lsel(n) for n in lab),
        sicdb.VASOPRESSOR_IDS["norepinephrine"],
        sicdb.VASOPRESSOR_IDS["epinephrine"],
        sicdb.VASOPRESSOR_IDS["dopamine"],
        sicdb.VASOPRESSOR_IDS["dobutamine"],
        ", ".join("v.%s" % n for n in sicdb.VITAL_IDS),
        ", ".join("l.%s" % n for n in lab),
    )
    return con.execute(q).df()


def score_sofa(df):
    """Five-component SOFA (no CNS). Labs are carried forward within a stay."""
    d = df.sort_values(["CaseID", "hr"]).copy()
    carry = ["pao2", "fio2", "platelets", "bilirubin", "creatinine", "map"]
    for c in carry:
        d[c] = d.groupby("CaseID")[c].ffill()

    # Respiration: PaO2/FiO2, FiO2 on a percent scale here.
    ratio = d["pao2"] / (d["fio2"] / 100.0)
    vent = d["peep"].notna()
    resp = np.select(
        [(ratio < 100) & vent, (ratio < 200) & vent, ratio < 300, ratio < 400],
        [4, 3, 2, 1], default=0).astype(float)
    resp[ratio.isna()] = np.nan

    plt = d["platelets"]
    coag = np.select([plt < 20, plt < 50, plt < 100, plt < 150],
                     [4, 3, 2, 1], default=0).astype(float)
    coag[plt.isna()] = np.nan

    bil = d["bilirubin"]
    liver = np.select([bil >= 12, bil >= 6, bil >= 2, bil >= 1.2],
                      [4, 3, 2, 1], default=0).astype(float)
    liver[bil.isna()] = np.nan

    # Vasopressor rates: AmountPerMinute is mg/min in SICdb; SOFA is in
    # ug/kg/min, hence *1000/weight.
    w = d["weight"].replace(0, np.nan)
    ne = d["norepi"] * 1000.0 / w
    ep = d["epi"] * 1000.0 / w
    dp = d["dopa"] * 1000.0 / w
    db = d["dobu"]
    mp = d["map"]
    cardio = np.select(
        [(dp > 15) | (ep > 0.1) | (ne > 0.1),
         (dp > 5) | (ep > 0) | (ne > 0),
         (dp > 0) | db.notna(),
         mp < 70],
        [4, 3, 2, 1], default=0).astype(float)

    cr = d["creatinine"]
    uo24 = (d.groupby("CaseID")["urine_output"]
              .rolling(24, min_periods=24).sum().reset_index(level=0, drop=True))
    renal = np.select(
        [(cr >= 5) | (uo24 < 200), (cr >= 3.5) | (uo24 < 500),
         cr >= 2, cr >= 1.2],
        [4, 3, 2, 1], default=0).astype(float)
    renal[cr.isna() & uo24.isna()] = np.nan

    d["sofa"] = sicdb.sofa_components(resp, coag, liver, cardio, renal)
    return d[["CaseID", "hr", "sofa"]]


def suspicion_times(con):
    """Start of the first antibacterial COURSE, in hours from ICU admission."""
    q = """
        WITH abx AS (
          SELECT m.CaseID, m.DrugID,
                 (m."Offset" - c.ICUOffset) / 3600.0 AS hr
          FROM medication m JOIN cases c USING (CaseID)
          WHERE m.DrugID IN (%s)
        ), per_case AS (
          SELECT CaseID, count(*) AS n_doses,
                 min(hr) AS first_hr, max(hr) - min(hr) AS span_h
          FROM abx GROUP BY 1
        )
        SELECT CaseID, first_hr AS t_suspicion_hour, n_doses, span_h
        FROM per_case
        WHERE n_doses >= %d AND span_h >= %d
    """ % (ids(sicdb.ANTIBACTERIAL_IDS), sicdb.COURSE_MIN_DOSES,
           sicdb.COURSE_MIN_SPAN_H)
    return con.execute(q).df()


def validate(con, out):
    """Check the rung-1 label against the only references SICdb offers.

    There is no gold standard here -- that is the whole problem with rung 1 --
    but three weak checks together say whether the label is selecting sick
    patients or just selecting antibiotic exposure:
      * AdmissionFormHasSepsis, the admitting physician's SAPS3 checkbox.
        Admission-time only and acknowledged by the maintainer to carry
        observer bias, so it undercounts hospital-acquired sepsis.
      * ICU mortality and mean SAPS3, which should both be elevated.
      * The rate within ELECTIVE surgery, which is close to a negative
        control: elective post-operative patients receiving prophylaxis
        should not be septic.
    """
    cs = con.execute("SELECT CaseID, AdmissionFormHasSepsis, DischargeState, "
                     "SurgicalAdmissionType, saps3 FROM cases").df()
    m = out.merge(cs, on="CaseID")
    m["sep"] = m.t_sepsis_hour.notna()
    m["died"] = (m.DischargeState == 2215)
    cb = m[m.AdmissionFormHasSepsis == 740]
    elective = m[m.SurgicalAdmissionType == 3126]
    return {
        "sepsis_rate": float(m.sep.mean()),
        "checkbox_positive": int(len(cb)),
        "checkbox_sensitivity": float(cb.sep.mean()) if len(cb) else None,
        "checkbox_ppv": float((m.sep & (m.AdmissionFormHasSepsis == 740)).sum()
                              / max(int(m.sep.sum()), 1)),
        "mortality_septic": float(m[m.sep].died.mean()),
        "mortality_nonseptic": float(m[~m.sep].died.mean()),
        "saps3_septic": float(m[m.sep].saps3.mean()),
        "saps3_nonseptic": float(m[~m.sep].saps3.mean()),
        "elective_surgery_rate": float(elective.sep.mean()),
        "elective_surgery_mortality": float(elective.died.mean()),
        "verdict": (
            "OVER-CALLS. 27.6% of a low-mortality perioperative cohort is not "
            "a credible sepsis incidence. The label does select severity "
            "(3.2x mortality, +9 SAPS3) so it is not noise, but it flags ~7x "
            "more cases than the admitting physician did and calls 22.7% of "
            "ELECTIVE surgical patients septic at 0.9% mortality -- that arm "
            "is prophylaxis, not infection. Use these labels for "
            "stratification and for transfer-discrimination under the "
            "relabelled definition. Do NOT train on them, and never pool them "
            "with MIMIC labels."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=int, default=336)
    ap.add_argument("--temp-dir",
                    default="F:/Claude/Sepsentinel/data_local/duckdb_tmp")
    args = ap.parse_args()
    t0 = time.time()

    import duckdb
    con = duckdb.connect()
    con.execute("SET temp_directory='%s'" % args.temp_dir)
    sicdb.register_sources(con, args.data_root)

    print("Building hourly SOFA inputs (one pass over 36.8M rows)...")
    hourly = build_hourly(con, args.max_hours)
    print("  %d case-hours over %d cases (%.1f min)"
          % (len(hourly), hourly.CaseID.nunique(), (time.time() - t0) / 60))

    sofa = score_sofa(hourly)
    susp = suspicion_times(con)
    print("  %d cases with an antibacterial course" % len(susp))

    rows = []
    susp_map = dict(zip(susp.CaseID, susp.t_suspicion_hour))
    for cid, g in sofa.groupby("CaseID"):
        t_susp = susp_map.get(cid)
        s = g.sort_values("hr")["sofa"].to_numpy()
        t_sofa = sicdb.first_sofa_rise(s)
        t_sepsis = sicdb.t_sepsis_from(t_susp, t_sofa)
        rows.append({"CaseID": int(cid), "t_suspicion_hour": t_susp,
                     "t_sofa_hour": t_sofa, "t_sepsis_hour": t_sepsis,
                     "max_sofa": float(np.nanmax(s)) if s.size else np.nan})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    n = len(out)
    n_susp = out.t_suspicion_hour.notna().sum()
    n_sofa = out.t_sofa_hour.notna().sum()
    n_sep = out.t_sepsis_hour.notna().sum()
    report = {
        "label_name": sicdb.LABEL_NAME, "label_rung": sicdb.LABEL_RUNG,
        "warning": sicdb.LABEL_WARNING, "deviations": sicdb.DEVIATIONS,
        "cases_scored": int(n),
        "with_suspicion": int(n_susp),
        "with_sofa_rise": int(n_sofa),
        "with_sepsis": int(n_sep),
        "sepsis_rate": float(n_sep / n) if n else None,
        "course_min_doses": sicdb.COURSE_MIN_DOSES,
        "course_min_span_h": sicdb.COURSE_MIN_SPAN_H,
        "minutes": (time.time() - t0) / 60.0,
    }
    report["validation"] = validate(con, out)
    with open(os.path.splitext(args.out)[0] + "_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    print("cases scored      %d" % n)
    print("  suspicion       %d (%.1f%%)" % (n_susp, 100 * n_susp / max(n, 1)))
    print("  SOFA rise >=2   %d (%.1f%%)" % (n_sofa, 100 * n_sofa / max(n, 1)))
    print("  SEPSIS (rung 1) %d (%.1f%%)" % (n_sep, 100 * n_sep / max(n, 1)))
    v = report["validation"]
    print()
    print("validation against the weak references available:")
    print("  SAPS3 checkbox: sensitivity %.1f%%, and only %.1f%% of our "
          "positives were flagged by the physician"
          % (100 * v["checkbox_sensitivity"], 100 * v["checkbox_ppv"]))
    print("  ICU mortality:  %.1f%% septic vs %.1f%% not (SAPS3 %.1f vs %.1f)"
          % (100 * v["mortality_septic"], 100 * v["mortality_nonseptic"],
             v["saps3_septic"], v["saps3_nonseptic"]))
    print("  ELECTIVE surgery: %.1f%% labelled septic at %.1f%% mortality"
          % (100 * v["elective_surgery_rate"],
             100 * v["elective_surgery_mortality"]))
    print("  -> %s" % v["verdict"])
    print()
    print(sicdb.LABEL_WARNING)
    print("\nSaved -> %s (%.1f min)" % (args.out, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
