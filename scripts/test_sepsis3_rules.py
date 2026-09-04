#!/usr/bin/env python
"""Rule tests for sepsentinel/data/sepsis3.py on a synthetic mini-MIMIC.

The demo database can only show that the labeller runs and gives plausible
prevalence; it cannot show that the Challenge timing rules are the ones being
applied. This builds a tiny MIMIC-shaped database with hand-placed events and
asserts the resulting onsets.

Run: python scripts/test_sepsis3_rules.py   (exits non-zero on failure)
"""

import datetime as dt
import gzip
import os
import shutil
import sys
import tempfile

import duckdb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sepsis3

BASE = dt.datetime(2150, 1, 1, 0, 0, 0)
LOS_HOURS = 200


def ts(hour):
    return (BASE + dt.timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")


def write_csv(root, rel, header, rows):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join("" if v is None else str(v) for v in r) + "\n")


def build_fixture(root):
    """Six stays, each exercising one branch of the definition.

    Every stay shares the same SOFA machinery: platelets 300 (coagulation 0)
    at hour 0, dropping to 80 (coagulation 2) at the stay's "sofa hour", which
    is the only source of a >=2-point rise.
    """
    stays = [1, 2, 3, 4, 5, 6]
    write_csv(root, "icu/icustays.csv.gz",
              ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
              [(s, 100 + s, s, ts(0), ts(LOS_HOURS), LOS_HOURS / 24.0)
               for s in stays])
    write_csv(root, "hosp/patients.csv.gz",
              ["subject_id", "gender", "anchor_age", "anchor_year"],
              [(s, "M", 60, 2150) for s in stays])

    # platelet drop hour per stay
    sofa_hour = {1: 12, 2: 12, 3: 40, 4: 12, 5: 4, 6: None}
    lab_rows = []
    for s in stays:
        lab_rows.append((s, sepsis3.LAB_PLATELET, ts(0), 300))
        if sofa_hour[s] is not None:
            for h in range(sofa_hour[s], LOS_HOURS, 6):   # keep it low
                lab_rows.append((s, sepsis3.LAB_PLATELET, ts(h), 80))
    write_csv(root, "hosp/labevents.csv.gz",
              ["subject_id", "itemid", "charttime", "valuenum"], lab_rows)

    # antibiotics: stay 1 abx h10 (culture h20, +10h -> pair)
    #              stay 2 abx h80 (culture h5, +75h -> NO pair)
    #              stay 3 abx h10 (culture h20 -> pair, but SOFA at h40 is
    #                              outside t_susp+12h)
    #              stay 4 no antibiotics at all
    #              stay 5 abx h20, culture h20 -> suspicion h20, SOFA rise at
    #                              h4 -> inside [t_susp-24, t_susp+12], and
    #                              min(4, 20) = 4 -> onset driven by SOFA
    #              stay 6 abx h10, culture h20 -> pair, but no SOFA rise
    abx = {1: 10, 2: 80, 3: 10, 5: 20, 6: 10}
    write_csv(root, "hosp/prescriptions.csv.gz",
              ["subject_id", "hadm_id", "starttime", "stoptime", "drug",
               "route"],
              [(s, 100 + s, ts(h), ts(h + 8), "Vancomycin", "IV")
               for s, h in abx.items()])
    write_csv(root, "hosp/emar.csv.gz",
              ["subject_id", "hadm_id", "charttime", "medication", "event_txt"],
              [])

    cultures = {1: 20, 2: 5, 3: 20, 5: 20, 6: 20}
    write_csv(root, "hosp/microbiologyevents.csv.gz",
              ["subject_id", "hadm_id", "chartdate", "charttime",
               "micro_specimen_id", "spec_type_desc"],
              [(s, 100 + s, ts(h), ts(h), 900 + s, "BLOOD CULTURE")
               for s, h in cultures.items()])

    # empty-but-well-typed event tables
    write_csv(root, "icu/chartevents.csv.gz",
              ["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
               "value", "valuenum", "valueuom"], [])
    write_csv(root, "icu/inputevents.csv.gz",
              ["subject_id", "hadm_id", "stay_id", "starttime", "endtime",
               "itemid", "amount", "rate", "rateuom", "patientweight",
               "statusdescription"], [])
    write_csv(root, "icu/outputevents.csv.gz",
              ["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
               "value", "valueuom"], [])
    write_csv(root, "icu/procedureevents.csv.gz",
              ["subject_id", "hadm_id", "stay_id", "starttime", "endtime",
               "itemid", "value", "valueuom", "statusdescription"], [])


EXPECTED = {
    1: ("septic", 10.0, "abx h10 + culture h20 (culture <=24h after abx); "
                        "SOFA rise h12 inside [t_susp-24, t_susp+12]"),
    2: ("control", None, "culture h5, abx h80: abx is >72h after the culture, "
                         "so the pair does not form"),
    3: ("control", None, "suspicion h10 but the SOFA rise is at h40, outside "
                         "t_susp+12h"),
    4: ("control", None, "SOFA rise but no antibiotics -> no suspicion"),
    5: ("septic", 4.0, "suspicion h20, SOFA rise h4 (inside t_susp-24h); "
                       "t_sepsis = min(20, 4) = 4"),
    6: ("control", None, "suspicion h10 but SOFA never rises"),
}


def main():
    root = tempfile.mkdtemp(prefix="sepsis3_fixture_")
    try:
        build_fixture(root)
        con = duckdb.connect()
        df = sepsis3.run(con, root, verbose=False).set_index("stay_id")

        failures = []
        print("stay  expected        got             note")
        for stay_id, (kind, onset, why) in EXPECTED.items():
            row = df.loc[stay_id]
            got = row["t_sepsis_hour"]
            got = None if pd.isna(got) else float(got)
            ok = (got == onset)
            print("%-5d %-15s %-15s %s"
                  % (stay_id,
                     kind if onset is None else "septic @ %.0fh" % onset,
                     "control" if got is None else "septic @ %.0fh" % got,
                     "OK" if ok else "FAIL"))
            print("      why: %s" % why)
            if not ok:
                failures.append(stay_id)

        print()
        if failures:
            print("FAILED on stays: %s" % failures)
            return 1
        print("All %d rule cases passed." % len(EXPECTED))
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
