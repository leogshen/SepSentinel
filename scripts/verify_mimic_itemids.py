#!/usr/bin/env python
"""Verify every itemid this project relies on against the live MIMIC-IV
dictionaries, and run the IL-6 census (DATA_ACCESS_SPEC.md checklist 1-3).

Checks:
  1. the database version/scale actually present (row counts of the small
     dictionary tables, ICU stay count);
  2. every §5 extraction itemid and every SOFA itemid in sepsis3.py exists,
     with its label printed so a wrong-but-existing id is visible;
  3. the interleukin census in d_labitems AND d_items, with usage counts.

Writes a markdown report; exits non-zero if any itemid is missing.

Usage:
    python scripts/verify_mimic_itemids.py --data-root F:/.../mimic-iv-3.1 \
        --out results/mimic31_itemid_verification.md
"""

import argparse
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sepsis3
from scripts import extract_mimic  # noqa: F401  (only for its itemid tables)

IL6_PATTERNS = ["%interleukin%", "%il-6%", "%il6%", "%procalcitonin%"]


def expected_items():
    """(itemid, table, purpose) for every id the pipeline depends on."""
    items = []
    for iid, name in extract_mimic.CHART_ITEMS.items():
        items.append((iid, "d_items", "extract: %s" % name))
    for iid, name in extract_mimic.LAB_ITEMS.items():
        items.append((iid, "d_labitems", "extract: %s" % name))
    for iid in sepsis3.MAP_ITEMS:
        items.append((iid, "d_items", "SOFA: MAP"))
    items.append((sepsis3.FIO2_ITEM, "d_items", "SOFA: FiO2"))
    items.append((sepsis3.PAO2_CHART_ITEM, "d_items", "SOFA: PaO2 (arterial)"))
    for part, iid in sepsis3.GCS_ITEMS.items():
        items.append((iid, "d_items", "SOFA: GCS %s" % part))
    for iid in sepsis3.VENT_ITEMS:
        items.append((iid, "d_items", "SOFA: ventilation"))
    for iid in sepsis3.URINE_ITEMS:
        items.append((iid, "d_items", "SOFA: urine output"))
    for iid, name in sepsis3.VASO_ITEMS.items():
        items.append((iid, "d_items", "SOFA: %s" % name))
    for iid, name in [(sepsis3.LAB_PLATELET, "platelets"),
                      (sepsis3.LAB_BILIRUBIN, "bilirubin"),
                      (sepsis3.LAB_CREATININE, "creatinine"),
                      (sepsis3.LAB_PAO2, "PaO2")]:
        items.append((iid, "d_labitems", "SOFA: %s" % name))
    # de-duplicate, keeping the merged purpose text
    merged = {}
    for iid, table, purpose in items:
        key = (iid, table)
        merged.setdefault(key, []).append(purpose)
    return sorted((iid, table, "; ".join(sorted(set(p))))
                  for (iid, table), p in merged.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/mimic_itemid_verification.md")
    args = ap.parse_args()
    root = args.data_root.rstrip("/\\").replace("\\", "/")

    con = duckdb.connect()
    con.execute("CREATE VIEW d_items AS SELECT * FROM "
                "read_csv_auto('%s/icu/d_items.csv.gz')" % root)
    con.execute("CREATE VIEW d_labitems AS SELECT * FROM "
                "read_csv_auto('%s/hosp/d_labitems.csv.gz')" % root)
    con.execute("CREATE VIEW icustays AS SELECT * FROM "
                "read_csv_auto('%s/icu/icustays.csv.gz')" % root)
    con.execute("CREATE VIEW patients AS SELECT * FROM "
                "read_csv_auto('%s/hosp/patients.csv.gz')" % root)

    lines = ["# MIMIC-IV itemid verification and IL-6 census", "",
             "Source: `%s`" % root, ""]

    n_stays = con.execute("SELECT COUNT(*) FROM icustays").fetchone()[0]
    n_subj = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    n_di = con.execute("SELECT COUNT(*) FROM d_items").fetchone()[0]
    n_dl = con.execute("SELECT COUNT(*) FROM d_labitems").fetchone()[0]
    lines += ["## Scale", "",
              "| ICU stays | Patients | d_items rows | d_labitems rows |",
              "|---|---|---|---|",
              "| %d | %d | %d | %d |" % (n_stays, n_subj, n_di, n_dl), ""]
    print("ICU stays %d | patients %d | d_items %d | d_labitems %d"
          % (n_stays, n_subj, n_di, n_dl))

    lines += ["## Itemids", "", "| itemid | table | used for | label in DB |",
              "|---|---|---|---|"]
    missing = []
    for iid, table, purpose in expected_items():
        row = con.execute("SELECT label FROM %s WHERE itemid = %d"
                          % (table, iid)).fetchone()
        label = row[0] if row else "*** MISSING ***"
        if row is None:
            missing.append((iid, table, purpose))
        lines.append("| %d | %s | %s | %s |" % (iid, table, purpose, label))
    lines.append("")

    # --- IL-6 census -------------------------------------------------------
    lines += ["## Interleukin / procalcitonin census", ""]
    where = " OR ".join("LOWER(label) LIKE '%s'" % p for p in IL6_PATTERNS)
    hits = []
    for table, ev_table, ev_root in [
            ("d_labitems", "labevents", "hosp"), ("d_items", "chartevents", "icu")]:
        rows = con.execute("SELECT itemid, label FROM %s WHERE %s"
                           % (table, where)).fetchall()
        if not rows:
            lines.append("- `%s`: no matching items." % table)
            continue
        for iid, label in rows:
            try:
                cnt = con.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT subject_id) FROM "
                    "read_csv_auto('%s/%s/%s.csv.gz') WHERE itemid = %d"
                    % (root, ev_root, ev_table, iid)).fetchone()
            except Exception as exc:                     # table not extracted
                cnt = ("?", "? (%s)" % type(exc).__name__)
            hits.append((table, iid, label, cnt[0], cnt[1]))
            lines.append("- `%s` %d **%s**: %s rows, %s subjects"
                         % (table, iid, label, cnt[0], cnt[1]))
    if not hits:
        lines.append("")
        lines.append("**Confirmed: MIMIC-IV carries no interleukin items.** "
                     "The IL-6 bridge must come from ImmPort SDY1662.")
    lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("Saved report -> %s" % args.out)

    if missing:
        print("\nMISSING ITEMIDS:")
        for iid, table, purpose in missing:
            print("  %d (%s) - %s" % (iid, table, purpose))
        return 1
    print("All %d itemids verified present." % len(expected_items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
