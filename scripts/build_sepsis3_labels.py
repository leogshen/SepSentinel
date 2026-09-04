#!/usr/bin/env python
"""Build Sepsis-3 (Challenge-rule) labels for MIMIC-IV ICU stays.

Writes a labels CSV (one row per qualifying stay, t_sepsis_hour NULL for
controls) plus a markdown report with the attrition counts, the onset-time
distribution, and the documented mimic-code deviations
(DATA_ACCESS_SPEC.md section 3 / checklist step 4).

Usage:
    python scripts/build_sepsis3_labels.py \
        --data-root F:/Claude/Sepsentinel/data_local/mimic-iv-clinical-database-demo-2.2 \
        --out results/sepsis3_labels_demo.csv
    python scripts/build_sepsis3_labels.py --data-root C:/data/mimic-iv-3.1 \
        --limit-stays 1000 --out results/sepsis3_labels_mve.csv
"""

import argparse
import os
import sys
import time

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sepsis3

EARLY_ONSET_EXCLUSION_H = 4   # spec section 2: onset at/before ICU hour 4


def summarize(df, root, elapsed):
    """Build the markdown report (ASCII only: Windows console is cp1252)."""
    n = len(df)
    septic = df[df["sepsis3"] == 1]
    onset = septic["t_sepsis_hour"].to_numpy(dtype=float)
    early = int((onset <= EARLY_ONSET_EXCLUSION_H).sum())
    mc = df["t_sepsis_hour_mimiccode"].to_numpy(dtype=float)
    n_mc = int(np.isfinite(mc).sum())
    both = np.isfinite(df["t_sepsis_hour"].to_numpy(dtype=float)) & np.isfinite(mc)
    delta = (mc[both] - df["t_sepsis_hour"].to_numpy(dtype=float)[both])

    lines = [
        "# Sepsis-3 label build report",
        "",
        "Source: `%s`" % root,
        "Built in %.1fs by `scripts/build_sepsis3_labels.py`." % elapsed,
        "",
        "## Cohort and labels",
        "",
        "| Quantity | Value |",
        "|---|---|",
        "| Qualifying ICU stays (age >= %d, LOS >= %dh) | %d |"
        % (sepsis3.MIN_AGE, sepsis3.MIN_LOS_HOURS, n),
        "| Stays with a suspicion-of-infection pair | %d (%.1f%%) |"
        % (int(df["first_suspicion_hour"].notna().sum()),
           100.0 * df["first_suspicion_hour"].notna().mean()),
        "| Septic stays (Challenge rule) | %d (%.1f%%) |"
        % (len(septic), 100.0 * len(septic) / max(n, 1)),
        "| ... onset at or before ICU hour %d (excluded by spec section 2) | %d |"
        % (EARLY_ONSET_EXCLUSION_H, early),
        "| ... usable septic stays after that exclusion | %d (%.1f%%) |"
        % (len(septic) - early, 100.0 * (len(septic) - early) / max(n, 1)),
        "| Septic stays under the mimic-code variant | %d (%.1f%%) |"
        % (n_mc, 100.0 * n_mc / max(n, 1)),
        "",
    ]

    if len(onset):
        qs = np.percentile(onset, [0, 25, 50, 75, 100])
        lines += [
            "## Onset time (hours from ICU intime, unshifted)",
            "",
            "min %.1f | p25 %.1f | median %.1f | p75 %.1f | max %.1f"
            % tuple(qs),
            "",
            "SOFA at onset: median %.1f (rise of median %.1f points vs the "
            "preceding 24h minimum)."
            % (float(septic["sofa_at_onset"].median()),
               float(septic["sofa_delta_at_onset"].median())),
            "",
            "Onset driven by suspicion vs by the SOFA rise: %d vs %d stays."
            % (int((septic["t_suspicion_hour"] <= septic["t_sofa_hour"]).sum()),
               int((septic["t_suspicion_hour"] > septic["t_sofa_hour"]).sum())),
            "",
        ]
    if both.sum():
        lines += [
            "## Challenge rule vs mimic-code rule",
            "",
            "Both rules fire on %d stays; the mimic-code onset is a median "
            "%.1f h later (p25 %.1f, p75 %.1f). Agreement within 1h: %.0f%%."
            % (int(both.sum()), float(np.median(delta)),
               float(np.percentile(delta, 25)), float(np.percentile(delta, 75)),
               100.0 * float((np.abs(delta) <= 1).mean())),
            "",
        ]

    lines += ["## Deviations from mimic-code", ""]
    lines += ["%d. %s" % (i + 1, d) for i, d in enumerate(sepsis3.DEVIATIONS)]
    lines += [""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True, help="labels CSV path")
    ap.add_argument("--report", default=None,
                    help="markdown report path (default: <out>.report.md)")
    ap.add_argument("--limit-stays", type=int, default=None)
    ap.add_argument("--memory-limit", default=None,
                    help="DuckDB memory limit, e.g. 24GB")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    con = duckdb.connect()
    if args.memory_limit:
        con.execute("SET memory_limit='%s'" % args.memory_limit)
    if args.threads:
        con.execute("SET threads=%d" % args.threads)

    t0 = time.time()
    print("Building Sepsis-3 labels from %s" % args.data_root)
    df = sepsis3.run(con, args.data_root, limit_stays=args.limit_stays)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    report_path = args.report or (os.path.splitext(args.out)[0] + "_report.md")
    report = summarize(df, args.data_root, elapsed)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print()
    print(report)
    print("Saved labels -> %s" % args.out)
    print("Saved report -> %s" % report_path)


if __name__ == "__main__":
    main()
