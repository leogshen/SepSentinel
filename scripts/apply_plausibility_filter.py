#!/usr/bin/env python
"""Blank out-of-range cells in an existing extraction pickle.

Purpose: measure what the missing extraction-time plausibility filter cost,
without paying for a re-extraction. NOT a substitute for filtering at
extraction -- an hour bin where an outlier was already blended into a median
cannot be recovered here -- so a null result from this script is a LOWER bound
on the harm, not proof of none.

Usage:
    python scripts/apply_plausibility_filter.py in.pkl out.pkl
"""

import pickle
import sys

import numpy as np

PLAUS = {"heart_rate": (20, 250), "spo2": (50, 100),
         "respiratory_rate": (2, 60), "temperature": (30, 43),
         "map": (20, 200), "sbp": (30, 300), "gcs": (3, 15),
         "fio2": (21, 100), "urine_output": (0, 2500),
         "lactate": (0, 30), "ph": (6.5, 7.8), "creatinine": (0, 25),
         "wbc": (0, 100), "platelets": (0, 1200), "bilirubin": (0, 60),
         "pao2": (30, 700), "bun": (0, 200), "glucose": (20, 1000)}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as fh:
        eps = pickle.load(fh)
    feats = eps[0]["features"]
    n_fixed = 0
    for e in eps:
        sig = e["signals"]
        for f, (lo, hi) in PLAUS.items():
            if f not in feats:
                continue
            j = feats.index(f)
            v = sig[:, j]
            bad = ~np.isnan(v) & ((v < lo) | (v > hi))
            if bad.any():
                n_fixed += int(bad.sum())
                v[bad] = np.nan
                sig[:, j] = v
    print("blanked %d out-of-range cells" % n_fixed)
    with open(dst, "wb") as fh:
        pickle.dump(eps, fh)


if __name__ == "__main__":
    main()
