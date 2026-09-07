#!/usr/bin/env python
"""Does cross-field cleaning change the results?

The per-channel plausibility filter cost was measured at +0.003 to +0.007
AUROC. This asks the same question for the violations a range filter cannot
see: SBP below MAP, implausibly narrow pulse pressure, SpO2/PaO2
contradictions, hourly urine volumes above 1000 mL, and heart-rate traces
frozen for a day.

Blanks the offending cells (both members of an inconsistent pair, since
which one is wrong is unknowable) and refits the flat baselines.

Usage:
    python scripts/crossfield_impact.py --episodes in.pkl --out-dir results/x
"""

import argparse
import os
import pickle
import subprocess
import sys

import numpy as np


def clean(eps, feats):
    idx = {f: feats.index(f) for f in feats}
    blanked = 0
    for e in eps:
        s = e["signals"]

        if "sbp" in idx and "map" in idx:
            sbp, mp = s[:, idx["sbp"]], s[:, idx["map"]]
            bad = ~np.isnan(sbp) & ~np.isnan(mp) & ((sbp < mp) | (mp > 0.95 * sbp))
            # both members go: which of the pair is wrong is unknowable
            blanked += 2 * int(bad.sum())
            sbp[bad] = np.nan
            mp[bad] = np.nan

        if "spo2" in idx and "pao2" in idx:
            spo2, pao2 = s[:, idx["spo2"]], s[:, idx["pao2"]]
            bad = (~np.isnan(spo2) & ~np.isnan(pao2)
                   & (spo2 <= 90) & (pao2 >= 200))
            blanked += 2 * int(bad.sum())
            spo2[bad] = np.nan
            pao2[bad] = np.nan

        if "urine_output" in idx:
            uo = s[:, idx["urine_output"]]
            bad = ~np.isnan(uo) & (uo > 1000)
            blanked += int(bad.sum())
            uo[bad] = np.nan

        if "heart_rate" in idx:
            hr = s[:, idx["heart_rate"]]
            m = ~np.isnan(hr)
            if m.sum() >= 24:
                vals, pos = hr[m], np.where(m)[0]
                run_start = 0
                for i in range(1, len(vals) + 1):
                    if i == len(vals) or vals[i] != vals[run_start]:
                        if i - run_start >= 24:      # frozen trace
                            hr[pos[run_start:i]] = np.nan
                            blanked += i - run_start
                        run_start = i
    return blanked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    n = clean(eps, list(eps[0]["features"]))
    print("blanked %d cells by cross-field rules" % n)

    tmp = os.path.join(args.out_dir, "_crossfield_cleaned.pkl")
    with open(tmp, "wb") as fh:
        pickle.dump(eps, fh)

    subprocess.run([sys.executable, "scripts/run_baselines.py",
                    "--episodes", tmp, "--out-dir", args.out_dir,
                    "--models", "logreg,xgboost"], check=True)
    os.remove(tmp)


if __name__ == "__main__":
    main()
