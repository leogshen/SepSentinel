#!/usr/bin/env python
"""Project a MIMIC episode pickle onto the pooled 17-feature layout.

The pooled encoder must see identical channels from both sources, in the same
order. MIMIC's extended set is 18 features including GCS; SICdb has no usable
GCS. Rather than re-extracting MIMIC (~15 min), this slices the existing
pickle's signal matrix down to sicdb.POOLED_FEATURES and rewrites `features`
and `vitals` to match.

Why GCS is dropped rather than carried as missing: a channel that is 100%
absent in one site and largely present in the other is a perfect hospital
label. The pooled encoder would be able to read the site off the observation
mask alone, which is exactly the shortcut the multi-source arm is supposed to
be testing against.

It also reports the measurement-density gap between the two sources, which is
the other harmonisation risk and a number the pretraining arms need on record.

Usage:
    python scripts/make_pooled.py \
        --mimic results/mimic31_full_ext_prodrome.pkl \
        --sicdb results/sicdb_episodes.pkl \
        --out results/mimic31_pooled17.pkl
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sepsentinel.data import sicdb


def density(episodes, features):
    """Per-channel observed fraction, before any imputation."""
    tot = np.zeros(len(features))
    obs = np.zeros(len(features))
    for e in episodes:
        s = e["signals"]
        tot += s.shape[0]
        obs += np.sum(~np.isnan(s), axis=0)
    return obs / np.maximum(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--sicdb", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    feats = list(sicdb.POOLED_FEATURES)
    vitals = list(sicdb.POOLED_VITALS)

    with open(args.mimic, "rb") as fh:
        mim = pickle.load(fh)
    src = list(mim[0]["features"])
    missing = [f for f in feats if f not in src]
    if missing:
        raise SystemExit("MIMIC pickle lacks pooled features: %s" % missing)
    cols = [src.index(f) for f in feats]
    print("MIMIC %d features -> pooled %d (dropping %s)"
          % (len(src), len(feats), [f for f in src if f not in feats]))

    out = []
    for e in mim:
        out.append(dict(e, signals=e["signals"][:, cols].copy(),
                        features=list(feats), vitals=list(vitals)))
    with open(args.out, "wb") as fh:
        pickle.dump(out, fh)

    with open(args.sicdb, "rb") as fh:
        sic = pickle.load(fh)
    if list(sic[0]["features"]) != feats:
        raise SystemExit("SICdb features do not match the pooled layout")

    dm = density(out, feats)
    ds = density(sic, feats)
    report = {
        "pooled_features": feats, "vitals": vitals,
        "dropped_from_mimic": [f for f in src if f not in feats],
        "mimic_episodes": len(out), "sicdb_episodes": len(sic),
        "mimic_septic": int(sum(e["label"] for e in out)),
        "sicdb_septic_rung1": int(sum(e["label"] for e in sic)),
        "observed_fraction": {
            f: {"mimic": float(a), "sicdb": float(b),
                "ratio_sicdb_over_mimic": float(b / a) if a > 0 else None}
            for f, a, b in zip(feats, dm, ds)},
    }
    with open(os.path.splitext(args.out)[0] + "_density.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    print("%-20s %8s %8s %7s" % ("channel", "MIMIC", "SICdb", "ratio"))
    for f, a, b in zip(feats, dm, ds):
        print("%-20s %8.3f %8.3f %7s"
              % (f, a, b, "%.1fx" % (b / a) if a > 0 else "n/a"))
    print()
    print("MIMIC episodes %d (%d septic) | SICdb episodes %d (%d rung-1)"
          % (len(out), report["mimic_septic"], len(sic),
             report["sicdb_septic_rung1"]))
    print("Saved -> %s" % args.out)


if __name__ == "__main__":
    main()
