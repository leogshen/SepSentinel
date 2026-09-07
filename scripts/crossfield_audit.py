#!/usr/bin/env python
"""Cross-field plausibility audit.

Per-channel range filters catch a value that is impossible on its own
(11,337 bpm). They cannot catch a value that is impossible only in context:
a systolic pressure BELOW the mean arterial pressure in the same hour is
physiologically impossible, but 60 and 90 are individually plausible numbers.

This measures how much of that second class exists in the extracted grid, so
"is range filtering enough?" has an answer rather than an assumption.

Usage:
    python scripts/crossfield_audit.py --episodes results/...pkl
"""

import argparse
import pickle
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    args = ap.parse_args()

    with open(args.episodes, "rb") as fh:
        eps = pickle.load(fh)
    feats = eps[0]["features"]
    idx = {f: feats.index(f) for f in feats}
    sig = np.concatenate([e["signals"] for e in eps], axis=0)
    n_hours = sig.shape[0]

    def col(f):
        return sig[:, idx[f]] if f in idx else None

    checks = []

    sbp, mp = col("sbp"), col("map")
    if sbp is not None and mp is not None:
        both = ~np.isnan(sbp) & ~np.isnan(mp)
        checks.append(("SBP < MAP (impossible ordering)",
                       int((both & (sbp < mp)).sum()), int(both.sum())))
        checks.append(("MAP > 0.95*SBP (implausibly narrow pulse pressure)",
                       int((both & (mp > 0.95 * sbp)).sum()), int(both.sum())))

    spo2, pao2 = col("spo2"), col("pao2")
    if spo2 is not None and pao2 is not None:
        both = ~np.isnan(spo2) & ~np.isnan(pao2)
        # SpO2 <= 90 with a high PaO2 is internally inconsistent
        checks.append(("SpO2 <=90% with PaO2 >=200 mmHg (inconsistent pair)",
                       int((both & (spo2 <= 90) & (pao2 >= 200)).sum()),
                       int(both.sum())))

    hr, rr = col("heart_rate"), col("respiratory_rate")
    if hr is not None and rr is not None:
        both = ~np.isnan(hr) & ~np.isnan(rr)
        checks.append(("RR > HR (possible but rare; flag for review)",
                       int((both & (rr > hr)).sum()), int(both.sum())))

    uo = col("urine_output")
    if uo is not None:
        obs = ~np.isnan(uo)
        checks.append(("urine output >1000 mL in one hour",
                       int((obs & (uo > 1000)).sum()), int(obs.sum())))

    gcs = col("gcs")
    if gcs is not None:
        obs = ~np.isnan(gcs)
        checks.append(("GCS outside 3-15",
                       int((obs & ((gcs < 3) | (gcs > 15))).sum()),
                       int(obs.sum())))

    fio2 = col("fio2")
    if fio2 is not None:
        obs = ~np.isnan(fio2)
        checks.append(("FiO2 < 21% (below room air)",
                       int((obs & (fio2 < 21)).sum()), int(obs.sum())))

    # Physiologically impossible constancy: a vital identical for >24h running
    flat = 0
    checked = 0
    j = idx.get("heart_rate")
    if j is not None:
        for e in eps:
            v = e["signals"][:, j]
            m = ~np.isnan(v)
            if m.sum() < 24:
                continue
            checked += 1
            vals = v[m]
            # longest run of identical values
            run = best = 1
            for a, b in zip(vals[:-1], vals[1:]):
                run = run + 1 if a == b else 1
                best = max(best, run)
            if best >= 24:
                flat += 1

    print("Cross-field audit of %s" % args.episodes)
    print("%d episodes, %d patient-hours\n" % (len(eps), n_hours))
    print("%-58s %10s %12s %8s" % ("check", "violations", "evaluable", "rate"))
    for name, bad, tot in checks:
        print("%-58s %10d %12d %7.4f%%"
              % (name, bad, tot, 100.0 * bad / max(tot, 1)))
    print("\n%-58s %10d %12d %7.4f%%"
          % ("heart rate identical for >=24 consecutive readings", flat,
             checked, 100.0 * flat / max(checked, 1)))


if __name__ == "__main__":
    main()
