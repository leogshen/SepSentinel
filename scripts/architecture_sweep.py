#!/usr/bin/env python
"""Two untried levers: sequence architecture, and trajectory features.

This project has never run a GRU, LSTM or TCN on MIMIC-IV -- those were only
ever run on PhysioNet with the old 4-feature stage-1 setup -- and YAIB's best
published MIMIC-IV sepsis model is a **GRU** at 0.836, not a Transformer.
Separately, `sepentinel/data/trajectory.py` has been in the repository since
before this work started and has never been evaluated: per feature it adds
diff_1h, a causal rolling mean_6h, and dev_6h (deviation from the patient's
own recent baseline), which is exactly the "patient-relative baseline" the
stronger papers use.

This sweeps model x feature-set on identical episodes, split and metrics, and
reports recall-at-burden first, with both precisions, per the project's
stated priority.

Trajectory features are computed AFTER Strategy B imputation and BEFORE
normalization, on the value channels only -- deriving a rolling mean of a
binary observation mask would be meaningless.

Usage:
    python scripts/architecture_sweep.py --episodes results/...pkl \
        --out-dir results/architecture_sweep --models gru,transformer,tcn \
        --feature-modes strategy_b,trajectory
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED, BATCH_SIZE, LR, PATIENCE
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.data.trajectory import compute_trajectory
from sepsentinel.model_b.gru import SepsisGRU
from sepsentinel.model_b.tcn import SepsisTCN
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import (
    build_preprocessor, curve_for, at_burden, BURDEN_POINTS,
)
from experiment5_recall_study import collect_patient_predictions, build_raw_map

BUILDERS = {
    "transformer": lambda d: SepsisTransformer(input_dim=d),
    "gru": lambda d: SepsisGRU(input_dim=d),
    "tcn": lambda d: SepsisTCN(input_dim=d),
}


def add_trajectory(data, n_raw, window=6):
    """Append diff/rolling-mean/deviation channels for the VALUE channels.

    The preprocessed layout is [values | masks | deltas]; trajectory features
    are meaningful only on the values, so masks and deltas pass through
    untouched. Normalisation of the new channels uses TRAIN statistics only,
    computed by the caller and passed in.
    """
    out = []
    for d in data:
        sig = d["signals"]
        traj = compute_trajectory(sig[:, :n_raw], window=window)
        out.append(dict(d, signals=np.concatenate([sig, traj], axis=1)))
    return out


def normalise_new(train, others, start_col):
    """Z-score the appended channels with train-only mean/std."""
    stack = np.concatenate([d["signals"][:, start_col:] for d in train])
    mu = np.nanmean(stack, axis=0)
    sd = np.nanstd(stack, axis=0)
    sd[sd < 1e-6] = 1.0
    for group in [train] + list(others):
        for d in group:
            d["signals"][:, start_col:] = (
                (d["signals"][:, start_col:] - mu) / sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="gru,transformer")
    ap.add_argument("--feature-modes", default="strategy_b,trajectory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = build_raw_map(splits)
    pre = build_preprocessor(episodes, "all")
    base = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    n_raw, n_ch = pre.n_raw, pre.n_channels
    y_train = np.concatenate([d["labels"] for d in base["train"]])
    pos_weight = float((1 - y_train.mean()) / max(y_train.mean(), 1e-9))
    print("%d raw features -> %d Strategy B channels; pos_weight %.1f"
          % (n_raw, n_ch, pos_weight))

    results = {}
    for fmode in args.feature_modes.split(","):
        fmode = fmode.strip()
        if fmode == "strategy_b":
            data = base
        elif fmode == "trajectory":
            data = {k: add_trajectory(v, n_raw) for k, v in base.items()}
            normalise_new(data["train"], [data["val"], data["test"]], n_ch)
        else:
            raise SystemExit("unknown feature mode: %s" % fmode)
        in_dim = data["train"][0]["signals"].shape[1]
        print("\n=== features: %s (%d channels) ===" % (fmode, in_dim))

        for mname in args.models.split(","):
            mname = mname.strip()
            tag = "%s__%s" % (mname, fmode)
            t0 = time.time()
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model = BUILDERS[mname](in_dim)
            trainer = Trainer(model, device=args.device, pos_weight=pos_weight,
                              checkpoint_dir=os.path.join(args.out_dir, tag))
            trainer.fit(data["train"], data["val"], epochs=args.epochs,
                        batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)

            res = collect_patient_predictions(model, data["test"], raw_map,
                                              args.device)
            y = np.concatenate([p["labels"] for p in res])
            p = np.concatenate([p["probs"] for p in res])
            curve = curve_for(res)
            c = at_burden(curve, 1.0)
            results[tag] = {
                "model": mname, "features": fmode, "in_dim": in_dim,
                "auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "at_1_per_day": c, "minutes": (time.time() - t0) / 60.0}
            print("%-24s AUROC %.4f AUPRC %.4f | at <=1.0/day: recall %.2f, "
                  "ts-prec %.3f, pt-prec %.3f, capture>=6h %.3f, lead %s h  "
                  "(%.1f min)"
                  % (tag, results[tag]["auroc"], results[tag]["auprc"],
                     c["patient_recall"], c["timestep_precision"],
                     c["patient_precision"], c["capture_6h"],
                     "%.1f" % c["median_lead_time_h"]
                     if c["median_lead_time_h"] is not None else "n/a",
                     results[tag]["minutes"]))
            with open(os.path.join(args.out_dir, "sweep.json"), "w") as fh:
                json.dump(results, fh, indent=2)

    print("\n%-24s %7s %7s %7s %8s %8s %7s" %
          ("run", "AUROC", "AUPRC", "recall", "ts-prec", "pt-prec", "cap6h"))
    for tag, r in sorted(results.items(),
                         key=lambda kv: -kv[1]["at_1_per_day"]["patient_recall"]):
        c = r["at_1_per_day"]
        print("%-24s %7.4f %7.4f %7.2f %8.3f %8.3f %7.3f"
              % (tag, r["auroc"], r["auprc"], c["patient_recall"],
                 c["timestep_precision"], c["patient_precision"],
                 c["capture_6h"]))
    print("\nSaved -> %s" % os.path.join(args.out_dir, "sweep.json"))


if __name__ == "__main__":
    main()
