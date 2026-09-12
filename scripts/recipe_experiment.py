#!/usr/bin/env python
"""Three recipe changes from TECHNIQUES.md, tested factorially over seeds.

  1. pos_weight ON vs OFF. We train with pos_weight ~37. HiRID's Table 5
     reports balanced class weighting HURTING every binary task by up to
     4.2 AUPRC, their released configs set LOSS_WEIGHT = None, and Yeche et
     al. show tuned weighted-CE and focal loss reduce exactly to plain CE.
  2. Temporal Label Smoothing (Yeche et al., ICML 2023). Replace the hard
     label with a smooth function of time-to-onset, so an hour 11 h out is
     not asked to look identical to an hour 1 h out. Reported +1.5 AUPRC and
     +9.7 event recall on an analogous task.
  3. Seeds. The architecture sweep spanned 0.010 AUROC across six runs while
     HiRID reports 0.3-0.8 AUPRC of seed-to-seed std over 10 seeds -- so that
     sweep could not distinguish architecture from noise. Everything here is
     run over multiple seeds and reported as mean +/- std.

TLS is applied to the TRAINING split only. Validation and test keep hard
labels, so early stopping and every reported metric stay on the real target
-- soft labels would make AUROC undefined.

Exponential smoothing, from the paper, with u = hours until onset and
h = W/2:

    d = -(1/g) * ln(1 - exp(-2gh))
    A = -exp(-g(2h - d))
    q(u) = exp(-g(u - d)) + A     for 0 < u <= 2h

which gives q(0) = 1 at onset and q(2h) = 0 at the window edge.

Usage:
    python scripts/recipe_experiment.py --episodes results/...pkl \
        --out-dir results/recipe_experiment --seeds 42,123,456
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
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import (
    build_preprocessor, curve_for, at_burden,
)
from experiment5_recall_study import collect_patient_predictions, build_raw_map


def tls_targets(labels, window_h, gamma):
    """Soft targets ramping to 1 at onset, 0 at the far edge of the window.

    Our positives already occupy [t_sepsis - W, t_sepsis), so the last
    positive hour IS the onset hour and position within the run of ones
    gives hours-until-onset directly -- no need to carry t_sepsis around.
    """
    out = np.asarray(labels, dtype=np.float32).copy()
    pos = np.flatnonzero(out > 0)
    if pos.size == 0:
        return out
    h = window_h / 2.0
    d = -(1.0 / gamma) * np.log(1.0 - np.exp(-2.0 * gamma * h))
    A = -np.exp(-gamma * (2.0 * h - d))
    last = pos[-1]                      # the hour immediately before onset
    u = (last - pos).astype(np.float32)  # 0 at onset, growing backwards
    q = np.exp(-gamma * (u - d)) + A
    out[pos] = np.clip(q, 0.0, 1.0)
    return out


def soften(data, window_h, gamma):
    return [dict(d, labels=tls_targets(d["labels"], window_h, gamma))
            for d in data]


def score(model, data, raw_map, device):
    """Test-set metrics for an already-fitted model."""
    res = collect_patient_predictions(model, data["test"], raw_map, device)
    y = np.concatenate([p["labels"] for p in res])
    p = np.concatenate([p["probs"] for p in res])
    c = at_burden(curve_for(res), 1.0)
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "recall": c["patient_recall"],
            "ts_precision": c["timestep_precision"],
            "pt_precision": c["patient_precision"],
            "capture_6h": c["capture_6h"],
            "lead": c["median_lead_time_h"]}


def run_one(data, raw_map, in_dim, pos_weight, seed, device, epochs, ckpt,
            resume=False):
    """Fit one run, or score an existing checkpoint if --resume and one exists.

    The Sep-8 run trained all three tls_posw seeds and died before writing
    their arm to recipe.json, so the checkpoints on disk are complete and only
    the metrics were lost. Scoring them back is inference-only.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SepsisTransformer(input_dim=in_dim)
    saved = os.path.join(ckpt, "best_model.pt")
    if resume and os.path.exists(saved):
        model.load_state_dict(torch.load(saved, weights_only=True))
        model.to(device)
        print("    resumed from %s (no retraining)" % saved)
        return score(model, data, raw_map, device)
    trainer = Trainer(model, device=device, pos_weight=pos_weight,
                      checkpoint_dir=ckpt)
    trainer.fit(data["train"], data["val"], epochs=epochs,
                batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)
    return score(model, data, raw_map, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--window-h", type=float, default=12.0,
                    help="prodrome window the episodes were built with")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="keep arms already in recipe.json, and score any "
                         "existing per-seed checkpoint instead of retraining")
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
    in_dim = base["train"][0]["signals"].shape[1]
    y_tr = np.concatenate([d["labels"] for d in base["train"]])
    pw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    print("%d channels, positive rate %.4f, pos_weight would be %.1f"
          % (in_dim, y_tr.mean(), pw))

    soft_train = soften(base["train"], args.window_h, args.gamma)
    ex = [d for d in soft_train if d["labels"].max() > 0][0]["labels"]
    nz = ex[ex > 0]
    print("TLS example target ramp (onset last): %s"
          % np.array2string(nz[-6:], precision=3))

    seeds = [int(s) for s in args.seeds.split(",")]
    configs = {
        "baseline_posw":        (base, pw),
        "no_posweight":         (base, None),
        "tls_posw":             ({**base, "train": soft_train}, pw),
        "tls_no_posweight":     ({**base, "train": soft_train}, None),
    }

    results = {}
    json_path = os.path.join(args.out_dir, "recipe.json")
    if args.resume and os.path.exists(json_path):
        with open(json_path) as fh:
            results = json.load(fh)
        print("resuming: %d arm(s) already recorded (%s)"
              % (len(results), ", ".join(results)))

    for name, (data, pos_weight) in configs.items():
        if name in results:
            print("%-18s already complete, skipping" % name)
            continue
        runs = []
        for seed in seeds:
            t0 = time.time()
            r = run_one(data, raw_map, in_dim, pos_weight, seed, args.device,
                        args.epochs, os.path.join(args.out_dir,
                                                  "%s_seed%d" % (name, seed)),
                        resume=args.resume)
            r["seed"] = seed
            r["minutes"] = (time.time() - t0) / 60.0
            runs.append(r)
            print("  %-18s seed %3d: AUROC %.4f AUPRC %.4f recall %.2f "
                  "ts-prec %.3f pt-prec %.3f capture>=6h %.3f  (%.1f min)"
                  % (name, seed, r["auroc"], r["auprc"], r["recall"],
                     r["ts_precision"], r["pt_precision"], r["capture_6h"],
                     r["minutes"]))
        agg = {}
        for k in ("auroc", "auprc", "recall", "ts_precision", "pt_precision",
                  "capture_6h", "lead"):
            vals = [r[k] for r in runs if r[k] is not None]
            agg[k] = (float(np.mean(vals)), float(np.std(vals)))
        results[name] = {"runs": runs, "mean_sd": agg}
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)

    print("\n%-18s %14s %14s %13s %13s %13s"
          % ("config", "AUROC", "AUPRC", "recall", "ts-prec", "capture>=6h"))
    for name, r in results.items():
        a = r["mean_sd"]
        print("%-18s %7.4f+/-%.4f %7.4f+/-%.4f %6.3f+/-%.3f %6.3f+/-%.3f "
              "%6.3f+/-%.3f"
              % (name, a["auroc"][0], a["auroc"][1], a["auprc"][0],
                 a["auprc"][1], a["recall"][0], a["recall"][1],
                 a["ts_precision"][0], a["ts_precision"][1],
                 a["capture_6h"][0], a["capture_6h"][1]))
    print("\nSaved -> %s" % os.path.join(args.out_dir, "recipe.json"))


if __name__ == "__main__":
    main()
