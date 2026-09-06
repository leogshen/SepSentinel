#!/usr/bin/env python
"""Minimum viable extraction run (DATA_ACCESS_SPEC.md section 11, steps 4-5).

Takes an episode pickle from scripts/extract_mimic.py and runs the existing
pipeline end to end -- grouped subject-level split, Strategy B preprocessing,
SepsisTransformer, corrected patient-level evaluation -- then checks the
section-11 acceptance criteria and writes a markdown report.

Frozen conventions (experiment2_imputation.py): split seed 42, training seeds
{42,123,456}, epochs 50, batch 32, lr 1e-3, patience 7.

No recall target is imposed. Results are reported threshold-free (AUROC,
AUPRC) and at equal ALERT BURDEN -- the constraint a unit actually imposes.
scripts/operating_curves.py emits the full curves.

Usage:
    python scripts/run_mve.py --episodes results/mimic_mve_episodes.pkl \
        --out-dir results/mve_mimiciv31
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
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer
from experiment2_imputation import (
    FEATURES as ALL_FEATURES, SPLIT_SEED, EPOCHS, BATCH_SIZE, LR, PATIENCE,
)
from experiment3_feature_ablation import AblationPreprocessor, EXPERIMENTS
from experiment5_recall_study import (
    collect_patient_predictions, build_raw_map,
)
from scripts.operating_curves import curve_for, at_burden, BURDEN_POINTS

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]

# Section 11 acceptance criteria.
PHYSIONET_PATIENT_PREV = 8.8   # %
PHYSIONET_TIMESTEP_PREV = 2.2  # %
AUROC_SANITY = (0.70, 0.85)

# PhysioNet Config I reference numbers (corrected, post-2026-08-19 bugfix).
REF = {"auroc": "0.814 +/- 0.004", "auprc": "0.144", "precision": "0.093",
       "alerts": "1.7", "lead": "23.5"}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_sd(values):
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def fmt(v):
    return "%.3f" % v if isinstance(v, float) else str(v)


def jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--run-name", default="MVE run",
                    help="title for the report (the same runner is used for "
                         "the MVE and the full-cohort run)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    assert episodes[0]["features"] == ALL_FEATURES, (
        "episode feature layout does not match the canonical list; "
        "AblationPreprocessor indexes into it positionally")

    n_pos = sum(e["label"] for e in episodes)
    all_labels = np.concatenate([e["labels"] for e in episodes])
    patient_prev = 100.0 * n_pos / len(episodes)
    timestep_prev = 100.0 * float(all_labels.mean())
    print("Episodes %d (%d septic, %.1f%% patient / %.1f%% timestep prevalence)"
          % (len(episodes), n_pos, patient_prev, timestep_prev))

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    for k in ("train", "val", "test"):
        print("  %-5s %4d episodes, %4d subjects, %3d septic"
              % (k, len(splits[k]), len(set(e["subject_id"] for e in splits[k])),
                 sum(e["label"] for e in splits[k])))
    groups = {k: set(e["subject_id"] for e in splits[k]) for k in splits}
    assert not (groups["train"] & groups["val"]), "subject leak train/val"
    assert not (groups["train"] & groups["test"]), "subject leak train/test"
    assert not (groups["val"] & groups["test"]), "subject leak val/test"

    pre = AblationPreprocessor(ALL_FEATURES, CONFIG_I_FEATURES)
    data = {"train": pre.fit_transform(splits["train"])}
    data["val"] = pre.transform(splits["val"])
    data["test"] = pre.transform(splits["test"])

    # pos_weight recomputed from THIS dataset's train split (leakage rule 5).
    y_train = np.concatenate([d["labels"] for d in data["train"]])
    pos_weight = float((1 - y_train.mean()) / max(y_train.mean(), 1e-9))
    print("Strategy B: %d channels; train pos_weight %.2f"
          % (pre.n_channels, pos_weight))

    raw_map = build_raw_map(splits)
    seeds = [int(x) for x in args.seeds.split(",")]
    per_seed = []
    for seed in seeds:
        set_seed(seed)
        model = SepsisTransformer(input_dim=pre.n_channels)
        trainer = Trainer(
            model, device=args.device, pos_weight=pos_weight,
            checkpoint_dir=os.path.join(args.out_dir, "checkpoints_seed%d" % seed))
        trainer.fit(data["train"], data["val"], epochs=args.epochs,
                    batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)

        results = collect_patient_predictions(model, data["test"], raw_map,
                                              args.device)
        y_true = np.concatenate([p["labels"] for p in results])
        y_prob = np.concatenate([p["probs"] for p in results])
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
        curve = curve_for(results)
        per_seed.append({"seed": seed, "auroc": auroc, "auprc": auprc,
                         "curve": curve})
        print("  seed %d: AUROC %.3f AUPRC %.3f" % (seed, auroc, auprc))
        for b in BURDEN_POINTS:
            c = at_burden(curve, b)
            if c is not None:
                print("      at <=%.1f alerts/pt-day: recall %.2f, "
                      "median lead %s h, capture>=6h %.2f"
                      % (b, c["patient_recall"], fmt(c["median_lead_time_h"]),
                         c["capture_6h"]))

    auroc_m, auroc_s = mean_sd(r["auroc"] for r in per_seed)
    auprc_m, auprc_s = mean_sd(r["auprc"] for r in per_seed)
    print("\nTest AUROC %.3f +/- %.3f | AUPRC %.3f +/- %.3f"
          % (auroc_m, auroc_s, auprc_m, auprc_s))

    def at_budget(budget, key):
        vals = [at_burden(r["curve"], budget) for r in per_seed]
        vals = [v[key] for v in vals if v is not None and v[key] is not None]
        return mean_sd(vals) if vals else (float("nan"), float("nan"))

    checks = [
        ("patient prevalence within 2x of PhysioNet %.1f%%" % PHYSIONET_PATIENT_PREV,
         0.5 * PHYSIONET_PATIENT_PREV <= patient_prev <= 2 * PHYSIONET_PATIENT_PREV,
         "%.1f%%" % patient_prev),
        ("timestep prevalence within 2x of PhysioNet %.1f%%" % PHYSIONET_TIMESTEP_PREV,
         0.5 * PHYSIONET_TIMESTEP_PREV <= timestep_prev <= 2 * PHYSIONET_TIMESTEP_PREV,
         "%.1f%%" % timestep_prev),
        ("mean test AUROC in [%.2f, %.2f]" % AUROC_SANITY,
         AUROC_SANITY[0] <= auroc_m <= AUROC_SANITY[1],
         "%.3f +/- %.3f" % (auroc_m, auroc_s)),
        ("no subject leakage across splits", True, "verified"),
        ("pipeline runs end to end", True, "yes"),
    ]

    lines = ["# %s report (DATA_ACCESS_SPEC sections 10-11)" % args.run_name, "",
             "Episodes: `%s`" % args.episodes,
             "Seeds %s, device %s, %.1f min."
             % (args.seeds, args.device, (time.time() - t0) / 60.0), "",
             "## Acceptance criteria", "",
             "| Criterion | Value | Pass |", "|---|---|---|"]
    for name, ok, val in checks:
        lines.append("| %s | %s | %s |" % (name, val, "PASS" if ok else "FAIL"))

    lines += ["", "## Metrics (%d seeds, mean +/- sd)" % len(seeds), "",
              "| Metric | This run | PhysioNet Config I |", "|---|---|---|",
              "| Test AUROC | %.3f +/- %.3f | %s |"
              % (auroc_m, auroc_s, REF["auroc"]),
              "| Test AUPRC | %.3f +/- %.3f | %s |"
              % (auprc_m, auprc_s, REF["auprc"]), "",
              "No recall target is imposed. Each row below is the best patient "
              "recall reachable inside the stated false-alert budget.", "",
              "| Alerts/patient-day budget | Patient recall | Median lead (h) | "
              "Capture >=6h | Capture >=12h |", "|---|---|---|---|---|"]
    for b in BURDEN_POINTS:
        lines.append("| <= %.1f | %.2f +/- %.2f | %.1f +/- %.1f | %.2f | %.2f |"
                     % (b, at_budget(b, "patient_recall")[0],
                        at_budget(b, "patient_recall")[1],
                        at_budget(b, "median_lead_time_h")[0],
                        at_budget(b, "median_lead_time_h")[1],
                        at_budget(b, "capture_6h")[0],
                        at_budget(b, "capture_12h")[0]))

    lines += ["", "## Split", "",
              "| Split | Episodes | Subjects | Septic |", "|---|---|---|---|"]
    for k in ("train", "val", "test"):
        lines.append("| %s | %d | %d | %d |"
                     % (k, len(splits[k]),
                        len(set(e["subject_id"] for e in splits[k])),
                        sum(e["label"] for e in splits[k])))
    lines.append("")

    report = "\n".join(lines)
    with open(os.path.join(args.out_dir, "MVE_REPORT.md"), "w",
              encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"seeds": per_seed, "auroc_mean": auroc_m, "auroc_sd": auroc_s,
                   "auprc_mean": auprc_m, "auprc_sd": auprc_s,
                   "patient_prevalence_pct": patient_prev,
                   "timestep_prevalence_pct": timestep_prev,
                   "acceptance": {n: bool(ok) for n, ok, _ in checks}},
                  fh, indent=2)
    print()
    print(report)
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
