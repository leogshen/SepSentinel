#!/usr/bin/env python
"""Minimum viable extraction run (DATA_ACCESS_SPEC.md section 11, steps 4-5).

Takes an episode pickle from scripts/extract_mimic.py and runs the existing
pipeline end to end -- grouped subject-level split, Strategy B preprocessing,
SepsisTransformer, corrected patient-level evaluation -- then checks the
section-11 acceptance criteria and writes a markdown report.

Frozen conventions (experiment2_imputation.py): split seed 42, training seeds
{42,123,456}, epochs 50, batch 32, lr 1e-3, patience 7.

The operating point reported is the section-10 primary one: the threshold set
for 70% PATIENT recall (not timestep recall), which is what the PhysioNet
baseline numbers are quoted at.

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
    collect_patient_predictions, compute_timestep_metrics,
    compute_early_warning_metrics, build_raw_map,
)

CONFIG_I_FEATURES = EXPERIMENTS["I"]["features"]
PRIMARY_RECALL = 0.70          # the deployable operating point (section 10)

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


def threshold_at_patient_recall(patient_results, target_recall):
    """Strictest threshold whose PATIENT recall still reaches target_recall."""
    best = None
    for t in np.arange(0.005, 0.995, 0.005):
        ew = compute_early_warning_metrics(patient_results, float(t))
        if ew["patient_recall"] >= target_recall:
            best = (float(t), ew)          # keep going: higher t is stricter
    if best is None:                       # unreachable at any threshold
        best = (0.005, compute_early_warning_metrics(patient_results, 0.005))
    return best


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
        thr, ew = threshold_at_patient_recall(results, PRIMARY_RECALL)
        ts = compute_timestep_metrics(y_true, y_prob, thr)
        per_seed.append({"seed": seed, "auroc": auroc, "auprc": auprc,
                         "threshold": thr, "timestep": jsonable(ts),
                         "early_warning": jsonable(ew)})
        print("  seed %d: AUROC %.3f AUPRC %.3f | at %.0f%% patient recall: "
              "precision %.3f, %.2f alerts/patient-day, median lead %s h"
              % (seed, auroc, auprc, 100 * ew["patient_recall"],
                 ts.get("precision", float("nan")),
                 ew["alerts_per_patient_day"], fmt(ew["median_lead_time_h"])))

    auroc_m, auroc_s = mean_sd(r["auroc"] for r in per_seed)
    auprc_m, auprc_s = mean_sd(r["auprc"] for r in per_seed)
    prec_m, prec_s = mean_sd(r["timestep"].get("precision", float("nan"))
                             for r in per_seed)
    alerts_m, alerts_s = mean_sd(r["early_warning"]["alerts_per_patient_day"]
                                 for r in per_seed)
    leads = [r["early_warning"]["median_lead_time_h"] for r in per_seed
             if r["early_warning"]["median_lead_time_h"] is not None]
    lead_m, lead_s = mean_sd(leads) if leads else (float("nan"), float("nan"))
    recall_m = mean_sd(r["early_warning"]["patient_recall"] for r in per_seed)[0]
    capture = {h: mean_sd(r["early_warning"]["capture_rate_by_lead_hour"][str(h)]
                          for r in per_seed)[0] for h in (3, 6, 12)}
    print("\nTest AUROC %.3f +/- %.3f | AUPRC %.3f +/- %.3f"
          % (auroc_m, auroc_s, auprc_m, auprc_s))

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
              "| Metric | MIMIC-IV MVE | PhysioNet Config I |", "|---|---|---|",
              "| Test AUROC | %.3f +/- %.3f | %s |"
              % (auroc_m, auroc_s, REF["auroc"]),
              "| Test AUPRC | %.3f +/- %.3f | %s |"
              % (auprc_m, auprc_s, REF["auprc"]), "",
              "At the primary operating point (threshold set for %.0f%% patient "
              "recall; achieved %.0f%%):" % (100 * PRIMARY_RECALL, 100 * recall_m),
              "",
              "| Metric | MIMIC-IV MVE | PhysioNet Config I |", "|---|---|---|",
              "| Timestep precision | %.3f +/- %.3f | %s |"
              % (prec_m, prec_s, REF["precision"]),
              "| False alerts / patient-day | %.2f +/- %.2f | %s |"
              % (alerts_m, alerts_s, REF["alerts"]),
              "| Median lead time (h) | %.1f +/- %.1f | %s |"
              % (lead_m, lead_s, REF["lead"])]
    for h in (3, 6, 12):
        lines.append("| Capture >=%dh before onset | %.2f | - |" % (h, capture[h]))

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
