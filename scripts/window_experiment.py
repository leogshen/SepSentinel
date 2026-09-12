#!/usr/bin/env python
"""Which prodrome window is the useful target? Choose with XGBoost, confirm
with the Transformer.

The prodrome window W defines the problem: positives are confined to
[t_sepsis - W, t_sepsis), and W has been fixed at 12 h since 2026-09-06
without ever being tuned. Every downstream claim -- lead time, capture,
alert burden -- is conditional on that arbitrary choice.

Design, and why it is cheap:

  1. Sweep W in {3, 6, 12, 24} with flat XGBoost, which trains in ~1 min.
     Select on the VALIDATION patients, never on test.
  2. Train the Transformer only at the selected W*.

That ordering is the whole point: it answers "does attention add value under
the best problem definition" without spending GPU time on every weak
configuration. Three of the four windows never see a sequence model.

Two properties make the cross-W comparison sound:

  * Signals do not depend on W. AblationPreprocessor.transform() reads only
    ep["signals"], so the design matrix is built ONCE and only the label
    vector changes per window. Nothing about normalisation, forward-fill or
    the split shifts underneath the comparison.
  * Lead time and capture are W-invariant. compute_early_warning_metrics
    resolves t_sepsis from ep["t_sepsis_hour"], the raw unshifted clinical
    onset, not from the label geometry -- so "how many hours before sepsis
    did we alarm" means the same thing at W=3 and W=24.

The timestep metrics are NOT W-invariant: prevalence runs 0.7% at W=3 to
4.3% at W=24, and AUPRC scales with prevalence. AUPRC lift over the
prevalence floor is reported alongside raw AUPRC for that reason, and the
selection criterion is a patient-level deployment metric, not AUPRC.

Reporting: AUROC is optimistic at ~1-4% positive rate and is never quoted
alone. Every operating point carries timestep precision / recall / F1 and
patient precision / recall.

Usage:
    python scripts/window_experiment.py \
        --episodes results/mimic31_full_ext_prodrome.pkl \
        --out-dir results/window_experiment
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
from sepsentinel.data.gridding import make_labels
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, curve_for, at_burden,
)
from experiment5_recall_study import collect_patient_predictions, build_raw_map

WINDOW_DEFAULT = "3,6,12,24"
METRIC_KEYS = ("patient_recall", "patient_precision", "timestep_precision",
               "timestep_recall", "timestep_f1", "median_lead_time_h",
               "capture_3h", "capture_6h", "capture_12h",
               "alerts_per_patient_day", "threshold")


def relabel(episodes, window_h, label_shift_hours=6):
    """Per-episode labels for a different prodrome window.

    Returns new dicts sharing the original `signals` array -- the sweep needs
    four label vectors, not four copies of a 269 MB design matrix.
    """
    out = []
    for ep in episodes:
        lab = make_labels(len(ep["labels"]), ep["t_sepsis_hour"],
                          label_shift_hours, window_h)
        pos = np.nansum(lab) > 0
        out.append(dict(ep, labels=lab, label=1 if pos else 0,
                        onset_step=int(np.argmax(lab > 0)) if pos else None))
    return out


def usable_across_windows(episodes, windows, label_shift_hours=6):
    """Episode ids that stay septic at EVERY candidate window.

    A stay whose record ends before t_sepsis - W has no positive hour at that
    W, so it would silently move from the septic denominator to the healthy
    one and the patient-level metrics would not be comparable across arms.
    There are a handful; dropping them from all arms keeps every denominator
    identical, which is worth more than the two episodes.
    """
    drop = set()
    for ep in episodes:
        if ep["t_sepsis_hour"] is None:
            continue
        for w in windows:
            lab = make_labels(len(ep["labels"]), ep["t_sepsis_hour"],
                              label_shift_hours, w)
            if np.nansum(lab) == 0:
                drop.add(ep["patient_id"])
                break
    return drop


def window_geometry(relabelled):
    """How much of the record the window actually covers, per septic episode.

    The interpretive trap of a wide W: when the positive window is longer than
    the record, a septic episode has NO negative hours, and the target quietly
    stops being "which hour is prodromal" and becomes "is this patient septic
    at all". At W=24 with a median record of 35 h and median onset at 23 h
    this is not hypothetical, so the fraction is reported with every arm.
    """
    sep = [e for e in relabelled if e["label"] == 1]
    npos = np.array([float(np.nansum(e["labels"])) for e in sep])
    nlen = np.array([float(len(e["labels"])) for e in sep])
    return {"n_septic": len(sep),
            "fully_positive_frac": float(np.mean(npos >= nlen)),
            "median_positive_hours": float(np.median(npos)),
            "median_covered_frac": float(np.median(npos / nlen))}


def swap_labels(preprocessed, label_by_pid):
    """Attach a window's labels to the already-built design matrix."""
    out = []
    for d in preprocessed:
        lab = label_by_pid[d["patient_id"]]
        out.append(dict(d, labels=lab, label=1 if np.nansum(lab) > 0 else 0))
    return out


def point(curve, burden):
    """Best-patient-recall operating point inside the alert budget."""
    c = at_burden(curve, burden)
    if c is None:
        return None
    return {k: c[k] for k in METRIC_KEYS}


def fit_xgboost(train, test_sets, raw_map, seed):
    """One XGBoost fit, scored on each named split. Returns dict of results."""
    import xgboost as xgb
    X = np.concatenate([d["signals"] for d in train], axis=0)
    y = np.concatenate([d["labels"] for d in train], axis=0)
    assert not np.isnan(y).any(), (
        "NaN labels: these episodes carry post-onset hours, which this script "
        "does not handle. Re-extract with --post-onset-truncate-h 0.")
    spw = float((1 - y.mean()) / max(y.mean(), 1e-9))
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=seed, scale_pos_weight=spw,
                            eval_metric="logloss", tree_method="hist")
    clf.fit(X, y)

    out = {"train_positive_rate": float(y.mean())}
    for name, data in test_sets.items():
        Xs = np.concatenate([d["signals"] for d in data], axis=0)
        ys = np.concatenate([d["labels"] for d in data], axis=0)
        prob = clf.predict_proba(Xs)[:, 1]
        out[name] = summarise(ys, prob,
                              patient_results_from_probs(data, raw_map, prob))
    return out


def summarise(y, prob, patient_results, burdens=(0.5, 1.0, 2.0)):
    """Threshold-free discrimination plus operating points at each burden."""
    prev = float(np.mean(y))
    auprc = float(average_precision_score(y, prob))
    curve = curve_for(patient_results)
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "auprc": auprc,
        "prevalence": prev,
        "auprc_lift": auprc / prev if prev else float("nan"),
        "n_positive": int(np.sum(y)),
        "n_rows": int(y.size),
        "at_burden": {str(b): point(curve, b) for b in burdens},
    }


def train_transformer(data, raw_map, in_dim, seed, device, epochs, ckpt_dir,
                      pos_weight):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SepsisTransformer(input_dim=in_dim)
    trainer = Trainer(model, device=device, pos_weight=pos_weight,
                      checkpoint_dir=ckpt_dir)
    trainer.fit(data["train"], data["val"], epochs=epochs,
                batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)
    return model


def score_transformer(model, data, raw_map, device):
    res = collect_patient_predictions(model, data, raw_map, device)
    y = np.concatenate([p["labels"] for p in res])
    p = np.concatenate([p["probs"] for p in res])
    return summarise(y, p, res)


def fmt(v, spec="%.3f"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else spec % v


def burden_table(rows, burden, label_col="Window"):
    """Markdown table of operating points, precision/recall/F1 first."""
    b = str(burden)
    lines = ["| %s | TS prec. | TS recall | TS F1 | Patient recall | "
             "Patient prec. | Median lead (h) | Cap >=6h | Cap >=12h | Thr |"
             % label_col,
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in rows:
        c = res["at_burden"][b]
        if c is None:
            lines.append("| %s | (burden unreachable) | | | | | | | | |" % name)
            continue
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (name, fmt(c["timestep_precision"]),
                        fmt(c["timestep_recall"]), fmt(c["timestep_f1"]),
                        fmt(c["patient_recall"], "%.3f"),
                        fmt(c["patient_precision"]),
                        fmt(c["median_lead_time_h"], "%.1f"),
                        fmt(c["capture_6h"]), fmt(c["capture_12h"]),
                        fmt(c["threshold"], "%.2f")))
    return lines


def discrimination_table(rows, label_col="Window"):
    lines = ["| %s | Prevalence | AUROC | AUPRC | AUPRC lift | Pos. rows |"
             % label_col, "|---|---|---|---|---|---|"]
    for name, res in rows:
        lines.append("| %s | %.4f | %.4f | %.4f | %.1fx | %d |"
                     % (name, res["prevalence"], res["auroc"], res["auprc"],
                        res["auprc_lift"], res["n_positive"]))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--windows", default=WINDOW_DEFAULT)
    ap.add_argument("--seeds", default="42,123,456",
                    help="Transformer seeds at the selected window")
    ap.add_argument("--burden", type=float, default=1.0,
                    help="alert budget (false alerts per nonseptic "
                         "patient-day) the selection is made inside")
    ap.add_argument("--label-shift-hours", type=float, default=6.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--features", default="all", choices=["all", "config_i"])
    ap.add_argument("--stage", default="all",
                    choices=["select", "transformer", "all"],
                    help="'select' stops after the XGBoost sweep")
    ap.add_argument("--force-window", type=float, default=None,
                    help="skip selection and train the Transformer at this W")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "windows.json")
    t_start = time.time()

    windows = [float(w) for w in args.windows.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    print("Loaded %d episodes (%d septic at W=12)"
          % (len(episodes), sum(e["label"] for e in episodes)))

    dropped = usable_across_windows(episodes, windows, args.label_shift_hours)
    if dropped:
        episodes = [e for e in episodes if e["patient_id"] not in dropped]
        print("Dropped %d episode(s) with no positive hour at some candidate "
              "window, so every arm shares one denominator" % len(dropped))

    # Split and preprocess ONCE. Signals are label-independent, so every
    # window reuses this exact design matrix.
    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_by_w = {}
    pre = build_preprocessor(episodes, args.features)
    base = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    in_dim = base["train"][0]["signals"].shape[1]
    print("Design matrix: %d channels; %d/%d/%d episodes train/val/test"
          % (in_dim, len(base["train"]), len(base["val"]), len(base["test"])))

    results = {"config": {"episodes": args.episodes, "windows": windows,
                          "burden": args.burden, "seeds": seeds,
                          "features": args.features, "in_dim": in_dim,
                          "split_seed": SPLIT_SEED,
                          "dropped_episodes": len(dropped)},
               "select": {}, "final": {}}

    # ---- Stage 1: XGBoost sweep, selected on validation --------------------
    for w in windows:
        t0 = time.time()
        relabelled = relabel(episodes, w, args.label_shift_hours)
        by_pid = {e["patient_id"]: e["labels"] for e in relabelled}
        raw_by_w[w] = {e["patient_id"]: e for e in relabelled}
        data = {k: swap_labels(v, by_pid) for k, v in base.items()}
        res = fit_xgboost(data["train"],
                          {"val": data["val"], "test": data["test"]},
                          raw_by_w[w], seed=42)
        res["geometry"] = window_geometry(relabelled)
        res["minutes"] = (time.time() - t0) / 60.0
        results["select"]["W%g" % w] = res
        v = res["val"]
        c = v["at_burden"][str(args.burden)]
        print("  W=%-4g prevalence %.4f  fully-positive septic %.0f%%  "
              "val AUROC %.4f AUPRC %.4f (%.1fx)  %s  (%.1f min)"
              % (w, v["prevalence"], res["geometry"]["fully_positive_frac"] * 100,
                 v["auroc"], v["auprc"], v["auprc_lift"],
                 "val recall@%.1f %.3f  TS-F1 %.3f  lead %sh"
                 % (args.burden, c["patient_recall"], c["timestep_f1"],
                    fmt(c["median_lead_time_h"], "%.1f")) if c else
                 "burden unreachable", res["minutes"]))
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)

    # Selection: best validation patient recall inside the alert budget.
    def val_recall(w):
        c = results["select"]["W%g" % w]["val"]["at_burden"][str(args.burden)]
        return -1.0 if c is None else c["patient_recall"]

    if args.force_window is not None:
        w_star = args.force_window
        rule = "forced by --force-window"
    else:
        w_star = max(windows, key=val_recall)
        rule = ("max validation patient recall at <=%.1f alerts per nonseptic "
                "patient-day" % args.burden)
    # Robustness: would a different objective have chosen differently?
    alt = {}
    for obj in ("capture_6h", "capture_12h", "timestep_f1"):
        def key(w, o=obj):
            c = results["select"]["W%g" % w]["val"]["at_burden"][str(args.burden)]
            return -1.0 if c is None or c[o] is None else c[o]
        alt[obj] = max(windows, key=key)
    alt["auprc_lift"] = max(
        windows, key=lambda w: results["select"]["W%g" % w]["val"]["auprc_lift"])

    results["selection"] = {"window": w_star, "rule": rule,
                            "alternative_objectives": alt}
    print("\nSelected W* = %g h (%s)" % (w_star, rule))
    print("Other objectives would have chosen: %s"
          % ", ".join("%s -> %g" % (k, v) for k, v in alt.items()))
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    if args.stage == "select":
        print("\nStage 'select' complete -> %s" % json_path)
        return

    # ---- Stage 2: Transformer at W* only -----------------------------------
    relabelled = relabel(episodes, w_star, args.label_shift_hours)
    by_pid = {e["patient_id"]: e["labels"] for e in relabelled}
    raw_star = {e["patient_id"]: e for e in relabelled}
    data = {k: swap_labels(v, by_pid) for k, v in base.items()}
    y_tr = np.concatenate([d["labels"] for d in data["train"]])
    pw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    print("\nTraining Transformer at W=%g (positive rate %.4f, pos_weight %.1f)"
          % (w_star, y_tr.mean(), pw))

    per_seed = []
    for seed in seeds:
        t0 = time.time()
        ckpt = os.path.join(args.out_dir, "transformer_W%g_seed%d"
                            % (w_star, seed))
        model = train_transformer(data, raw_star, in_dim, seed, args.device,
                                  args.epochs, ckpt, pw)
        res = score_transformer(model, data["test"], raw_star, args.device)
        res["seed"] = seed
        res["minutes"] = (time.time() - t0) / 60.0
        per_seed.append(res)
        c = res["at_burden"][str(args.burden)]
        print("  seed %3d: AUROC %.4f AUPRC %.4f  recall %s  TS-F1 %s  "
              "(%.1f min)"
              % (seed, res["auroc"], res["auprc"],
                 fmt(c["patient_recall"]) if c else "n/a",
                 fmt(c["timestep_f1"]) if c else "n/a", res["minutes"]))
        results["final"]["transformer_seeds"] = per_seed
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)

    # Mean +/- sd across seeds, so the XGBoost comparison is against a
    # distribution rather than one draw.
    agg = {}
    for k in ("auroc", "auprc", "auprc_lift"):
        vals = [r[k] for r in per_seed]
        agg[k] = [float(np.mean(vals)), float(np.std(vals))]
    for k in METRIC_KEYS:
        vals = [r["at_burden"][str(args.burden)][k] for r in per_seed
                if r["at_burden"][str(args.burden)] is not None
                and r["at_burden"][str(args.burden)][k] is not None]
        if vals:
            agg[k] = [float(np.mean(vals)), float(np.std(vals))]
    results["final"]["transformer_mean_sd"] = agg
    results["final"]["xgboost_test"] = results["select"]["W%g" % w_star]["test"]
    results["final"]["window"] = w_star

    # ---- Report -------------------------------------------------------------
    sel_rows = [("W=%g h" % w, results["select"]["W%g" % w]["val"])
                for w in windows]
    test_rows = [("W=%g h" % w, results["select"]["W%g" % w]["test"])
                 for w in windows]
    xgb_star = results["final"]["xgboost_test"]
    c_x = xgb_star["at_burden"][str(args.burden)]
    c_t = agg

    lines = [
        "# Prodrome window: which target definition is the useful one?", "",
        "Episodes: `%s`  (%d after dropping %d with no positive hour at some "
        "candidate window)" % (args.episodes, len(episodes), len(dropped)),
        "Grouped subject-level split, seed %d. Design matrix built once and "
        "shared across windows; only the label vector changes." % SPLIT_SEED,
        "Total runtime %.1f min." % ((time.time() - t_start) / 60.0), "",
        "**Selected W\\* = %g h** -- %s." % (w_star, rule), "",
        "Lead time and capture are measured against the raw clinical onset "
        "`t_sepsis_hour`, so they mean the same thing at every W. Timestep "
        "metrics are not comparable across W without care: prevalence runs "
        "%.2f%% to %.2f%% across the sweep, and AUPRC scales with it, which "
        "is why AUPRC lift over the prevalence floor is shown too."
        % (min(r["prevalence"] for _, r in sel_rows) * 100,
           max(r["prevalence"] for _, r in sel_rows) * 100), "",
        "## Stage 1 -- XGBoost sweep, VALIDATION patients", "",
        "Selection was made here and nowhere else.", "",
    ]
    lines += discrimination_table(sel_rows)
    lines += ["", "### Window geometry", "",
              "A septic episode with no negative hours contributes nothing to "
              "*when* the alarm should fire, only to *whether*. Where that "
              "fraction is large the target has drifted from early warning "
              "toward patient-level classification.", "",
              "| Window | Septic episodes | Fully positive | Median positive "
              "hours | Median fraction of record |",
              "|---|---|---|---|---|"]
    for w in windows:
        g = results["select"]["W%g" % w]["geometry"]
        lines.append("| W=%g h | %d | %.1f%% | %.0f | %.2f |"
                     % (w, g["n_septic"], g["fully_positive_frac"] * 100,
                        g["median_positive_hours"], g["median_covered_frac"]))
    lines += ["", "At <=%.1f false alerts per nonseptic patient-day:"
              % args.burden, ""]
    lines += burden_table(sel_rows, args.burden)
    lines += ["", "Under other objectives the choice would have been: %s."
              % ", ".join("%s -> W=%g h" % (k, v) for k, v in alt.items()), ""]

    lines += ["## Stage 1 (reference) -- the same XGBoost fits on TEST", "",
              "Shown for completeness; nothing was selected on these.", ""]
    lines += discrimination_table(test_rows)
    lines += [""]
    lines += burden_table(test_rows, args.burden)

    lines += ["", "## Stage 2 -- does attention add value at W\\* = %g h?" % w_star,
              "",
              "Transformer trained only at the selected window, %d seeds, "
              "reported mean +/- sd on TEST. XGBoost is the single fit from "
              "stage 1, same split, same channels." % len(seeds), "",
              "| Model | AUROC | AUPRC | AUPRC lift |", "|---|---|---|---|",
              "| XGBoost (flat) | %.4f | %.4f | %.1fx |"
              % (xgb_star["auroc"], xgb_star["auprc"], xgb_star["auprc_lift"]),
              "| Transformer | %.4f +/- %.4f | %.4f +/- %.4f | %.1fx |"
              % (agg["auroc"][0], agg["auroc"][1], agg["auprc"][0],
                 agg["auprc"][1], agg["auprc_lift"][0]), "",
              "At <=%.1f false alerts per nonseptic patient-day:"
              % args.burden, "",
              "| Model | TS prec. | TS recall | TS F1 | Patient recall | "
              "Patient prec. | Median lead (h) | Cap >=6h | Cap >=12h |",
              "|---|---|---|---|---|---|---|---|---|"]
    if c_x is not None:
        lines.append("| XGBoost (flat) | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (fmt(c_x["timestep_precision"]),
                        fmt(c_x["timestep_recall"]), fmt(c_x["timestep_f1"]),
                        fmt(c_x["patient_recall"]), fmt(c_x["patient_precision"]),
                        fmt(c_x["median_lead_time_h"], "%.1f"),
                        fmt(c_x["capture_6h"]), fmt(c_x["capture_12h"])))

    def pm(k, spec="%.3f"):
        if k not in c_t:
            return "n/a"
        return (spec + " +/- " + spec) % (c_t[k][0], c_t[k][1])
    lines.append("| Transformer | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (pm("timestep_precision"), pm("timestep_recall"),
                    pm("timestep_f1"), pm("patient_recall"),
                    pm("patient_precision"), pm("median_lead_time_h", "%.1f"),
                    pm("capture_6h"), pm("capture_12h")))
    lines.append("")

    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "WINDOW.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)
    print("\nSaved -> %s" % os.path.join(args.out_dir, "WINDOW.md"))


if __name__ == "__main__":
    main()
