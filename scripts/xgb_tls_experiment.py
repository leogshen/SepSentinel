#!/usr/bin/env python
"""Temporal Label Smoothing on flat XGBoost at W=12, against its hard-label
baseline.

TLS raised the Transformer's patient recall from 0.494 to 0.566
(results/recipe_experiment/recipe.json). XGBoost is the model currently
delivering better early warning, so the question is whether that gain
transfers. It is a hypothesis, not an expectation: the Transformer result
says nothing about a tree ensemble, whose loss, capacity and inductive bias
all differ.

Everything is held fixed except the training targets: same episodes, same
grouped split, same 38 Strategy B channels, same XGBoost hyperparameters,
same seed. Only `y` and the sample weights change.

Three methodological requirements, each of which the earlier scripts got
wrong and each of which changes the numbers:

1. THRESHOLDS ARE FROZEN ON VALIDATION. `operating_curves.at_burden()`
   returns max(patient_recall) over thresholds inside the burden budget --
   run on test, that is a maximum taken on the test split, and it flatters
   every number it produces. Here the operating threshold is chosen on the
   VALIDATION curve and then applied unchanged to test. The realised test
   burden is reported alongside, because it will not land exactly on budget.

2. SOFT-LABEL WEIGHTING IS DERIVED, NOT ASSUMED. XGBoost's
   `scale_pos_weight` multiplies the gradient of rows whose label is exactly
   1; with soft targets in (0,1) it would apply to almost nothing. The
   correct object is weighted cross-entropy with a soft target, and it
   reduces exactly to a per-row weight and label:

       positive mass  a = spw * q        negative mass  b = 1 - q
       a*(-log p) + b*(-log(1-p)) = (a+b) * CE(p, a/(a+b))
       => weight w = a + b,  label y = a / (a + b)

   At q=1 this gives (w, y) = (spw, 1) and at q=0 it gives (1, 0), so the
   hard-label arm is the same formula with q in {0,1} and the two arms are
   weighted on one consistent scale. Passing q straight to XGBoost with
   scale_pos_weight would NOT be this, which is why it is spelled out.

3. TLS IS GENERATED FROM ACTUAL ONSET TIMES. recipe_experiment.tls_targets()
   anchors u=0 to the last positive row, assuming it is the onset hour. On
   this cohort the true hours-until-onset at that row has median 1.0 h (max
   11 h), so the ramp sat ~1 h late. Here u = t_sepsis_hour - hour_index,
   read from the episode.

Reported with paired patient bootstrap intervals: test patients are
resampled with replacement and BOTH arms are recomputed on the same
resample, so the interval is on the difference and the between-arm
correlation is preserved. An unpaired interval on each arm separately would
be far wider and would not answer the question.

AUROC is optimistic at 2.6% prevalence and is never quoted alone; every
operating point carries timestep precision / recall / F1 and patient
precision / recall.

Usage:
    python scripts/xgb_tls_experiment.py \
        --episodes results/mimic31_full_ext_prodrome.pkl \
        --out-dir results/xgb_tls
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.splitting import grouped_patient_split
from scripts.operating_curves import (
    build_preprocessor, patient_results_from_probs, curve_for, spend_burden,
)

LEAD_POINTS = (3, 6, 12)


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

def tls_from_onset(n_hours, t_sepsis_hour, window_h, gamma):
    """Exponential TLS ramp keyed to the true clinical onset.

    u = t_sepsis_hour - hour_index, hours until onset. Positives occupy
    0 < u <= window_h. From Yeche et al. (ICML 2023) with 2h = window_h:

        d = -(1/g) ln(1 - exp(-g * W))
        A = -exp(-g * (W - d))
        q(u) = exp(-g (u - d)) + A

    which is 1 at onset and 0 at the far edge of the window.
    """
    q = np.zeros(n_hours, dtype=np.float32)
    if t_sepsis_hour is None:
        return q
    g, W = gamma, float(window_h)
    d = -(1.0 / g) * np.log(1.0 - np.exp(-g * W))
    A = -np.exp(-g * (W - d))
    u = t_sepsis_hour - np.arange(n_hours, dtype=np.float32)
    inside = (u > 0) & (u <= W)
    q[inside] = np.clip(np.exp(-g * (u[inside] - d)) + A, 0.0, 1.0)
    return q


def weighted_targets(q, spw):
    """Weighted cross-entropy with soft targets, as (label, weight).

    See requirement 2 in the module docstring. Exact, and reduces to
    scale_pos_weight semantics when q is in {0, 1}.
    """
    a = spw * q
    b = 1.0 - q
    w = a + b
    return (a / np.maximum(w, 1e-12)).astype(np.float32), w.astype(np.float32)


# --------------------------------------------------------------------------
# Metrics at a FIXED threshold
# --------------------------------------------------------------------------

def per_patient_stats(patient_results, thr):
    """Vectorisable per-patient quantities at one fixed threshold.

    Everything the bootstrap needs is reduced to per-patient scalars here, so
    a resample is array indexing rather than a Python loop over patients.
    """
    n = len(patient_results)
    septic = np.zeros(n, bool)
    alarmed = np.zeros(n, bool)
    lead = np.full(n, np.nan)
    caught = {h: np.zeros(n, bool) for h in LEAD_POINTS}
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n)
    n_alerts = np.zeros(n); length = np.zeros(n)

    for i, p in enumerate(patient_results):
        probs = np.asarray(p["probs"])
        labels = np.asarray(p["labels"])
        pred = probs >= thr
        septic[i] = (p["label"] == 1)
        length[i] = p["length"]
        n_alerts[i] = float(pred.sum())
        tp[i] = float((pred & (labels == 1)).sum())
        fp[i] = float((pred & (labels == 0)).sum())
        fn[i] = float((~pred & (labels == 1)).sum())
        if not septic[i]:
            continue
        idx = np.flatnonzero(pred)
        if idx.size == 0:
            continue
        alarmed[i] = True
        first = int(idx[0])
        t_sepsis = p.get("t_sepsis_hour")
        if t_sepsis is None:
            t_sepsis = p["onset_step"] + 6
        lead[i] = t_sepsis - first
        for h in LEAD_POINTS:
            caught[h][i] = first <= t_sepsis - h
    return {"septic": septic, "alarmed": alarmed, "lead": lead,
            "caught": caught, "tp": tp, "fp": fp, "fn": fn,
            "n_alerts": n_alerts, "length": length}


def reduce_stats(st, idx=None):
    """Collapse per-patient stats (optionally a resample) into metrics."""
    if idx is None:
        idx = np.arange(len(st["septic"]))
    sep = st["septic"][idx]
    heal = ~sep
    n_sep = int(sep.sum())
    tp = st["tp"][idx].sum(); fp = st["fp"][idx].sum(); fn = st["fn"][idx].sum()
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan
    al = st["alarmed"][idx]          # set for septic patients only
    ever = st["n_alerts"][idx] > 0   # any class, for patient precision
    n_ever = int(ever.sum())
    healthy_hours = st["length"][idx][heal].sum()
    healthy_alerts = st["n_alerts"][idx][heal].sum()
    leads = st["lead"][idx]
    leads = leads[~np.isnan(leads)]
    out = {
        "patient_recall": float(al[sep].sum() / n_sep) if n_sep else np.nan,
        "patient_precision": float((ever & sep).sum() / n_ever) if n_ever else np.nan,
        "timestep_precision": float(prec), "timestep_recall": float(rec),
        "timestep_f1": float(f1),
        "median_lead_time_h": float(np.median(leads)) if leads.size else np.nan,
        "alerts_per_patient_day": float(24.0 * healthy_alerts / healthy_hours)
        if healthy_hours else np.nan,
    }
    for h in LEAD_POINTS:
        out["capture_%dh" % h] = (float(st["caught"][h][idx][sep].sum() / n_sep)
                                  if n_sep else np.nan)
    return out


def freeze_threshold(results, burden):
    """Operating threshold that spends the alert budget on the given split.

    Uses spend_burden, not at_burden: maximising recall optimises the very
    quantity being reported, so it does not transfer across splits and leaves
    arms at different realised burdens. Pinning burden depends only on the
    control patients' alarm rate and transfers stably, which is what keeps
    the arms comparable after the threshold is frozen.
    """
    c = spend_burden(curve_for(results), burden)
    return None if c is None else float(c["threshold"])


def paired_bootstrap(stats_a, stats_b, n_boot, seed, keys):
    """CI on (b - a) with test patients resampled jointly.

    Both arms are evaluated on the SAME resampled patient set, so the
    per-patient correlation between the two models is preserved and the
    interval is on the paired difference. Resampling is at the patient level,
    which is the unit of independence here (hours within a stay are not).
    """
    rng = np.random.default_rng(seed)
    n = len(stats_a["septic"])
    diffs = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = reduce_stats(stats_a, idx)
        rb = reduce_stats(stats_b, idx)
        for k in keys:
            diffs[k][b] = rb[k] - ra[k]
    out = {}
    for k in keys:
        d = diffs[k][~np.isnan(diffs[k])]
        out[k] = {"mean_diff": float(np.mean(d)),
                  "lo95": float(np.percentile(d, 2.5)),
                  "hi95": float(np.percentile(d, 97.5)),
                  "p_gt_0": float(np.mean(d > 0))}
    return out


# --------------------------------------------------------------------------

def fit_predict(X_tr, y, w, seed, X_eval):
    """Native-API XGBoost so both arms share one code path.

    XGBClassifier infers discrete classes from `y` and rejects targets in
    (0,1), so the soft arm cannot go through it. Using xgb.train for BOTH
    arms keeps the hard baseline on exactly the same estimator rather than
    comparing across two different wrappers. Parameters match the project's
    standard XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1).
    """
    import xgboost as xgb
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
              "tree_method": "hist", "eval_metric": "logloss", "seed": seed}
    bst = xgb.train(params, xgb.DMatrix(X_tr, label=y, weight=w),
                    num_boost_round=200)
    return {k: bst.predict(xgb.DMatrix(v)) for k, v in X_eval.items()}


def fmt(v, spec="%.3f"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else spec % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--window-h", type=float, default=12.0,
                    help="prodrome window the episodes were built with")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", default="all", choices=["all", "config_i"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    with open(args.episodes, "rb") as fh:
        episodes = pickle.load(fh)
    print("Episodes %d (%d septic)"
          % (len(episodes), sum(e["label"] for e in episodes)))

    splits = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    raw_map = {ep["patient_id"]: ep for s in splits.values() for ep in s}
    pre = build_preprocessor(episodes, args.features)
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    print("%d channels; %d/%d/%d episodes"
          % (data["train"][0]["signals"].shape[1], len(data["train"]),
             len(data["val"]), len(data["test"])))

    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    y_hard = {k: np.concatenate([d["labels"] for d in v])
              for k, v in data.items()}
    spw = float((1 - y_hard["train"].mean()) / max(y_hard["train"].mean(), 1e-9))
    print("train positive rate %.4f, scale_pos_weight %.1f"
          % (y_hard["train"].mean(), spw))

    # TLS targets for the TRAIN split only; val/test keep hard labels so
    # thresholds and every reported metric stay on the real target.
    q_train = np.concatenate([
        tls_from_onset(len(d["labels"]), raw_map[d["patient_id"]]["t_sepsis_hour"],
                       args.window_h, args.gamma)
        for d in data["train"]])
    ex = q_train[q_train > 0]
    print("TLS ramp sample (sorted, last 6): %s"
          % np.array2string(np.sort(ex)[-6:], precision=3))
    print("TLS positive mass %.1f vs hard %.1f (%.1f%% of hard)"
          % (q_train.sum(), y_hard["train"].sum(),
             100 * q_train.sum() / y_hard["train"].sum()))

    # TLS redistributes positive mass toward onset AND shrinks it: the ramp
    # sums to well under the hard-label mass, so a plain TLS-vs-hard contrast
    # confounds "shape of the target" with "less total positive weight". The
    # third arm rescales spw so total positive mass matches the hard arm,
    # isolating the shape effect.
    mass_ratio = float(y_hard["train"].sum() / max(q_train.sum(), 1e-9))
    configs = [("hard", y_hard["train"], spw),
               ("tls", q_train, spw),
               ("tls_massmatched", q_train, spw * mass_ratio)]
    print("mass-matched arm uses spw %.1f (x%.2f)"
          % (spw * mass_ratio, mass_ratio))

    arms = {}
    for name, q, arm_spw in configs:
        t1 = time.time()
        yy, ww = weighted_targets(q.astype(np.float64), arm_spw)
        probs = fit_predict(X["train"], yy, ww, args.seed,
                            {k: X[k] for k in ("val", "test")})
        res = {k: patient_results_from_probs(data[k], raw_map, probs[k])
               for k in ("val", "test")}
        thr = freeze_threshold(res["val"], args.burden)
        st = per_patient_stats(res["test"], thr)
        # Secondary view: threshold re-pinned so every arm spends the SAME
        # test burden. This uses test data, but only the control patients'
        # alarm rate -- never the septic outcomes being compared -- so it is
        # far weaker than selecting a threshold to maximise test recall. It
        # is reported alongside, not instead of, the frozen-threshold view.
        thr_eq = freeze_threshold(res["test"], args.burden)
        st_eq = per_patient_stats(res["test"], thr_eq)
        arms[name] = {
            "threshold_from_val": thr,
            "threshold_test_equalised": thr_eq,
            "test_equalised": reduce_stats(st_eq),
            "_stats_eq": st_eq,
            "scale_pos_weight": arm_spw,
            "auroc": float(roc_auc_score(y_hard["test"], probs["test"])),
            "auprc": float(average_precision_score(y_hard["test"], probs["test"])),
            "test": reduce_stats(st),
            "_stats": st,
            "minutes": (time.time() - t1) / 60.0,
        }
        m = arms[name]["test"]
        print("  %-16s thr(val)=%.2f  AUROC %.4f  recall %.3f  TS-F1 %.3f  "
              "lead %s  burden %.2f  (%.1f min)"
              % (name, thr, arms[name]["auroc"], m["patient_recall"],
                 m["timestep_f1"], fmt(m["median_lead_time_h"], "%.1f"),
                 m["alerts_per_patient_day"], arms[name]["minutes"]))

    keys = ("patient_recall", "patient_precision", "timestep_precision",
            "timestep_recall", "timestep_f1", "median_lead_time_h",
            "capture_3h", "capture_6h", "capture_12h",
            "alerts_per_patient_day")
    print("\nPaired patient bootstrap, %d resamples..." % args.n_boot)
    contrasts, contrasts_eq = {}, {}
    for name in ("tls", "tls_massmatched"):
        contrasts[name] = paired_bootstrap(
            arms["hard"]["_stats"], arms[name]["_stats"],
            args.n_boot, args.seed, keys)
        contrasts_eq[name] = paired_bootstrap(
            arms["hard"]["_stats_eq"], arms[name]["_stats_eq"],
            args.n_boot, args.seed, keys)

    results = {
        "config": {"episodes": args.episodes, "window_h": args.window_h,
                   "gamma": args.gamma, "burden": args.burden,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "scale_pos_weight": spw, "split_seed": SPLIT_SEED,
                   "tls_mass_fraction_of_hard":
                       float(q_train.sum() / y_hard["train"].sum())},
        "arms": {k: {kk: vv for kk, vv in v.items()
                     if not kk.startswith("_stats")}
                 for k, v in arms.items()},
        "paired_bootstrap_vs_hard": contrasts,
        "paired_bootstrap_vs_hard_test_equalised": contrasts_eq,
    }
    with open(os.path.join(args.out_dir, "xgb_tls.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    label = {"patient_recall": "**Patient recall**",
             "patient_precision": "Patient precision",
             "timestep_precision": "Timestep precision",
             "timestep_recall": "Timestep recall",
             "timestep_f1": "Timestep F1",
             "median_lead_time_h": "Median lead (h)",
             "capture_3h": "Capture >=3h", "capture_6h": "Capture >=6h",
             "capture_12h": "Capture >=12h",
             "alerts_per_patient_day": "Realised burden (alerts/pt-day)"}

    lines = [
        "# Temporal Label Smoothing on flat XGBoost (W=%g h)" % args.window_h,
        "", "Episodes: `%s`. Grouped subject split, seed %d. Only the "
        "training targets differ between arms -- same channels, same XGBoost "
        "settings (native API, 200 rounds, depth 6, eta 0.1), same seed."
        % (args.episodes, SPLIT_SEED),
        "Built in %.1f min." % ((time.time() - t0) / 60.0), "",
        "Soft-label weighting is the exact weighted-CE reduction "
        "`w = spw*q + (1-q)`, `y = spw*q/w`, which reduces to "
        "`scale_pos_weight` semantics at q in {0,1}, so both arms are "
        "weighted on one scale. TLS is keyed to `t_sepsis_hour`, not to the "
        "last positive row. **Operating thresholds are frozen on validation** "
        "and applied unchanged to test, so no test-set maximum enters these "
        "numbers; the realised test burden is reported rather than assumed.",
        "",
        "The TLS ramp carries only **%.1f%%** of the hard-label positive "
        "mass, so a plain TLS-vs-hard contrast confounds the *shape* of the "
        "target with *less total positive weight*. The third arm rescales "
        "`spw` by %.2fx to match total mass, isolating the shape effect."
        % (100 * q_train.sum() / y_hard["train"].sum(), mass_ratio), "",
        "## Test set, threshold frozen on validation", "",
        "| | Hard | TLS | TLS mass-matched | TLS - hard (95% CI) | "
        "TLS-mm - hard (95% CI) |", "|---|---|---|---|---|---|",
    ]
    for k in keys:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        cells = []
        for nm in ("tls", "tls_massmatched"):
            b = contrasts[nm][k]
            star = "" if (b["lo95"] <= 0 <= b["hi95"]) else " **"
            cells.append("%+.3f [%+.3f, %+.3f]%s"
                         % (b["mean_diff"], b["lo95"], b["hi95"], star))
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (label[k], fmt(arms["hard"]["test"][k], spec),
                        fmt(arms["tls"]["test"][k], spec),
                        fmt(arms["tls_massmatched"]["test"][k], spec),
                        cells[0], cells[1]))

    lines += ["", "`**` marks a paired 95%% interval excluding zero. "
              "Intervals are on the difference, from %d patient-level "
              "resamples in which every arm sees the same resampled "
              "patients." % args.n_boot, "",
              "## Test set, burden equalised across arms", "",
              "Same models, threshold re-pinned so every arm spends the same "
              "test burden. This touches test data, but only the control "
              "patients' alarm rate -- never the septic outcomes being "
              "compared -- so it is far weaker than choosing a threshold to "
              "maximise test recall. Read it as the equal-burden view that "
              "the frozen-threshold table above cannot quite deliver.", "",
              "| | Hard | TLS | TLS mass-matched | TLS - hard (95% CI) | "
              "TLS-mm - hard (95% CI) |", "|---|---|---|---|---|---|"]
    for k in keys:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        cells = []
        for nm in ("tls", "tls_massmatched"):
            b = contrasts_eq[nm][k]
            star = "" if (b["lo95"] <= 0 <= b["hi95"]) else " **"
            cells.append("%+.3f [%+.3f, %+.3f]%s"
                         % (b["mean_diff"], b["lo95"], b["hi95"], star))
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (label[k], fmt(arms["hard"]["test_equalised"][k], spec),
                        fmt(arms["tls"]["test_equalised"][k], spec),
                        fmt(arms["tls_massmatched"]["test_equalised"][k], spec),
                        cells[0], cells[1]))

    lines += ["",
              "| Arm | Thr (val) | Thr (burden-eq) | spw | AUROC | AUPRC |",
              "|---|---|---|---|---|---|"]
    for nm in ("hard", "tls", "tls_massmatched"):
        a = arms[nm]
        lines.append("| %s | %.2f | %.2f | %.1f | %.4f | %.4f |"
                     % (nm, a["threshold_from_val"],
                        a["threshold_test_equalised"], a["scale_pos_weight"],
                        a["auroc"], a["auprc"]))
    lines.append("")

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "XGB_TLS.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
