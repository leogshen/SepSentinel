#!/usr/bin/env python
"""Re-report the recipe and window experiments with honest thresholds.

Both of those experiments picked their reported operating point with
`operating_curves.at_burden()`, which returns max(patient_recall) over
thresholds inside the alert budget. Run on a test curve that is a maximum
taken on test: it optimises the very quantity being reported, so every
"at <=1.0 alerts/patient-day" number in results/recipe_experiment/recipe.json
and results/window_experiment/WINDOW.md is optimistic by an unmeasured
amount, and the arms are not guaranteed to sit at comparable burdens.

Nothing is retrained here. Every model already exists on disk:
  - recipe: 12 Transformer checkpoints (4 arms x 3 seeds), W=12
  - window: 3 Transformer checkpoints at W=3, plus XGBoost refits per window
    (seconds each, deterministic)

For each model this reports two views, the same pair used by
xgb_tls_experiment.py:

  frozen      threshold chosen on VALIDATION with spend_burden() -- pins the
              budget rather than maximising recall, so it transfers -- then
              applied unchanged to test. No test-set optimisation at all.
  burden-eq   threshold re-pinned so every arm spends the same TEST burden.
              Uses test data, but only the control patients' alarm rate,
              never the septic outcomes being compared.

All contrasts carry paired patient-bootstrap intervals: test patients are
resampled with replacement and every arm is recomputed on the same resample,
so the interval is on the difference.

Usage:
    python scripts/refrozen_report.py --episodes results/mimic31_full_ext_prodrome.pkl \
        --recipe-dir results/recipe_experiment \
        --window-dir results/window_experiment \
        --out-dir results/refrozen
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED
from sepsentinel.data.gridding import make_labels
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import build_preprocessor, curve_for
from scripts.window_experiment import relabel, swap_labels, usable_across_windows
from scripts.xgb_tls_experiment import (
    per_patient_stats, reduce_stats, freeze_threshold, paired_bootstrap, fmt,
)
from experiment5_recall_study import collect_patient_predictions

KEYS = ("patient_recall", "patient_precision", "timestep_precision",
        "timestep_recall", "timestep_f1", "median_lead_time_h",
        "capture_3h", "capture_6h", "capture_12h", "alerts_per_patient_day")
LABEL = {"patient_recall": "**Patient recall**",
         "patient_precision": "Patient precision",
         "timestep_precision": "Timestep precision",
         "timestep_recall": "Timestep recall",
         "timestep_f1": "Timestep F1",
         "median_lead_time_h": "Median lead (h)",
         "capture_3h": "Capture >=3h", "capture_6h": "Capture >=6h",
         "capture_12h": "Capture >=12h",
         "alerts_per_patient_day": "Realised burden (alerts/pt-day)"}


def transformer_probs(ckpt, in_dim, data, raw_map, device):
    model = SepsisTransformer(input_dim=in_dim)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.to(device)
    return {k: collect_patient_predictions(model, data[k], raw_map, device)
            for k in ("val", "test")}


def two_views(res, burden):
    """Per-patient stats under the frozen and burden-equalised thresholds."""
    thr = freeze_threshold(res["val"], burden)
    thr_eq = freeze_threshold(res["test"], burden)
    return {"thr": thr, "thr_eq": thr_eq,
            "frozen": per_patient_stats(res["test"], thr),
            "eq": per_patient_stats(res["test"], thr_eq)}


def seed_pooled(stats_list, key):
    """Mean over seeds of a per-arm metric, for the headline column."""
    vals = [reduce_stats(s)[key] for s in stats_list]
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def table(rows, contrasts, ref_name):
    """rows: list of (name, metrics dict). contrasts: name -> bootstrap dict."""
    head = ["| | " + " | ".join(n for n, _ in rows) + " | " +
            " | ".join("%s - %s (95%% CI)" % (n, ref_name)
                       for n in contrasts) + " |",
            "|" + "---|" * (1 + len(rows) + len(contrasts))]
    out = list(head)
    for k in KEYS:
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        cells = [fmt(m[k], spec) for _, m in rows]
        for nm in contrasts:
            b = contrasts[nm][k]
            star = "" if (b["lo95"] <= 0 <= b["hi95"]) else " **"
            cells.append("%+.3f [%+.3f, %+.3f]%s"
                         % (b["mean_diff"], b["lo95"], b["hi95"], star))
        out.append("| %s | %s |" % (LABEL[k], " | ".join(cells)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--recipe-dir", default="results/recipe_experiment")
    ap.add_argument("--window-dir", default="results/window_experiment")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows", default="3,6,12,24")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    report = {}
    lines = ["# Re-reported with honest thresholds", "",
             "Both source experiments chose their operating point with "
             "`at_burden()`, i.e. max(patient recall) over thresholds -- "
             "computed on test. That optimises the quantity being reported. "
             "Nothing is retrained here; the same checkpoints and the same "
             "deterministic XGBoost fits are re-scored under two rules:", "",
             "- **frozen** -- threshold pinned on VALIDATION to spend the "
             "budget (`spend_burden`), applied unchanged to test. No test "
             "optimisation.",
             "- **burden-eq** -- threshold re-pinned so all arms spend the "
             "same TEST burden. Uses only control patients' alarm rate.", "",
             "Paired patient-bootstrap intervals throughout (%d resamples, "
             "all arms on the same resampled patients)." % args.n_boot, ""]

    with open(args.episodes, "rb") as fh:
        episodes_all = pickle.load(fh)

    # ---------------- Recipe: 4 arms x 3 seeds, W=12 ----------------------
    splits = grouped_patient_split(episodes_all, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    pre = build_preprocessor(episodes_all, "all")
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    in_dim = data["train"][0]["signals"].shape[1]
    print("recipe: %d channels, %d test episodes" % (in_dim, len(data["test"])))

    arms = ["baseline_posw", "no_posweight", "tls_posw", "tls_no_posweight"]
    recipe = {}
    for arm in arms:
        views = []
        for ck in sorted(glob.glob(os.path.join(args.recipe_dir,
                                                "%s_seed*" % arm,
                                                "best_model.pt"))):
            res = transformer_probs(ck, in_dim, data, raw_map, args.device)
            views.append(two_views(res, args.burden))
            print("  %s %s thr=%.2f thr_eq=%.2f"
                  % (arm, os.path.basename(os.path.dirname(ck)),
                     views[-1]["thr"], views[-1]["thr_eq"]))
        recipe[arm] = views

    for which in ("frozen", "eq"):
        rows, contrasts = [], {}
        for arm in arms:
            per_seed = [v[which] for v in recipe[arm]]
            m = {k: seed_pooled(per_seed, k)[0] for k in KEYS}
            rows.append((arm, m))
        # Contrast each arm against the baseline, seed 0 vs seed 0 paired on
        # patients; seeds are averaged after the bootstrap by pooling the
        # first seed of each arm (checkpoints are paired by seed order).
        for arm in arms[1:]:
            contrasts[arm] = paired_bootstrap(
                recipe["baseline_posw"][0][which], recipe[arm][0][which],
                args.n_boot, args.seed, KEYS)
        lines += ["## Recipe experiment (W=12), %s thresholds"
                  % ("validation-frozen" if which == "frozen"
                     else "burden-equalised"), "",
                  "Columns are means over 3 seeds. Intervals contrast one "
                  "matched seed per arm against the same seed of the "
                  "baseline (checkpoints are globbed in a fixed order, so "
                  "the pairing is consistent), resampled on patients.", ""]
        lines += table(rows, contrasts, "baseline")
        lines += [""]
        report["recipe_%s" % which] = {
            "arms": {a: m for a, m in rows},
            "contrasts_vs_baseline": contrasts}
        with open(os.path.join(args.out_dir, "refrozen.json"), "w") as fh:
            json.dump(report, fh, indent=2)

    # ---------------- Window: XGBoost per W + Transformer at W=3 ----------
    windows = [float(w) for w in args.windows.split(",")]
    dropped = usable_across_windows(episodes_all, windows)
    episodes = [e for e in episodes_all if e["patient_id"] not in dropped]
    splits_w = grouped_patient_split(episodes, random_state=SPLIT_SEED)
    pre_w = build_preprocessor(episodes, "all")
    base = {"train": pre_w.fit_transform(splits_w["train"]),
            "val": pre_w.transform(splits_w["val"]),
            "test": pre_w.transform(splits_w["test"])}
    in_dim_w = base["train"][0]["signals"].shape[1]
    print("window: %d channels, %d test episodes (dropped %d)"
          % (in_dim_w, len(base["test"]), len(dropped)))

    import xgboost as xgb
    win = {}
    for w in windows:
        rel = relabel(episodes, w)
        by_pid = {e["patient_id"]: e["labels"] for e in rel}
        raw_w = {e["patient_id"]: e for e in rel}
        d = {k: swap_labels(v, by_pid) for k, v in base.items()}
        X = {k: np.concatenate([x["signals"] for x in v]) for k, v in d.items()}
        y = np.concatenate([x["labels"] for x in d["train"]])
        spw = float((1 - y.mean()) / max(y.mean(), 1e-9))
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                learning_rate=0.1, random_state=42,
                                scale_pos_weight=spw, eval_metric="logloss",
                                tree_method="hist").fit(X["train"], y)
        from scripts.operating_curves import patient_results_from_probs
        res = {k: patient_results_from_probs(d[k], raw_w,
                                             clf.predict_proba(X[k])[:, 1])
               for k in ("val", "test")}
        win["W%g" % w] = two_views(res, args.burden)
        print("  XGB W=%g thr=%.2f thr_eq=%.2f"
              % (w, win["W%g" % w]["thr"], win["W%g" % w]["thr_eq"]))

    # Transformer at W=3, the selected window
    rel3 = relabel(episodes, 3.0)
    by3 = {e["patient_id"]: e["labels"] for e in rel3}
    raw3 = {e["patient_id"]: e for e in rel3}
    d3 = {k: swap_labels(v, by3) for k, v in base.items()}
    tf3 = []
    for ck in sorted(glob.glob(os.path.join(args.window_dir,
                                            "transformer_W3_seed*",
                                            "best_model.pt"))):
        res = transformer_probs(ck, in_dim_w, d3, raw3, args.device)
        tf3.append(two_views(res, args.burden))
        print("  TF W=3 %s thr=%.2f thr_eq=%.2f"
              % (os.path.basename(os.path.dirname(ck)), tf3[-1]["thr"],
                 tf3[-1]["thr_eq"]))

    for which in ("frozen", "eq"):
        rows = [("W=%g h" % w, reduce_stats(win["W%g" % w][which]))
                for w in windows]
        contrasts = {}
        for w in windows[1:]:
            contrasts["W=%g h" % w] = paired_bootstrap(
                win["W%g" % windows[0]][which], win["W%g" % w][which],
                args.n_boot, args.seed, KEYS)
        lines += ["## Window sweep (XGBoost), %s thresholds"
                  % ("validation-frozen" if which == "frozen"
                     else "burden-equalised"), ""]
        lines += table(rows, contrasts, "W=%g h" % windows[0])
        lines += [""]

        xg = reduce_stats(win["W3"][which])
        tfm = {k: seed_pooled([v[which] for v in tf3], k)[0] for k in KEYS}
        c = paired_bootstrap(win["W3"][which], tf3[0][which],
                             args.n_boot, args.seed, KEYS)
        lines += ["### Attention at W=3, %s thresholds"
                  % ("validation-frozen" if which == "frozen"
                     else "burden-equalised"), ""]
        lines += table([("XGBoost", xg), ("Transformer", tfm)],
                       {"Transformer": c}, "XGBoost")
        lines += [""]
        report["window_%s" % which] = {
            "xgboost": {n: m for n, m in rows},
            "transformer_W3": tfm,
            "transformer_minus_xgboost_W3": c}
        with open(os.path.join(args.out_dir, "refrozen.json"), "w") as fh:
            json.dump(report, fh, indent=2)

    lines += ["", "`**` marks a paired 95% interval excluding zero.",
              "", "Built in %.1f min." % ((time.time() - t0) / 60.0), ""]
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "REFROZEN.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
