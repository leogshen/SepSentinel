#!/usr/bin/env python
"""Four arms: does pooling MIMIC and SICdb for pretraining buy anything?

  1. xgboost      flat baseline, no encoder
  2. scratch      supervised Transformer, no pretraining
  3. mae_mimic    MAE pretraining on MIMIC only, then fine-tune
  4. mae_pooled   MAE pretraining on MIMIC + SICdb, then fine-tune

The supervised task, the split and the evaluation are identical across arms.
Arms 3 and 4 are COMPUTE-MATCHED: the same number of pretraining steps, so
the contrast isolates the contribution of SICdb rather than of extra compute.

The pretext is NOT the experiment-6 recipe, which was measured to be
defective (see sepsentinel/... and TECHNIQUES.md Topic B). Two fixes:

  * OBSERVATION-RESTRICTED LOSS. Experiment 6 built its reconstruction target
    from post-preprocessing channels, so causally forward-filled and
    median-imputed values were targets and the model was rewarded for
    reproducing imputation artifacts. Here the loss is computed only at
    positions that were genuinely observed in the RAW episode, which is
    carried alongside the preprocessed tensor.
  * CONTIGUOUS BLOCK MASKING. Experiment 6 masked i.i.d. per timestep, so on
    a forward-filled staircase the unmasked neighbouring hour was the answer.
    Here whole spans of MASK_BLOCK_H hours are removed at once.

Two further guards the multi-source arm needs:

  * SEALED SICdb HOLD-OUT. A fraction of SICdb patients never enters
    pretraining, so SICdb remains available as external validation. Once a
    patient has been pretrained on, it cannot serve that role.
  * HOSPITAL-SHORTCUT PROBE. A logistic probe on the frozen encoder tries to
    predict which database a window came from. High AUROC means the encoder
    spent capacity on site identity. The density report shows the risk is
    concentrated in the blood-gas channels (lactate 10x, PaO2 7.6x, pH 6.7x
    denser in SICdb), so this is measured rather than assumed away.

SICdb labels are never used: pretraining is self-supervised and fine-tuning
is on MIMIC. Those labels are rung 1 and measured to over-call by 3-7x.

Usage:
    python scripts/pooled_pretrain.py \
        --mimic results/mimic31_pooled17.pkl \
        --sicdb results/sicdb_episodes.pkl \
        --out-dir results/pooled_pretrain
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED, BATCH_SIZE, LR, PATIENCE
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import build_preprocessor, patient_results_from_probs
from scripts.xgb_tls_experiment import (
    per_patient_stats, reduce_stats, freeze_threshold, paired_bootstrap, fmt,
)
from experiment5_recall_study import collect_patient_predictions

MASK_RATIO = 0.40
MASK_BLOCK_H = 6          # contiguous hours removed per masked span
SICDB_SEALED_FRAC = 0.30  # never pretrained on; reserved for external use
KEYS = ("capture_6h", "capture_12h", "capture_3h", "patient_recall",
        "patient_precision", "timestep_precision", "timestep_recall",
        "timestep_f1", "median_lead_time_h", "alerts_per_patient_day")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def with_observed(preprocessed, raw_by_pid, n_raw):
    """Attach the RAW observation mask to each preprocessed episode.

    AblationPreprocessor.transform() drops everything except signals/labels/
    length/patient_id/label, so observability has to be recovered from the raw
    episode. Without this the MAE cannot tell an observation from a
    forward-filled carry, which is the defect in experiment 6.
    """
    out = []
    for d in preprocessed:
        raw = raw_by_pid[d["patient_id"]]["signals"]
        obs = (~np.isnan(raw)).astype(np.float32)
        out.append(dict(d, observed=obs[:len(d["signals"])]))
    return out


def pad_batch(items, device):
    n = len(items)
    T = max(len(d["signals"]) for d in items)
    C = items[0]["signals"].shape[1]
    K = items[0]["observed"].shape[1]
    x = np.zeros((n, T, C), dtype=np.float32)
    ob = np.zeros((n, T, K), dtype=np.float32)
    ln = np.zeros(n, dtype=np.int64)
    for i, d in enumerate(items):
        t = len(d["signals"])
        x[i, :t] = d["signals"]
        ob[i, :t] = d["observed"]
        ln[i] = t
    return (torch.from_numpy(x).to(device), torch.from_numpy(ob).to(device),
            torch.from_numpy(ln))


def block_mask(shape_bt, lengths, ratio, block_h, device, gen):
    """(B, T) boolean: True where a contiguous span is masked out."""
    B, T = shape_bt
    m = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        L = int(lengths[i])
        if L <= 1:
            continue
        n_blocks = max(1, int(round(ratio * L / block_h)))
        starts = torch.randint(0, max(1, L - 1), (n_blocks,),
                               generator=gen, device=device)
        for s in starts.tolist():
            m[i, s:min(s + block_h, L)] = True
    return m


# --------------------------------------------------------------------------
# Pretraining
# --------------------------------------------------------------------------

class MAE(nn.Module):
    """Encoder + a light decoder, discarded after pretraining."""

    def __init__(self, in_dim, n_values, d_model=64):
        super().__init__()
        self.backbone = SepsisTransformer(input_dim=in_dim, d_model=d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, 32), nn.GELU(),
                                     nn.Linear(32, n_values))

    def forward(self, x, lengths):
        h = self.backbone.encoder(x, lengths)
        return self.decoder(h)


def pretrain(episodes, in_dim, n_values, steps, device, seed, batch_size,
             lr=1e-4, log_every=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    model = MAE(in_dim, n_values).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    n = len(episodes)
    hist = []
    model.train()
    for step in range(steps):
        idx = rng.integers(0, n, batch_size)
        x, obs, ln = pad_batch([episodes[i] for i in idx], device)
        mask = block_mask(x.shape[:2], ln, MASK_RATIO, MASK_BLOCK_H, device, gen)
        xin = x.clone()
        xin[mask] = 0.0
        pred = model(xin, ln)
        # Loss only where (a) the span was masked and (b) the value was really
        # observed in the raw record.
        tgt = x[:, :, :n_values]
        w = mask.unsqueeze(-1) & (obs[:, :, :n_values] > 0)
        if w.any():
            loss = ((pred - tgt) ** 2)[w].mean()
        else:
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % log_every == 0:
            hist.append({"step": step + 1, "loss": float(loss.item())})
            print("    step %5d/%d  masked-observed MSE %.4f"
                  % (step + 1, steps, loss.item()))
    return model.backbone, hist


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def finetune_and_score(init_encoder, data, raw_map, in_dim, seed, device,
                       epochs, ckpt, pos_weight, burden):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SepsisTransformer(input_dim=in_dim)
    if init_encoder is not None:
        model.encoder.load_state_dict(init_encoder.encoder.state_dict())
    tr = Trainer(model, device=device, pos_weight=pos_weight,
                 checkpoint_dir=ckpt)
    tr.fit(data["train"], data["val"], epochs=epochs, batch_size=BATCH_SIZE,
           lr=LR, patience=PATIENCE)
    res = {k: collect_patient_predictions(model, data[k], raw_map, device)
           for k in ("val", "test")}
    thr = freeze_threshold(res["val"], burden)
    st = per_patient_stats(res["test"], thr)
    y = np.concatenate([p["labels"] for p in res["test"]])
    p = np.concatenate([p["probs"] for p in res["test"]])
    return model, st, {"threshold_from_val": thr,
                       "auroc": float(roc_auc_score(y, p)),
                       "auprc": float(average_precision_score(y, p))}


def shortcut_probe(encoder, mimic_eps, sicdb_eps, device, seed, n=1500):
    """Can a linear probe on frozen features tell the two databases apart?

    MUST be read against the random-encoder control below. The two databases
    differ in measurement density by up to 10x on the blood-gas channels, so
    their INPUTS are already separable and a high probe AUROC on a random
    encoder says nothing about what pretraining did. Only the gap between a
    pretrained encoder and a random one is evidence that pretraining spent
    capacity on site identity.
    """
    rng = np.random.default_rng(seed)
    feats, lab = [], []
    encoder.eval()
    with torch.no_grad():
        for eps, y in ((mimic_eps, 0), (sicdb_eps, 1)):
            pick = rng.choice(len(eps), size=min(n, len(eps)), replace=False)
            for i in pick:
                x, obs, ln = pad_batch([eps[i]], device)
                h = encoder.encoder(x, ln)[0, :int(ln[0])]
                feats.append(h.mean(0).cpu().numpy())
                lab.append(y)
    X = np.stack(feats); Y = np.array(lab)
    cut = int(0.7 * len(Y)); order = rng.permutation(len(Y))
    tr, te = order[:cut], order[cut:]
    clf = LogisticRegression(max_iter=2000).fit(X[tr], Y[tr])
    return float(roc_auc_score(Y[te], clf.predict_proba(X[te])[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True)
    ap.add_argument("--sicdb", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--pretrain-batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--burden", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "pooled.json")
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]

    with open(args.mimic, "rb") as fh:
        mim = pickle.load(fh)
    with open(args.sicdb, "rb") as fh:
        sic = pickle.load(fh)
    feats = list(mim[0]["features"])
    assert list(sic[0]["features"]) == feats, "feature layout mismatch"
    n_raw = len(feats)
    print("pooled layout: %d features | MIMIC %d eps | SICdb %d eps"
          % (n_raw, len(mim), len(sic)))

    # MIMIC split first: the supervised task and every reported number live
    # here, and no SICdb patient may leak into it.
    splits = grouped_patient_split(mim, random_state=SPLIT_SEED)
    raw_map = {e["patient_id"]: e for s in splits.values() for e in s}
    pre = build_preprocessor(mim, "all")
    data = {"train": pre.fit_transform(splits["train"]),
            "val": pre.transform(splits["val"]),
            "test": pre.transform(splits["test"])}
    in_dim = data["train"][0]["signals"].shape[1]
    print("channels after Strategy B: %d" % in_dim)

    # Seal a SICdb hold-out BEFORE any pretraining touches it.
    sic_subj = sorted({e["subject_id"] for e in sic})
    rng = np.random.default_rng(SPLIT_SEED)
    sealed = set(rng.choice(sic_subj,
                            size=int(SICDB_SEALED_FRAC * len(sic_subj)),
                            replace=False).tolist())
    sic_pre = [e for e in sic if e["subject_id"] not in sealed]
    sic_sealed = [e for e in sic if e["subject_id"] in sealed]
    print("SICdb: %d episodes for pretraining, %d SEALED for external use"
          % (len(sic_pre), len(sic_sealed)))

    # Pretraining corpora use the MIMIC TRAIN split only -- val and test
    # patients must stay unseen by the encoder for the usual claim to hold.
    sic_pp = pre.transform(sic_pre)
    raw_sic = {e["patient_id"]: e for e in sic_pre}
    pool_mimic = with_observed(data["train"], raw_map, n_raw)
    pool_sicdb = with_observed(sic_pp, raw_sic, n_raw)
    data_obs = {k: with_observed(v, raw_map, n_raw) for k, v in data.items()}

    y_tr = np.concatenate([d["labels"] for d in data["train"]])
    pw = float((1 - y_tr.mean()) / max(y_tr.mean(), 1e-9))
    print("MIMIC train positive rate %.4f, pos_weight %.1f" % (y_tr.mean(), pw))

    # Random-encoder control for the shortcut probe, measured BEFORE any
    # pretraining so it cannot be contaminated.
    torch.manual_seed(SPLIT_SEED)
    _rand = SepsisTransformer(input_dim=in_dim).to(args.device)
    probes = {"random_control": shortcut_probe(_rand, pool_mimic, pool_sicdb,
                                               args.device, SPLIT_SEED)}
    print("shortcut probe, RANDOM encoder control: AUROC %.3f"
          % probes["random_control"])

    results = {"config": {
        "mimic": args.mimic, "sicdb": args.sicdb, "features": feats,
        "in_dim": in_dim, "pretrain_steps": args.pretrain_steps,
        "pretrain_batch": args.pretrain_batch, "mask_ratio": MASK_RATIO,
        "mask_block_h": MASK_BLOCK_H, "sicdb_sealed_frac": SICDB_SEALED_FRAC,
        "sicdb_pretrain_episodes": len(sic_pre),
        "sicdb_sealed_episodes": len(sic_sealed),
        "burden": args.burden, "seeds": seeds, "split_seed": SPLIT_SEED},
        "arms": {}}

    # ---- Arm 1: flat XGBoost ------------------------------------------
    import xgboost as xgb
    X = {k: np.concatenate([d["signals"] for d in v]) for k, v in data.items()}
    ytr = np.concatenate([d["labels"] for d in data["train"]])
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            random_state=42, scale_pos_weight=pw,
                            eval_metric="logloss",
                            tree_method="hist").fit(X["train"], ytr)
    pr = {k: patient_results_from_probs(data[k], raw_map,
                                        clf.predict_proba(X[k])[:, 1])
          for k in ("val", "test")}
    thr = freeze_threshold(pr["val"], args.burden)
    st_xgb = per_patient_stats(pr["test"], thr)
    yte = np.concatenate([d["labels"] for d in data["test"]])
    pte = clf.predict_proba(X["test"])[:, 1]
    results["arms"]["xgboost"] = dict(
        reduce_stats(st_xgb), threshold_from_val=thr,
        auroc=float(roc_auc_score(yte, pte)),
        auprc=float(average_precision_score(yte, pte)))
    m = results["arms"]["xgboost"]
    print("\nxgboost      cap6h %.3f  recall %.3f  burden %.2f  AUROC %.4f"
          % (m["capture_6h"], m["patient_recall"],
             m["alerts_per_patient_day"], m["auroc"]))

    # ---- Arms 2-4: encoder, one pretrained encoder per arm/seed -------
    stats = {"xgboost": [st_xgb]}
    for arm in ("scratch", "mae_mimic", "mae_pooled"):
        per_seed, arm_stats = [], []
        for seed in seeds:
            t1 = time.time()
            enc, hist = None, None
            if arm != "scratch":
                corpus = pool_mimic if arm == "mae_mimic" \
                    else pool_mimic + pool_sicdb
                print("\n  [%s seed %d] pretraining on %d episodes, %d steps"
                      % (arm, seed, len(corpus), args.pretrain_steps))
                # Reconstruct only the VALUE channels. Strategy B lays the
                # tensor out as [values(n_raw), lab_masks, lab_deltas], so the
                # first n_raw columns align position-for-position with the raw
                # observation mask. Reconstructing the mask/delta channels
                # would be asking the model to predict bookkeeping.
                enc, hist = pretrain(corpus, in_dim, n_raw,
                                     args.pretrain_steps, args.device, seed,
                                     args.pretrain_batch)
                if arm not in probes:
                    probes[arm] = shortcut_probe(enc, pool_mimic, pool_sicdb,
                                                 args.device, seed)
                    print("    hospital-shortcut probe AUROC %.3f "
                          "(random-encoder control %.3f)"
                          % (probes[arm], probes["random_control"]))
            ck = os.path.join(args.out_dir, "%s_seed%d" % (arm, seed))
            model, st, meta = finetune_and_score(
                enc, data, raw_map, in_dim, seed, args.device, args.epochs,
                ck, pw, args.burden)
            r = dict(reduce_stats(st), **meta, seed=seed,
                     minutes=(time.time() - t1) / 60.0,
                     pretrain_loss=hist[-1]["loss"] if hist else None)
            per_seed.append(r); arm_stats.append(st)
            print("  %-11s seed %3d: cap6h %.3f  recall %.3f  burden %.2f  "
                  "AUROC %.4f  (%.1f min)"
                  % (arm, seed, r["capture_6h"], r["patient_recall"],
                     r["alerts_per_patient_day"], r["auroc"], r["minutes"]))
            results["arms"].setdefault(arm, {})["runs"] = per_seed
            with open(json_path, "w") as fh:
                json.dump(results, fh, indent=2)
        stats[arm] = arm_stats
        agg = {}
        for k in list(KEYS) + ["auroc", "auprc"]:
            vals = [r[k] for r in per_seed if r.get(k) is not None]
            if vals:
                agg[k] = [float(np.mean(vals)), float(np.std(vals))]
        results["arms"][arm]["mean_sd"] = agg
    results["shortcut_probe_auroc"] = probes

    # ---- Contrasts ----------------------------------------------------
    contrasts = {}
    for arm in ("mae_mimic", "mae_pooled"):
        contrasts["%s_vs_scratch" % arm] = paired_bootstrap(
            stats["scratch"][0], stats[arm][0], args.n_boot, 42, KEYS)
    contrasts["mae_pooled_vs_mae_mimic"] = paired_bootstrap(
        stats["mae_mimic"][0], stats["mae_pooled"][0], args.n_boot, 42, KEYS)
    contrasts["scratch_vs_xgboost"] = paired_bootstrap(
        stats["xgboost"][0], stats["scratch"][0], args.n_boot, 42, KEYS)
    results["paired_bootstrap"] = contrasts
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- Report -------------------------------------------------------
    label = {"capture_6h": "**Capture >=6h**", "capture_12h": "Capture >=12h",
             "capture_3h": "Capture >=3h", "patient_recall": "Patient recall",
             "patient_precision": "Patient precision",
             "timestep_precision": "Timestep precision",
             "timestep_recall": "Timestep recall", "timestep_f1": "Timestep F1",
             "median_lead_time_h": "Median lead (h)",
             "alerts_per_patient_day": "Realised burden (alerts/pt-day)"}
    arms = ["xgboost", "scratch", "mae_mimic", "mae_pooled"]

    def cell(arm, k):
        if arm == "xgboost":
            return fmt(results["arms"][arm][k],
                       "%.2f" if k in ("median_lead_time_h",
                                       "alerts_per_patient_day") else "%.3f")
        a = results["arms"][arm]["mean_sd"].get(k)
        if not a:
            return "n/a"
        spec = "%.2f" if k in ("median_lead_time_h",
                               "alerts_per_patient_day") else "%.3f"
        return (spec + " +/- " + spec) % (a[0], a[1])

    lines = [
        "# Pooled pretraining: four arms", "",
        "Supervised task, split and evaluation identical across arms. Arms 3 "
        "and 4 are compute-matched at %d pretraining steps, so the contrast "
        "isolates SICdb rather than extra compute. Thresholds are frozen on "
        "VALIDATION (`spend_burden`) at <=%.1f alerts per nonseptic "
        "patient-day. Capture >=6h is the governing metric: the alert goes to "
        "a nurse or response team, whose action needs hours to matter."
        % (args.pretrain_steps, args.burden), "",
        "MIMIC %d episodes (%d septic). SICdb %d episodes pretrained on, "
        "**%d sealed** and never seen, so SICdb survives as external "
        "validation. SICdb labels are never used anywhere: they are rung 1 "
        "and over-call by 3-7x."
        % (len(mim), sum(e["label"] for e in mim), len(sic_pre),
           len(sic_sealed)), "",
        "Pretext: contiguous %d-hour block masking, loss restricted to "
        "positions genuinely OBSERVED in the raw record. Both fix measured "
        "defects in the experiment-6 recipe." % MASK_BLOCK_H, "",
        "| | " + " | ".join(arms) + " |", "|" + "---|" * (len(arms) + 1),
    ]
    for k in list(KEYS) + ["auroc", "auprc"]:
        lines.append("| %s | %s |"
                     % (label.get(k, k), " | ".join(cell(a, k) for a in arms)))
    lines += ["", "## Paired differences (95% CI, patient bootstrap)", "",
              "| Contrast | " + " | ".join(label[k] for k in
                                           ("capture_6h", "patient_recall",
                                            "median_lead_time_h")) + " |",
              "|---|---|---|---|"]
    for name, b in contrasts.items():
        row = []
        for k in ("capture_6h", "patient_recall", "median_lead_time_h"):
            d = b[k]
            star = "" if (d["lo95"] <= 0 <= d["hi95"]) else " **"
            row.append("%+.3f [%+.3f, %+.3f]%s"
                       % (d["mean_diff"], d["lo95"], d["hi95"], star))
        lines.append("| %s | %s |" % (name, " | ".join(row)))
    lines += ["", "## Hospital-shortcut probe", "",
              "Logistic probe on frozen encoder features, predicting which "
              "database a window came from. Read every row against the "
              "RANDOM-ENCODER CONTROL: the two databases differ by up to 10x "
              "in blood-gas measurement density, so their inputs are already "
              "separable and a high AUROC on an untrained encoder is a "
              "property of the data, not of pretraining. Only the GAP above "
              "the control is evidence that pretraining spent capacity on "
              "site identity.", ""]
    ctrl = probes.get("random_control")
    for arm, auc in probes.items():
        if arm == "random_control":
            lines.append("- **random encoder (control): AUROC %.3f**" % auc)
        else:
            lines.append("- %s: AUROC %.3f (%+.3f vs control)"
                         % (arm, auc, auc - ctrl))
    lines += ["", "Built in %.1f min." % ((time.time() - t0) / 60), ""]

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "POOLED.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print()
    print(text)


if __name__ == "__main__":
    main()
