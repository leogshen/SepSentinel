#!/usr/bin/env python
"""Is hand-built missingness encoding the bottleneck?

The hypothesis: our Strategy B representation (causal forward-fill + binary
mask + hours-since-last) is a hand-built version of what GRU-D and friends
learn, and the hand-built version is what is capping us -- not the sequence
model on top of it.

Three pieces of evidence motivate the test:
  * our own ablation -- raw current-hour values 0.640, + mask/delta 0.722,
    + full causal Transformer 0.736: representation is worth ~6x the
    architecture;
  * Jin & Lee 2026 report +0.037 AUROC on MIMIC-IV v3.1 from adding binary
    missing-indicators alone;
  * arXiv:2601.16516 finds encoder design matters far more than the
    pretrained backbone for irregular ICU series.

Arms, all on identical episodes/split/metrics, increasingly informed about
missingness:

  A  values_only        forward-filled values, no mask, no delta
  B  strategy_b         + binary mask + hours-since-last  (what we ship now)
  C  learned_decay      GRU-D-style: the model learns, per channel, how fast
                        a stale observation should decay toward the empirical
                        mean, instead of being forward-filled forever
  D  learned_embedding  a small learned embedding of (was-measured,
                        time-since) per channel, concatenated to the value --
                        the "missing token" idea, learned rather than
                        hand-coded

A -> B measures what our hand-built encoding is worth. B -> C/D measures what
a LEARNED encoding adds on top. If B -> C/D is large, missingness is
architectural and we should adopt an irregular-time-series encoder. If it is
near zero, the hand-built version is already capturing what is there and the
ceiling is genuinely the measurement sparsity.

Usage:
    python scripts/missingness_experiment.py --episodes results/...pkl \
        --out-dir results/missingness_experiment
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
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment2_imputation import SPLIT_SEED, BATCH_SIZE, LR, PATIENCE
from sepsentinel.data.preprocessing import collate_fn
from sepsentinel.data.splitting import grouped_patient_split
from sepsentinel.model_b.training import Trainer, SequenceDataset
from sepsentinel.model_b.transformer import SepsisTransformer
from scripts.operating_curves import (
    build_preprocessor, curve_for, at_burden, BURDEN_POINTS,
)
from experiment5_recall_study import collect_patient_predictions, build_raw_map


class LearnedDecayInput(nn.Module):
    """GRU-D-style learnable decay applied to the value channels.

    Strategy B forward-fills a stale value forever, asserting that a lactate
    drawn 40 hours ago still describes the patient. GRU-D instead decays a
    stale observation toward the population mean at a rate the model learns
    per channel:

        gamma = exp(-relu(W * delta))
        x_hat = gamma * x_last + (1 - gamma) * x_mean

    A channel measured continuously can learn gamma ~ 1 (trust the carried
    value); a channel measured every few days can learn a fast decay. The
    decay rate is the thing we currently hard-code as "carry it forever".
    """

    def __init__(self, n_channels, n_raw):
        super().__init__()
        self.n_raw = n_raw
        self.decay = nn.Linear(n_raw, n_raw)          # per-channel decay rate
        nn.init.zeros_(self.decay.bias)

    def forward(self, x, values, masks, deltas, means):
        gamma = torch.exp(-torch.relu(self.decay(deltas)))
        decayed = gamma * values + (1.0 - gamma) * means
        return torch.cat([decayed, masks, deltas], dim=-1)


class LearnedMissingEmbedding(nn.Module):
    """Learned per-channel embedding of the observation pattern.

    Instead of feeding (mask, delta) as raw numbers, embed them: the model
    gets a learned vector describing "this channel was measured / not, this
    long ago", which it can combine with the value however it likes. This is
    the closest thing to the "missing token" idea inside our current stack.
    """

    def __init__(self, n_raw, embed_dim=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2 * n_raw, n_raw * embed_dim), nn.ReLU(),
            nn.Linear(n_raw * embed_dim, n_raw * embed_dim))
        self.out_dim = n_raw + n_raw * embed_dim

    def forward(self, x, values, masks, deltas, means):
        e = self.embed(torch.cat([masks, deltas], dim=-1))
        return torch.cat([values, e], dim=-1)


class MissingnessTransformer(nn.Module):
    """SepsisTransformer with a swappable missingness front-end."""

    def __init__(self, n_raw, mode, d_model=64, **kw):
        super().__init__()
        self.mode = mode
        self.n_raw = n_raw
        if mode == "values_only":
            in_dim = n_raw
            self.front = None
        elif mode == "strategy_b":
            in_dim = 3 * n_raw
            self.front = None
        elif mode == "learned_decay":
            in_dim = 3 * n_raw
            self.front = LearnedDecayInput(3 * n_raw, n_raw)
        elif mode == "learned_embedding":
            self.front = LearnedMissingEmbedding(n_raw)
            in_dim = self.front.out_dim
        else:
            raise ValueError(mode)
        self.net = SepsisTransformer(input_dim=in_dim, d_model=d_model, **kw)
        self.register_buffer("means", torch.zeros(n_raw))

    def forward(self, x, lengths=None):
        n = self.n_raw
        values, masks, deltas = x[..., :n], x[..., n:2 * n], x[..., 2 * n:]
        if self.mode == "values_only":
            z = values
        elif self.mode == "strategy_b":
            z = x
        else:
            z = self.front(x, values, masks, deltas,
                           self.means.expand_as(values))
        return self.net(z, lengths)


def evaluate(model, data, raw_map, device):
    results = collect_patient_predictions(model, data, raw_map, device)
    y = np.concatenate([p["labels"] for p in results])
    p = np.concatenate([p["probs"] for p in results])
    return (float(roc_auc_score(y, p)), float(average_precision_score(y, p)),
            curve_for(results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes",
                    default="values_only,strategy_b,learned_decay,learned_embedding")
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
    data = {k: (pre.fit_transform(splits[k]) if k == "train"
                else pre.transform(splits[k])) for k in ("train", "val", "test")}

    n_raw = pre.n_raw
    n_ch = pre.n_channels
    print("features %d raw -> %d channels" % (n_raw, n_ch))
    if n_ch != 3 * n_raw:
        print("NOTE: layout is not a clean [values|masks|deltas] split "
              "(%d != 3x%d) -- dense vitals contribute value-only channels. "
              "The decay/embedding arms assume the clean layout, so this "
              "experiment needs a labs-only feature set to be exact."
              % (n_ch, n_raw))

    y_train = np.concatenate([d["labels"] for d in data["train"]])
    pos_weight = float((1 - y_train.mean()) / max(y_train.mean(), 1e-9))
    means = torch.tensor(
        np.nanmean(np.concatenate([d["signals"][:, :n_raw]
                                   for d in data["train"]]), axis=0),
        dtype=torch.float32)

    out = {}
    for mode in args.modes.split(","):
        mode = mode.strip()
        t0 = time.time()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MissingnessTransformer(n_raw, mode)
        model.means.copy_(means)
        trainer = Trainer(model, device=args.device, pos_weight=pos_weight,
                          checkpoint_dir=os.path.join(args.out_dir, mode))
        trainer.fit(data["train"], data["val"], epochs=args.epochs,
                    batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE)
        auroc, auprc, curve = evaluate(model, data["test"], raw_map, args.device)
        c = at_burden(curve, 1.0)
        out[mode] = {"auroc": auroc, "auprc": auprc, "at_1_per_day": c,
                     "minutes": (time.time() - t0) / 60.0}
        print("%-18s AUROC %.4f AUPRC %.4f | at <=1.0/day: recall %.2f, "
              "lead %s h, capture>=6h %.2f  (%.1f min)"
              % (mode, auroc, auprc, c["patient_recall"],
                 "%.1f" % c["median_lead_time_h"]
                 if c["median_lead_time_h"] is not None else "n/a",
                 c["capture_6h"], out[mode]["minutes"]))

    with open(os.path.join(args.out_dir, "missingness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved -> %s" % os.path.join(args.out_dir, "missingness.json"))


if __name__ == "__main__":
    main()
