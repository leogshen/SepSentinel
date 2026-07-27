# Experiment 2: Sparse-Lab Imputation Strategy Ablation
#
# Scientific question: For sparse clinical lab variables, which preprocessing
# strategy gives the best balance of causality, patient-specific preservation,
# predictive performance, and suitability for real-time deployment?
#
# Strategies:
#   A: Causal forward-fill + population median (value only)
#   B: A + observation mask + time-since-last-observation (missingness-aware)
#   C: Forward-fill + patient-first back-fill (non-causal, value only)
#   D: C + mask + delta (non-causal, missingness-aware)
#   E: Patient-wide mean fill (non-causal, value only)
#
# Frozen: split, Transformer arch, optimizer, LR, scheduler, loss, epochs,
#         early stopping, vital-sign preprocessing.
# Ablation differs ONLY in how sparse lab variables are handled.

import os
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    brier_score_loss,
)

from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import collate_fn, CLIP_RANGES
from sepsentinel.model_b.training import Trainer, SequenceDataset
from sepsentinel.model_b.transformer import SepsisTransformer

# ============================================================
# Configuration
# ============================================================

FEATURES = [
    "heart_rate", "spo2", "respiratory_rate", "temperature",
    "lactate", "ph", "creatinine", "wbc", "platelets", "bilirubin",
]
N_VITALS = 4
N_LABS = 6

STRATEGIES = ["A", "B", "C", "D", "E"]
SEEDS = [42, 123, 456]
SPLIT_SEED = 42

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 7
MIN_DELTA = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

RESULTS_DIR = "results/experiment2_imputation"

# Clinical metadata
FEATURE_META = {
    "heart_rate":       {"timescale": "seconds-minutes", "changing": "fast",     "type": "vital"},
    "spo2":             {"timescale": "seconds-minutes", "changing": "fast",     "type": "vital"},
    "respiratory_rate": {"timescale": "seconds-minutes", "changing": "fast",     "type": "vital"},
    "temperature":      {"timescale": "hours",           "changing": "moderate", "type": "vital"},
    "lactate":          {"timescale": "minutes-hours",   "changing": "fast",     "type": "lab"},
    "ph":               {"timescale": "hours",           "changing": "moderate", "type": "lab"},
    "creatinine":       {"timescale": "hours-days",      "changing": "slow",     "type": "lab"},
    "wbc":              {"timescale": "hours-days",      "changing": "slow",     "type": "lab"},
    "platelets":        {"timescale": "days",            "changing": "slow",     "type": "lab"},
    "bilirubin":        {"timescale": "days",            "changing": "slow",     "type": "lab"},
}


# ============================================================
# Feature Statistics
# ============================================================

def report_feature_statistics(episodes):
    """Print detailed per-feature statistics."""
    n_patients = len(episodes)
    print(f"\n  Feature Statistics ({n_patients} patients)")
    print(f"  {'='*90}")
    print(f"  {'Feature':>18s} | NaN%  | MedCount | %Pat w/0 | Timescale      | Changing  | MedGap")
    print(f"  {'-'*90}")

    all_signals = np.concatenate([e["signals"] for e in episodes], axis=0)

    for j, feat in enumerate(FEATURES):
        col = all_signals[:, j]
        nan_pct = np.isnan(col).mean() * 100

        # Per-patient stats
        counts = []
        gaps_all = []
        n_zero = 0
        for ep in episodes:
            observed = ~np.isnan(ep["signals"][:, j])
            n_obs = observed.sum()
            counts.append(n_obs)
            if n_obs == 0:
                n_zero += 1
            elif n_obs >= 2:
                obs_idx = np.where(observed)[0]
                gaps = np.diff(obs_idx)
                gaps_all.extend(gaps.tolist())

        med_count = np.median(counts)
        pct_zero = n_zero / n_patients * 100
        med_gap = np.median(gaps_all) if gaps_all else float('nan')

        meta = FEATURE_META[feat]
        print(f"  {feat:>18s} | {nan_pct:4.1f}% | {med_count:8.1f} | {pct_zero:7.1f}% | "
              f"{meta['timescale']:14s} | {meta['changing']:9s} | {med_gap:.1f}h")

    print(f"  {'='*90}")


# ============================================================
# Preprocessing Strategy
# ============================================================

class StrategyPreprocessor:
    """Unified preprocessor implementing strategies A-E.

    Vitals (indices 0..3) always get: causal forward-fill + training median.
    Labs (indices 4..9) differ by strategy:
      A: forward-fill + training median (value only)
      B: A + mask + delta
      C: forward-fill + patient-first back-fill (value only)
      D: C + mask + delta
      E: patient-wide mean fill (value only)
    """

    def __init__(self, strategy, features):
        assert strategy in ("A", "B", "C", "D", "E")
        self.strategy = strategy
        self.features = features
        self.n_features = len(features)

        self.has_missingness = strategy in ("B", "D")
        if self.has_missingness:
            # vitals(4) + lab_values(6) + lab_masks(6) + lab_deltas(6) = 22
            self.n_channels = N_VITALS + N_LABS * 3
        else:
            self.n_channels = self.n_features  # 10

        # Training statistics (fitted on train only)
        self.train_medians = None
        self.val_mean = None
        self.val_std = None
        self.delta_mean = None
        self.delta_std = None

    def fit(self, episodes):
        """Compute training statistics."""
        all_raw = np.concatenate([e["signals"] for e in episodes], axis=0)
        self.train_medians = np.nanmedian(all_raw, axis=0)

        # Process all training episodes to compute normalization stats
        all_values = []
        all_deltas = []
        for ep in episodes:
            channels = self._fill_episode(ep["signals"])
            values = channels[:, :self.n_features]
            all_values.append(values)
            if self.has_missingness:
                deltas = channels[:, self.n_features + N_LABS:]
                all_deltas.append(deltas)

        all_values = np.concatenate(all_values, axis=0)
        self.val_mean = all_values.mean(axis=0)
        self.val_std = all_values.std(axis=0)
        self.val_std[self.val_std == 0] = 1.0

        if self.has_missingness:
            all_deltas = np.concatenate(all_deltas, axis=0)
            self.delta_mean = all_deltas.mean(axis=0)
            self.delta_std = all_deltas.std(axis=0)
            self.delta_std[self.delta_std == 0] = 1.0

        return self

    def transform(self, episodes):
        """Apply preprocessing to episodes."""
        results = []
        for ep in episodes:
            channels = self._fill_episode(ep["signals"])
            values = channels[:, :self.n_features]

            # Z-score normalize values
            values_norm = (values - self.val_mean) / self.val_std

            if self.has_missingness:
                masks = channels[:, self.n_features:self.n_features + N_LABS]
                deltas = channels[:, self.n_features + N_LABS:]
                deltas_norm = (deltas - self.delta_mean) / self.delta_std

                # Layout: [vitals(4), lab_values(6), lab_masks(6), lab_deltas(6)]
                expanded = np.concatenate([
                    values_norm[:, :N_VITALS],
                    values_norm[:, N_VITALS:],
                    masks,
                    deltas_norm,
                ], axis=1).astype(np.float32)
            else:
                expanded = values_norm.astype(np.float32)

            results.append({
                "signals": expanded,
                "labels": ep["labels"].astype(np.float32),
                "length": len(expanded),
                "patient_id": ep["patient_id"],
                "label": ep["label"],
            })
        return results

    def fit_transform(self, episodes):
        return self.fit(episodes).transform(episodes)

    def _fill_episode(self, raw_signals):
        """Apply strategy-specific filling. Returns filled channels."""
        signals = raw_signals.copy()
        n_steps, n_feat = signals.shape

        # Record lab observation mask BEFORE any filling
        lab_observed = (~np.isnan(signals[:, N_VITALS:])).astype(np.float32)

        if self.strategy in ("A", "B"):
            self._fill_strategy_a(signals, n_steps)

        elif self.strategy in ("C", "D"):
            self._fill_strategy_c(signals, n_steps)

        elif self.strategy == "E":
            self._fill_strategy_e(signals, n_steps)

        # Clip to physiological ranges
        for j, feat in enumerate(self.features):
            if feat in CLIP_RANGES:
                lo, hi = CLIP_RANGES[feat]
                signals[:, j] = np.clip(signals[:, j], lo, hi)

        if self.has_missingness:
            # Compute deltas for labs
            lab_deltas = np.zeros((n_steps, N_LABS), dtype=np.float32)
            for j in range(N_LABS):
                last_obs = -1
                for i in range(n_steps):
                    if lab_observed[i, j] == 1.0:
                        lab_deltas[i, j] = 0.0
                        last_obs = i
                    else:
                        if last_obs >= 0:
                            lab_deltas[i, j] = float(i - last_obs)
                        else:
                            # Leading NaN: 1-indexed elapsed from sequence start
                            lab_deltas[i, j] = float(i + 1)

            # Return: [values(10), masks(6), deltas(6)]
            return np.concatenate([signals, lab_observed, lab_deltas], axis=1)
        else:
            return signals

    def _fill_strategy_a(self, signals, n_steps):
        """Forward-fill all + training median for remaining."""
        n_feat = signals.shape[1]
        # Forward-fill
        for j in range(n_feat):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]
        # Fill remaining (leading NaNs) with training median
        for j in range(n_feat):
            nans = np.isnan(signals[:, j])
            if nans.any():
                signals[nans, j] = self.train_medians[j]

    def _fill_strategy_c(self, signals, n_steps):
        """Vitals: forward-fill + median. Labs: forward-fill + patient-first back-fill."""
        # Vitals: same as strategy A
        for j in range(N_VITALS):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]
            nans = np.isnan(signals[:, j])
            if nans.any():
                signals[nans, j] = self.train_medians[j]

        # Labs: forward-fill + patient-first back-fill
        for j in range(N_VITALS, signals.shape[1]):
            # Forward-fill
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]
            # Back-fill leading NaNs with patient's first observation
            if np.isnan(signals[0, j]):
                first_val = None
                for i in range(n_steps):
                    if not np.isnan(signals[i, j]):
                        first_val = signals[i, j]
                        break
                if first_val is not None:
                    for i in range(n_steps):
                        if np.isnan(signals[i, j]):
                            signals[i, j] = first_val
                        else:
                            break
                else:
                    # Patient has no observations -> training median
                    signals[:, j] = self.train_medians[j]

    def _fill_strategy_e(self, signals, n_steps):
        """Vitals: forward-fill + median. Labs: patient-wide mean fill."""
        # Vitals: same as strategy A
        for j in range(N_VITALS):
            for i in range(1, n_steps):
                if np.isnan(signals[i, j]):
                    signals[i, j] = signals[i - 1, j]
            nans = np.isnan(signals[:, j])
            if nans.any():
                signals[nans, j] = self.train_medians[j]

        # Labs: fill with patient-wide mean of available measurements
        for j in range(N_VITALS, signals.shape[1]):
            col = signals[:, j]
            non_nan = col[~np.isnan(col)]
            if len(non_nan) > 0:
                patient_mean = non_nan.mean()
                nans = np.isnan(col)
                signals[nans, j] = patient_mean
            else:
                signals[:, j] = self.train_medians[j]


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_prob, threshold):
    """Compute all requested evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prevalence = float(y_true.mean())
    auprc_lift = auprc / prevalence if prevalence > 0 else 0

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    specificity = float(tn / max(tn + fp, 1))
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fp_per_1000 = float(fp / len(y_true) * 1000)
    pred_pos_rate = float(y_pred.mean())
    brier = brier_score_loss(y_true, y_prob)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "auprc_lift": float(auprc_lift),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": specificity,
        "f1": float(f1),
        "fp_per_1000": fp_per_1000,
        "pred_pos_rate": pred_pos_rate,
        "brier": float(brier),
        "threshold": float(threshold),
        "prevalence": prevalence,
    }


def find_best_f1_threshold(y_true, y_prob):
    """Find threshold maximizing F1 on validation predictions."""
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.01, 0.99, 0.005):
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def collect_predictions(model, data, device):
    """Run model on data and collect unpadded predictions."""
    loader = DataLoader(SequenceDataset(data), batch_size=BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)
    all_probs, all_labels = [], []
    with torch.no_grad():
        model.eval()
        for signals, labels, lengths, mask in loader:
            signals = signals.to(device)
            logits = model(signals, lengths)
            probs = torch.sigmoid(logits)
            for i in range(len(lengths)):
                sl = lengths[i].item()
                all_probs.append(probs[i, :sl].cpu().numpy())
                all_labels.append(labels[i, :sl].numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ============================================================
# Single Run
# ============================================================

def run_single(strategy, preprocessor, train_eps, val_eps, test_eps,
               seed, device):
    """Train and evaluate one strategy+seed combination."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Preprocess (preprocessor is already fitted on train)
    train_data = preprocessor.transform(train_eps)
    val_data = preprocessor.transform(val_eps)
    test_data = preprocessor.transform(test_eps)

    # Pos weight from training labels
    train_labels = np.concatenate([e["labels"] for e in train_data])
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = float(n_neg / max(n_pos, 1))

    # Model
    input_dim = preprocessor.n_channels
    model = SepsisTransformer(input_dim=input_dim)
    n_params = sum(p.numel() for p in model.parameters())

    # Checkpoint dir
    ckpt_dir = os.path.join(RESULTS_DIR, f"strategy_{strategy}", f"seed_{seed}")

    # Train
    trainer = Trainer(
        model, device=device,
        checkpoint_dir=ckpt_dir,
        pos_weight=pos_weight,
    )

    t0 = time.time()
    trainer.fit(
        train_data, val_data,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE, min_delta=MIN_DELTA,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
    )
    train_time = time.time() - t0

    # Validation: find best threshold + metrics
    val_labels, val_probs = collect_predictions(model, val_data, device)
    best_threshold = find_best_f1_threshold(val_labels, val_probs)
    val_metrics = compute_metrics(val_labels, val_probs, best_threshold)

    # Test: evaluate at best validation threshold
    test_labels, test_probs = collect_predictions(model, test_data, device)
    test_metrics = compute_metrics(test_labels, test_probs, best_threshold)

    result = {
        "strategy": strategy,
        "seed": seed,
        "input_dim": input_dim,
        "n_params": n_params,
        "train_time_sec": round(train_time, 1),
        "threshold": best_threshold,
        "val": val_metrics,
        "test": test_metrics,
    }

    print(f"    Val  AUROC={val_metrics['auroc']:.4f}  AUPRC={val_metrics['auprc']:.4f}  "
          f"F1={val_metrics['f1']:.4f}")
    print(f"    Test AUROC={test_metrics['auroc']:.4f}  AUPRC={test_metrics['auprc']:.4f}  "
          f"F1={test_metrics['f1']:.4f}  Brier={test_metrics['brier']:.4f}")

    return result


# ============================================================
# Comparison Table
# ============================================================

def print_comparison(all_results):
    """Print aggregated comparison across strategies."""
    print("\n" + "=" * 100)
    print("  COMPARISON TABLE (mean +/- std across seeds)")
    print("=" * 100)

    strategies_done = sorted(set(r["strategy"] for r in all_results))

    # Header
    header = (f"  {'Strategy':>8s} | {'Causal':>6s} | {'Channels':>8s} | "
              f"{'AUROC':>14s} | {'AUPRC':>14s} | {'F1':>14s} | "
              f"{'Brier':>14s} | {'FP/1000':>14s} | {'Time':>6s}")
    print(header)
    print(f"  {'-' * 96}")

    causal_map = {"A": "Yes", "B": "Yes", "C": "No", "D": "No", "E": "No"}
    summary_rows = []

    for strat in strategies_done:
        runs = [r for r in all_results if r["strategy"] == strat]
        n_seeds = len(runs)

        aurocs = [r["test"]["auroc"] for r in runs]
        auprcs = [r["test"]["auprc"] for r in runs]
        f1s = [r["test"]["f1"] for r in runs]
        briers = [r["test"]["brier"] for r in runs]
        fps = [r["test"]["fp_per_1000"] for r in runs]
        times = [r["train_time_sec"] for r in runs]
        channels = runs[0]["input_dim"]

        def fmt(vals):
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"

        row = (f"  {strat:>8s} | {causal_map[strat]:>6s} | {channels:>8d} | "
               f"{fmt(aurocs):>14s} | {fmt(auprcs):>14s} | {fmt(f1s):>14s} | "
               f"{fmt(briers):>14s} | {fmt([round(x,1) for x in fps]):>14s} | "
               f"{np.mean(times):5.0f}s")
        print(row)

        summary_rows.append({
            "strategy": strat,
            "causal": causal_map[strat],
            "channels": channels,
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auprc_mean": float(np.mean(auprcs)),
            "auprc_std": float(np.std(auprcs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "brier_mean": float(np.mean(briers)),
            "brier_std": float(np.std(briers)),
            "fp1000_mean": float(np.mean(fps)),
            "n_seeds": n_seeds,
        })

    return summary_rows


def answer_questions(summary):
    """Print answers to the 7 primary comparison questions."""
    print("\n" + "=" * 100)
    print("  PRIMARY COMPARISONS")
    print("=" * 100)

    def get(strat):
        return next(s for s in summary if s["strategy"] == strat)

    a, b, c, d, e = get("A"), get("B"), get("C"), get("D"), get("E")

    print(f"""
  1. Does mask+delta improve over value-only causal imputation?
     Strategy A (value-only):   AUROC={a['auroc_mean']:.4f}, AUPRC={a['auprc_mean']:.4f}
     Strategy B (mask+delta):   AUROC={b['auroc_mean']:.4f}, AUPRC={b['auprc_mean']:.4f}
     Delta AUROC: {b['auroc_mean'] - a['auroc_mean']:+.4f}
     Delta AUPRC: {b['auprc_mean'] - a['auprc_mean']:+.4f}

  2. Does first-observation back-fill improve retrospective metrics?
     Strategy A (causal):       AUROC={a['auroc_mean']:.4f}, AUPRC={a['auprc_mean']:.4f}
     Strategy C (back-fill):    AUROC={c['auroc_mean']:.4f}, AUPRC={c['auprc_mean']:.4f}
     Delta AUROC: {c['auroc_mean'] - a['auroc_mean']:+.4f}

  3. How much improvement is attributable to non-causal future info?
     Causal best (B):       AUROC={b['auroc_mean']:.4f}
     Non-causal best (D):   AUROC={d['auroc_mean']:.4f}
     Gap: {d['auroc_mean'] - b['auroc_mean']:+.4f}

  4. Does patient-mean filling inflate performance?
     Strategy A (causal):       AUROC={a['auroc_mean']:.4f}
     Strategy E (patient-mean): AUROC={e['auroc_mean']:.4f}
     Delta: {e['auroc_mean'] - a['auroc_mean']:+.4f}

  5. Best method for strict real-time deployment:
     Among causal strategies (A, B):
     A: AUROC={a['auroc_mean']:.4f}, AUPRC={a['auprc_mean']:.4f}
     B: AUROC={b['auroc_mean']:.4f}, AUPRC={b['auprc_mean']:.4f}

  6. Best method for retrospective benchmarking only:
     Among all strategies:
     C: AUROC={c['auroc_mean']:.4f}  D: AUROC={d['auroc_mean']:.4f}  E: AUROC={e['auroc_mean']:.4f}

  7. Fast-changing vs slow-changing labs:
     (Requires Strategy F - run after A-E if needed)
""")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 100)
    print("  EXPERIMENT 2: SPARSE-LAB IMPUTATION STRATEGY ABLATION")
    print("  Feature set: 4 vitals + 6 labs = 10 features")
    print("  Strategies: A (causal), B (causal+miss), C (backfill),")
    print("              D (backfill+miss), E (patient-mean)")
    print("  Seeds:", SEEDS)
    print("=" * 100)

    # -- Load data --
    print("\nLoading PhysioNet data (10 features)...")
    episodes = load_physionet(features=FEATURES, min_length=6)
    print(f"  {len(episodes)} episodes loaded")

    # -- Feature statistics --
    report_feature_statistics(episodes)

    # -- Split --
    print("\nSplitting (patient-level, seed=42)...")
    splits = patient_split(episodes, random_state=SPLIT_SEED)
    for name in ["train", "val", "test"]:
        n = len(splits[name])
        n_sep = sum(1 for e in splits[name] if e["label"] == 1)
        total_steps = sum(e["signals"].shape[0] for e in splits[name])
        print(f"  {name:5s}: {n} patients ({n_sep} septic), {total_steps:,} timesteps")

    # -- Run strategies --
    all_results = []
    results_path = os.path.join(RESULTS_DIR, "all_results.json")

    # Check for existing results to support resuming
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
        done = {(r["strategy"], r["seed"]) for r in all_results}
        print(f"\n  Resuming: {len(done)} runs already completed")
    else:
        done = set()

    for strategy in STRATEGIES:
        print(f"\n{'='*100}")
        print(f"  STRATEGY {strategy}")
        if strategy in ("C", "D", "E"):
            print("  ** NON-CAUSAL: uses future information. For retrospective comparison only. **")
        print(f"{'='*100}")

        # Fit preprocessor on training data (once per strategy)
        preprocessor = StrategyPreprocessor(strategy, FEATURES)
        preprocessor.fit(splits["train"])
        print(f"  Input channels: {preprocessor.n_channels}")

        for seed in SEEDS:
            if (strategy, seed) in done:
                print(f"\n  Seed {seed}: already completed, skipping")
                continue

            print(f"\n  Seed {seed}:")
            result = run_single(
                strategy, preprocessor,
                splits["train"], splits["val"], splits["test"],
                seed, device,
            )
            all_results.append(result)

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

    # -- Comparison --
    summary = print_comparison(all_results)

    # -- Answer questions --
    answer_questions(summary)

    # -- Save summary --
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All results saved to {RESULTS_DIR}/")
    print(f"  - all_results.json  (per-run details)")
    print(f"  - summary.json      (aggregated comparison)")


if __name__ == "__main__":
    main()
