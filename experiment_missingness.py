# Experiment 1: Causal Missing-Value Representation
#
# Controlled comparison: identical Transformer, optimizer, LR, scheduler,
# loss, epochs, seed, and patient split. The ONLY change is preprocessing:
#
# Baseline (4 channels):  HR, SpO2, Temp, RR
#                         Back-fill + forward-fill + mean impute + z-score
#
# Experiment (12 channels): HR, HR_mask, HR_delta,
#                           SpO2, SpO2_mask, SpO2_delta,
#                           Temp, Temp_mask, Temp_delta,
#                           RR, RR_mask, RR_delta
#                           Forward-fill only + mean impute + z-score
#                           mask: 1=observed, 0=imputed
#                           delta: timesteps since last observation
#
# Threshold tuning does NOT change AUROC or AUPRC.

import os
import time

import numpy as np
import torch

from sepsentinel.config.signals import STAGES
from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import SequencePreprocessor
from sepsentinel.data.preprocessing_missingness import MissingnessAwarePreprocessor
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.evaluation import evaluate_on_test
from sepsentinel.model_b.transformer import SepsisTransformer

RESULTS_DIR = "results/experiment1_missingness"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Reproducibility --
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -- Hyperparameters (frozen — identical to baseline) --
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 7
MIN_DELTA = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_pos_weight(episodes):
    all_labels = np.concatenate([e["labels"] for e in episodes])
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    return float(n_neg / max(n_pos, 1))


def find_best_f1_threshold(model, val_data, device):
    """Find best-F1 threshold on validation set."""
    from torch.utils.data import DataLoader
    from sepsentinel.data.preprocessing import collate_fn
    from sepsentinel.model_b.training import SequenceDataset

    loader = DataLoader(SequenceDataset(val_data), batch_size=BATCH_SIZE,
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

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    from sklearn.metrics import roc_auc_score, average_precision_score

    val_auroc = roc_auc_score(all_labels, all_probs)
    val_auprc = average_precision_score(all_labels, all_probs)

    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (all_probs >= t).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return val_auroc, val_auprc, best_t, best_f1


def full_test_eval(model, test_data, device, threshold=0.5):
    """Evaluate on test set with a specific threshold."""
    from torch.utils.data import DataLoader
    from sepsentinel.data.preprocessing import collate_fn
    from sepsentinel.model_b.training import SequenceDataset
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, confusion_matrix,
    )

    loader = DataLoader(SequenceDataset(test_data), batch_size=BATCH_SIZE,
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

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= threshold).astype(int)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    spec_tn = ((all_preds == 0) & (all_labels == 0)).sum()
    spec_fp = ((all_preds == 1) & (all_labels == 0)).sum()
    specificity = spec_tn / max(spec_tn + spec_fp, 1)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    fp_per_1000 = spec_fp / len(all_labels) * 1000
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "auroc": auroc, "auprc": auprc,
        "precision": prec, "recall": rec,
        "specificity": specificity, "f1": f1,
        "fp_per_1000": fp_per_1000,
        "confusion_matrix": cm,
        "threshold": threshold,
        "n_timesteps": len(all_labels),
        "n_positive": int(all_labels.sum()),
    }


def print_results(label, val_metrics, test_metrics):
    print(f"\n  {label}")
    print(f"  {'-' * 55}")
    print(f"  Validation:")
    print(f"    AUROC:          {val_metrics['auroc']:.4f}")
    print(f"    AUPRC:          {val_metrics['auprc']:.4f}")
    print(f"    Best threshold: {val_metrics['threshold']:.3f}")
    print(f"    F1 (at best t): {val_metrics['f1']:.4f}")
    print(f"  Held-out Test (at best validation threshold):")
    print(f"    AUROC:          {test_metrics['auroc']:.4f}")
    print(f"    AUPRC:          {test_metrics['auprc']:.4f}")
    print(f"    Precision:      {test_metrics['precision']:.4f}")
    print(f"    Recall:         {test_metrics['recall']:.4f}")
    print(f"    Specificity:    {test_metrics['specificity']:.4f}")
    print(f"    F1:             {test_metrics['f1']:.4f}")
    print(f"    FP/1000:        {test_metrics['fp_per_1000']:.1f}")
    cm = test_metrics['confusion_matrix']
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")


def main():
    print("=" * 70)
    print("  EXPERIMENT 1: CAUSAL MISSING-VALUE REPRESENTATION")
    print("  Only preprocessing changes. Everything else is frozen.")
    print("=" * 70)

    # -- Load data --
    print("\nLoading PhysioNet Stage 1 data...")
    episodes = load_physionet(stage=1, min_length=6)
    features = STAGES[1]

    print(f"  {len(episodes)} episodes, features: {features}")

    # -- Same split for both runs --
    print("Splitting (patient-level, seed=42)...")
    splits = patient_split(episodes, random_state=42)
    for name in ["train", "val", "test"]:
        n = len(splits[name])
        n_sep = sum(1 for e in splits[name] if e["label"] == 1)
        print(f"  {name:5s}: {n} patients ({n_sep} septic)")

    # ══════════════════════════════════════════════════════
    #  ARM 1: BASELINE (original preprocessing, 4 channels)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ARM 1: BASELINE (4 channels, with back-fill)")
    print("=" * 70)

    # Reset seeds before each arm
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    baseline_prep = SequencePreprocessor(features)
    baseline_train = baseline_prep.fit_transform(splits["train"])
    baseline_val = baseline_prep.transform(splits["val"])
    baseline_test = baseline_prep.transform(splits["test"])

    pos_weight = compute_pos_weight(baseline_train)
    print(f"  Pos weight: {pos_weight:.1f}")
    print(f"  Input channels: {baseline_train[0]['signals'].shape[1]}")

    baseline_model = SepsisTransformer(input_dim=4)
    n_params_baseline = sum(p.numel() for p in baseline_model.parameters())
    print(f"  Parameters: {n_params_baseline:,}")

    baseline_trainer = Trainer(
        baseline_model, device=DEVICE,
        checkpoint_dir=os.path.join(RESULTS_DIR, "checkpoints_baseline"),
        pos_weight=pos_weight,
    )

    t0 = time.time()
    baseline_history = baseline_trainer.fit(
        baseline_train, baseline_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE, min_delta=MIN_DELTA,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
    )
    baseline_time = time.time() - t0
    print(f"  Training time: {baseline_time:.1f}s")

    # Validation metrics + best threshold
    b_val_auroc, b_val_auprc, b_best_t, b_best_f1 = find_best_f1_threshold(
        baseline_model, baseline_val, DEVICE)
    baseline_val_metrics = {
        "auroc": b_val_auroc, "auprc": b_val_auprc,
        "threshold": b_best_t, "f1": b_best_f1,
    }

    # Test evaluation at best-F1 threshold
    baseline_test_metrics = full_test_eval(
        baseline_model, baseline_test, DEVICE, threshold=b_best_t)

    print_results("BASELINE", baseline_val_metrics, baseline_test_metrics)

    # ══════════════════════════════════════════════════════
    #  ARM 2: MISSINGNESS (new preprocessing, 12 channels)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ARM 2: MISSINGNESS-AWARE (12 channels, causal only)")
    print("=" * 70)

    # Reset seeds for identical initialization
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    miss_prep = MissingnessAwarePreprocessor(features)
    miss_train = miss_prep.fit_transform(splits["train"])
    miss_val = miss_prep.transform(splits["val"])
    miss_test = miss_prep.transform(splits["test"])

    # pos_weight is computed from labels, which are identical
    print(f"  Pos weight: {pos_weight:.1f} (same labels)")
    print(f"  Input channels: {miss_train[0]['signals'].shape[1]}")

    miss_model = SepsisTransformer(input_dim=12)
    n_params_miss = sum(p.numel() for p in miss_model.parameters())
    print(f"  Parameters: {n_params_miss:,}")
    print(f"  Parameter increase: {n_params_miss - n_params_baseline:,} "
          f"({(n_params_miss - n_params_baseline) / n_params_baseline * 100:.1f}%)")

    miss_trainer = Trainer(
        miss_model, device=DEVICE,
        checkpoint_dir=os.path.join(RESULTS_DIR, "checkpoints_missingness"),
        pos_weight=pos_weight,
    )

    t0 = time.time()
    miss_history = miss_trainer.fit(
        miss_train, miss_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE, min_delta=MIN_DELTA,
        scheduler_factor=SCHEDULER_FACTOR,
        scheduler_patience=SCHEDULER_PATIENCE,
    )
    miss_time = time.time() - t0
    print(f"  Training time: {miss_time:.1f}s")

    # Validation metrics + best threshold
    m_val_auroc, m_val_auprc, m_best_t, m_best_f1 = find_best_f1_threshold(
        miss_model, miss_val, DEVICE)
    miss_val_metrics = {
        "auroc": m_val_auroc, "auprc": m_val_auprc,
        "threshold": m_best_t, "f1": m_best_f1,
    }

    # Test evaluation at best-F1 threshold
    miss_test_metrics = full_test_eval(
        miss_model, miss_test, DEVICE, threshold=m_best_t)

    print_results("MISSINGNESS-AWARE", miss_val_metrics, miss_test_metrics)

    # ══════════════════════════════════════════════════════
    #  COMPARISON
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  Threshold tuning does NOT change AUROC or AUPRC.")
    print(f"  These are threshold-independent ranking metrics.\n")

    print(f"  {'Metric':<22s} {'Baseline':>10s} {'Missingness':>12s} {'Delta':>10s} {'Rel %':>8s}")
    print(f"  {'-' * 65}")

    comparisons = [
        ("Val AUROC",      baseline_val_metrics["auroc"],     miss_val_metrics["auroc"]),
        ("Val AUPRC",      baseline_val_metrics["auprc"],     miss_val_metrics["auprc"]),
        ("Val Best F1",    baseline_val_metrics["f1"],        miss_val_metrics["f1"]),
        ("Test AUROC",     baseline_test_metrics["auroc"],    miss_test_metrics["auroc"]),
        ("Test AUPRC",     baseline_test_metrics["auprc"],    miss_test_metrics["auprc"]),
        ("Test Precision",  baseline_test_metrics["precision"], miss_test_metrics["precision"]),
        ("Test Recall",     baseline_test_metrics["recall"],   miss_test_metrics["recall"]),
        ("Test Specificity", baseline_test_metrics["specificity"], miss_test_metrics["specificity"]),
        ("Test F1",         baseline_test_metrics["f1"],       miss_test_metrics["f1"]),
        ("Test FP/1000",    baseline_test_metrics["fp_per_1000"], miss_test_metrics["fp_per_1000"]),
    ]

    for name, base, exp in comparisons:
        delta = exp - base
        rel = delta / max(abs(base), 1e-9) * 100
        print(f"  {name:<22s} {base:>10.4f} {exp:>12.4f} {delta:>+10.4f} {rel:>+7.1f}%")

    print(f"\n  {'-' * 65}")
    print(f"  Best val threshold:  Baseline={baseline_val_metrics['threshold']:.3f}  "
          f"Missingness={miss_val_metrics['threshold']:.3f}")
    print(f"  Parameters:          Baseline={n_params_baseline:,}  "
          f"Missingness={n_params_miss:,} "
          f"(+{n_params_miss - n_params_baseline:,})")
    print(f"  Training time:       Baseline={baseline_time:.0f}s  "
          f"Missingness={miss_time:.0f}s")

    # -- Analysis notes --
    print(f"\n  {'-' * 65}")
    print(f"  ANALYSIS NOTES")
    print(f"  {'-' * 65}")
    print(f"  1. The only change is preprocessing: 4 channels -> 12 channels.")
    print(f"     Mask and delta channels explicitly encode missingness patterns.")
    print(f"  2. The input projection layer grows from 4->64 to 12->64,")
    print(f"     adding {n_params_miss - n_params_baseline} parameters ({(n_params_miss - n_params_baseline) / n_params_baseline * 100:.1f}% increase).")
    print(f"  3. Back-fill leakage has been removed in the experiment arm.")
    print(f"  4. Delta convention for leading NaNs: i+1 (1-indexed elapsed time")
    print(f"     from sequence start). delta=0 strictly means 'observed now'.")
    print(f"  5. Mask channels are NOT normalized (binary 0/1).")
    print(f"     Delta channels are z-score normalized using training-set stats.")

    print("\n" + "=" * 70)

    # -- Save results --
    results_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(results_path, "w") as f:
        for name, base, exp in comparisons:
            delta = exp - base
            rel = delta / max(abs(base), 1e-9) * 100
            f.write(f"{name},{base:.6f},{exp:.6f},{delta:+.6f},{rel:+.2f}%\n")
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
