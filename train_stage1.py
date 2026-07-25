# Module 7: Train and compare all Model B architectures on PhysioNet Stage 1.
#
# Models:
#   1. Random Forest (flat baseline)
#   2. XGBoost (flat baseline)
#   3. GRU (sequential)
#   4. TCN (sequential)
#   5. Transformer (sequential)
#
# Stage 1 features: HR, SpO2, Temp, RR (4 physiological signals)
# Dataset: PhysioNet/CinC 2019 Sepsis Challenge
#
# Usage:
#   python train_stage1.py                  # train all models
#   python train_stage1.py --models gru tcn # train specific models
#   python train_stage1.py --epochs 30      # override epochs

import argparse
import os
import time

import numpy as np
import torch

from sepsentinel.config.signals import STAGES
from sepsentinel.data.physionet import load_physionet, print_physionet_summary
from sepsentinel.data.splitting import patient_split, print_split_summary
from sepsentinel.data.preprocessing import SequencePreprocessor
from sepsentinel.model_b.registry import get_model
from sepsentinel.model_b.training import Trainer
from sepsentinel.model_b.evaluation import evaluate_on_test, print_evaluation
from sepsentinel.model_b.gru import SepsisGRU
from sepsentinel.model_b.tcn import SepsisTCN
from sepsentinel.model_b.transformer import SepsisTransformer

RESULTS_DIR = "results/stage1"
CHECKPOINT_DIR = "models/checkpoints"


def compute_pos_weight(train_episodes):
    """Compute class weight ratio for BCEWithLogitsLoss."""
    all_labels = np.concatenate([e["labels"] for e in train_episodes])
    n_pos = (all_labels == 1).sum()
    n_neg = (all_labels == 0).sum()
    if n_pos == 0:
        return 1.0
    return float(n_neg / n_pos)


def train_flat_model(model_name, train_data, test_data, features):
    """Train a flat model (RF or XGBoost) on flattened per-timestep features."""
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")

    # Flatten episodes to per-timestep samples
    X_train = np.concatenate([e["signals"] for e in train_data])
    y_train = np.concatenate([e["labels"] for e in train_data])
    X_test = np.concatenate([e["signals"] for e in test_data])
    y_test = np.concatenate([e["labels"] for e in test_data])

    print(f"  Train samples: {len(X_train):,} ({y_train.sum():,.0f} positive, "
          f"{y_train.mean()*100:.1f}% sepsis)")
    print(f"  Test samples:  {len(X_test):,}")

    # For imbalanced data, set scale_pos_weight for XGBoost
    kwargs = {}
    if model_name == "xgboost":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        kwargs["scale_pos_weight"] = float(n_neg / max(n_pos, 1))

    model = get_model(model_name, **kwargs)

    t0 = time.time()
    metrics = model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Internal accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"\n{metrics['report']}")

    # Evaluate on held-out test set
    y_probs = model.predict_proba(X_test)
    y_preds = (y_probs >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score, confusion_matrix,
    )

    test_metrics = {
        "auroc": roc_auc_score(y_test, y_probs),
        "auprc": average_precision_score(y_test, y_probs),
        "accuracy": accuracy_score(y_test, y_preds),
        "precision": precision_score(y_test, y_preds, zero_division=0),
        "recall": recall_score(y_test, y_preds, zero_division=0),
        "f1": f1_score(y_test, y_preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_preds),
        "training_time": elapsed,
    }

    print(f"\n  Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Test AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  Test F1:    {test_metrics['f1']:.4f}")

    return model, test_metrics


def train_sequential_model(model_class, model_name, train_data, val_data, test_data,
                           n_features, pos_weight, device, epochs, batch_size):
    """Train a PyTorch sequential model (GRU, TCN, Transformer)."""
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")

    model = model_class(input_dim=n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device}")
    print(f"  Pos weight: {pos_weight:.1f}")

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name)
    trainer = Trainer(model, device=device, checkpoint_dir=checkpoint_dir,
                      pos_weight=pos_weight)

    t0 = time.time()
    history = trainer.fit(
        train_data, val_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        patience=7,
    )
    elapsed = time.time() - t0

    print(f"\n  Total training time: {elapsed:.1f}s")

    # Evaluate on test set
    test_metrics = evaluate_on_test(model, test_data, batch_size=batch_size,
                                     device=device)
    test_metrics["training_time"] = elapsed
    test_metrics["n_params"] = n_params
    test_metrics["history"] = history

    print_evaluation(test_metrics)

    return model, test_metrics


def print_comparison(all_results):
    """Print a summary table comparing all models."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON — Stage 1 (HR, SpO2, Temp, RR)")
    print(f"{'='*70}")
    print(f"  {'Model':<15s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} "
          f"{'Recall':>7s} {'Prec':>7s} {'Time':>8s}")
    print(f"  {'-'*60}")

    for name, metrics in sorted(all_results.items(),
                                key=lambda x: x[1]["auroc"], reverse=True):
        t = metrics["training_time"]
        time_str = f"{t:.0f}s" if t < 120 else f"{t/60:.1f}m"
        print(f"  {name:<15s} {metrics['auroc']:>7.4f} {metrics['auprc']:>7.4f} "
              f"{metrics['f1']:>7.4f} {metrics['recall']:>7.4f} "
              f"{metrics['precision']:>7.4f} {time_str:>8s}")

    print(f"{'='*70}")

    best_name = max(all_results, key=lambda k: all_results[k]["auroc"])
    print(f"\n  Best by AUROC: {best_name} ({all_results[best_name]['auroc']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train Model B Stage 1")
    parser.add_argument("--models", nargs="+",
                        default=["random_forest", "xgboost", "gru", "tcn", "transformer"],
                        help="Models to train")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs for sequential models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--min-length", type=int, default=6,
                        help="Minimum episode length in hours")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Load data ---
    print("Loading PhysioNet Stage 1 data...")
    episodes = load_physionet(stage=1, min_length=args.min_length)
    print_physionet_summary(episodes)

    # --- Split ---
    print("\nSplitting data (patient-level stratified)...")
    splits = patient_split(episodes)
    print_split_summary(splits)

    # --- Preprocess ---
    features = STAGES[1]
    print(f"\nPreprocessing (features: {features})...")
    preprocessor = SequencePreprocessor(features)
    train_data = preprocessor.fit_transform(splits["train"])
    val_data = preprocessor.transform(splits["val"])
    test_data = preprocessor.transform(splits["test"])

    n_features = len(features)
    pos_weight = compute_pos_weight(train_data)
    print(f"  Pos weight (neg/pos ratio): {pos_weight:.1f}")

    # --- Train models ---
    all_results = {}

    flat_models = {"random_forest", "xgboost"}
    seq_models = {
        "gru": SepsisGRU,
        "tcn": SepsisTCN,
        "transformer": SepsisTransformer,
    }

    for model_name in args.models:
        if model_name in flat_models:
            _, metrics = train_flat_model(model_name, train_data, test_data, features)
            all_results[model_name] = metrics

        elif model_name in seq_models:
            _, metrics = train_sequential_model(
                seq_models[model_name], model_name,
                train_data, val_data, test_data,
                n_features=n_features,
                pos_weight=pos_weight,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            all_results[model_name] = metrics

        else:
            print(f"\n  WARNING: Unknown model '{model_name}', skipping.")

    # --- Comparison ---
    if len(all_results) > 1:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
