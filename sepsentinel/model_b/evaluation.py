# Model B evaluation — test set metrics and confusion matrix.

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from sepsentinel.data.preprocessing import collate_fn
from sepsentinel.model_b.training import SequenceDataset


@torch.no_grad()
def evaluate_on_test(model, test_data, batch_size=32, device="cpu", threshold=0.5):
    """Full evaluation on the held-out test set.

    Returns:
        dict with AUROC, AUPRC, accuracy, precision, recall, F1,
        confusion matrix, and raw predictions.
    """
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        SequenceDataset(test_data), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    all_probs = []
    all_labels = []

    for signals, labels, lengths, mask in loader:
        signals = signals.to(device)
        logits = model(signals, lengths)
        probs = torch.sigmoid(logits)

        for i in range(len(lengths)):
            seq_len = lengths[i].item()
            all_probs.append(probs[i, :seq_len].cpu().numpy())
            all_labels.append(labels[i, :seq_len].numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= threshold).astype(int)

    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "threshold": threshold,
    }


def print_evaluation(metrics):
    """Pretty-print test set evaluation results."""
    print("=" * 60)
    print("  TEST SET EVALUATION")
    print("=" * 60)
    print(f"  AUROC:      {metrics['auroc']:.4f}")
    print(f"  AUPRC:      {metrics['auprc']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Threshold:  {metrics['threshold']}")
    print()
    cm = metrics["confusion_matrix"]
    print("  Confusion Matrix:")
    print(f"                  Predicted 0    Predicted 1")
    print(f"    Actual 0     {cm[0, 0]:>10,}   {cm[0, 1]:>10,}")
    print(f"    Actual 1     {cm[1, 0]:>10,}   {cm[1, 1]:>10,}")
    print()
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("=" * 60)
