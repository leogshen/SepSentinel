# Threshold-behavior analysis for the trained Transformer model.
#
# Thresholds are selected on the VALIDATION set only.
# The held-out TEST set is evaluated exactly once per selected threshold.
# AUROC and AUPRC are threshold-independent and do not change.

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix,
)

from sepsentinel.config.signals import STAGES
from sepsentinel.data.physionet import load_physionet
from sepsentinel.data.splitting import patient_split
from sepsentinel.data.preprocessing import SequencePreprocessor, collate_fn
from sepsentinel.model_b.training import SequenceDataset
from sepsentinel.model_b.transformer import SepsisTransformer

RESULTS_DIR = "results/threshold_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ──

def collect_predictions(model, data, batch_size=32):
    """Run inference and collect valid (unpadded) predictions and labels."""
    loader = DataLoader(
        SequenceDataset(data), batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    all_probs, all_labels = [], []
    with torch.no_grad():
        for signals, labels, lengths, mask in loader:
            logits = model(signals, lengths)
            probs = torch.sigmoid(logits)
            for i in range(len(lengths)):
                sl = lengths[i].item()
                all_probs.append(probs[i, :sl].cpu().numpy())
                all_labels.append(labels[i, :sl].numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_at_threshold(labels, probs, threshold):
    """Compute all metrics for a single threshold."""
    preds = (probs >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    n = len(labels)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / n
    fpr = fp / max(fp + tn, 1)
    pred_pos_pct = (tp + fp) / n * 100
    fp_per_1000 = fp / n * 1000

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
        "pred_pos_pct": pred_pos_pct,
        "fp_per_1000": fp_per_1000,
    }


def print_threshold_row(m, label=""):
    """Print one row of threshold metrics."""
    tag = f"  [{label}]" if label else "  "
    print(f"{tag} t={m['threshold']:.3f}  "
          f"Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
          f"Spec={m['specificity']:.4f}  F1={m['f1']:.4f}  "
          f"FP/1k={m['fp_per_1000']:.1f}  PredPos={m['pred_pos_pct']:.1f}%")


def plot_confusion_matrix(m, ax, title):
    """Plot a single confusion matrix."""
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=11, color="white" if cm[i, j] > cm.max() * 0.5 else "black")


# ── Main ──

def main():
    print("=" * 70)
    print("  THRESHOLD-BEHAVIOR ANALYSIS")
    print("  Thresholds selected on VALIDATION set only.")
    print("  Test set evaluated once per selected threshold.")
    print("=" * 70)

    # ── Load and preprocess ──
    print("\nLoading data...")
    episodes = load_physionet(stage=1, min_length=6)
    splits = patient_split(episodes)
    features = STAGES[1]
    preprocessor = SequencePreprocessor(features)
    train_data = preprocessor.fit_transform(splits["train"])
    val_data = preprocessor.transform(splits["val"])
    test_data = preprocessor.transform(splits["test"])

    # ── Load model ──
    print("Loading trained Transformer...")
    model = SepsisTransformer(input_dim=len(features))
    model.load_state_dict(torch.load(
        "models/checkpoints/transformer/best_model.pt", weights_only=True))
    model.eval()

    # ── Collect predictions ──
    print("Running inference on validation set...")
    val_probs, val_labels = collect_predictions(model, val_data)
    print(f"  Val timesteps: {len(val_labels):,}  "
          f"Positive: {int(val_labels.sum()):,} ({val_labels.mean()*100:.2f}%)")

    print("Running inference on test set...")
    test_probs, test_labels = collect_predictions(model, test_data)
    print(f"  Test timesteps: {len(test_labels):,}  "
          f"Positive: {int(test_labels.sum()):,} ({test_labels.mean()*100:.2f}%)")

    # ── Threshold grid ──
    # Coarse grid + fine refinement near promising regions
    coarse = np.arange(0.01, 1.00, 0.01)
    fine = np.arange(0.05, 0.50, 0.005)  # refine the low-threshold region
    thresholds = np.unique(np.concatenate([coarse, fine]))
    thresholds.sort()

    print(f"\nEvaluating {len(thresholds)} thresholds on validation set...")

    # ── Step 3: Compute metrics for every threshold ──
    val_table = []
    for t in thresholds:
        val_table.append(metrics_at_threshold(val_labels, val_probs, t))

    # Save full table
    table_path = os.path.join(RESULTS_DIR, "threshold_table_validation.csv")
    with open(table_path, "w") as f:
        cols = list(val_table[0].keys())
        f.write(",".join(cols) + "\n")
        for row in val_table:
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"  Saved: {table_path}")

    # ── Step 4: Identify operating points ──
    print("\n" + "=" * 70)
    print("  SELECTED OPERATING POINTS (validation set)")
    print("=" * 70)

    # A. Max F1
    best_f1_row = max(val_table, key=lambda r: r["f1"])
    print_threshold_row(best_f1_row, "A: Max F1")

    # B. Max Youden's J
    for r in val_table:
        r["youden_j"] = r["recall"] + r["specificity"] - 1
    best_j_row = max(val_table, key=lambda r: r["youden_j"])
    print_threshold_row(best_j_row, f"B: Max Youden J={best_j_row['youden_j']:.4f}")

    # C. Lowest threshold achieving >= 80% sensitivity
    candidates_80 = [r for r in val_table if r["recall"] >= 0.80]
    if candidates_80:
        sens80_row = max(candidates_80, key=lambda r: r["threshold"])
        print_threshold_row(sens80_row, "C: >=80% sensitivity")
    else:
        sens80_row = None
        print("  [C] No threshold achieves 80% sensitivity on validation set.")

    # D. Lowest threshold achieving >= 90% sensitivity
    candidates_90 = [r for r in val_table if r["recall"] >= 0.90]
    if candidates_90:
        sens90_row = max(candidates_90, key=lambda r: r["threshold"])
        print_threshold_row(sens90_row, "D: >=90% sensitivity")
    else:
        sens90_row = None
        print("  [D] No threshold achieves 90% sensitivity on validation set.")

    # E. Best precision while maintaining >= 70% sensitivity
    candidates_70 = [r for r in val_table if r["recall"] >= 0.70]
    if candidates_70:
        prec70_row = max(candidates_70, key=lambda r: r["precision"])
        print_threshold_row(prec70_row, "E: Best prec @ >=70% sens")
    else:
        prec70_row = None
        print("  [E] No threshold achieves 70% sensitivity on validation set.")

    # Default threshold for comparison
    default_row = metrics_at_threshold(val_labels, val_probs, 0.5)
    default_row["youden_j"] = default_row["recall"] + default_row["specificity"] - 1
    print_threshold_row(default_row, "Default: t=0.500")

    selected = {
        "A: Max F1": best_f1_row,
        "B: Max Youden J": best_j_row,
        "C: >=80% sens": sens80_row,
        "D: >=90% sens": sens90_row,
        "E: Best prec@70% sens": prec70_row,
        "Default (0.5)": default_row,
    }
    selected = {k: v for k, v in selected.items() if v is not None}

    # ── Step 5: Plots ──

    # 5a. Metrics vs threshold
    print("\n  Generating metrics-vs-threshold plot...")
    ts = [r["threshold"] for r in val_table]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, [r["precision"] for r in val_table], label="Precision", linewidth=2)
    ax.plot(ts, [r["recall"] for r in val_table], label="Recall / Sensitivity", linewidth=2)
    ax.plot(ts, [r["specificity"] for r in val_table], label="Specificity", linewidth=2)
    ax.plot(ts, [r["f1"] for r in val_table], label="F1", linewidth=2, linestyle="--")

    colors = {"A: Max F1": "red", "B: Max Youden J": "purple",
              "C: >=80% sens": "green", "D: >=90% sens": "orange",
              "E: Best prec@70% sens": "brown", "Default (0.5)": "gray"}
    for name, row in selected.items():
        ax.axvline(x=row["threshold"], color=colors.get(name, "black"),
                   linestyle=":", alpha=0.7, label=f'{name} (t={row["threshold"]:.3f})')

    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Precision, Recall, Specificity, F1 vs Threshold (Validation Set)", fontsize=13)
    ax.legend(fontsize=9, loc="center right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_vs_threshold.png"), dpi=150)
    print(f"    Saved: {RESULTS_DIR}/metrics_vs_threshold.png")

    # 5b. PR curve with operating points
    print("  Generating PR curve with operating points...")
    pr_precision, pr_recall, _ = precision_recall_curve(val_labels, val_probs)
    prevalence = val_labels.mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(pr_recall, pr_precision, linewidth=2, color="#2196F3",
            label=f"PR curve (AUPRC={average_precision_score(val_labels, val_probs):.4f})")
    ax.axhline(y=prevalence, color="gray", linestyle="--", linewidth=1,
               label=f"Random (prevalence={prevalence:.4f})")

    markers = {"A: Max F1": "o", "B: Max Youden J": "s",
               "C: >=80% sens": "^", "D: >=90% sens": "v",
               "E: Best prec@70% sens": "D", "Default (0.5)": "X"}
    for name, row in selected.items():
        ax.scatter(row["recall"], row["precision"], s=120, zorder=5,
                   marker=markers.get(name, "o"), color=colors.get(name, "black"),
                   edgecolors="black", linewidth=1,
                   label=f'{name} (t={row["threshold"]:.3f})')

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve with Operating Points (Validation Set)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, max(pr_precision.max() * 1.1, 0.15)])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pr_curve_operating_points.png"), dpi=150)
    print(f"    Saved: {RESULTS_DIR}/pr_curve_operating_points.png")

    # 5c. Confusion matrices for each selected threshold
    print("  Generating confusion matrices...")
    n_selected = len(selected)
    n_cols = min(3, n_selected)
    n_rows = (n_selected + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_selected == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (name, row) in enumerate(selected.items()):
        plot_confusion_matrix(row, axes_flat[idx],
                              f'{name}\nt={row["threshold"]:.3f}')
    for idx in range(len(selected), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Confusion Matrices at Selected Thresholds (Validation Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_validation.png"), dpi=150)
    print(f"    Saved: {RESULTS_DIR}/confusion_matrices_validation.png")

    # ── Step 6: Apply selected thresholds to TEST set ──
    print("\n" + "=" * 70)
    print("  TEST SET EVALUATION (one-time application of validation-selected thresholds)")
    print("  NOTE: AUROC and AUPRC are threshold-independent and do not change.")
    print("=" * 70)

    test_auroc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)
    print(f"\n  Test AUROC: {test_auroc:.4f}  (unchanged by threshold)")
    print(f"  Test AUPRC: {test_auprc:.4f}  (unchanged by threshold)\n")

    print(f"  {'Name':<25s} {'Thresh':>6s} {'Prec':>7s} {'Recall':>7s} "
          f"{'Spec':>7s} {'F1':>7s} {'FP/1k':>7s} {'PredPos%':>8s}")
    print(f"  {'-'*75}")

    test_results = {}
    for name, val_row in selected.items():
        t = val_row["threshold"]
        tm = metrics_at_threshold(test_labels, test_probs, t)
        test_results[name] = tm
        print(f"  {name:<25s} {t:>6.3f} {tm['precision']:>7.4f} {tm['recall']:>7.4f} "
              f"{tm['specificity']:>7.4f} {tm['f1']:>7.4f} {tm['fp_per_1000']:>7.1f} "
              f"{tm['pred_pos_pct']:>7.1f}%")

    # Confusion matrices for test set
    print("\n  Generating test-set confusion matrices...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_selected == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (name, tm) in enumerate(test_results.items()):
        plot_confusion_matrix(tm, axes_flat[idx],
                              f'{name}\nt={tm["threshold"]:.3f}')
    for idx in range(len(test_results), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Confusion Matrices at Selected Thresholds (Test Set)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices_test.png"), dpi=150)
    print(f"    Saved: {RESULTS_DIR}/confusion_matrices_test.png")

    # Save test results table
    test_table_path = os.path.join(RESULTS_DIR, "test_results_selected_thresholds.csv")
    with open(test_table_path, "w") as f:
        f.write("name,threshold,precision,recall,specificity,f1,fp_per_1000,"
                "pred_pos_pct,tp,fp,fn,tn\n")
        for name, tm in test_results.items():
            f.write(f"{name},{tm['threshold']:.3f},{tm['precision']:.6f},"
                    f"{tm['recall']:.6f},{tm['specificity']:.6f},{tm['f1']:.6f},"
                    f"{tm['fp_per_1000']:.2f},{tm['pred_pos_pct']:.2f},"
                    f"{tm['tp']},{tm['fp']},{tm['fn']},{tm['tn']}\n")
    print(f"    Saved: {test_table_path}")

    # ── Step 7: Compare against default 0.5 ──
    print("\n" + "=" * 70)
    print("  COMPARISON AGAINST DEFAULT THRESHOLD 0.5")
    print("=" * 70)

    default_test = test_results.get("Default (0.5)")
    if default_test:
        best_f1_test = test_results.get("A: Max F1")
        best_j_test = test_results.get("B: Max Youden J")
        if best_f1_test:
            print(f"\n  Default (t=0.500):  F1={default_test['f1']:.4f}  "
                  f"Recall={default_test['recall']:.4f}  Prec={default_test['precision']:.4f}")
            print(f"  Max-F1  (t={best_f1_test['threshold']:.3f}):  "
                  f"F1={best_f1_test['f1']:.4f}  "
                  f"Recall={best_f1_test['recall']:.4f}  Prec={best_f1_test['precision']:.4f}")
            f1_delta = best_f1_test['f1'] - default_test['f1']
            print(f"  F1 improvement: {f1_delta:+.4f} ({f1_delta/max(default_test['f1'],1e-9)*100:+.1f}%)")
        if best_j_test:
            print(f"\n  Default (t=0.500):  Youden J = "
                  f"{default_test['recall'] + default_test['specificity'] - 1:.4f}")
            j_test = best_j_test['recall'] + best_j_test['specificity'] - 1
            j_default = default_test['recall'] + default_test['specificity'] - 1
            print(f"  Max-J   (t={best_j_test['threshold']:.3f}):  Youden J = {j_test:.4f}")
            print(f"  J improvement: {j_test - j_default:+.4f}")

    # ── Step 8: Recommendations ──
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    print("""
  IMPORTANT: Threshold selection does NOT change AUROC or AUPRC.
  These are threshold-independent ranking metrics. Changing the threshold
  only moves the operating point along the existing ROC and PR curves.

  Recommended operating thresholds:

  1. BALANCED RESEARCH REPORTING: Use the Max-F1 threshold.
     This provides the best tradeoff between precision and recall for
     reporting balanced binary classification performance. Suitable for
     paper comparisons and benchmarking.

  2. HIGH-SENSITIVITY EARLY WARNING: Use the >=90% sensitivity threshold.
     In clinical sepsis early warning, missing a sepsis event (false negative)
     is far more costly than a false alarm (false positive). This threshold
     prioritizes catching nearly all sepsis timesteps at the cost of higher
     false alarm rate. Suitable for bedside alerting where human review
     follows each alarm.

  Both thresholds were selected on the validation set and applied once to
  the held-out test set. No test data was used in threshold selection.
""")

    plt.close("all")
    print("  All results saved to results/threshold_analysis/")
    print("=" * 70)


if __name__ == "__main__":
    main()
