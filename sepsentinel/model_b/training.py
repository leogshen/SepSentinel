# Training loop for Model B with early stopping, LR scheduling,
# checkpointing, and per-epoch metric logging.

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

from sepsentinel.data.preprocessing import collate_fn


class SequenceDataset(Dataset):
    """Wraps preprocessed episodes for DataLoader."""

    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class Trainer:
    """Handles training, validation, early stopping, and checkpointing."""

    def __init__(self, model, device="cpu", checkpoint_dir="models/checkpoints",
                 pos_weight=None):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss with class weighting for imbalance
        weight = torch.tensor([pos_weight], dtype=torch.float32).to(device) \
            if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction="none")

        self.history = []

    def fit(self, train_data, val_data, epochs=50, batch_size=32, lr=1e-3,
            patience=7, min_delta=1e-4, scheduler_factor=0.5, scheduler_patience=3):
        """Train with early stopping and LR scheduling.

        Args:
            train_data, val_data: Lists of preprocessed episode dicts.
            epochs: Maximum training epochs.
            batch_size: Batch size.
            lr: Initial learning rate.
            patience: Early stopping patience (epochs without improvement).
            min_delta: Minimum AUROC improvement to count as improvement.
            scheduler_factor: LR reduction factor on plateau.
            scheduler_patience: Epochs to wait before reducing LR.

        Returns:
            Training history (list of per-epoch metric dicts).
        """
        train_loader = DataLoader(
            SequenceDataset(train_data), batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            SequenceDataset(val_data), batch_size=batch_size,
            shuffle=False, collate_fn=collate_fn
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=scheduler_factor,
            patience=scheduler_patience,
        )

        best_auroc = -1
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # --- Train ---
            train_loss = self._train_epoch(train_loader, optimizer)

            # --- Validate ---
            val_metrics = self._evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_auroc = val_metrics["auroc"]

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_auroc)
            elapsed = time.time() - t0

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_metrics["auprc"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "lr": current_lr,
                "time": elapsed,
            }
            self.history.append(record)

            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"AUROC={val_auroc:.4f}  "
                  f"AUPRC={val_metrics['auprc']:.4f}  "
                  f"sens={val_metrics['sensitivity']:.3f}  "
                  f"spec={val_metrics['specificity']:.3f}  "
                  f"lr={current_lr:.1e}  "
                  f"({elapsed:.1f}s)")

            # --- Checkpointing & early stopping ---
            if val_auroc > best_auroc + min_delta:
                best_auroc = val_auroc
                epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        # Load best model
        self._load_checkpoint("best_model.pt")
        print(f"\n  Best validation AUROC: {best_auroc:.4f}")
        return self.history

    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0
        n_samples = 0

        for signals, labels, lengths, mask in loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            optimizer.zero_grad()
            logits = self.model(signals, lengths)

            # Compute masked loss (ignore padding)
            loss_matrix = self.criterion(logits, labels)
            loss = (loss_matrix * mask).sum() / mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()

        return total_loss / n_samples

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0
        n_samples = 0

        for signals, labels, lengths, mask in loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(signals, lengths)
            probs = torch.sigmoid(logits)

            loss_matrix = self.criterion(logits, labels)
            total_loss += (loss_matrix * mask).sum().item()
            n_samples += mask.sum().item()

            # Collect valid (non-padded) predictions
            for i in range(len(lengths)):
                seq_len = lengths[i].item()
                all_probs.append(probs[i, :seq_len].cpu().numpy())
                all_labels.append(labels[i, :seq_len].cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Metrics
        auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        auprc = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

        preds = (all_probs >= 0.5).astype(int)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        tn = ((preds == 0) & (all_labels == 0)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)

        return {
            "loss": total_loss / n_samples,
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def _save_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def _load_checkpoint(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(path, weights_only=True))
