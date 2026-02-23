"""
train.py — Phase 5: Training loop with Adam, CosineAnnealingLR, early stopping.

Trains AttentiveLSTM jointly on SoH and RUL regression with joint MSE loss.
Logs per-epoch train/val loss and per-target RMSE to a CSV for analysis.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import config
from model import AttentiveLSTM
from utils import EarlyStopping, compute_metrics, set_seed


# ─────────────────────────────────────────────────────────────────
# 1. Single Epoch Training
# ─────────────────────────────────────────────────────────────────
def _train_epoch(
    model:     AttentiveLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> Tuple[float, float, float]:
    """
    Run one full training epoch.

    Returns:
        (mean_loss, soh_rmse, rul_rmse)
    """
    model.train()
    total_loss = 0.0
    all_soh_true, all_soh_pred = [], []
    all_rul_true, all_rul_pred = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)   # (B, L, F)
        y_batch = y_batch.to(device, non_blocking=True)   # (B, 2)

        optimizer.zero_grad(set_to_none=True)

        preds, _ = model(x_batch)                          # (B, 2)
        loss = criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients during early training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

        preds_np = preds.detach().cpu().numpy()
        y_np     = y_batch.cpu().numpy()
        all_soh_true.append(y_np[:, 0]);    all_soh_pred.append(preds_np[:, 0])
        all_rul_true.append(y_np[:, 1]);    all_rul_pred.append(preds_np[:, 1])

    n          = len(loader.dataset)
    mean_loss  = total_loss / n
    soh_rmse   = compute_metrics(np.concatenate(all_soh_true),
                                 np.concatenate(all_soh_pred))["RMSE"]
    rul_rmse   = compute_metrics(np.concatenate(all_rul_true),
                                 np.concatenate(all_rul_pred))["RMSE"]
    return mean_loss, soh_rmse, rul_rmse


# ─────────────────────────────────────────────────────────────────
# 2. Single Epoch Validation
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _val_epoch(
    model:     AttentiveLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate model on validation set.

    Returns:
        (mean_loss, soh_rmse, rul_rmse)
    """
    model.eval()
    total_loss = 0.0
    all_soh_true, all_soh_pred = [], []
    all_rul_true, all_rul_pred = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds, _ = model(x_batch)
        loss = criterion(preds, y_batch)

        total_loss += loss.item() * x_batch.size(0)

        preds_np = preds.cpu().numpy()
        y_np     = y_batch.cpu().numpy()
        all_soh_true.append(y_np[:, 0]);    all_soh_pred.append(preds_np[:, 0])
        all_rul_true.append(y_np[:, 1]);    all_rul_pred.append(preds_np[:, 1])

    n         = len(loader.dataset)
    mean_loss = total_loss / n
    soh_rmse  = compute_metrics(np.concatenate(all_soh_true),
                                np.concatenate(all_soh_pred))["RMSE"]
    rul_rmse  = compute_metrics(np.concatenate(all_rul_true),
                                np.concatenate(all_rul_pred))["RMSE"]
    return mean_loss, soh_rmse, rul_rmse


# ─────────────────────────────────────────────────────────────────
# 3. Trainer Class
# ─────────────────────────────────────────────────────────────────
class Trainer:
    """
    Encapsulates the full training procedure:
      - Optimizer: Adam with weight decay
      - Scheduler: CosineAnnealingLR
      - Loss:      MSELoss (joint on [SoH, RUL_norm])
      - Early stopping on validation MSE

    Args:
        model:         AttentiveLSTM instance.
        train_loader:  DataLoader for training data.
        val_loader:    DataLoader for validation data.
        device:        torch.device to train on.
        num_epochs:    Maximum training epochs.
        lr:            Initial Adam learning rate.
        weight_decay:  Adam L2 regularisation.
        scheduler_tmax: CosineAnnealing T_max period (epochs).
        patience:      Early stopping patience.
        log_path:      CSV file path for training log.
        model_path:    Path to save best model weights.
    """

    def __init__(
        self,
        model:          AttentiveLSTM,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        device:         torch.device,
        num_epochs:     int   = config.NUM_EPOCHS,
        lr:             float = config.LEARNING_RATE,
        weight_decay:   float = config.WEIGHT_DECAY,
        scheduler_tmax: int   = config.SCHEDULER_TMAX,
        patience:       int   = config.EARLY_STOP_PAT,
        log_path:       str   = str(config.TRAIN_LOG_PATH),
        model_path:     str   = str(config.MODEL_PATH),
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.num_epochs   = num_epochs
        self.log_path     = Path(log_path)

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=scheduler_tmax, eta_min=1e-6)
        self.stopper   = EarlyStopping(patience=patience, save_path=model_path)

    def train(self) -> List[Dict]:
        """
        Run the full training loop until early stopping or max epochs.

        Returns:
            history: List of per-epoch log dicts.
        """
        history: List[Dict] = []

        # Prepare CSV log file
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "val_loss",
                "train_soh_rmse", "train_rul_rmse",
                "val_soh_rmse",   "val_rul_rmse",
                "lr", "epoch_time_s",
            ])
            writer.writeheader()

            print(f"\n{'─'*70}")
            print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} "
                  f"{'SoH RMSE':>10} {'RUL RMSE':>10} {'LR':>10}")
            print(f"{'─'*70}")

            for epoch in range(1, self.num_epochs + 1):
                t0 = time.time()

                tr_loss, tr_soh, tr_rul = _train_epoch(
                    self.model, self.train_loader, self.criterion,
                    self.optimizer, self.device)

                va_loss, va_soh, va_rul = _val_epoch(
                    self.model, self.val_loader, self.criterion, self.device)

                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                elapsed = time.time() - t0
                row = {
                    "epoch":          epoch,
                    "train_loss":     tr_loss,
                    "val_loss":       va_loss,
                    "train_soh_rmse": tr_soh,
                    "train_rul_rmse": tr_rul,
                    "val_soh_rmse":   va_soh,
                    "val_rul_rmse":   va_rul,
                    "lr":             current_lr,
                    "epoch_time_s":   elapsed,
                }
                writer.writerow(row)
                f.flush()
                history.append(row)

                # Console output every 5 epochs or on last epoch
                if epoch % 5 == 0 or epoch == 1 or epoch == self.num_epochs:
                    print(f"{epoch:>6} {tr_loss:>12.6f} {va_loss:>12.6f} "
                          f"{va_soh:>10.5f} {va_rul:>10.5f} {current_lr:>10.2e}  "
                          f"[{elapsed:.1f}s]")

                # Early stopping check
                self.stopper(va_loss, self.model)
                if self.stopper.early_stop:
                    print(f"\n[Trainer] Early stopping triggered at epoch {epoch}.")
                    break

        print(f"{'─'*70}")
        print(f"[Trainer] Best val loss: {self.stopper.best_loss:.6f}")
        print(f"[Trainer] Model saved → {self.stopper.save_path}")
        return history

    def load_best_model(self) -> AttentiveLSTM:
        """Return model loaded with the best-epoch weights."""
        return self.stopper.load_best(self.model)
