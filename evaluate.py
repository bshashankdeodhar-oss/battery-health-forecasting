"""
evaluate.py — Phase 6: Load best model, run test inference, compute metrics, generate plots.

Produces:
  artifacts/soh_predictions.png
  artifacts/rul_predictions.png
  artifacts/attention_heatmap.png
  artifacts/training_curves.png
  Console: RMSE, MAE, R² for SoH and RUL
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model import AttentiveLSTM
from utils import compute_metrics, plot_predictions, plot_attention_heatmap, plot_training_curves


# ─────────────────────────────────────────────────────────────────
# 1. Inference Pass
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model:  AttentiveLSTM,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference over an entire DataLoader.

    Args:
        model:  Trained AttentiveLSTM (eval mode).
        loader: Test DataLoader.
        device: Compute device.

    Returns:
        y_true:       (N, 2) ground-truth [SoH, RUL_norm]
        y_pred:       (N, 2) model predictions
        attn_weights: (N, seq_len) attention weight matrices
    """
    model.eval()
    all_true, all_pred, all_attn = [], [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        preds, attn = model(x_batch)        # (B, 2), (B, L)

        all_true.append(y_batch.numpy())
        all_pred.append(preds.cpu().numpy())
        all_attn.append(attn.cpu().numpy())

    y_true       = np.concatenate(all_true, axis=0)        # (N, 2)
    y_pred       = np.concatenate(all_pred, axis=0)        # (N, 2)
    attn_weights = np.concatenate(all_attn, axis=0)        # (N, L)
    return y_true, y_pred, attn_weights


# ─────────────────────────────────────────────────────────────────
# 2. Print Metrics Table
# ─────────────────────────────────────────────────────────────────
def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[dict, dict]:
    """
    Print a formatted table of RMSE, MAE, R² for SoH and RUL (normalised).

    Args:
        y_true: (N, 2)  — columns [SoH, RUL_norm]
        y_pred: (N, 2)  — columns [SoH_hat, RUL_norm_hat]
    """
    soh_metrics = compute_metrics(y_true[:, 0], y_pred[:, 0])
    rul_metrics = compute_metrics(y_true[:, 1], y_pred[:, 1])

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"{'Metric':<12} {'SoH':>18} {'RUL (norm)':>18}")
    print(sep)
    for key in ("RMSE", "MAE", "R2"):
        print(f"{key:<12} {soh_metrics[key]:>18.6f} {rul_metrics[key]:>18.6f}")
    print(sep)
    print()
    return soh_metrics, rul_metrics


# ─────────────────────────────────────────────────────────────────
# 3. Full Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────
def evaluate(
    model:       AttentiveLSTM,
    test_loader: DataLoader,
    device:      torch.device,
    artifacts_dir: str = str(config.ARTIFACTS_DIR),
    log_path:    str   = str(config.TRAIN_LOG_PATH),
) -> Tuple[dict, dict]:
    """
    End-to-end evaluation:
      1. Inference on test set.
      2. Metrics computation and printing.
      3. Save all visualisation plots.

    Args:
        model:         Loaded best AttentiveLSTM.
        test_loader:   Test DataLoader.
        device:        Compute device.
        artifacts_dir: Directory to write PNGs.
        log_path:      CSV log from training (for curve plot).

    Returns:
        (soh_metrics, rul_metrics) — each a dict with RMSE, MAE, R²
    """
    art = Path(artifacts_dir)
    print("\n[Evaluate] Running inference on test set …")
    y_true, y_pred, attn_weights = run_inference(model, test_loader, device)

    # ── Metrics ──────────────────────────────────────────────────
    soh_m, rul_m = print_metrics(y_true, y_pred)

    # ── Plot: SoH ────────────────────────────────────────────────
    print("[Evaluate] Generating plots …")
    plot_predictions(
        actual    = y_true[:, 0],
        predicted = y_pred[:, 0],
        label     = "SoH",
        save_path = str(art / "soh_predictions.png"),
    )

    # ── Plot: RUL ─────────────────────────────────────────────────
    plot_predictions(
        actual    = y_true[:, 1],
        predicted = y_pred[:, 1],
        label     = "RUL (normalised)",
        save_path = str(art / "rul_predictions.png"),
    )

    # ── Plot: Attention Heatmap ──────────────────────────────────
    plot_attention_heatmap(
        weights   = attn_weights,
        save_path = str(art / "attention_heatmap.png"),
    )

    # ── Plot: Training Curves ────────────────────────────────────
    if Path(log_path).exists():
        plot_training_curves(
            log_path  = log_path,
            save_path = str(art / "training_curves.png"),
        )
    else:
        print(f"  [Skip] Training log not found: {log_path}")

    print(f"\n[Evaluate] All artifacts saved to: {art}")
    return soh_m, rul_m


# ─────────────────────────────────────────────────────────────────
# 4. Convenience: Load Model from Checkpoint
# ─────────────────────────────────────────────────────────────────
def load_model(
    checkpoint_path: str = str(config.MODEL_PATH),
    device:          torch.device = torch.device("cpu"),
) -> AttentiveLSTM:
    """
    Instantiate AttentiveLSTM and load saved weights.

    Args:
        checkpoint_path: Path to best_model.pt saved by EarlyStopping.
        device:          Device to map parameters to.

    Returns:
        Model in eval mode.
    """
    model = AttentiveLSTM()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[Evaluate] Model loaded from {checkpoint_path}")
    return model
