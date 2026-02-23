"""
utils.py — Shared utilities: metrics, early stopping, plotting, reproducibility.
"""

import io
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for headless runs)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────
# 2. Regression Metrics
# ─────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute RMSE, MAE, and R² for a single regression target.

    Args:
        y_true: Ground-truth array, shape (N,)
        y_pred: Predicted array,    shape (N,)

    Returns:
        dict with keys 'RMSE', 'MAE', 'R2'
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    r2     = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ─────────────────────────────────────────────────────────────────
# 3. Early Stopping
# ─────────────────────────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when validation loss does not improve for `patience` epochs.
    Saves the best model weights automatically.
    """

    def __init__(self, patience: int = 15, delta: float = 1e-6,
                 save_path: str = "artifacts/best_model.pt") -> None:
        self.patience   = patience
        self.delta      = delta
        self.save_path  = Path(save_path)
        self.best_loss  = np.inf
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load best saved weights into model and return it."""
        model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        return model


def _save_fig(fig: Figure, save_path: str) -> None:
    """
    Render a matplotlib Figure to PNG via BytesIO and write to disk.
    Avoids all PIL file-open logic (which fails on Python 3.14 + Windows paths
    with spaces when called from certain internal matplotlib code paths).
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    Path(save_path).write_bytes(buf.read())


# ─────────────────────────────────────────────────────────────────
# 4. Plotting — Predictions
# ─────────────────────────────────────────────────────────────────
def plot_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: str,
    save_path: str,
    unit: str = "",
) -> None:
    """
    Scatter + line plot of predicted vs actual values.
    Uses pure OO matplotlib API — no pyplot global state.
    """
    actual    = actual.flatten()
    predicted = predicted.flatten()
    sort_idx  = np.argsort(actual)
    act_s     = actual[sort_idx]
    pred_s    = predicted[sort_idx]

    fig = Figure(figsize=(14, 5))
    fig.suptitle(f"{label} – Predicted vs Actual", fontsize=14, fontweight="bold")
    axes = fig.subplots(1, 2)

    # ── Left: scatter ────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(actual, predicted, alpha=0.5, s=12, color="steelblue", label="Samples")
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {label} {unit}")
    ax.set_ylabel(f"Predicted {label} {unit}")
    ax.set_title("Scatter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: time-series overlay ────────────────────────────────
    ax2 = axes[1]
    ax2.plot(act_s,  label="Actual",    linewidth=1.5, color="steelblue")
    ax2.plot(pred_s, label="Predicted", linewidth=1.5, color="orange", linestyle="--")
    ax2.set_xlabel("Sample index (sorted by actual)")
    ax2.set_ylabel(f"{label} {unit}")
    ax2.set_title("Sorted Overlay")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path)
    print(f"  [Saved] {save_path}")


# ─────────────────────────────────────────────────────────────────
# 5. Plotting — Attention Heatmap
# ─────────────────────────────────────────────────────────────────
def plot_attention_heatmap(
    weights: np.ndarray,
    save_path: str,
    max_samples: int = 50,
) -> None:
    """
    Visualise attention weight distributions across sequence steps.

    Args:
        weights:     Attention weights array, shape (N, seq_len)
        save_path:   PNG save path
        max_samples: Max rows to display in heatmap
    """
    weights = weights[:max_samples]            # (M, seq_len)
    n_samples, seq_len = weights.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention Weight Analysis", fontsize=14, fontweight="bold")

    # ── Left: heatmap ────────────────────────────────────────────
    im = axes[0].imshow(weights, aspect="auto", cmap="viridis",
                        interpolation="nearest")
    axes[0].set_xlabel("Sequence Position (Cycle Index within Window)")
    axes[0].set_ylabel("Sample Index")
    axes[0].set_title("Attention Weights Heatmap")
    plt.colorbar(im, ax=axes[0], label="Weight")

    # ── Right: mean attention profile ────────────────────────────
    mean_w = weights.mean(axis=0)    # (seq_len,)
    std_w  = weights.std(axis=0)
    x_pos  = np.arange(seq_len)

    axes[1].fill_between(x_pos, mean_w - std_w, mean_w + std_w,
                         alpha=0.25, color="coral", label="±1 std")
    axes[1].plot(x_pos, mean_w, color="coral", linewidth=2, label="Mean attention")
    axes[1].axhline(1.0 / seq_len, color="grey", linestyle="--",
                    linewidth=1, label="Uniform baseline")
    axes[1].set_xlabel("Sequence Position (Cycle Index within Window)")
    axes[1].set_ylabel("Average Attention Weight")
    axes[1].set_title("Mean Attention Profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    with open(save_path, "wb") as _f:
        fig.savefig(_f, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ─────────────────────────────────────────────────────────────────
# 6. Training Curve Plotter
# ─────────────────────────────────────────────────────────────────
def plot_training_curves(log_path: str, save_path: str) -> None:
    """
    Plot train/val loss curves and per-target RMSE from the CSV training log.

    Args:
        log_path:  Path to training_log.csv
        save_path: PNG save path
    """
    import pandas as pd

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(df["epoch"], df["val_soh_rmse"], label="SoH RMSE (val)")
    axes[1].plot(df["epoch"], df["train_soh_rmse"], label="SoH RMSE (train)", linestyle="--")
    axes[1].plot(df["epoch"], df["val_rul_rmse"], label="RUL RMSE (val)")
    axes[1].plot(df["epoch"], df["train_rul_rmse"], label="RUL RMSE (train)", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Per-Target RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    with open(save_path, "wb") as _f:
        fig.savefig(_f, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    print(f"  [Saved] {save_path}")
