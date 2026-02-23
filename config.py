"""
config.py — Centralized configuration for Battery Health Forecasting System.
All hyperparameters, paths, and constants live here.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "Battery Dataset" / "Battery Dataset"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH    = ARTIFACTS_DIR / "scaler.pkl"
MODEL_PATH     = ARTIFACTS_DIR / "best_model.pt"
TRAIN_LOG_PATH = ARTIFACTS_DIR / "training_log.csv"

# ── Dataset ───────────────────────────────────────────────────────────────────
# Nominal capacity used for C-rate calculation (Ah) – XJTU cells ≈ 2 Ah
RATED_CAPACITY     = 2.0
SAMPLING_RATE_HZ   = 1          # 1 Hz → dt = 1 second
EOL_SOH_THRESHOLD  = 0.80       # End-of-Life defined at SoH ≤ 0.80

# ── Sequence ──────────────────────────────────────────────────────────────────
SEQ_LEN     = 50                # sliding-window length (cycles)
FEATURE_DIM = 6                 # [dis_med_V, chg_med_V, dis_cap_Ah, chg_time_norm, energy_eff, cycle_idx_norm]

# ── Split Ratios (battery-level, to prevent leakage) ─────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN_SIZE    = 64
NUM_LSTM_LAYERS = 2
DROPOUT        = 0.30
ATTN_HIDDEN    = 32             # intermediate attention projection size
OUTPUT_SIZE    = 2              # [SoH, RUL_norm]

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
NUM_EPOCHS     = 200
SCHEDULER_TMAX = 50             # CosineAnnealing period
EARLY_STOP_PAT = 15             # patience (epochs without improvement)
SEED           = 42

# ── Misc ──────────────────────────────────────────────────────────────────────
NUM_WORKERS  = 0                # DataLoader workers (0 = main process, safe on Windows)
PIN_MEMORY   = False            # enable only if CUDA GPU available
DEVICE       = "cuda"           # falls back to cpu in train.py if not available
