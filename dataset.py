"""
dataset.py — Phase 3: PyTorch Dataset with sliding-window sequence creation.

Sliding window scheme:
    For a battery with N cycles and sequence length L:
        window k  →  input features[k : k+L]  (shape L × F)
                  →  target [SoH[k+L-1], RUL_norm[k+L-1]] (shape 2)

Battery-level train/val/test split prevents sequence leakage across batteries.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

import config
from feature_engineering import FEATURE_COLUMNS


# ─────────────────────────────────────────────────────────────────
# 1. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────
class BatterySequenceDataset(Dataset):
    """
    Creates overlapping sliding windows of `seq_len` consecutive cycles
    per battery and packs them into (x, y) tensors.

    Args:
        df:          Feature DataFrame (already scaled) containing
                     FEATURE_COLUMNS, 'SoH', 'RUL_norm', 'battery_id'.
        seq_len:     Number of cycles per input window (default 50).
        feature_cols: Ordered list of feature column names forming the input.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = config.SEQ_LEN,
        feature_cols: List[str] = None,
    ) -> None:
        super().__init__()
        self.seq_len      = seq_len
        self.feature_cols = feature_cols or FEATURE_COLUMNS

        # Validate required columns
        required = set(self.feature_cols) | {"SoH", "RUL_norm", "battery_id"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        self._windows: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build_windows(df)

    def _build_windows(self, df: pd.DataFrame) -> None:
        """
        Iterate over each battery, sort by cycle order, then slide a window
        of length `seq_len` with stride 1 across its cycle sequence.
        """
        for batt_id, grp in df.groupby("battery_id", sort=False):
            grp = grp.sort_values("cycle_number").reset_index(drop=True)
            n   = len(grp)
            if n < self.seq_len:
                # Battery too short for even one window — skip
                continue

            feats   = grp[self.feature_cols].values.astype(np.float32)  # (N, F)
            soh_arr = grp["SoH"].values.astype(np.float32)              # (N,)
            rul_arr = grp["RUL_norm"].values.astype(np.float32)         # (N,)

            for start in range(n - self.seq_len + 1):
                end    = start + self.seq_len
                x      = feats[start:end]                               # (L, F)
                # Target: predicted at the LAST step of the window
                y      = np.array([soh_arr[end - 1], rul_arr[end - 1]], dtype=np.float32)
                self._windows.append((x, y))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self._windows[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


# ─────────────────────────────────────────────────────────────────
# 2. Battery-Level Train / Val / Test Split
# ─────────────────────────────────────────────────────────────────
def split_batteries(
    battery_ids: List[str],
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio:   float = config.VAL_RATIO,
    seed:        int   = config.SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly split battery IDs into train / val / test groups.

    Splitting at battery level ensures that no sequence from a battery in the
    test set shares cycles with sequences in the training set.

    Args:
        battery_ids: Full list of unique battery identifiers.
        train_ratio: Fraction for training set.
        val_ratio:   Fraction for validation set.
        seed:        RNG seed for reproducibility.

    Returns:
        (train_ids, val_ids, test_ids)
    """
    rng   = np.random.default_rng(seed)
    ids   = list(battery_ids)
    rng.shuffle(ids)

    n       = len(ids)
    n_train = int(np.floor(n * train_ratio))
    n_val   = int(np.floor(n * val_ratio))

    train_ids = ids[:n_train]
    val_ids   = ids[n_train : n_train + n_val]
    test_ids  = ids[n_train + n_val:]

    print(f"[Split] Batteries → Train: {len(train_ids)}  Val: {len(val_ids)}  "
          f"Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────
# 3. DataLoader Factory
# ─────────────────────────────────────────────────────────────────
def build_dataloaders(
    df_scaled: pd.DataFrame,
    train_ids: List[str],
    val_ids:   List[str],
    test_ids:  List[str],
    seq_len:   int = config.SEQ_LEN,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a pre-scaled feature DataFrame.

    Args:
        df_scaled:   Scaled feature DataFrame with 'battery_id' column.
        train_ids:   Battery IDs assigned to training set.
        val_ids:     Battery IDs assigned to validation set.
        test_ids:    Battery IDs assigned to test set.
        seq_len:     Sliding window length.
        batch_size:  Mini-batch size.
        num_workers: DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    def _subset(ids: List[str]) -> pd.DataFrame:
        return df_scaled[df_scaled["battery_id"].isin(ids)].copy()

    df_train = _subset(train_ids)
    df_val   = _subset(val_ids)
    df_test  = _subset(test_ids)

    ds_train = BatterySequenceDataset(df_train, seq_len=seq_len)
    ds_val   = BatterySequenceDataset(df_val,   seq_len=seq_len)
    ds_test  = BatterySequenceDataset(df_test,  seq_len=seq_len)

    print(f"[Dataset] Windows → Train: {len(ds_train)}  Val: {len(ds_val)}  "
          f"Test: {len(ds_test)}")

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader, test_loader
