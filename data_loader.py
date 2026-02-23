"""
data_loader.py — Phase 1: Load all XJTU .mat files using their columnar summary format.

XJTU .mat Structure
═══════════════════
Each file contains two top-level keys:
  'data'    — (optional) raw time-series per cycle stored as a nested cell/struct
  'summary' — STRUCT with per-cycle summary arrays (one value per cycle)

The 'summary' struct always has these fields (verified from Batch-1/2C_battery-1.mat):
  cycle_life               : float  — total number of cycles until end
  description              : str    — experiment description
  charge_capacity_Ah       : (N,)   — charge capacity per cycle (Ah)
  discharge_capacity_Ah    : (N,)   — discharge capacity per cycle (Ah) [primary SoH signal]
  charge_energy_Wh         : (N,)   — charge energy per cycle (Wh)
  discharge_energy_Wh      : (N,)   — discharge energy per cycle (Wh)
  charge_median_voltage    : (N,)   — median voltage during charge (V)
  discharge_median_voltage : (N,)   — median voltage during discharge (V)  [key health proxy]
  charge_time              : (N,)   — time taken to charge (s)
  total_time               : (N,)   — cumulative time (s)
  ... (additional fields vary by file version)

Since raw V/I/T time-series may not always be present, we derive features from the
summary arrays instead — this is robust across all XJTU batch variants.
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io


# ─────────────────────────────────────────────────────────────────
# 1. Data Structure
# ─────────────────────────────────────────────────────────────────
@dataclass
class BatterySummary:
    """
    Holds per-cycle summary statistics for one battery extracted from the
    XJTU .mat 'summary' struct.  All arrays have length = number_of_cycles.
    """
    battery_id:   str
    batch_label:  str
    n_cycles:     int

    # Core capacity signals (Ah)
    discharge_capacity: np.ndarray   # shape (N,) — used for SoH
    charge_capacity:    np.ndarray   # shape (N,)

    # Voltage proxies (V)
    discharge_median_voltage: np.ndarray   # shape (N,)
    charge_median_voltage:    np.ndarray   # shape (N,)

    # Energy (Wh)
    discharge_energy: np.ndarray     # shape (N,)
    charge_energy:    np.ndarray     # shape (N,)

    # Time (s)
    charge_time:  np.ndarray         # shape (N,)
    total_time:   np.ndarray         # shape (N,) — cumulative


# ─────────────────────────────────────────────────────────────────
# 2. Helpers
# ─────────────────────────────────────────────────────────────────
def _safe_array(struct_obj, *field_variants, length: int = 0) -> np.ndarray:
    """
    Extract a 1-D float64 array from a scipy STRUCT object, trying multiple
    field name variants (case-insensitive).  Returns a zero array of `length`
    if not found.
    """
    if not hasattr(struct_obj, "_fieldnames"):
        return np.zeros(length, dtype=np.float64)

    available = {fn.lower(): fn for fn in struct_obj._fieldnames}
    for variant in field_variants:
        canonical = available.get(variant.lower())
        if canonical is not None:
            raw = getattr(struct_obj, canonical)
            try:
                arr = np.atleast_1d(np.array(raw, dtype=np.float64)).flatten()
                if arr.size > 0:
                    return arr
            except (ValueError, TypeError):
                continue
    return np.zeros(max(length, 1), dtype=np.float64)


def _unwrap(obj):
    """Unwrap a 0-d numpy scalar or single-element object array."""
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        if obj.dtype == object and obj.size == 1:
            return obj.flat[0]
    return obj


# ─────────────────────────────────────────────────────────────────
# 3. Parse Summary Struct from One .mat File
# ─────────────────────────────────────────────────────────────────
def _parse_summary(mat_data: dict, battery_id: str, batch_label: str) -> Optional[BatterySummary]:
    """
    Extract per-cycle summary arrays from the 'summary' key of a XJTU .mat dict.

    The 'summary' struct is always present in XJTU files and contains the
    smoothed/aggregated per-cycle statistics that serve as our feature base.
    """
    SKIP = {"__header__", "__version__", "__globals__"}

    # ── Locate the summary struct ─────────────────────────────────
    summary_obj = None
    for key in mat_data:
        if key in SKIP:
            continue
        if key.lower() == "summary":
            summary_obj = _unwrap(mat_data[key])
            break

    # Fall back: look for any struct with capacity-like fields
    if summary_obj is None or not hasattr(summary_obj, "_fieldnames"):
        for key in mat_data:
            if key in SKIP:
                continue
            candidate = _unwrap(mat_data[key])
            if hasattr(candidate, "_fieldnames"):
                fnames_lower = {f.lower() for f in candidate._fieldnames}
                if any("capacity" in fn for fn in fnames_lower):
                    summary_obj = candidate
                    break

    if summary_obj is None or not hasattr(summary_obj, "_fieldnames"):
        warnings.warn(f"[{battery_id}] No parseable summary struct found.")
        return None

    # ── Read cycle_life to determine array length ─────────────────
    cycle_life_raw = getattr(summary_obj, "cycle_life", None)
    if cycle_life_raw is not None:
        try:
            n_cycles = int(float(np.atleast_1d(cycle_life_raw).flat[0]))
        except Exception:
            n_cycles = 0
    else:
        n_cycles = 0

    # ── Extract per-cycle arrays ──────────────────────────────────
    dis_cap = _safe_array(summary_obj,
                          "discharge_capacity_Ah", "discharge_capacity",
                          "dis_cap", "capacity",
                          length=n_cycles)
    chg_cap = _safe_array(summary_obj,
                          "charge_capacity_Ah", "charge_capacity",
                          "chg_cap",
                          length=n_cycles)
    dis_v   = _safe_array(summary_obj,
                          "discharge_median_voltage", "discharge_voltage",
                          "dis_voltage", "dis_vol",
                          length=n_cycles)
    chg_v   = _safe_array(summary_obj,
                          "charge_median_voltage", "charge_voltage",
                          "chg_voltage", "chg_vol",
                          length=n_cycles)
    dis_e   = _safe_array(summary_obj,
                          "discharge_energy_Wh", "discharge_energy",
                          "dis_energy",
                          length=n_cycles)
    chg_e   = _safe_array(summary_obj,
                          "charge_energy_Wh", "charge_energy",
                          "chg_energy",
                          length=n_cycles)
    chg_t   = _safe_array(summary_obj,
                          "charge_time", "chg_time",
                          "charge_time_s",
                          length=n_cycles)
    tot_t   = _safe_array(summary_obj,
                          "total_time", "tot_time",
                          "time",
                          length=n_cycles)

    # Determine actual cycle count from longest non-zero array
    actual_n = max(dis_cap.size, chg_cap.size, dis_v.size)
    if actual_n == 0:
        warnings.warn(f"[{battery_id}] All summary arrays are empty.")
        return None

    if n_cycles == 0:
        n_cycles = actual_n

    # Pad/trim all arrays to the same length
    def _align(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.size >= n:
            return arr[:n]
        return np.pad(arr, (0, n - arr.size), constant_values=np.nan)

    n = n_cycles
    dis_cap = _align(dis_cap, n)
    chg_cap = _align(chg_cap, n)
    dis_v   = _align(dis_v,   n)
    chg_v   = _align(chg_v,   n)
    dis_e   = _align(dis_e,   n)
    chg_e   = _align(chg_e,   n)
    chg_t   = _align(chg_t,   n)
    tot_t   = _align(tot_t,   n)

    # Validate discharge capacity is present (critical for SoH)
    if np.all(dis_cap == 0) or np.all(np.isnan(dis_cap)):
        warnings.warn(f"[{battery_id}] discharge_capacity is all-zero or NaN — skipping.")
        return None

    return BatterySummary(
        battery_id=battery_id,
        batch_label=batch_label,
        n_cycles=n,
        discharge_capacity=dis_cap,
        charge_capacity=chg_cap,
        discharge_median_voltage=dis_v,
        charge_median_voltage=chg_v,
        discharge_energy=dis_e,
        charge_energy=chg_e,
        charge_time=chg_t,
        total_time=tot_t,
    )


# ─────────────────────────────────────────────────────────────────
# 4. Load a Single .mat File
# ─────────────────────────────────────────────────────────────────
def load_mat_file(path: str) -> dict:
    """
    Load a MATLAB .mat file safely using scipy.io.loadmat.

    Args:
        path: Absolute or relative path to the .mat file.

    Returns:
        Dictionary produced by scipy.io.loadmat.

    Raises:
        IOError: If the file cannot be parsed.
    """
    try:
        mat = scipy.io.loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
            mat_dtype=True,
        )
        return mat
    except Exception as exc:
        raise IOError(f"Failed to load '{path}': {exc}") from exc


# ─────────────────────────────────────────────────────────────────
# 5. Build Unified Dataset Across All Batteries
# ─────────────────────────────────────────────────────────────────
def load_battery_dataset(
    root_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, "BatterySummary"]]:
    """
    Recursively discover all .mat files under `root_dir`, load them, extract
    per-cycle summary records, and return:
      - A flat summary DataFrame (one row per cycle) with metadata + capacity.
      - A dict mapping battery_id → BatterySummary for downstream feature engineering.

    DataFrame columns:
        battery_id, batch_label, cycle_number, discharge_capacity

    Args:
        root_dir: Path to the dataset root (contains Batch-{N} sub-folders).

    Returns:
        (df_summary, summary_dict)
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {root}")

    print(f"[DataLoader] Found {len(mat_files)} .mat files in '{root}'")

    all_rows: List[dict] = []
    summary_dict: Dict[str, BatterySummary] = {}

    for mat_path in mat_files:
        batch_label = mat_path.parent.name               # e.g. "Batch-1"
        battery_id  = f"{batch_label}/{mat_path.stem}"   # e.g. "Batch-1/2C_battery-1"

        try:
            mat_data = load_mat_file(mat_path)
        except IOError as exc:
            warnings.warn(str(exc))
            continue

        batt = _parse_summary(mat_data, battery_id=battery_id, batch_label=batch_label)

        if batt is None:
            warnings.warn(f"[{battery_id}] Could not parse summary — skipping.")
            continue

        summary_dict[battery_id] = batt

        for i in range(batt.n_cycles):
            all_rows.append({
                "battery_id":          battery_id,
                "batch_label":         batch_label,
                "cycle_number":        i + 1,
                "discharge_capacity":  float(batt.discharge_capacity[i]),
            })

        print(f"  \u2713  {battery_id:45s}  {batt.n_cycles:4d} cycles  "
              f"cap_init={batt.discharge_capacity[0]:.4f} Ah")

    if not all_rows:
        raise RuntimeError(
            "No cycles could be extracted from any .mat file.\n"
            "Check that DATA_DIR points to the folder containing Batch-{N} sub-dirs."
        )

    df_summary = pd.DataFrame(all_rows)
    print(f"\n[DataLoader] Total batteries : {len(summary_dict)}")
    print(f"[DataLoader] Total cycles    : {len(df_summary)}")
    return df_summary, summary_dict
