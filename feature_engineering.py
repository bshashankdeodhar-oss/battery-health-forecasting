"""
feature_engineering.py — Phase 2: Per-cycle feature extraction, SoH/RUL computation, normalisation.

Features are derived from the XJTU per-cycle summary arrays stored in BatterySummary.
No raw time-series processing is needed — the .mat 'summary' struct already provides
cycle-level statistics.

Feature set (6 features):
  1. discharge_median_voltage   — proxy for average cell degradation
  2. charge_median_voltage      — reflection of internal resistance growth
  3. discharge_capacity_Ah      — direct capacity signal (before SoH normalisation)
  4. charge_time_norm           — normalised charge time (longer = degraded)
  5. energy_efficiency          — discharge_energy / charge_energy per cycle
  6. cycle_index_norm           — cycle number normalised to [0, 1] over battery lifespan
"""

import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_loader import BatterySummary
import config


# ─────────────────────────────────────────────────────────────────
# 1. Feature Column List
# ─────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "discharge_median_voltage",    # V  — main degradation proxy
    "charge_median_voltage",       # V  — IR growth indicator
    "discharge_capacity_Ah",       # Ah — direct capacity (scaled later)
    "charge_time_norm",            # normalised charge time [0, 1] within battery
    "energy_efficiency",           # Wh_dis / Wh_chg  — roundtrip efficiency
    "cycle_index_norm",            # cycle age normalised to [0, 1]
]


# ─────────────────────────────────────────────────────────────────
# 2. SoH and RUL Computation
# ─────────────────────────────────────────────────────────────────
def _is_variable_dod(capacities: np.ndarray, threshold: float = 0.50) -> bool:
    """
    Detect if a battery uses a Variable Depth-of-Discharge (VDoD) protocol.

    Satellite batteries interleave full reference discharges (~2 Ah) with
    short partial discharges (0.1–1.9 Ah). The signature is a very wide
    spread in per-cycle capacity:  cap_min / cap_max < 0.50.

    Normal batteries (2C, 3C, R2.5, R3, RW) discharge close to their full
    nominal capacity every cycle, so cap_min / cap_max > 0.70.
    """
    cap_min = np.nanmin(capacities)
    cap_max = np.nanmax(capacities)
    if cap_max <= 0:
        return False
    return (cap_min / cap_max) < threshold


def _impute_vdod_capacities(
    capacities: np.ndarray,
    full_cycle_min_frac: float = 0.70,
) -> np.ndarray:
    """
    For VDoD (satellite-type) batteries: replace partial-discharge cycle
    capacities with the capacity of the last full reference cycle.

    This converts the mix of [1.991, 0.111, 0.445, 1.922, 0.756, ...]
    into    [1.991, 1.991, 1.991, 1.922, 1.922, ...]
    so that SoH = imputed_cap / cap_init tracks actual capacity FADE
    (from the reference cycles) rather than per-cycle DoD.

    Args:
        capacities:          Raw per-cycle discharge capacity array.
        full_cycle_min_frac: Cycle is 'full' if cap >= this * max_cap.

    Returns:
        Array of same length with partial cycles filled from last reference.
    """
    peak_cap = np.nanmax(capacities)
    is_full = capacities >= full_cycle_min_frac * peak_cap

    imputed = capacities.copy().astype(np.float64)
    last_ref = float(capacities[is_full][0]) if np.any(is_full) else float(capacities[0])

    for i in range(len(imputed)):
        if is_full[i]:
            last_ref = float(imputed[i])      # update reference with latest full cycle
        else:
            imputed[i] = last_ref             # substitute partial cycle with reference

    return imputed


def compute_soh_rul(
    capacities: np.ndarray,
    eol_threshold: float = config.EOL_SOH_THRESHOLD,
    battery_id: str = "",
    verbose_diag: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute State-of-Health and Remaining Useful Life for one battery.

    SoH FORMULA (unchanged for all normal batteries):
        SoH[k] = discharge_capacity[k] / cap_init

    TARGETED VDoD FIX (satellite batteries only):
        Detected by cap_min / cap_max < 0.50 (wide DoD spread).
        Partial-discharge cycles are forward-filled from the last
        full reference cycle before applying SoH = cap / cap_init.
        This preserves degradation in the reference cycles without
        the 0.056 fake-collapse from a 0.111 Ah partial discharge.

    Args:
        capacities:    1-D array of per-cycle discharge capacity (Ah).
        eol_threshold: SoH fraction defining end-of-life (default 0.80).
        battery_id:    Battery name string for diagnostic prints.
        verbose_diag:  If True, print first/last 10 capacities and SoH bounds.

    Returns:
        (soh_array, rul_array, eol_cycle_index)
    """
    # ── 1. NaN imputation ────────────────────────────────────────────────
    caps = pd.Series(capacities, dtype=np.float64).ffill().bfill().values
    n = len(caps)

    if caps[0] <= 0:
        raise ValueError(f"[{battery_id}] Initial capacity <= 0 ({caps[0]:.4f} Ah).")

    # ── 2. Targeted VDoD fix (satellite batteries only, explicit rule) ──
    if "satellite" in battery_id.lower():
        working_caps = _impute_vdod_capacities(caps)
        vdod_flag = True
    else:
        working_caps = caps
        vdod_flag = False

    # ── 3. Original SoH formula — unchanged for all batteries ────────────
    cap_init = float(working_caps[0])
    if cap_init <= 0:
        raise ValueError(f"[{battery_id}] cap_init <= 0 after imputation ({cap_init:.4f}).")

    soh = (working_caps / cap_init).astype(np.float32)

    # ── 4. Optional diagnostic print ─────────────────────────────────────
    if verbose_diag:
        tag = "[VDoD-fixed]" if vdod_flag else "[normal]"
        print(f"\n  DIAG {tag} {battery_id}")
        print(f"    raw caps first 10 : {caps[:10].tolist()}")
        print(f"    raw caps last  10 : {caps[-10:].tolist()}")
        if vdod_flag:
            print(f"    imputed caps first10: {working_caps[:10].tolist()}")
        print(f"    cap_init = {cap_init:.4f} Ah")
        print(f"    SoH min={soh.min():.4f}  max={soh.max():.4f}\n")

    # ── 5. EOL detection ─────────────────────────────────────────────────
    below_eol = np.where(soh <= eol_threshold)[0]
    if len(below_eol) > 0:
        eol_idx = int(below_eol[0])
    else:
        eol_idx = n - 1
        warnings.warn(
            f"EOL threshold ({eol_threshold:.0%}) not reached; "
            f"SoH min = {soh.min():.3f}. Using last cycle as EOL proxy."
        )

    rul = np.maximum(eol_idx - np.arange(n), 0).astype(np.float32)
    return soh, rul, eol_idx





# ─────────────────────────────────────────────────────────────────
# 3. Build Full Feature Matrix for All Batteries
# ─────────────────────────────────────────────────────────────────
def build_feature_matrix(
    summary_dict: Dict[str, BatterySummary],
    eol_threshold: float = config.EOL_SOH_THRESHOLD,
) -> pd.DataFrame:
    """
    Build a per-cycle feature DataFrame with SoH, RUL, and feature columns
    across all batteries, derived from BatterySummary columnar arrays.

    Args:
        summary_dict:   Dict mapping battery_id → BatterySummary.
        eol_threshold:  SoH fraction for EOL.

    Returns:
        DataFrame with columns:
            battery_id, batch_label, cycle_number,
            [FEATURE_COLUMNS],
            SoH, RUL, eol_cycle, is_past_eol
    """
    frames: List[pd.DataFrame] = []

    for battery_id, batt in summary_dict.items():
        n = batt.n_cycles
        if n < 2:
            warnings.warn(f"[{battery_id}] Fewer than 2 cycles — skipping.")
            continue

        # ── Per-cycle features ────────────────────────────────────

        # (a) discharge & charge median voltage — already per-cycle scalars
        dis_v = batt.discharge_median_voltage          # (N,)
        chg_v = batt.charge_median_voltage             # (N,)

        # (b) discharge capacity
        dis_cap = batt.discharge_capacity              # (N,) Ah

        # (c) normalised charge time  (0→1 within battery lifespan)
        chg_t = batt.charge_time.copy()
        chg_t_max = np.nanmax(chg_t)
        chg_t_norm = (chg_t / chg_t_max) if chg_t_max > 0 else np.zeros(n)

        # (d) energy efficiency = discharge_energy / charge_energy per cycle
        dis_e = batt.discharge_energy
        chg_e = batt.charge_energy
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(chg_e > 0, dis_e / chg_e, 0.0)
        # Clip to sensible range [0, 1]
        eff = np.clip(eff, 0.0, 1.0)

        # (e) cycle index normalised
        cycle_idx_norm = np.arange(n, dtype=np.float32) / max(n - 1, 1)

        # Fill NaN in all feature arrays with forward-fill then 0
        def _ffill(arr: np.ndarray) -> np.ndarray:
            s = pd.Series(arr).ffill().bfill().fillna(0.0)
            return s.values

        df_bat = pd.DataFrame({
            "battery_id":               battery_id,
            "batch_label":              batt.batch_label,
            "cycle_number":             np.arange(1, n + 1),
            "discharge_median_voltage": _ffill(dis_v),
            "charge_median_voltage":    _ffill(chg_v),
            "discharge_capacity_Ah":    _ffill(dis_cap),
            "charge_time_norm":         _ffill(chg_t_norm),
            "energy_efficiency":        _ffill(eff.astype(np.float32)),
            "cycle_index_norm":         cycle_idx_norm,
        })

        # ── SoH / RUL ─────────────────────────────────────────────
        # Print diagnostics for the first normal and first satellite battery.
        _IS_SAT = "satellite" in battery_id.lower()
        _diag = getattr(build_feature_matrix, "_diag_normal_done", False) is False and not _IS_SAT or \
                getattr(build_feature_matrix, "_diag_sat_done",    False) is False and _IS_SAT
        try:
            soh, rul, eol_idx = compute_soh_rul(
                batt.discharge_capacity,
                eol_threshold,
                battery_id=battery_id,
                verbose_diag=_diag,
            )
            if _diag and not _IS_SAT:
                build_feature_matrix._diag_normal_done = True
            if _diag and _IS_SAT:
                build_feature_matrix._diag_sat_done = True
        except ValueError as exc:
            warnings.warn(f"[{battery_id}] SoH computation failed: {exc} — skipping.")
            continue


        df_bat["SoH"]         = soh
        df_bat["RUL"]         = rul
        df_bat["eol_cycle"]   = eol_idx + 1         # 1-indexed
        df_bat["is_past_eol"] = (np.arange(n) > eol_idx).astype(int)

        frames.append(df_bat)
        print(f"  \u2713  {battery_id:45s}  SoH [{soh.min():.3f}, {soh.max():.3f}]  "
              f"EOL @ cycle {eol_idx + 1}")

    if not frames:
        raise RuntimeError("Feature matrix is empty — check data loading step.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n[FeatEng] Feature matrix shape: {df.shape}  |  "
          f"Batteries: {df['battery_id'].nunique()}")
    return df


# ─────────────────────────────────────────────────────────────────
# 4. Normalisation — StandardScaler
# ─────────────────────────────────────────────────────────────────
def fit_and_save_scaler(
    df_train: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    save_path: str = str(config.SCALER_PATH),
) -> StandardScaler:
    """
    Fit a StandardScaler on train-set features and save it for deployment.
    Only fit on training data to prevent leakage.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols].values)
    joblib.dump(scaler, save_path)
    print(f"[FeatEng] Scaler saved → {save_path}")
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: List[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Apply a fitted scaler to the feature columns of a DataFrame."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


def load_scaler(load_path: str = str(config.SCALER_PATH)) -> StandardScaler:
    """Load a persisted StandardScaler from disk."""
    return joblib.load(load_path)


# ─────────────────────────────────────────────────────────────────
# 5. RUL Normalisation (per-battery)
# ─────────────────────────────────────────────────────────────────
def normalise_rul_per_battery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise RUL to [0, 1] per battery so loss gradients for SoH and RUL
    stay in comparable ranges during joint training.

    RUL_norm = RUL / max_RUL_in_battery

    Args:
        df: Feature DataFrame containing 'RUL', 'battery_id', 'eol_cycle' columns.

    Returns:
        New DataFrame with additional column 'RUL_norm'.
    """
    df = df.copy()
    df["RUL_norm"] = df.groupby("battery_id", group_keys=False).apply(
        lambda grp: grp["RUL"] / max(grp["RUL"].max(), 1.0),
        include_groups=False,
    )
    return df
