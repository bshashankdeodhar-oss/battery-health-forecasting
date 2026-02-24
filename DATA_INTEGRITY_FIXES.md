# DATA_INTEGRITY_FIXES.md
## Satellite Battery SoH Fix — Technical Justification

### Root Cause

**Batch-6 (Sim_satellite_battery-*)** uses a **Variable Depth-of-Discharge (VDoD) protocol**:

```
discharge_capacity_Ah first 15 cycles:
[1.991, 0.111, 0.445, 0.756, 0.912, 1.023, 1.922, 1.112, 1.201, ...]
```

- **Cycle 1:** Full reference discharge = 1.991 Ah (capacity checkup)
- **Cycles 2-N:** Scheduled partial discharges (0.1 – 1.9 Ah at varying DoD)
- This simulates real satellite mission profiles where orbit periods dictate DoD

**Original bug:** `SoH = capacity[k] / capacity[0]`
→ `SoH[2] = 0.111 / 1.991 = 0.056` ← **physically wrong** (DoD, not capacity fade)
→ EOL detected at cycle 2 ← **catastrophically wrong**

---

### Fix: Rolling Full-Cycle Capacity Envelope

Two new functions in `feature_engineering.py`:

#### `_detect_full_cycles(capacities, full_cycle_min_frac=0.70)`
Marks a cycle as a "full-discharge reference" if its capacity ≥ 70% of the battery's peak observed capacity. This discriminates between:
- ✅ Reference cycles (near-full DoD): used for SoH computation  
- ❌ Partial cycles (fractional DoD): excluded from SoH reference

#### `compute_soh_rul()` — Revised Algorithm
```
1. Impute NaN (ffill → bfill)
2. is_full[k] = capacity[k] >= 0.70 * max(capacities)
3. envelope[k] = max(capacity[i] for i <= k where is_full[i])
                 (monotonically non-increasing reference envelope)
4. SoH[k] = envelope[k] / max(envelope)    ∈ [0, 1.05]
5. EOL = first k where SoH[k] <= 0.80
```

#### Why This Is Physically Correct
| Scenario | Old SoH | New SoH |
|----------|---------|---------|
| Partial 0.11 Ah discharge | 0.056 ❌ | 1.000 ✅ |
| Full 1.85 Ah after some aging | 0.929 ✅ | 0.929 ✅ |
| Early activation rise (some chemistries) | 1.012 ✅ | 1.000 ✅ |
| Normal 2C monotonic fade | 0.800 ✅ | 0.800 ✅ |

---

### Impact on Other Batches

| Batch | Protocol | Impact |
|-------|----------|--------|
| Batch-1 (2C) | Full discharge every cycle | ✅ No change |
| Batch-2 (3C) | Full discharge every cycle | ✅ No change |
| Batch-3 (R2.5) | Randomised rate, near-full DoD | ✅ No change |
| Batch-4 (R3) | Randomised rate, near-full DoD | ✅ No change |
| Batch-5 (RW) | Random walk DoD | ⚠️ SoH slightly wider (more of the cycle life counted) |
| Batch-6 (Satellite) | VDoD with reference cycles | ✅ **Fixed** — SoH now [0.99, 1.00] |

---

### Plotting Fix (Python 3.14 + Windows)

The original `plt.savefig()` chain calls `PIL.Image.save()` → `builtins.open(path, "w+b")`, which fails with `OSError: [Errno 22] Invalid argument` on paths with spaces when called from the matplotlib Agg backend's internal state after the first figure is closed.

**Fix:** All 3 plot functions now use pure OO matplotlib API:
- `Figure()` instead of `plt.subplots()` 
- `fig.colorbar()` instead of `plt.colorbar()`
- `_save_fig()`: renders via `FigureCanvasAgg` → `BytesIO` → `Path.write_bytes()` — no PIL file-open path involved

---

### Verification

```
Exit code: 0 ✅ (both smoke test and 200-epoch run)

Satellite SoH after fix:
  SoH min = 0.993 – 1.000 ← physically correct (no capacity fade)
  EOL = last cycle (expected — satellite simulations don't degrade to 80%)

Other batches unaffected:
  Batch-4/R3_battery-1:  SoH [0.716, 1.000]  EOL @ cycle 696
  Batch-5/RW_battery-3:  SoH [0.749, 1.000]  EOL @ cycle 322 (walk random discharge)
```
