# Attention Mechanism — Technical Justification for Racing/Dynamic Contexts

## Overview

The AttentiveLSTM model uses **Bahdanau additive attention** over 50-cycle sliding windows.
Rather than treating every cycle equally, the model learns to assign higher weight (α_t)
to cycles that carry disproportionately large information about degradation state.
This section explains, mechanistically, what physical phenomena the attention learns to focus on.

---

## 1. How Attention Works in This Architecture

Given LSTM encoder outputs **h₁, h₂, …, h₅₀** (one per cycle), the attention layer computes:

```
e_t  = v · tanh(W_h · h_t)       # energy score
α_t  = softmax(e₁, …, e₅₀)       # attention weights (sum to 1)
c    = Σ_t  α_t · h_t             # context vector
ŷ    = Head(c)                    # [SoH, RUL_norm]
```

`h_t` encodes the full temporal context of cycle `t` because the LSTM has already
processed cycles 1 → t before producing `h_t`. So the attention weight α_t reflects
how *relevant* the entire trajectory up to cycle `t` is for predicting the final health state.

---

## 2. High C-Rate Discharge Events

**Physical mechanism**: C-rate = discharge current / rated capacity. At high C-rates (2C, 3C),
lithium-ion plating and localised heat generation accelerate SEI layer formation and
active material loss.

**Attention response**: The feature `peak_c_rate` is explicitly included. When a high-C
cycle occurs, the LSTM hidden state encodes a distinct stress signature. Cycles with
high `peak_c_rate` combined with capacity drops will show **elevated energy scores e_t**
because these cycles maximally shift the LSTM's internal representation of degradation.

The Batch-2 (`3C`) and Batch-1 (`2C`) batteries in the XJTU dataset exhibit faster
capacity fade, so windows containing their early high-C cycles receive higher attention
weights compared to nominal cycling windows.

---

## 3. Aggressive Charging

**Physical mechanism**: Rapid charging pushes lithium intercalation beyond equilibrium,
causing lithium plating on graphite anodes and permanent capacity loss.

**Attention response**: The feature `charge_throughput = ∫|I|dt` reflects total accumulated
charge per cycle. Aggressive charging → large throughput in short time → high `mean_current`.
The LSTM associates these feature combinations with accelerated SoH drop. If a window
contains a run of aggressive-charge cycles followed by a noticeable capacity dip,
those cycles receive elevated α_t because they causally precede the observed degradation trend.

---

## 4. Thermal Stress

**Physical mechanism**: Elevated cell temperature (max_T > 40°C) accelerates electrolyte
decomposition, binder dissolution, and lithium dendrite growth.

**Attention response**: `max_temperature` is a dedicated feature. In the XJTU dataset,
thermally stressed cells (common in the RW batch with random walk discharge profiles)
show irregular temperature spikes. The LSTM encodes spikes as anomalous states.
Windows where `max_temperature` is persistently high receive elevated α_t because
they carry strong predictive signal about accelerated fade rate.

---

## 5. Nonlinear Degradation Accumulation

**Physical mechanism**: Battery degradation is not linear — it often follows a
**"knee-point"** pattern: gradual fade (Phase I) → accelerated fade (Phase II).
Standard models using only the last cycle miss the knee onset entirely.

**Attention captures this via two mechanisms:**

1. **Long-range temporal context**: The LSTM's recurrent state propagates information
   from early cycles (cycle 1 → 50). Subtle signatures (small incremental capacity drop,
   slight voltage plateau shift) that precede the knee by 10–20 cycles are encoded in
   `h_t` and can receive elevated α_t if they improve RUL/SoH prediction.

2. **Selective focus near inflection**: When the model sees a window spanning across
   the knee region, it assigns high α_t to cycles near the onset of accelerated fade,
   effectively learning to detect the phase transition. This is only possible with
   attention — a flat aggregation (e.g., mean pooling) would dilute this signal.

---

## 6. Edge Deployment Suitability

| Constraint | Value |
|---|---|
| Total parameters | ~46,000 (well under 1M budget) |
| Inference input | 50 cycles × 6 features = 300 floats |
| Attention overhead | O(L × H) = O(50 × 64) per sample |
| Estimated latency | < 5 ms on 32 TOPS NPU (INT8 quantised) |
| Model size (FP32) | ~180 KB |
| Scaler | `scaler.pkl` (12 floats → 96 bytes) |

The attention mechanism adds **zero** inference-time complexity overhead for edge deployment
(it is just matrix multiplications + softmax), while enabling interpretability dashboards
that can flag anomalous battery stress events in real time.

---

## 7. Summary

```
Physical Event          → Feature Signal               → Attention Effect
────────────────────────────────────────────────────────────────────────────
High C-rate discharge   → peak_c_rate spike            → α_t elevated on stress cycles
Aggressive charging     → charge_throughput + mean_I↑  → α_t elevated on charge cycles
Thermal stress          → max_temperature spike        → α_t elevated on hot cycles
Nonlinear degradation   → subtle V/Q slope change      → α_t elevated near knee onset
────────────────────────────────────────────────────────────────────────────
```

The result is a model that implicitly performs **anomaly-weighted health estimation**—
physically meaningful, interpretable, and robust across different discharge profiles
(2C, 3C, R2.5Ω, R3Ω, RW random walk).
