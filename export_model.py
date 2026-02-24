"""
export_model.py — Export the trained AttentiveLSTM as a self-contained
                  TorchScript file that can be loaded with:

    model = torch.jit.load("artifacts/battery_model_scripted.pt")
    soh, rul_norm = model(x)   # x: (batch, seq_len, feature_dim)

No model.py or config.py needed by the person loading the model.
Run once from the project root after training:

    python export_model.py

Outputs
-------
  artifacts/battery_model_scripted.pt   — TorchScript (self-contained)
  artifacts/scaler.pkl                  — StandardScaler for 6 features
  artifacts/model_card.txt              — Architecture & usage summary
"""

import sys
import torch
import pickle
from pathlib import Path

# ── Locate project files ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from model import AttentiveLSTM

CHECKPOINT  = config.MODEL_PATH                         # artifacts/best_model.pt
SCRIPT_OUT  = config.ARTIFACTS_DIR / "battery_model_scripted.pt"
SCALER_OUT  = config.SCALER_PATH                        # artifacts/scaler.pkl
CARD_OUT    = config.ARTIFACTS_DIR / "model_card.txt"

# ── Load checkpoint ────────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CHECKPOINT}")
state = torch.load(CHECKPOINT, map_location="cpu")

# ── Reconstruct model ──────────────────────────────────────────────────────────
model = AttentiveLSTM(
    input_size   = config.FEATURE_DIM,
    hidden_size  = config.HIDDEN_SIZE,
    num_layers   = config.NUM_LSTM_LAYERS,
    attn_hidden  = config.ATTN_HIDDEN,
    output_size  = config.OUTPUT_SIZE,
    dropout      = 0.0,              # disable dropout for inference
)
model.load_state_dict(state)
model.eval()

# ── Thin wrapper: return predictions only (no attention weights) ──────────────
#   The raw model returns (predictions, attn_weights). For dashboard use the
#   consumer normally only needs [SoH, RUL_norm], so this keeps the API clean.
class _InferenceWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred, _attn = self.base(x)
        return pred

wrapper = _InferenceWrapper(model)
wrapper.eval()

# ── TorchScript via tracing ────────────────────────────────────────────────────
#   Trace with a dummy (batch=1, seq=SEQ_LEN, features=FEATURE_DIM) input
dummy = torch.zeros(1, config.SEQ_LEN, config.FEATURE_DIM)
with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)

# Verify the trace produces the correct output shape
out = traced(dummy)
assert out.shape == (1, config.OUTPUT_SIZE), \
    f"Unexpected output shape: {out.shape}"
print(f"  Trace verified: input {tuple(dummy.shape)} → output {tuple(out.shape)}")

# ── Save TorchScript ───────────────────────────────────────────────────────────
traced.save(str(SCRIPT_OUT))
print(f"  [Saved] TorchScript  → {SCRIPT_OUT}")

# ── Confirm scaler exists ──────────────────────────────────────────────────────
if SCALER_OUT.exists():
    print(f"  [Ready] Scaler       → {SCALER_OUT}")
else:
    print("  [WARN] scaler.pkl not found — run full pipeline first.")

# ── Write model card ───────────────────────────────────────────────────────────
param_count = sum(p.numel() for p in model.parameters())

card = f"""
======================================================
  Battery Health Forecasting -- Model Card
======================================================

Architecture:  AttentiveLSTM
Parameters:    {param_count:,}
Input shape:   (batch, seq_len={config.SEQ_LEN}, features={config.FEATURE_DIM})
Output:        (batch, 2)  ->  [SoH, normalised_RUL]

Input features (in order):
  0  discharge_median_voltage  (V)
  1  charge_median_voltage     (V)
  2  discharge_capacity_Ah     (Ah)
  3  charge_time_norm          (0-1, normalised within battery lifetime)
  4  energy_efficiency         (discharge_Wh / charge_Wh, 0-1)
  5  cycle_index_norm          (0-1, cycle / total_cycles)

All 6 features must be scaled with scaler.pkl (StandardScaler)
before passing to the model.

Output interpretation:
  output[:, 0]  = SoH    in [0, 1]  (State of Health)
  output[:, 1]  = RUL_n  in [0, 1]  (Remaining Useful Life, normalised)

To use (Python):
------------------------------------------------------
  import torch, pickle, numpy as np

  # Load model
  model  = torch.jit.load("battery_model_scripted.pt")
  model.eval()

  # Load scaler
  with open("scaler.pkl", "rb") as f:
      scaler = pickle.load(f)

  # Prepare a sequence: numpy array (seq_len=50, 6 features)
  raw_seq = np.array(...)            # shape (50, 6)
  scaled  = scaler.transform(raw_seq)
  x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 50, 6)

  with torch.no_grad():
      pred   = model(x)              # (1, 2)
      soh    = pred[0, 0].item()
      rul_n  = pred[0, 1].item()

Dataset:  XJTU Battery Dataset (55 batteries, ~27,600 charge-discharge cycles)
Training: AttentiveLSTM (2-layer LSTM + temporal attention), Adam, CosineAnnealing

Files needed:
  battery_model_scripted.pt   -- this TorchScript model
  scaler.pkl                  -- feature scaler (required)
======================================================
"""

CARD_OUT.write_text(card.strip(), encoding="utf-8")

print(f"  [Saved] Model card   → {CARD_OUT}")

print(f"\n✓  Export complete.  Share these two files with your colleague:")
print(f"   {SCRIPT_OUT.name}")
print(f"   {SCALER_OUT.name}")
