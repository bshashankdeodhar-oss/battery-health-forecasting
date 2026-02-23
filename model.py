"""
model.py — Phase 4: AttentiveLSTM architecture for SoH/RUL regression.

Architecture:
    Input  : (batch, seq_len, feature_dim)
    LSTM   : 2-layer bidirectional=False, hidden=64, dropout=0.30
    Attention: Bahdanau additive style — learns a scalar weight per time step
    Context: weighted sum over time → (batch, hidden)
    Head   : Linear(hidden → 2) → [SoH, RUL_norm]
    Output : predictions (batch, 2) + attention_weights (batch, seq_len)

Total parameter budget: < 1,000,000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config


# ─────────────────────────────────────────────────────────────────
# 1. Additive Attention Module
# ─────────────────────────────────────────────────────────────────
class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention over a sequence of LSTM hidden states.

    Given encoder outputs h ∈ R^(batch × seq × hidden), computes a context
    vector c as a weighted mean:
        e_t  = v · tanh(W_h · h_t)          ← energy score
        α_t  = softmax(e_t)                  ← attention weight
        c    = Σ_t α_t · h_t                ← context vector

    This is NOT self-attention (no Q/K/V projections); computational cost
    is O(seq_len × hidden) — very lightweight for seq_len=50.

    Args:
        hidden_size: LSTM hidden dimension.
        attn_hidden: Intermediate projection size.
    """

    def __init__(self, hidden_size: int = config.HIDDEN_SIZE,
                 attn_hidden: int = config.ATTN_HIDDEN) -> None:
        super().__init__()
        self.W_h = nn.Linear(hidden_size, attn_hidden, bias=True)
        self.v   = nn.Linear(attn_hidden, 1,           bias=False)

    def forward(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_out: (batch, seq_len, hidden_size)

        Returns:
            context:     (batch, hidden_size)  — weighted context vector
            weights:     (batch, seq_len)       — attention distribution
        """
        # Energy: (batch, seq_len, attn_hidden) → (batch, seq_len, 1)
        energy  = torch.tanh(self.W_h(encoder_out))   # (B, L, A)
        scores  = self.v(energy).squeeze(-1)           # (B, L)
        weights = F.softmax(scores, dim=-1)            # (B, L)  — sums to 1

        # Context vector: weighted sum over time
        context = torch.bmm(weights.unsqueeze(1), encoder_out)  # (B, 1, H)
        context = context.squeeze(1)                             # (B, H)

        return context, weights


# ─────────────────────────────────────────────────────────────────
# 2. Full AttentiveLSTM Model
# ─────────────────────────────────────────────────────────────────
class AttentiveLSTM(nn.Module):
    """
    Two-layer LSTM with Bahdanau attention for SoH / RUL prediction.

    Forward pass returns both predictions and attention weights so that
    evaluation code can visualise where the model focuses during inference.

    Args:
        input_size:   Number of input features per time step (F).
        hidden_size:  LSTM hidden state dimension.
        num_layers:   Number of stacked LSTM layers.
        dropout:      Dropout probability between LSTM layers.
        attn_hidden:  Attention projection size.
        output_size:  Number of regression targets (2: SoH and RUL_norm).
    """

    def __init__(
        self,
        input_size:  int = config.FEATURE_DIM,
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers:  int = config.NUM_LSTM_LAYERS,
        dropout:     float = config.DROPOUT,
        attn_hidden: int = config.ATTN_HIDDEN,
        output_size: int = config.OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Layer normalisation on input features (stabilises training)
        self.input_norm = nn.LayerNorm(input_size)

        # ── Stacked LSTM ─────────────────────────────────────────
        # Note: dropout is applied between layers (not after last layer)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Dropout after LSTM output (regularisation) ────────────
        self.dropout = nn.Dropout(p=dropout)

        # ── Attention ─────────────────────────────────────────────
        self.attention = BahdanauAttention(hidden_size, attn_hidden)

        # ── Regression Head ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Weight initialisation
        self._init_weights()

        # ── Parameter count report ────────────────────────────────
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] AttentiveLSTM | Total params: {total_params:,} | "
              f"Trainable: {trainable:,}")
        assert total_params < 1_000_000, (
            f"Model exceeds 1M parameter budget! ({total_params:,})"
        )

    def _init_weights(self) -> None:
        """Xavier uniform init for LSTM and linear layers."""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (standard trick to reduce vanishing gradient)
                n = param.shape[0]
                param.data[n // 4 : n // 2].fill_(1.0)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim).

        Returns:
            predictions:    (batch, 2)       — [SoH_hat, RUL_norm_hat]
            attn_weights:   (batch, seq_len) — attention distribution over cycles
        """
        # Layer norm on features
        x = self.input_norm(x)                            # (B, L, F)

        # LSTM encoding
        enc_out, _ = self.lstm(x)                         # (B, L, H)
        enc_out     = self.dropout(enc_out)

        # Attention
        context, attn_weights = self.attention(enc_out)   # (B, H), (B, L)

        # Regression
        predictions = self.head(context)                  # (B, 2)

        return predictions, attn_weights

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
