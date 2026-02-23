"""
main.py — Orchestrates all phases of the Battery Health Forecasting pipeline.

Usage:
    python main.py                         # full pipeline with defaults
    python main.py --data_dir "Battery Dataset" --epochs 100
    python main.py --smoke_test            # 3 epochs, quick validation
    python main.py --eval_only             # skip training, evaluate saved model

Phases:
    1. Data Loading      (data_loader.py)
    2. Feature Eng.     (feature_engineering.py)
    3. Dataset Build    (dataset.py)
    4. Model Build      (model.py)
    5. Training         (train.py)
    6. Evaluation       (evaluate.py)
"""

import argparse
import sys
from pathlib import Path

# ── Windows UTF-8 console fix ─────────────────────────────────────
# Prevents UnicodeEncodeError when printing ═/✓ on cp1252 terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

import config
from utils import set_seed


# ─────────────────────────────────────────────────────────────────
# 0. CLI Argument Parsing
# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Battery Health Forecasting — AttentiveLSTM Pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, default=str(config.DATA_DIR),
        help="Root directory containing Batch-{N} subdirectories with .mat files.",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.NUM_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=config.BATCH_SIZE,
        help="Mini-batch size for DataLoaders.",
    )
    parser.add_argument(
        "--lr", type=float, default=config.LEARNING_RATE,
        help="Adam initial learning rate.",
    )
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Run 3 epochs only for a quick sanity check.",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; load best_model.pt and evaluate directly.",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true",
        help="Disable CUDA even if available.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# 1. Device Selection
# ─────────────────────────────────────────────────────────────────
def get_device(no_cuda: bool = False) -> torch.device:
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    return device


# ─────────────────────────────────────────────────────────────────
# 2. Data Pipeline (Phases 1–3)
# ─────────────────────────────────────────────────────────────────
def build_data_pipeline(args: argparse.Namespace):
    """
    Phases 1–3: Load, engineer features, scale, split, build DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, feature_dim
    """
    from data_loader import load_battery_dataset
    from feature_engineering import (
        build_feature_matrix, fit_and_save_scaler, apply_scaler,
        normalise_rul_per_battery, FEATURE_COLUMNS,
    )
    from dataset import split_batteries, build_dataloaders


    # ── Phase 1: Load ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("PHASE 1 — DATA LOADING")
    print("═" * 70)
    _, summary_dict = load_battery_dataset(args.data_dir)

    # ── Phase 2: Feature Engineering ─────────────────────────────
    print("\n" + "═" * 70)
    print("PHASE 2 — FEATURE ENGINEERING")
    print("═" * 70)
    df = build_feature_matrix(summary_dict)
    df = normalise_rul_per_battery(df)

    # Sanity checks
    assert "SoH" in df.columns, "SoH column missing from feature matrix."
    assert "RUL_norm" in df.columns, "RUL_norm column missing from feature matrix."
    assert df[FEATURE_COLUMNS].isnull().sum().sum() == 0, \
        "NaN values found in feature columns — check data loading."

    # ── Phase 3: Sequence Dataset ─────────────────────────────────
    print("\n" + "═" * 70)
    print("PHASE 3 — SEQUENCE DATASET CREATION")
    print("═" * 70)

    battery_ids = df["battery_id"].unique().tolist()
    train_ids, val_ids, test_ids = split_batteries(
        battery_ids, seed=args.seed
    )

    # Fit scaler only on training set to prevent leakage
    df_train_raw = df[df["battery_id"].isin(train_ids)]
    scaler = fit_and_save_scaler(df_train_raw, feature_cols=FEATURE_COLUMNS)

    df_scaled = apply_scaler(df, scaler, feature_cols=FEATURE_COLUMNS)

    train_loader, val_loader, test_loader = build_dataloaders(
        df_scaled     = df_scaled,
        train_ids     = train_ids,
        val_ids       = val_ids,
        test_ids      = test_ids,
        batch_size    = args.batch_size,
        num_workers   = config.NUM_WORKERS,
    )

    feature_dim = len(FEATURE_COLUMNS)
    return train_loader, val_loader, test_loader, feature_dim


# ─────────────────────────────────────────────────────────────────
# 3. Main Entry Point
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    device = get_device(args.no_cuda)
    set_seed(args.seed)

    if args.smoke_test:
        args.epochs = 3
        print("[Main] *** SMOKE TEST MODE — 3 epochs only ***")

    # ── Build data pipeline ───────────────────────────────────────
    train_loader, val_loader, test_loader, feature_dim = build_data_pipeline(args)

    # ── Phase 4: Model ────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("PHASE 4 — MODEL ARCHITECTURE")
    print("═" * 70)
    from model import AttentiveLSTM
    model = AttentiveLSTM(input_size=feature_dim)

    # ── Phase 5: Training ─────────────────────────────────────────
    if not args.eval_only:
        print("\n" + "═" * 70)
        print("PHASE 5 — TRAINING")
        print("═" * 70)
        from train import Trainer
        trainer = Trainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            device       = device,
            num_epochs   = args.epochs,
            lr           = args.lr,
        )
        trainer.train()
        model = trainer.load_best_model()
        model.to(device)

    else:
        # Load pre-trained weights
        from evaluate import load_model
        if not config.MODEL_PATH.exists():
            print(f"[Main] ERROR: No checkpoint found at {config.MODEL_PATH}")
            print("       Run without --eval_only to train first.")
            sys.exit(1)
        model = load_model(str(config.MODEL_PATH), device=device)

    # ── Phase 6: Evaluation ───────────────────────────────────────
    print("\n" + "═" * 70)
    print("PHASE 6 — EVALUATION")
    print("═" * 70)
    from evaluate import evaluate
    soh_metrics, rul_metrics = evaluate(
        model        = model,
        test_loader  = test_loader,
        device       = device,
        artifacts_dir = str(config.ARTIFACTS_DIR),
        log_path      = str(config.TRAIN_LOG_PATH),
    )

    # ── Final Summary ─────────────────────────────────────────────
    print("═" * 70)
    print("PIPELINE COMPLETE")
    print(f"  SoH — RMSE: {soh_metrics['RMSE']:.5f}  MAE: {soh_metrics['MAE']:.5f}  R²: {soh_metrics['R2']:.5f}")
    print(f"  RUL — RMSE: {rul_metrics['RMSE']:.5f}  MAE: {rul_metrics['MAE']:.5f}  R²: {rul_metrics['R2']:.5f}")
    print(f"  Artifacts → {config.ARTIFACTS_DIR}")
    print("═" * 70)


if __name__ == "__main__":
    main()
