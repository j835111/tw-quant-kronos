#!/bin/bash
set -e

# Multi-Horizon GBDT backtest evaluation script for RunPod.
# Integrates Z-Score blending (0.6 : 0.4) and XReg (multiplier=2.0) adjustment.

echo "=== Running Walk-Forward Backtest for Multi-Horizon GBDT Ensemble (With XReg) ==="
PYTHONPATH=. python3 finetune_tw/backtest_xgb_ensemble.py \
  --xgb_model_full finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full_mh.json \
  --xgb_model_raw finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_raw_mh.json \
  --weight 0.6 \
  --xreg_enabled \
  --xreg_mult 2.0

echo "=== Backtest finished successfully! ==="
