#!/bin/bash
set -e

# Multi-Horizon GBDT Training script for RunPod (50GB RAM)
# Evaluates Option 2: Multi-Horizon composite target GBDT training.

echo "=== Starting Multi-Horizon GBDT Training (full model) ==="
PYTHONUNBUFFERED=1 python3 -m finetune_tw.train_xgb_streaming \
  --train finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train.parquet \
  --val finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train.parquet finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_val.parquet \
  --features full --selection-metric rank_ic --early_stopping_rounds 40 \
  --train-start 2015-05-22 --train-end 2022-12-30 \
  --train-exclude-range 2021-01-04:2021-06-30 \
  --val-range 2021-01-04:2021-06-30 --val-range 2023-01-03:2024-06-28 \
  --top-k 10 \
  --mh_enabled \
  --train_targets finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train_targets.parquet \
  --val_targets finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_val_targets.parquet \
  --mem-limit-gb 40.0 \
  --out finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full_mh.json

echo ""
echo "=== Starting Multi-Horizon GBDT Training (raw model) ==="
PYTHONUNBUFFERED=1 python3 -m finetune_tw.train_xgb_streaming \
  --train finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train.parquet \
  --val finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train.parquet finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_val.parquet \
  --features raw --selection-metric rank_ic --early_stopping_rounds 40 \
  --train-start 2015-05-22 --train-end 2022-12-30 \
  --train-exclude-range 2021-01-04:2021-06-30 \
  --val-range 2021-01-04:2021-06-30 --val-range 2023-01-03:2024-06-28 \
  --top-k 10 \
  --mh_enabled \
  --train_targets finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_train_targets.parquet \
  --val_targets finetune_tw/outputs/tw_daily/round6_artifacts/embeddings_val_targets.parquet \
  --mem-limit-gb 40.0 \
  --out finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_raw_mh.json

echo "=== All Multi-Horizon trainings finished successfully! ==="
