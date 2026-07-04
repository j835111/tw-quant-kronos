#!/bin/bash
set -e
cd /root/Kronos
CFG=finetune_tw/configs/config_tw_daily.yaml

echo "=== Step 1: Extract train embeddings (8 parallel workers) ==="
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2015-01-01 --end 2016-02-15 --horizon 5 --out /root/emb_train_0.parquet > /root/log_train_0.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2016-02-16 --end 2017-04-01 --horizon 5 --out /root/emb_train_1.parquet > /root/log_train_1.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2017-04-02 --end 2018-05-17 --horizon 5 --out /root/emb_train_2.parquet > /root/log_train_2.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2018-05-18 --end 2019-07-02 --horizon 5 --out /root/emb_train_3.parquet > /root/log_train_3.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2019-07-03 --end 2020-08-16 --horizon 5 --out /root/emb_train_4.parquet > /root/log_train_4.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2020-08-17 --end 2021-10-01 --horizon 5 --out /root/emb_train_5.parquet > /root/log_train_5.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2021-10-02 --end 2022-11-16 --horizon 5 --out /root/emb_train_6.parquet > /root/log_train_6.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2022-11-17 --end 2023-12-31 --horizon 5 --out /root/emb_train_7.parquet > /root/log_train_7.log 2>&1 &
wait
echo "Train workers done, merging..."
python3 -c "
import pandas as pd
dfs = [pd.read_parquet(f'/root/emb_train_{i}.parquet') for i in range(8)]
merged = pd.concat(dfs, ignore_index=True)
merged.to_parquet('/root/embeddings_train.parquet', index=False)
print(f'merged train: {len(merged)} rows')
"

echo "=== Step 2: Extract val embeddings (4 parallel workers) ==="
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2024-01-01 --end 2024-02-15 --horizon 5 --out /root/emb_val_0.parquet > /root/log_val_0.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2024-02-16 --end 2024-04-01 --horizon 5 --out /root/emb_val_1.parquet > /root/log_val_1.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2024-04-02 --end 2024-05-16 --horizon 5 --out /root/emb_val_2.parquet > /root/log_val_2.log 2>&1 &
python3 -m finetune_tw.extract_embeddings --config $CFG --model pretrained --start 2024-05-17 --end 2024-06-30 --horizon 5 --out /root/emb_val_3.parquet > /root/log_val_3.log 2>&1 &
wait
echo "Val workers done, merging..."
python3 -c "
import pandas as pd
dfs = [pd.read_parquet(f'/root/emb_val_{i}.parquet') for i in range(4)]
merged = pd.concat(dfs, ignore_index=True)
merged.to_parquet('/root/embeddings_val.parquet', index=False)
print(f'merged val: {len(merged)} rows')
"

echo "=== Step 3: Train XGBoost ==="
python3 -m finetune_tw.train_xgb_lambdarank --train /root/embeddings_train.parquet --val /root/embeddings_val.parquet --out /root/xgb_round6.json

echo "=== Step 4: Backtest ==="
python3 -m finetune_tw.backtest_xgb_embedding --config $CFG --model pretrained --xgb_model /root/xgb_round6.json --hold_days_list 5 --top_k 10

echo "=== ALL DONE ==="
