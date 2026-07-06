# Round 6 Artifact 評估

**定位：Round 6 後續改進評估**

## 評估範圍

輸入產物：

- `embeddings_train.parquet`：2,141,404 筆，2015-05-22 至 2023-12-29
- `embeddings_val.parquet`：135,323 筆，2024-01-01 至 2024-06-28
- `xgb_round6.json`：836 個特徵、200 輪 boosting、best iteration 190

本次評估使用快取的 validation embeddings 重現 XGBoost 結果，並分析四個
raw technical features 在 train 與 validation 的行為。現有產物不含測試期的
逐股分數，因此無法直接證明 2026-Q2 發生的機制。

## 結果重現

快取產物可重現文件記載的 validation 結果：

| Trees | Mean rank-IC | IC-IR | IC 為正的日期比例 | Mean top-10 excess |
|---|---:|---:|---:|---:|
| 全部 200 trees | 0.066281 | 0.636 | 70.0% | +0.216% |
| Best iteration 0-190 | 0.066450 | 0.637 | 70.8% | +0.213% |

使用全部 200 trees 與停在 iteration 190 的差異可忽略，不足以解釋測試期
回測表現不佳。

## 交易日曆缺陷

`extract_embeddings.py` 使用 `pd.bdate_range`，而不是 TWSE 實際交易日。
這會加入非交易日，並重複使用前一個交易日的 context：

- Train：147 / 2,246 個日期不是 TWSE 交易日（6.5%）
- Validation：13 / 130 個日期不是 TWSE 交易日（10.0%）
- Validation 有 12 個日期的 score state 完全重複；2024-01-01 也是非交易日，
  但其前一個 state 位於 validation 範圍之外
- 2024-02-05 的農曆年前 context 被計算八次
  （2024-02-05 至 2024-02-14 的 business-day entries）

過濾後剩下 117 個真實 validation 交易日：

| Trees | Mean rank-IC | IC-IR | IC 為正的日期比例 | Mean top-10 excess |
|---|---:|---:|---:|---:|
| 全部 200 trees | 0.072783 | 0.705 | 72.6% | +0.300% |
| Best iteration 0-190 | 0.072909 | 0.704 | 73.5% | +0.293% |

此缺陷並未灌高本次 validation IC；過濾後的 IC 反而更高。但它改變了
train 與 validation 的樣本權重，後續實驗比較前必須先移除。

## 反轉曝險

以下指標只使用真實 TWSE 交易日：

| 特徵 | Train label IC | Validation score IC | Validation top-10 percentile | XGB total-gain share |
|---|---:|---:|---:|---:|
| MA5 distance | -0.0417 | -0.3415 | 36.5% | 20.6% |
| MA20 distance | -0.0291 | -0.1963 | 45.1% | 9.2% |
| 10-day momentum | -0.0218 | -0.1471 | 44.7% | 3.9% |
| Volume ratio | -0.0009 | -0.0504 | 60.0% | 4.0% |

四個 raw features 占 XGBoost total gain 的 37.6%，同時也是 gain 最高的四個
單一特徵。其中最強的 MA5 distance 明確讓模型偏向近期落後股。因此，
anti-momentum 診斷同時得到歷史 labels 與 fitted model 支持，不只是
2026-Q2 回測後的事後推測。

## 全市場排序與 Top-Tail 的落差

在 117 個真實 validation 交易日：

- Full-universe mean rank-IC：0.0728
- Mean top-10 excess h5 return：+0.300%
- Top-10 與實際報酬前十名的重疊率：0.855%
- 平均約 1,041 檔股票下，隨機選取的重疊率期望值約為 0.96%

模型在 validation 具備有效的全市場截面排序能力，但幾乎無法辨識真正的
前十名贏家。這與 objective mismatch 一致：LambdaRankIC 最佳化所有 pairs，
實際策略卻只交易前 1%。

## 改進優先序

1. 先將 train 與 validation 過濾為真實 TWSE 交易日，再重新訓練 XGBoost。
   現有 Parquet 已足夠，不需要重新執行 GPU embedding extraction。
2. 執行 embedding-only 與 raw-feature-only ablation。Raw features 占 37.6%
   gain 且主導 feature ranking，因此不應在完成歸因前優先修改 pooling。
3. 除 rank-IC 外，加入 top-10 excess return 或 NDCG@10 作為選模指標。
   Full-universe IC 並未與實際部署策略完全對齊。
4. 加入橫截面與多時間尺度動能特徵，但同時保留 momentum 與 reversal signals。
   Validation 的極端區域呈現非單調關係，不能直接刪除所有反轉特徵。
5. 使用涵蓋不同市場 regime 的多段 validation windows。
6. 在完整證實 2026-Q2 機制前，先匯出測試期逐股分數。

## 輸出檔案

- `round6_val_scores.parquet`
- `round6_artifact_evaluation.json`
- `round6_factor_evaluation.json`
- `round6_trading_day_filtered_evaluation.json`
- `round6_daily_diagnostics.csv`
- `round6_daily_diagnostics_trading_days.csv`
- `round6_factor_daily_diagnostics.csv`
- `round6_factor_daily_diagnostics_trading_days.csv`
- `round6_factor_period_diagnostics.csv`
- `round6_period_diagnostics.csv`
