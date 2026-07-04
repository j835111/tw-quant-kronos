# Kronos 台股實盤操作指南

策略：**Round 6 Batch 3c `full` 模型**（Kronos embedding + XGBoost LambdaRankIC），
top_k=10, hold_days=5（真實 next-open 回測 Sharpe **1.336**、年化 31.17%、MaxDD 27.21%，
為目前 Round 0-6 全系列最佳紀錄，取代舊版純 Kronos Round 0 top-k 策略）。

詳細方法論、診斷與回測驗證見 `docs/round6-embedding-xgb-lambdarank-improvements.md`。

> 舊版純 Kronos（Round 0, top_k=10/hold_days=3, `signal_today.py`）已停用；
> `finetune_tw/grid_search_backtest.py` 那份 Sharpe 1.92 數字來自另一套（非 next-open）
> 回測方法，跟本文件與上述 improvements 文件用的 next-open 執行框架不是同一套量尺，
> 不可直接比較。

## 前置準備（一次性）

```bash
cd /mnt/d/project/Kronos
pip install -r requirements.txt

# 確認 DB 存在（已 commit 到 git）
ls finetune_tw/data/tw_stocks.db

# 確認 Round 6 Batch 3c production 模型存在（若無，從 HF 復原，見下方「模型復原」）
ls finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json
```

### 模型復原（若本地沒有 checkpoint）

```bash
hf download j835111/kronos-tw-finetune \
    --revision round6-batch3c-full-production \
    --include "round6_xgb/production/*" \
    --local-dir /tmp/kronos-batch3c-production
mkdir -p finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results
cp /tmp/kronos-batch3c-production/round6_xgb/production/xgb_batch3c_full.json \
   /tmp/kronos-batch3c-production/round6_xgb/production/xgb_batch3c_full.summary.json \
   finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/
```

## 每日例行作業

### 每個交易日收盤後（13:35+）

```bash
python -m finetune_tw.download_data \
    --config finetune_tw/configs/config_tw_daily.yaml \
    --update
```

約 2-3 分鐘，更新 DB 至今日收盤價。

### 每 5 個交易日：取訊號

```bash
# 無持倉 / 首次執行
python -m finetune_tw.signal_today_xgb \
    --config finetune_tw/configs/config_tw_daily.yaml \
    --xgb_model finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json \
    --top_k 10

# 已有持倉：帶入目前代碼，自動顯示換股建議
python -m finetune_tw.signal_today_xgb \
    --config finetune_tw/configs/config_tw_daily.yaml \
    --xgb_model finetune_tw/outputs/tw_daily/round6_artifacts/batch3c_results/xgb_batch3c_full.json \
    --top_k 10 \
    --holdings 2330,2317,2454,2382,2308,2357,2412,3231,2379,2345
```

`--model` 預設為 `pretrained`（未微調的 `NeoQuasar/Kronos-base`），**必須維持這個預設值**——
Batch 3c 全程用這顆 backbone 訓練，換成 `round0` 等微調版會有 train/inference 分佈不一致的問題。

輸出範例：

```
=== Round 6 Batch 3c 選股訊號（Kronos embedding + XGBoost）===
  embedding backbone：pretrained  |  xgb_model：.../xgb_batch3c_full.json
  特徵數：858  |  top_k=10
  訊號日：2026-06-24

【選股結果】XGBoost 分數 top 10（2026-06-24 訊號）
排名      代碼        分數
----------------------------
   1      2330      +1.2345
   2      2317      +0.9876
   ...

【換股建議】（目前持倉: [...]）
  繼續持有：{2330, 2317, ...}
  賣出：    {2454, 2382}
  買入：    {3231, 2379}
```

**注意**：XGBoost 分數是相對排序用的原始分數，不是預測報酬率，數值大小本身沒有意義，只用來排名選 top_k。

### 隔日開盤後（9:00-9:10）

1. 按「賣出」清單平倉
2. 等成交後，按「買入」清單建倉
3. 10 支等權配置（每支 10% 資金）

## 操作時間表

| 時間 | 動作 |
|------|------|
| 每天 13:35 後 | `download_data --update` |
| 每 5 個交易日收盤後 | `signal_today_xgb.py` 取換股清單 |
| 隔日 9:00-9:10 | 依清單下單（賣舊買新） |

建議固定在**每週同一天**換股，避開節假日干擾。

## 交易成本估算

| 項目 | 費率 |
|------|------|
| 買進手續費 | 0.1425%（可談折扣至 ~0.05%） |
| 賣出手續費 | 0.1425% |
| 證券交易稅（賣出）| 0.3% |
| 每次換手往返合計 | ~0.585% |

hold_days=5 換手頻率比舊版 hold_days=3 低，成本侵蝕相對較輕；仍需以實單觀察滑點與流動性影響，
尤其 top_k=10 集中在少數幾檔股票時。

## 最低資金建議

- 台股一張 = 1000 股，均價 50 元 ≈ 5 萬元
- 10 支等權：**建議 100 萬以上**，避免零股問題與過度集中

## 無 GPU 環境

`signal_today_xgb.py` 內部仍會呼叫 Kronos 抽 embedding（`--model pretrained`），CPU 環境下
1000+ 檔股票推論可能需要數十分鐘，建議在收盤後執行，隔天開盤前完成即可。

## 策略沿革

| 版本 | 方法 | hold=5d Sharpe | 年化 | MaxDD |
|------|------|---:|---:|---:|
| Round 0（純 Kronos，round0 微調） | `signal_today.py` | 1.115 | 38.6% | 35.0% |
| Round 6 M1 舊版（pretrained embedding + 舊 XGBoost） | 已棄用 | 0.340 | 5.5% | 30.3% |
| **Round 6 Batch 3c `full`（現行）** | `signal_today_xgb.py` | **1.336** | 31.2% | **27.2%** |

Batch 3c `raw`（純技術指標 + cs_rank，完全不需要 Kronos）Sharpe 1.104，作為不依賴 GPU 的低成本
備援策略保留，但主力仍建議用 `full`。完整比較與方法論見
`docs/round6-embedding-xgb-lambdarank-improvements.md`。
