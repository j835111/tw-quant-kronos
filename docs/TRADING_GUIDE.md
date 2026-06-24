# Kronos 台股實盤操作指南

策略：top_k=10, hold_days=3，使用 Round 0 模型（Sharpe 1.92, 年化 85.6%）

## 前置準備（一次性）

```bash
cd /mnt/d/project/Kronos
pip install -r requirements.txt

# 確認 DB 存在（已 commit 到 git）
ls finetune_tw/data/tw_stocks.db
```

## 每日例行作業

### 每個交易日收盤後（13:35+）

```bash
python -m finetune_tw.download_data \
    --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \
    --update
```

約 2-3 分鐘，更新 DB 至今日收盤價。

### 每 3 個交易日：取訊號

```bash
# 無持倉 / 首次執行
python -m finetune_tw.signal_today \
    --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \
    --model round0 \
    --top_k 10 \
    --hold_days 3

# 已有持倉：帶入目前代碼，自動顯示換股建議
python -m finetune_tw.signal_today \
    --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \
    --model round0 \
    --top_k 10 \
    --hold_days 3 \
    --holdings 2330,2317,2454,2382,2308,2357,2412,3231,2379,2345
```

輸出範例：

```
=== Kronos 選股訊號 ===
  模型：round0  |  top_k=10  |  hold_days=3
  訊號日：2026-06-24  （預測 +3 個交易日後的收盤報酬）

【選股結果】預測報酬 top 10
排名      代碼    預測   +3日報酬
---------------------------------
   1      2330          +3.85%
   2      2317          +3.42%
   ...

【換股建議】
  繼續持有：{2330, 2317, ...}
  賣出：    {2454, 2382}
  買入：    {3231, 2379}
```

### 隔日開盤後（9:00-9:10）

1. 按「賣出」清單平倉
2. 等成交後，按「買入」清單建倉
3. 10 支等權配置（每支 10% 資金）

## 操作時間表

| 時間 | 動作 |
|------|------|
| 每天 13:35 後 | `download_data --update` |
| 每 3 個交易日收盤後 | `signal_today.py` 取換股清單 |
| 隔日 9:00-9:10 | 依清單下單（賣舊買新） |

建議固定在**週一、週四**換股，避開節假日干擾。

## 交易成本估算

| 項目 | 費率 |
|------|------|
| 買進手續費 | 0.1425%（可談折扣至 ~0.05%） |
| 賣出手續費 | 0.1425% |
| 證券交易稅（賣出）| 0.3% |
| 每次換手往返合計 | ~0.585% |

以平均 50% 換手率估算，年化成本侵蝕約 **15-20%**。  
回測年化 85.6%，扣除後仍有可觀空間，但需實單驗證滑點影響。

## 最低資金建議

- 台股一張 = 1000 股，均價 50 元 ≈ 5 萬元
- 10 支等權：**建議 100 萬以上**，避免零股問題與過度集中

## 無 GPU 環境

`signal_today.py` 自動 fallback 至 CPU，274 支股票推論約需 **30-60 分鐘**。  
建議在收盤後執行，隔天開盤前完成即可。

## 策略參數對照表（Grid Search 結果）

| top_k | hold_days | Sharpe | 年化報酬 | Max DD |
|-------|-----------|--------|----------|--------|
| 10    | 3         | 1.92   | 85.6%    | 25.5%  |
| 20    | 3         | 1.89   | ~68%     | 23.1%  |
| 10    | 5         | ~1.80  | ~65%     | ~22%   |

**若交易成本偏高或資金較小**，建議改用 `top_k=20, hold_days=3`：換手率相近但分散度更高，MaxDD 較低。
