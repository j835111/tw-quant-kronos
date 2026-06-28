# PRD: Volume Confidence Filter（預測成交量過濾）

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## Problem Statement

現在選股只考慮「預測報酬排名最高的 top_k 支」，但沒有考慮**預測成交量**。  
模型輸出的 `volume` 欄位完全未被使用。  
風險：某些股票信號強但預測成交量低 → 執行時滑點大、難以建倉。

**文獻支持：** 「Enhancing Intraday Momentum via Volume-Based Information Uncertainty（MDPI 2025）」：結合開盤報酬信號與 volume-based 不確定性，在高不確定性 regime 下準確度 71.43%，Sharpe 3.02。  
核心邏輯：volume 高 = 市場參與度高 = 信號更可信。

---

## User Stories

- 作為研究者，我希望選出的 top_k 股票在預測成交量上有最低門檻，避免持有「沒有人交易」的股票
- 作為策略部署者，我希望這個過濾是可選的（flag），不強制破壞目前的 baseline

---

## Requirements

### Functional (MoSCoW)

**Must:**
- [ ] 在 `compute_raw_signals_open()` 中，對每個 (date, sym) 提取預測 volume（entry 當天，`pred["volume"].iloc[0]`）
- [ ] 計算當日全 universe 的預測 volume 分布，排除低於第 N percentile 的股票（`N=25` 為默認）
- [ ] 過濾在**信號計算階段**進行，不是在 top_k 選擇後

**Should:**
- [ ] 以 `--vol-filter-pct N` flag 控制 percentile（預設 25，設為 0 即關閉）
- [ ] Log 每個 rebal date 被過濾掉的股數

**Won't:**
- 修改模型
- 用 volume 做加權（那是一個不同的機制，與 M1 的 ATR 加權可以正交組合）

---

## Acceptance Criteria

- [ ] 在 top_k=10, hold=5d 下，過濾後 Sharpe ≥ 1.356（不退步）
- [ ] 每個 rebal date 平均被過濾掉 10-30% 股票（過多說明門檻太高）
- [ ] `--vol-filter-pct 0` 的結果與原始 v2 完全一致（regression test）

---

## Technical Approach

**修改檔案：** `finetune_tw/backtest_next_open.py`

```python
def compute_raw_signals_open(predictor, cfg, rebal_dates, pred_len, symbols,
                              vol_filter_pct: float = 25.0):
    ...
    for rebal_date in rebal_dates:
        date_preds = {}
        date_vols  = {}  # ← 新增

        for b in range(0, len(batch_syms), BATCH_SIZE):
            ...
            for sym, pred in zip(..., preds):
                if pred is not None and len(pred) >= pred_len:
                    pred_opens = pred["open"].reset_index(drop=True)
                    date_preds[sym] = pred_opens.iloc[1:].reset_index(drop=True) / pred_opens.iloc[0] - 1
                    date_vols[sym]  = float(pred["volume"].iloc[0])   # ← entry 當天預測量

        # Volume filter
        if vol_filter_pct > 0 and date_vols:
            vols = np.array(list(date_vols.values()))
            threshold = np.percentile(vols, vol_filter_pct)
            filtered = {sym for sym, v in date_vols.items() if v < threshold}
            date_preds = {sym: s for sym, s in date_preds.items() if sym not in filtered}
            if filtered:
                print(f"  [vol-filter] {rebal_date}: removed {len(filtered)} low-vol symbols")

        raw_preds[rebal_date] = date_preds
```

**DECISION NEEDED:**
1. 用 `volume.iloc[0]`（入場當天預測量）還是 `volume.mean()`（持倉期平均預測量）？
2. percentile 基準是「當日全 universe」還是「rolling 歷史 universe 平均」？（當日更動態，歷史更穩定）

---

## Risks

| 風險 | 程度 | 緩解 |
|------|------|------|
| 模型的 volume 預測可能比 price 更不準 | MEDIUM | 先做 EDA：看 pred_vol vs actual_vol 的相關性 |
| 低 volume 股票不一定是壞信號（可能是冷門但漲幅大的） | MEDIUM | 從 25th percentile 開始，逐步調高 |
| 過濾後 top_k 可能選不到 10 支 | LOW | 若過濾後剩餘 < top_k，降低門檻或直接用剩餘全部 |

---

## Success Metrics

| 指標 | 基準 | 目標 |
|------|------|------|
| Sharpe | 1.356 | ≥ 1.356 |
| MaxDD | 35% | ≤ 32% |
| 每 rebal date 過濾比例 | 0% | 10-30% |

---

## Open Questions

1. 是否要先做一次 EDA：比較 `pred["volume"]` 和實際成交量的相關性？（確認模型的 volume 預測是否有信息量）
2. Volume filter 和 M1 ATR sizing 要同時上嗎？建議分開實驗以隔離效果。
