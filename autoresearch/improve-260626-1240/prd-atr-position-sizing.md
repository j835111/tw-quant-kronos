# PRD: ATR Position Sizing（預測 ATR 波動加權）

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## Problem Statement

當前 `backtest_next_open.py` 對 top_k 所有股票使用**等權重**持倉（每檔 1/top_k = 10%）。  
模型輸出中已包含 `high` 和 `low` 欄位——但完全沒有被使用。  
結果：MaxDD = 35%，超過目標 20%。高波動股（振幅 2.4%）和低波動股（振幅 1.3%）佔用同樣資金，導致組合風險不均衡。

**核心洞見：** 把等權重改成「1/pred_ATR 加權」，讓每檔股票對組合貢獻的*風險*相同，而非*資金*相同。

---

## User Stories

- 作為研究者，我希望每筆持倉的預期風險貢獻相同，這樣高波動個股不會主導組合的最大回撤
- 作為策略部署者，我希望不需要重新訓練模型就能改善 MaxDD，快速驗證效果

---

## Requirements

### Functional (MoSCoW)

**Must:**
- [ ] 計算每個 rebalancing date 所有 top_k 候選股票的預測 ATR
- [ ] `pred_atr[sym] = (pred["high"].iloc[H] - pred["low"].iloc[H]) / pred["close"].iloc[H]`  其中 H = `hold_days - 1`（持有期最後一天）
- [ ] 等權重替換為 `weight[sym] = 1 / pred_atr[sym]`，再 normalize 到總和=1
- [ ] 在選股（top_k 過濾後）套用加權，不改變選哪些股的邏輯

**Should:**
- [ ] 加上 ATR clip：`pred_atr = max(pred_atr, min_atr_threshold)` 防止除零或極端值  
  建議 `min_atr_threshold = 0.003`（0.3%，等同於漲跌幅極限 10% 的 3%）
- [ ] Log 每個 rebal date 的有效 ATR 分布（min/max/mean）供調試

**Won't (in this version):**
- 改變 top_k 選股邏輯（那是 M3 的工作）
- 修改模型或重訓

### Non-functional

- 執行時間增加 < 5%（只是簡單的除法運算）
- 結果可重現（seed 固定）

---

## Acceptance Criteria

- [ ] `backtest_next_open v2`（open 信號）在 top_k=10, hold=5d 下 MaxDD ≤ 25%（目標：最終降到 20%）
- [ ] Sharpe 不下降（ATR 加權理論上只影響分散度，不應大幅改變選股 alpha）
- [ ] 程式碼：等權版本作為 `--equal-weight` flag 保留，便於對比

---

## Technical Approach

**修改檔案：** `finetune_tw/backtest_next_open.py`

### 關鍵改動位置

在 `signals_to_holdings()` 函式（或其呼叫端）中，選出 top_k 後：

```python
def compute_atr_weights(raw_preds: dict, hold_days: int, selected_syms: list) -> dict:
    """Return normalized inverse-ATR weights for selected symbols."""
    MIN_ATR = 0.003
    weights = {}
    for sym in selected_syms:
        pred = raw_preds.get(sym)
        if pred is None:
            weights[sym] = 1.0
            continue
        h = min(hold_days - 1, len(pred["high"]) - 1)
        hi = pred["high"].iloc[h]
        lo = pred["low"].iloc[h]
        cl = pred["close"].iloc[h]
        atr = max((hi - lo) / cl, MIN_ATR) if cl > 0 else MIN_ATR
        weights[sym] = 1.0 / atr
    # Normalize
    total = sum(weights.values())
    return {sym: w / total for sym, w in weights.items()}
```

在 `build_portfolio_returns()` 中，把目前的：
```python
weights = {sym: 1.0 / top_k for sym in selected}
```
改為：
```python
weights = compute_atr_weights(raw_preds, hold_days, selected)
```

### 需要傳遞的額外參數

`build_portfolio_returns()` 目前沒有 `raw_preds`。需要在 `run_backtest_next_open()` 中把 `raw_preds` 傳下去，或者在 `signals_to_holdings()` 時就附加 weight 信息。

**DECISION NEEDED:** 要把 weights 計算放在 `signals_to_holdings()` 還是 `build_portfolio_returns()`？前者更乾淨但需要 `signals_to_holdings()` 也拿到 `raw_preds`；後者侵入性較小。

---

## Risks & Confidence

| 風險 | 程度 | 緩解 |
|------|------|------|
| 模型的 high/low 預測可能很不準 | MEDIUM | 先用 `backtest_next_open` 驗證，看 ATR 預測分布是否合理 |
| ATR 加權後集中在低波動股，可能降低 alpha | LOW | 保留 `--equal-weight` flag 快速對比 |
| 計算 ATR 時 hold_days 的 H 選哪一天 | LOW | 用持有期最後一天（`H = hold_days - 1`）最能代表退出時的風險 |

**Evidence tier:** PRIMARY（我們自己的模型輸出數據）+ SECONDARY（ATR sizing 業界實踐）

---

## Success Metrics

| 指標 | 基準（v2 等權） | 目標 |
|------|----------------|------|
| MaxDD | 35% | ≤ 25% |
| Sharpe | 1.356 | ≥ 1.356（不退步）|
| Annual Return | 50% | ≥ 45% |

---

## Open Questions

1. ATR 用持倉期第幾天的預測？（H=0 = 入場那天，H=hold_days-1 = 最後一天，H=mean 平均）
2. 要不要同時做「高波動排除」（即 M3 的 volume filter 概念但用 ATR）vs 僅加權？
3. 是否要記錄每筆交易的實際持倉權重到輸出 JSON，供後續分析？
