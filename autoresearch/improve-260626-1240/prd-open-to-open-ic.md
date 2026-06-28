# PRD: Open-to-Open IC Early Stopping

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## Problem Statement

現在的 IC validation（`ic_validation.py`）計算的是 **close-to-close** IC：
- 預測信號：`pred["close"][h] / ctx_close - 1`
- 實際報酬：`actual_close[h] / ctx_close - 1`

但我們的信號層（v2）和執行層都是 **open-to-open**：
- 信號：`pred_open[h+1] / pred_open[0] - 1`
- 實際執行：`actual_open[T+h+1] / actual_open[T+1] - 1`

這是訓練目標（close-to-close IC → early stopping）與部署目標（open-to-open return）的不對齊。  

**理論依據：** The Label Horizon Paradox (ICML 2026, arxiv:2602.03395) 論文的核心主張：訓練標籤應該解耦於推理目標，最優監督信號由 signal-noise 競爭決定。

**實驗依據：** 我們的 v1→v2 實驗：只改信號層（close→open），Sharpe 從 1.196 → 1.356（+13.4%）。若 early stopping 也改用 open-to-open IC，下一次重訓的起點和 epoch 選擇都將更對齊。

---

## User Stories

- 作為研究者，我希望 early stopping 選出的 checkpoint 是「最能預測 open-to-open 報酬」的 epoch，而不是「最能預測 close-to-close 報酬」的 epoch
- 作為訓練工程師，我希望修改集中在 `ic_validation.py`，不需要重構 `train_predictor.py` 的主迴圈

---

## Requirements

### Functional (MoSCoW)

**Must:**
- [ ] `_collect_rows_for_date()` 改為返回 `(sym, pred_open_arr, pred_open_t1, last_date)` 而非 `pred_close_arr`
  - `pred_open_arr` = `pred["open"].values`（length = pred_len）
  - `pred_open_t1` = `pred["open"].iloc[0]`（= 預測的 open[T+1]，基準點）
- [ ] `validate_predictor_ic()` 改為：
  ```python
  # 信號：pred_open[h+1] / pred_open[0] - 1
  pred_returns.append(pred_open[h+1] / pred_open_t1 - 1.0)
  # 實際：actual_open[h+1] / actual_open[0] - 1
  actual_returns.append(actual_open[h+1] / actual_open_t1 - 1.0)
  ```
  其中 h 的 range 改為 0..horizons-1（讓 h=0 對應 open[T+2]/open[T+1]-1）
- [ ] `validate_predictor_ic_ir()` 同樣改用 open-to-open
- [ ] `actual_lookup` 函式需要返回 **open** 序列而非 close 序列（傳給這兩個函式）
- [ ] `build_ctx_fn` 需要返回 `ctx_open`（最後已知開盤價）而非 `ctx_close`

**Should:**
- [ ] `pred_len` 需要是 `val_ic_horizons + 1`（因為 open[T+h+1] 需要多一步），確認 config 的 `pred_len` 夠大
- [ ] 在 log 中標示 IC 計算模式（`ic_mode: open-to-open`）

**Won't:**
- 改變 IC 計算的評估 horizon 數（還是 `val_ic_horizons`）
- 改變 EarlyStopper 的邏輯

### Non-functional

- 訓練速度影響 < 10%（只是改計算的 reference column）
- 修改後 IC 值可能和以前不可直接比較（open IC vs close IC 不同 scale），**這是預期行為**

---

## Acceptance Criteria

- [ ] 用修改後的 `ic_validation.py` 跑一次重訓，訓練 log 中 `val_ic` 和 `ic_ir_h5` 有非零值
- [ ] Backtest 用 `backtest_next_open v2`（open 信號）評估：Sharpe ≥ 1.4（vs 現在 1.356）
- [ ] Early stopping 選出的 epoch 和舊版本不同（說明確實在優化不同目標）

---

## Technical Approach

**修改檔案：** `finetune_tw/ic_validation.py`，`finetune_tw/train_predictor.py`

### ic_validation.py 關鍵改動

```python
def _collect_rows_for_date(predict_batch_fn, val_universe, date, cfg, build_ctx_fn, batch_size=64):
    """Return (sym, pred_open_arr, pred_open_t1, last_date) per symbol."""
    ...
    for offset, pred in enumerate(preds):
        if pred is None or len(pred) < getattr(cfg, "val_ic_horizons", cfg.pred_len) + 1:
            continue
        i = start + offset
        rows.append((syms[i],
                     pred["open"].values.astype(float),   # ← 改 close→open
                     float(pred["open"].iloc[0]),          # ← pred open[T+1] 基準
                     last_dates[i]))
    return rows


def validate_predictor_ic(...) -> float:
    horizons = min(cfg.val_ic_horizons, cfg.pred_len - 1)  # -1 因為需要 open[h+1]
    for date in val_dates:
        rows = _collect_rows_for_date(...)
        for horizon in range(horizons):
            pred_returns, actual_returns = [], []
            for sym, pred_open, pred_open_t1, last_date in rows:
                actual_open = np.asarray(actual_lookup(sym, last_date, cfg.pred_len), dtype=float)
                if len(actual_open) <= horizon + 1:
                    continue
                if pred_open_t1 <= 0 or actual_open[0] <= 0:
                    continue
                pred_returns.append(pred_open[horizon + 1] / pred_open_t1 - 1.0)   # ← open-to-open
                actual_returns.append(actual_open[horizon + 1] / actual_open[0] - 1.0)
```

### train_predictor.py 關鍵改動

`actual_fn` 和 `ctx_fn` 需要改回傳 open 序列：

```python
def actual_fn(sym, last_date, n):
    """Return next n business days' OPEN prices after last_date."""
    df = query_symbol(cfg.db_path, sym, start=..., end=...)
    # 取 last_date 之後的 open（不是 close）
    future = df[df["date"] > last_date]["open"].values[:n]
    return future

def ctx_fn(sym, date):
    ...
    # 改回傳 ctx_open 而非 ctx_close
    ctx_open = ctx_df["open"].iloc[-1]
    return ctx_df, x_ts, y_ts, last_date, ctx_open  # ← 最後一個是 ctx_open
```

**DECISION NEEDED:**  
IC horizon h=0 現在代表 `open[T+2]/open[T+1]-1`（1天）或 `open[T+h+2]/open[T+1]-1`？  
建議：h=0 → 1 天後；h=4 → 5 天後（ic_ir_h5 target）。

---

## Risks & Confidence

| 風險 | 程度 | 緩解 |
|------|------|------|
| Open 序列 actual_lookup 需要確認 DB 有 open 欄位 | LOW | DB 已有 open 欄位（`query_symbol` 返回 OHLCV）|
| IC 值會比以前低（open 預測比 close 更難） | MEDIUM | 這是正確的——之前 IC 偏高是因為測的是更容易的 close 任務 |
| pred_len 不夠（需要多 1 步） | MEDIUM | 確認 `cfg.pred_len >= cfg.val_ic_horizons + 1`，加斷言 |

**Evidence tier:** PRIMARY（我們的 v1→v2 實驗）+ PRIMARY（Label Horizon Paradox ICML 2026）

---

## Success Metrics

| 指標 | 基準 | 目標 |
|------|------|------|
| Backtest Sharpe (v2, 5d) | 1.356 | ≥ 1.4 |
| Train log ic_ir_h5 | 0.40（close-to-close） | 任何非零值（新 metric，不可直接比較）|
| MaxDD（配合 M1） | 35% | ≤ 20% |

---

## Open Questions

1. 要不要同時把訓練 loss 的 horizon 也改？（現在 token CE 是對 `pred_len=10` 步都算損失）
2. 是否保留 `validate_predictor_ic()` 的 close-to-close 版本作為診斷指標（即使不用它做 early stopping）？
3. `val_ic_horizons` config 要不要從 10 改成 5（只評估 h1-h5，因為 h=5d 是我們的部署設定）？
