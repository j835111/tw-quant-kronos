# Plan: Kronos TW 模型改進（ATR sizing + Volume filter + Open-to-open IC）

**Branch:** feature/atr-vol-open-ic  
**Merge base:** 37f63b38fe9bbb1797600c2bd409ae51d34f3a00  
**PRDs:** autoresearch/improve-260626-1240/prd-*.md

## Global Constraints

- Python 3.10+，無新套件依賴（只用已有的 pandas、numpy、torch）
- 不修改模型結構（model/kronos.py）
- 不修改 tokenizer
- 所有改動必須有對應的 unit test（pytest）
- 新函式不加多行 docstring；必要的話一行說明
- 不改變現有 backtest CLI 的預設行為（改動必須是 opt-in 或向後相容）
- 不破壞 `pytest tests/` 現有測試

## Tasks

### Task 1: M1 — ATR Position Sizing（`backtest_next_open.py`）

**Files:** `finetune_tw/backtest_next_open.py`  
**PRD:** `autoresearch/improve-260626-1240/prd-atr-position-sizing.md`

目標：把 `run_backtest_next_open()` 的等權重持倉改為「1/pred_ATR 加權」，用模型已輸出的 `pred["high"]` 和 `pred["low"]` 計算每檔股票的預測振幅。

**具體要求：**

1. 新增函式 `compute_atr_weights(raw_preds: dict, hold_days: int, selected_syms: list, min_atr: float = 0.003) -> dict`
   - 對每個 sym：取 `pred["high"].iloc[H]`、`pred["low"].iloc[H]`、`pred["close"].iloc[H]`，其中 `H = min(hold_days - 1, len(pred) - 1)`
   - `pred_atr = max((high - low) / close, min_atr)` if close > 0 else min_atr
   - `weights[sym] = 1.0 / pred_atr`
   - Normalize：`total = sum(weights.values()); return {sym: w/total for sym, w in weights.items()}`
   - 如果 sym 在 raw_preds 中找不到，給 weight = 1.0（再 normalize）

2. 在 `run_backtest_next_open()` 新增參數 `use_atr_weights: bool = False`

3. 在 `build_portfolio_returns()` 的呼叫前，如果 `use_atr_weights=True`，對每個 rebal_date 用 `compute_atr_weights()` 計算 weights，傳給 `build_portfolio_returns()`

4. `build_portfolio_returns()` 必須接受可選的 `weights: dict | None = None` 參數：
   - `None` 時維持等權重（向後相容）
   - 傳入 dict 時用指定 weights（dict 的 sum 應已為 1.0）

5. CLI: `--atr-weights` flag，啟用 `use_atr_weights=True`

6. 測試：`tests/finetune_tw/test_backtest_next_open.py`（若不存在則建立）
   - `test_compute_atr_weights_basic`：3 支股票，不同振幅，驗證 weights sum=1，高振幅股 weight < 低振幅股
   - `test_compute_atr_weights_min_atr_clamp`：振幅=0 時被 clamp 到 min_atr
   - `test_compute_atr_weights_missing_sym`：sym 不在 raw_preds 中，給 1.0 再 normalize
   - `test_build_portfolio_returns_equal_weight`：weights=None 等權重不變（regression）
   - `test_build_portfolio_returns_custom_weights`：傳 weights dict 時按指定比例計算

### Task 2: M3 — Volume Confidence Filter（`backtest_next_open.py`）

**Files:** `finetune_tw/backtest_next_open.py`  
**PRD:** `autoresearch/improve-260626-1240/prd-volume-filter.md`  
**依賴：** Task 1 完成後執行（同一個檔案）

在 `compute_raw_signals_open()` 中新增 volume 過濾，排除預測成交量低於第 N percentile 的股票。

**具體要求：**

1. `compute_raw_signals_open()` 新增參數 `vol_filter_pct: float = 0.0`（預設關閉，0 = 不過濾）

2. 每個 rebal_date 計算完 date_preds 後：
   ```python
   if vol_filter_pct > 0 and date_vols:
       vols = np.array(list(date_vols.values()))
       threshold = np.percentile(vols, vol_filter_pct)
       before = len(date_preds)
       date_preds = {sym: s for sym, s in date_preds.items()
                     if date_vols.get(sym, 0) >= threshold}
       removed = before - len(date_preds)
       if removed > 0:
           print(f"  [vol-filter] {rebal_date}: removed {removed} low-vol symbols")
   ```

3. `date_vols[sym] = float(pred["volume"].iloc[0])`（預測 entry 當天成交量）在 signals 計算迴圈內一併收集

4. `run_backtest_next_open()` 新增參數 `vol_filter_pct: float = 0.0`，傳給 `compute_raw_signals_open()`

5. CLI: `--vol-filter-pct FLOAT`，預設 0.0（不過濾）

6. 測試：加到 `tests/finetune_tw/test_backtest_next_open.py`
   - `test_vol_filter_disabled`：vol_filter_pct=0 時不過濾，結果與不傳相同
   - `test_vol_filter_removes_low_vol`：5 支股票，2 支 volume 很低，設 pct=40，確認那 2 支被排除
   - `test_vol_filter_100pct_keeps_nothing`：pct=100 時所有股票都被移除，回傳空 dict

### Task 3: M2 — Open-to-Open IC Early Stopping（`ic_validation.py` + `train_predictor.py`）

**Files:** `finetune_tw/ic_validation.py`、`finetune_tw/train_predictor.py`  
**PRD:** `autoresearch/improve-260626-1240/prd-open-to-open-ic.md`

把 IC validation 從 close-to-close 改成 open-to-open，對齊模型的部署目標。

**具體要求（ic_validation.py）：**

1. `_collect_rows_for_date()` 返回 `(sym, pred_open_arr, pred_open_t1, last_date)`
   - `pred_open_arr`: `pred["open"].values.astype(float)` （length >= pred_len）
   - `pred_open_t1`: `float(pred["open"].iloc[0])`  ← open[T+1] 的預測基準
   - 把 `pred["close"].values` 的收集改成 `pred["open"].values`
   - `ctx_close` 改名為 `ctx_ref`（傳入的語義不變，保持接口相容）

2. `validate_predictor_ic()` 改用 open-to-open：
   - `pred_returns.append(pred_open[h+1] / pred_open_t1 - 1.0)` （h 從 0 開始，h+1 是下一步）
   - `actual_returns.append(actual_open[h+1] / actual_open_t1 - 1.0)`
   - `actual_open`: 從 `actual_lookup(sym, last_date, pred_len)` 取得的 open 序列
   - `actual_open_t1 = actual_open[0]` 為基準
   - `horizons = min(cfg.val_ic_horizons, cfg.pred_len - 1)`（-1 確保 h+1 不越界）
   - 若 `pred_open_t1 <= 0` 或 `actual_open_t1 <= 0` 則 skip

3. `validate_predictor_ic_ir()` 同樣改用 open-to-open（target_horizon 語義不變，但現在算的是 open[T+h+1]/open[T+1]-1）：
   - `pred_returns.append(pred_open[h+1] / pred_open_t1 - 1.0)`
   - `actual_returns.append(actual_open[h+1] / actual_open_t1 - 1.0)`
   - 其中 `h = min(target_horizon, cfg.pred_len - 1) - 1`（0-indexed，注意邊界）

**具體要求（train_predictor.py）：**

4. `actual_fn(sym, last_date, n)` 改回傳 **open** 序列（不是 close）：
   ```python
   # 改成：
   future_df = df[df["date"] > pd.Timestamp(last_date)].head(n)
   return future_df["open"].values  # ← open 不是 close
   ```

5. `ctx_fn(sym, date)` 最後一個回傳值改為 `ctx_open`（最後已知開盤價）：
   ```python
   ctx_open = float(ctx_df["open"].iloc[-1])  # ← open 不是 close
   return ctx_df, x_ts, y_ts, last_date, ctx_open
   ```
   （`_collect_rows_for_date` 的 `ctx_close` 引數改名已在 ic_validation.py 中，這裡對應）

6. 測試：`tests/finetune_tw/test_ic_validation.py`（若不存在則建立）
   - `test_rank_ic_basic`：已有或補充基本 rank_ic 測試
   - `test_collect_rows_returns_open`：mock predict_batch_fn，驗證回傳的 pred_open_arr 是 open 欄位而非 close
   - `test_validate_ic_open_to_open`：5 symbols，mock 預測 open 序列，驗證 IC 計算用 open[h+1]/open[0]-1
   - `test_validate_ic_ir_open_to_open`：同上，驗證 IC-IR

### Task 4: N1 — Auxiliary Ranking Loss（`train_predictor.py`）

**Files:** `finetune_tw/train_predictor.py`  
**PRD:** `autoresearch/improve-260626-1240/prd-ranking-loss.md`  
**依賴：** Task 3 完成後執行（同一個檔案，要在 M2 的改動基礎上加）

在 predictor fine-tuning 的 training loop 中加入輔助 rank IC loss。

**具體要求：**

1. 新增函式 `differentiable_rank_ic_loss(pred_scores: torch.Tensor, actual_scores: torch.Tensor) -> torch.Tensor`
   - 用 soft rank / negative Pearson correlation（對 rank 的 differentiable approximation）
   - 實作：`-(z_pred * z_actual).mean()`，其中 `z = (x - x.mean()) / (x.std() + 1e-8)`
   - 若 batch size < 2，回傳 `torch.tensor(0.0, device=pred_scores.device)`

2. 在 config（`Config` 類別或 yaml 讀取）新增兩個參數：
   - `ranking_loss_alpha: float = 0.0`（預設 0 = 關閉）
   - `ranking_loss_horizon: int = 5`（哪個 horizon 的 open-to-open 作為 ranking target）

3. 在 training loop 的 loss 計算後，若 `cfg.ranking_loss_alpha > 0`：
   - 從 batch 中取出同 token sequence 的位置 `H = cfg.ranking_loss_horizon`（0-indexed）
   - `pred_scores`：從 model logits 用 argmax decode 出 H 步的 open price（用 tokenizer.decode_s1_s2()），再計算 open[H]/open[0]-1
   - `actual_scores`：從 batch_y 取對應的 actual open return（需要 batch 包含 actual future open）
   - `total_loss = token_loss + cfg.ranking_loss_alpha * ranking_loss`

4. **DECISION（由實作者判斷）**：如果取得 actual future open 代價過大（需要大幅修改 Dataset），可以先做 soft 版本：只對 pred_scores（decode 出的預測報酬）做自我 rank consistency loss（pending actual target），並在 commit message 中說明。

5. Config yaml 需新增對應欄位（`config_tw_daily_rtx6000.yaml` 加上 `ranking_loss_alpha: 0.0` 和 `ranking_loss_horizon: 5`）

6. 測試：`tests/finetune_tw/test_ranking_loss.py`
   - `test_differentiable_rank_ic_loss_perfect`：pred 和 actual 完全相同時 loss 應接近 -1.0
   - `test_differentiable_rank_ic_loss_reversed`：pred 和 actual 完全相反時 loss 應接近 +1.0（最差）
   - `test_differentiable_rank_ic_loss_single_item`：batch_size=1 時回傳 0.0
   - `test_ranking_loss_alpha_zero_no_effect`：alpha=0 時 total_loss == token_loss

## Implementation Notes

- Task 1 和 Task 2 都在 `backtest_next_open.py`，必須按序（Task 2 在 Task 1 完成後）
- Task 3 和 Task 4 都在 `train_predictor.py`，必須按序（Task 4 在 Task 3 完成後）
- Task 1/2 和 Task 3/4 之間沒有依賴，但為簡化 review，按 1→2→3→4 順序執行
