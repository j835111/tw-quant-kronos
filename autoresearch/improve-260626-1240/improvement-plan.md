# Kronos TW 模型改進計劃

**ICP 過濾：** 台股量化動能策略研究者，部署限制：次日開盤掛單執行  
**評估基準：** `backtest_next_open v2`（open 信號 + open 執行），top_k=10，hold=5d  
**現狀：** Sharpe 1.356，Ann 50%，MaxDD 35%  
**目標：** Sharpe ≥ 1.5，MaxDD ≤ 20%

---

## ✅ Must-have（直接有效，高信心）

### M1 — 預測 ATR position sizing（降低 MaxDD）
**評分：** #1 Must-have | 信心：HIGH | 不需要重新訓練  
**問題：** MaxDD 35% 遠超目標 20%，且所有持倉大小目前相等（top_k 各 10%）。  
**機制：** 在 `signals_to_holdings()` 或 `build_portfolio_returns()` 中，用模型預測的 `(high - low) / close` 作為每檔股票的前向波動估計，position size ∝ 1/pred_ATR，再做 normalize：
```python
pred_atr = (pred["high"] - pred["low"]) / pred["close"]
# 用 hold=5d 對應的那一天預測值（iloc[4]）
weights[sym] = 1.0 / pred_atr_h5[sym]
weights /= weights.sum()  # normalize to 100%
```
**預期效果：** MaxDD 降至 20-25%（從 35%），Sharpe 可能小幅提升因為降低了高波動股的曝險  
**文獻：** arxiv:2605.19278, ATR scaling studies, 2025 backtests: +34% profitability  
**工作量：** 小（修改 `backtest_next_open.py`，無需重訓）

---

### M2 — 開盤到開盤 IC early stopping（重新對齊訓練目標）
**評分：** #2 Must-have | 信心：HIGH | 需要 1 次重訓  
**問題：** 現在的 `ic_validation.py` 計算 `pred["close"][h] / ctx_close - 1` vs actual close-to-close return，但我們的信號是 open-to-open，執行也是 open-to-open。訓練時選最好的 epoch 用的是錯誤的指標。  
**機制：** 修改 `validate_predictor_ic()` 和 `validate_predictor_ic_ir()` 改用 open-to-open：
```python
# 現在：
pred_returns.append(pred_close[h] / ctx_close - 1.0)
actual_returns.append(actual_close[h] / ctx_close - 1.0)

# 改成：
pred_returns.append(pred_open[h+1] / pred_open[0] - 1.0)  # open[T+h+1]/open[T+1]-1
actual_returns.append(actual_open[h+1] / actual_open[0] - 1.0)
```
`build_ctx_fn` 需要傳入 `ctx_open` 而不是 `ctx_close`，`actual_lookup` 需要返回 open 序列。  
**預期效果：** Sharpe 從 1.356 提升到 1.4-1.6（讓 early stopping 直接優化部署目標）  
**文獻：** Label Horizon Paradox (arxiv:2602.03395, ICML 2026)，我們自己的 v1→v2 實驗（+0.16 Sharpe）  
**工作量：** 中（修改 `ic_validation.py` + `train_predictor.py` 的 ctx 構建 + 重訓一次，約 20 個 epoch，4-6 小時）

---

### M3 — Volume 信心過濾器（無需重訓）
**評分：** #3 Must-have | 信心：MEDIUM | 不需要重新訓練  
**問題：** 在 top_k=10 的選股中，某些股票信號強但預測成交量極低，實際執行時滑點大。  
**機制：** 在 `compute_raw_signals_open()` 中，選股時排除預測 volume 落在底 25% 的股票：
```python
pred_vol_h1 = pred["volume"].iloc[0]
# 排除預測 volume 太低的（避免流動性差的股票）
if pred_vol_h1 < vol_threshold:  # threshold = 全 universe vol 的 25th percentile
    continue
```
或者：volume 加權 signal（signal × log(pred_vol)）  
**預期效果：** 減少低流動性持倉 → 降低 MaxDD，提升可執行性  
**文獻：** [Enhancing Intraday Momentum via Volume-Based Uncertainty, MDPI 2025]；Sharpe 3.02 in high-confidence regime  
**工作量：** 小（修改 backtest，一行過濾）

---

## 🔧 Nice-to-have（高效益，需要重訓）

### N1 — 輔助排名損失函數（Auxiliary Ranking Loss）
**評分：** #4 | 信心：HIGH | 需要重訓 + 架構修改  
**問題：** 現在的訓練損失是 token cross-entropy，本質是點估計，不直接優化股票間的相對排名。IC-IR early stopping 只在 eval 時看排名，訓練時完全沒有排名信號。  
**機制：** 在 `train_predictor.py` 的 loss 計算中加入輔助 IC loss：
```python
# 每個 batch 包含 B 支股票的同日預測
token_loss, _, _ = model.head.compute_loss(...)

# 輔助：預測排名 vs 實際排名的 rank correlation loss
pred_close_h5 = decode_batch_closes(batch, horizon=5)
actual_close_h5 = batch_targets[:, 5]
ic_loss = -rank_ic_differentiable(pred_close_h5, actual_close_h5)

total_loss = token_loss + alpha * ic_loss  # alpha ∈ [0.1, 1.0]
```
**預期效果：** IC 從 0.04 → 0.06-0.08，Sharpe 從 1.4 → 1.6+  
**文獻：** arxiv:2510.14156 (CIKM 2025)，MiM-StocR arxiv:2509.10461（ApproxNDCG listwise 在 CSI 上 SOTA）  
**工作量：** 大（需要讓訓練 batch 包含同日多股、設計 differentiable IC loss、重訓）

### N2 — 擴大 IC 驗證集（更穩定的 early stopping）
**評分：** #5 | 信心：MEDIUM | 需要重訓  
**問題：** 300 symbols × 20 dates = IC-IR 估計方差過高（導致 Round 2 選到錯誤 epoch）。  
**機制：** config 改 `ic_val_symbols: 500, ic_val_dates: 40`  
**預期效果：** Early stopping 更穩定，減少 epoch 選擇噪音  
**工作量：** 小（config 修改）+ 每 epoch 增加 50% 驗證時間

### N3 — Label Horizon 掃描（bi-level proxy label 搜尋）
**評分：** #6 | 信心：HIGH | 需要多次重訓  
**問題：** 最優訓練 label 的 horizon 可能不是 h=5d，可能是 h=3d 或 h=7d  
**機制：** 分別訓練 IC validation at h=3, h=5, h=7 的版本，比較 open-to-open backtest  
**文獻：** arxiv:2602.03395 (ICML 2026)  
**工作量：** 大（3 次重訓 × 4-6 小時）

---

## 🚀 Moonshot（長期，高收益高風險）

### S1 — 添加 price-space MSE 輔助損失
**評分：** #7 | 信心：MEDIUM  
**問題：** BSQ 離散化會丟失細粒度價格信息，且 token CE 與價格 MSE 不一致  
**機制：** 解碼預測 token 到 price space，計算 MSE(pred_price, actual_price)，加入訓練：
```python
decoded_prices = tokenizer.decode(pred_s1, pred_s2)
price_mse_loss = F.mse_loss(decoded_prices, target_prices)
total_loss = token_loss + beta * price_mse_loss
```
**工作量：** 非常大（需要修改模型結構、重訓 tokenizer）

### S2 — 連續回歸 head（放棄 BSQ 離散化）
**評分：** #8 | 信心：LOW（根本架構改變）  
**機制：** 參考 Chronos-2/TimesFM，用 quantile head 替代 BSQ，直接預測 price distribution  
**工作量：** 極大（論文級工作量）

---

## 建議執行順序

```
Week 1（無需重訓，快速驗證）：
  M1 (ATR sizing) → 跑 backtest → 確認 MaxDD 改善
  M3 (Volume filter) → 跑 backtest → 確認選股品質

Week 2-3（一次重訓）：
  M2 (open-to-open IC) + N2 (擴大驗證集) → 同時改，一起重訓
  → 評估：Sharpe 是否 > 1.5？MaxDD 是否 < 20%？

Week 4+（如果還不夠）：
  N1 (ranking loss) → 更大架構改動
  N3 (label horizon scan) → 多次實驗
```
