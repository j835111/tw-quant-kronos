# Improvement Plan — Kronos TW Round 6（全新方向）
> 研究日期：2026-07-01 | 依據：10 次 web research 迭代 + 完整歷史記錄

---

## 戰略轉向說明

**五輪 fine-tuning 的核心教訓**：我們一直在試圖讓 Kronos 預測台股的未來報酬——而大量文獻（arXiv:2511.18578 等）和我們自己的實驗（Rounds 1-5）都證明，**pretrained TSFM fine-tuning 在金融回報預測上系統性失敗**。

**新策略**：停止嘗試讓 Kronos 直接預測，改用 Kronos 作為**特徵抽取器**，並搭配**專門為金融排名設計的 loss function（LambdaRankIC）**。

---

## Must-Have（高優先）

---

### M1 — Kronos Embedding → XGBoost + LambdaRankIC
**信心：HIGH | 工程難度：MEDIUM（1-2 天）| 預算：低（本機 CPU/GPU）**

**原理**：凍結 Kronos-base predictor，提取每支股票的 OHLCV 序列在 Transformer 最後一層的 hidden state（d_model=512 向量），接著用 XGBoost 搭配 LambdaRankIC loss 直接優化 Rank IC。

**為何這是突破**：
- 完全繞過 catastrophic forgetting（Kronos 不更新）
- LambdaRankIC 在低 SNR / heavy-tail noise 下一致優於 regression 和 ListMLE（arXiv:2605.00501）
- Kronos token space 的 IC~0.04 弱，但 hidden state 的 512 維向量可能含有截面排名信息，XGBoost 可非線性提取
- 訓練在 CPU 上幾分鐘可完成

**實作步驟**：

```python
# Step 1: 提取 Kronos hidden states（新建 extract_embeddings.py）
# 對每個 (date, symbol) 取 lookback_window=90 的 OHLCV
# 送入 KronosPredictor → 在 decode 前取 transformer 最後層 hidden state
# 輸出：dict[date][symbol] = np.array(d_model,)  or  mean-pooled token sequence

# Step 2: 用 XGBoost + LambdaRankIC 訓練排名模型
# 訓練集：2015-01-01 → 2024-06-30
# 特徵：hidden state (512) + 可選 raw OHLCV statistics (mean, std, skew of close returns, volume)
# 目標：open-to-open return at h=5d
# Loss：LambdaRankIC (custom XGBoost objective, 參考 arXiv:2605.00501 梯度公式)
# 每日 group：同一個 date 的所有 symbols 為一個 query group

# Step 3: 推理方式與現有 backtest_next_open.py 相同（top_k=10, hold=5d）
```

**關鍵決策**：
- Hidden state 取法：選 `s1/s2 token sequence 的 last token position` vs `mean pooling`
  → 建議先試 mean pooling（穩定），再試 last position
- XGBoost LambdaRankIC：用 [2605.00501] Equation 5 的 lambda gradient，或先用近似版 `rank:ndcg` 評估基線

**檔案**：
- 新建：`finetune_tw/extract_embeddings.py`
- 新建：`finetune_tw/train_xgb_lambdarank.py`
- 使用：`finetune_tw/data_loader.py` 的 DB 讀取邏輯

---

## Nice-to-Have（中優先）

---

### N1 — L2-SP 正則化
**信心：MEDIUM | 工程難度：LOW（2 小時）**

在 fine-tuning loss 中加一行正則化，懲罰偏離 pretrained 的程度：
```python
# train_predictor.py, training loop 中
l2_sp_loss = sum((p - p0).pow(2).sum() for p, p0 in zip(model.parameters(), pretrained_params))
loss = ce_loss + lambda_sp * l2_sp_loss
# lambda_sp = 1e-4（初始值，可 grid search 0.01, 0.001, 0.0001）
```

比 EWC 更簡單（不需 Fisher matrix），理論上可限制 val_loss 上升。

---

### N2 — MoFO Optimizer（替換 AdamW）
**信心：MEDIUM | 工程難度：MEDIUM（4-6 小時）**

```python
# 替換 optimizer，只更新動量最大的 top-K% 參數
# 參考：https://github.com/YChen-zzz/MoFO
```

比 FPT Selective Freeze 更靈活（不是靜態凍結 self_attn），讓模型自己決定哪些參數需要更新。

---

### N3 — SSPT 台股持續預訓練
**信心：MEDIUM | 工程難度：HIGH（3-5 天）**

在 Kronos-base 上加三個自監督任務，使用台股資料持續預訓練：
1. **股票代碼分類**：給 90 天序列，預測是哪支股票（1091-class classification）
2. **產業分類**：預測產業別（electronics, finance, etc.）
3. **移動平均預測**：預測 20MA, 60MA 值

這讓 Kronos 先學台股的身份特徵，再 fine-tune 預測任務，理論上比直接從 pretrained 開始更好。

---

## DECISION NEEDED

1. **M1 的 hidden state 取法**：mean pooling vs last token vs CLS-equivalent？
   - 建議：mean pooling 先跑，last token 作為 ablation

2. **M1 的特徵工程**：純 hidden state vs hidden state + raw features（MA, momentum, volume ratio）？
   - 建議：先試純 hidden state，如果效果弱再加 raw features

---

## 執行建議

```
Week 1:
  Day 1-5: M1（extract_embeddings.py + train_xgb_lambdarank.py）

Week 2:
  Day 1: 比較 M1 vs Round 0 基準
  Day 2+: 視結果決定 N1/N2/N3
```

---

## 目標對照

| 指標 | Round 0 基準 | 目標 |
|------|------------|------|
| open/open Sharpe（hold=5d）| 1.12 | ≥ 1.5 |
| Annual Return | 38.6% | > 15% |
| MaxDD | 35% | < 20% |

---

## 參考文獻

| Paper | 方向 | arXiv |
|-------|------|-------|
| LambdaRankIC | M1 ranking loss | 2605.00501 |
| Re(Visiting) TSFMs in Finance | 戰略依據 | 2511.18578 |
| MoFO | N2 optimizer | 2407.20999 |
| SSPT | N3 pretraining | 2506.16746 |
| EWC Done Right | N1 regularization | 2603.18596 |
| Pretrained TSFM for Financial Return Forecasting | 現況對照 | 2606.27100 |
