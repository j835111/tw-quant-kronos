# PRD: Auxiliary Ranking Loss（輔助排名損失函數）

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## Problem Statement

Kronos predictor 的訓練損失是 **token cross-entropy**（離散 BSQ token 的預測準確率）。  
這是 pointwise 損失：對每個 token 獨立最小化錯誤，不考慮不同股票之間的相對排名。

但選股任務的核心目標是**截面排名**：只要能正確識別出「這批股票中哪幾支漲最多」就夠了，絕對預測精度反而不重要。

**文獻依據：**
- *On Evaluating Loss Functions for Stock Ranking* (arxiv:2510.14156, CIKM 2025)：系統性驗證 pairwise/listwise > pointwise 損失函數用於 Transformer 股票排名
- *MiM-StocR* (arxiv:2509.10461)：Adaptive-k ApproxNDCG（listwise）+ 動能多任務 → CSI 50/100/300 SOTA

**當前數據：** Round 0 val_loss=3.6440，IC=0.041，pretrained baseline IC=0.050。Fine-tuning 反而使 IC 下降，說明 token CE loss 優化方向與 IC 目標相反。

---

## User Stories

- 作為研究者，我希望訓練時 batch 包含多支股票，讓模型在訓練過程中就學習截面排名，而不只是逐個 token 預測
- 作為工程師，我希望 ranking loss 是輔助性的（alpha 權重可調），不破壞原有的 token 預測能力

---

## Requirements

### Functional (MoSCoW)

**Must:**
- [ ] 訓練 batch 必須包含同一日期的多支股票（current: 每個樣本獨立，可能來自不同日期）
  - **這是最大的架構改動**：需要修改 `MultiStockDataset` 或加一個 grouped batch sampler
- [ ] 實現 differentiable rank IC loss：
  ```python
  def rank_ic_loss(pred_scores: Tensor, actual_scores: Tensor) -> Tensor:
      """Differentiable approximation of -rank IC.
      使用 soft rank (Blondel et al.) 或簡單 Pearson on ranks + STE.
      """
      # 簡單版：-Pearson(pred_scores, actual_scores)
      pred_z = (pred_scores - pred_scores.mean()) / (pred_scores.std() + 1e-8)
      actual_z = (actual_scores - actual_scores.mean()) / (actual_scores.std() + 1e-8)
      return -(pred_z * actual_z).mean()
  ```
- [ ] 總損失：`total_loss = token_loss + alpha * ranking_loss`，`alpha` 為 config 參數（建議初始值 0.1）
- [ ] `pred_scores` 從預測 open[T+h+1]/open[T+1]-1 提取（h=5d target）

**Should:**
- [ ] 加入 config 參數 `ranking_loss_alpha: 0.1` 和 `ranking_loss_horizon: 5`
- [ ] 每 N steps log 一次 ranking_loss 和 token_loss 比例
- [ ] 如果 batch 內只有 1 支股票，ranking_loss 自動設為 0（rank IC 無法計算）

**Won't (in this version):**
- 實現完整 ApproxNDCG（複雜度高，留給 N1 進階版）
- 修改 tokenizer

### Non-functional

- 訓練速度影響 < 20%（grouped batch 有額外開銷）
- 顯存影響需要測試（batch 包含多支股票可能需要較小的 per-stock 序列長度）

---

## Acceptance Criteria

- [ ] 訓練 log 中 `ranking_loss` 非零且在訓練過程中下降
- [ ] `val_ic`（open-to-open，配合 M2）比 Round 0 的 0.041 高
- [ ] `backtest_next_open v2` Sharpe ≥ 1.4
- [ ] `alpha=0` 復現 Round 0 行為（regression test）

---

## Technical Approach

**修改檔案：** `finetune_tw/train_predictor.py`，`finetune_tw/dataset.py`

### Step 1: 修改 Dataset/DataLoader 讓 batch 按日期組織

**DECISION NEEDED:** 最小改動方案是：不改 Dataset，改 collate_fn，讓每個 batch 只包含同一日期的樣本。  
或者：在 loss 計算時，只對 batch 內共享同一 `x_stamp[-1]`（最後一個時間戳）的樣本計算 ranking loss。

```python
# 在 train loop 中：
if batch_size_same_date >= 2:
    # 抽取同日期的樣本
    same_date_mask = (batch_x_stamp[:, -1] == batch_x_stamp[0, -1])
    if same_date_mask.sum() >= 2:
        pred_opens = decode_opens(logits, same_date_mask)  # shape: (N,)
        actual_opens = batch_targets[same_date_mask, target_h]  # shape: (N,)
        ranking_loss = rank_ic_loss(pred_opens, actual_opens)
    else:
        ranking_loss = 0.0
else:
    ranking_loss = 0.0

total_loss = token_loss + cfg.ranking_loss_alpha * ranking_loss
```

### Step 2: decode_opens 函式

從 token logits 解碼出 open price 的連續估計：
```python
def decode_opens_from_logits(model, tokenizer, logits_s1, logits_s2, x_mean, x_std, horizon=5):
    """Greedy decode predicted tokens → open price at horizon h."""
    s1 = logits_s1.argmax(-1)  # (B, seq_len)
    s2 = logits_s2.argmax(-1)
    # Decode 前 horizon+1 步
    preds = tokenizer.decode(s1[:, :horizon+1], s2[:, :horizon+1])  # (B, horizon+1, 6)
    open_h = preds[:, horizon, 0]  # column 0 = open
    open_0 = preds[:, 0, 0]
    # open-to-open return at horizon h
    return (open_h / (open_0 + 1e-8)) - 1.0
```

**注意：** 這個 decode step 可能很慢（需要通過 tokenizer decode），且 argmax 是不可微的（STE 可以緩解）。  
**DECISION NEEDED:** 用 STE（straight-through estimator）還是用 logit 直接線性組合 codebook？

---

## Risks & Confidence

| 風險 | 程度 | 緩解 |
|------|------|------|
| argmax 不可微，ranking loss 梯度可能很小 | HIGH | 用 STE 或 Gumbel-Softmax；或直接用 logit 的加權平均 codebook |
| 同日期樣本在 batch 內太少（< 5 支）導致 ranking IC 無意義 | MEDIUM | 設最小閾值；或改 sampler 確保每日至少 20 支 |
| alpha 調太大導致 token 預測崩潰 | MEDIUM | alpha 從 0.01 開始，gradual increase |
| 顯存增加 | LOW | 比現在的 batch 多一個 decode pass |

**Evidence tier:** SECONDARY (arxiv papers) — 這是本 PRD 中唯一需要重訓且架構改動最大的項目。

---

## Success Metrics

| 指標 | 基準（Round 0） | 目標 |
|------|----------------|------|
| val_ic (open-to-open) | ~0.04 | ≥ 0.05 |
| Backtest Sharpe | 1.356 | ≥ 1.5 |
| MaxDD（配合 M1）| 35% | ≤ 20% |
| token val_loss | 3.64 | ≤ 3.5（不應顯著退步）|

---

## Open Questions

1. `decode_opens` 用 greedy（argmax）還是 soft decode（weighted sum over codebook）？前者快但不可微；後者可微但需要改模型。
2. 什麼時候開始加 ranking loss？是從第 1 個 epoch 就加，還是 warmup 後再加（token loss 先穩定）？
3. ranking loss 的 horizon 要固定在 h=5 還是混合 h=3,5,7？
4. 要不要先做實驗確認 M2（open-to-open IC early stop）對 Round 0 的提升，再決定是否需要 N1？（N1 的成本比 M2 大很多）
