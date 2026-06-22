# Improvement Plan — finetune_tw Round 2
**日期:** 2026-06-22 | **目標:** Sharpe ≥ 1.5, Annual Return > 15%, Max DD < 20%

---

## Must-Have（必須執行，風險低、影響高）

### M1. 修正 Early Stopping 指標 [HIGH 信心] [影響：極高]
**現況：** `EarlyStopper(metric="val_ic")` — 均值噪音大（σ≈0.08 >> IC本身）
**改法：** 切換為 `ic_ir_h5`（IC / σ(IC)，只計算 h5 horizon）
**需要：** `ic_validation.py` 新增 per-horizon IC 標準差計算 → 回傳 IC-IR@h5
**預期效益：** early stop 改選信號*一致性*最佳 checkpoint，而非單次 IC 最高

```python
# ic_validation.py 新增
def validate_predictor_ic_ir(predict_batch_fn, ..., target_horizon=5) -> float:
    """Return IC-IR = mean_IC / std_IC at target_horizon only."""
    ...
    ics_at_h = [ic_at_date_h5 for date in val_dates]
    return np.mean(ics_at_h) / (np.std(ics_at_h) + 1e-8)
```

---

### M2. 擴大 IC 驗證樣本 [HIGH 信心] [影響：高]
**現況：** `ic_val_symbols=150, ic_val_dates=8` → 1200 樣本，σ(IC)≈0.08
**改法：** `ic_val_symbols=300, ic_val_dates=20` → 6000 樣本，σ(IC)↓ ~0.035

```yaml
# config_tw_daily_rtx6000.yaml 改動
ic_val_symbols: 300
ic_val_dates: 20
early_stop_patience: 3   # 樣本多了，允許多等一輪
```

**代價：** 每 epoch 末 IC 驗證時間 ×2.5，但 RTX Pro 6000 可承受
**預期效益：** SNR 提升 2.3×，early stop 訊號從噪音中分離出來

---

### M3. 訓練起點改為 Round 0 Predictor [HIGH 信心] [影響：極高]
**現況：** `pretrained_predictor: "NeoQuasar/Kronos-base"`（從未接觸台股）
**改法：** 從 `j835111/kronos-tw-finetune` revision `round-0` 出發

```yaml
pretrained_predictor: "j835111/kronos-tw-finetune"
hf_revision: "round-0"
hf_subfolder: "predictor/best_model"
```

**理由：** Round 0 已學到台股動量效應（h5 IC 反超 pretrained）。從 Round 0 出發 + IC-IR early stop，可在此基礎上進一步強化一致性，而非從零重學。

---

### M4. 延長訓練 + Warmup+Cosine Decay [HIGH 信心] [影響：高]
**現況：** `basemodel_epochs=6, OneCycleLR, lr=1e-5`（6 epochs 遠不夠）
**改法：**

```yaml
basemodel_epochs: 20
predictor_lr: 5e-5       # peak LR（warmup 後）
```

```python
# train_predictor.py 替換 scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.predictor_lr,
    steps_per_epoch=train_steps_per_epoch, epochs=cfg.basemodel_epochs,
    pct_start=0.05,     # 5% warmup（原本 3%）
    div_factor=25,      # initial_lr = peak/25 = 2e-6
    final_div_factor=1e4,  # final_lr = peak/10000 = 5e-9
    anneal_strategy='cos',
)
```

**理由：** 文獻一致建議 warmup+cosine 優於固定 LR；peak LR 5e-5（而非 1e-5）讓模型在正確方向有足夠梯度更新空間。Early stop patience=3 保護不過擬合。

---

### M5. 增量訓練（不清空 checkpoint）[HIGH 信心] [影響：中]
**現況：** molab 重啟後 `_clear_stale` 可能誤刪 checkpoint
**改法：** 確認 `hf_utils.push_best_model()` 在每次 best checkpoint 更新時同步 push，不依賴本地持久化

```python
# train_predictor.py is_best 分支補上
if is_best:
    model.save_pretrained(str(save_dir / "best_model"))
    from finetune_tw.hf_utils import push_best_model
    push_best_model(save_dir / "best_model", cfg, subfolder="predictor/best_model")
    # 同時 push train_log
    push_best_model(log_path, cfg, subfolder="predictor/train_log.csv")
```

---

## Nice-to-Have（中等影響，需要較多實作，建議 Round 3 嘗試）

### N1. Pairwise Ranking Loss 輔助訓練 [MEDIUM 信心] [影響：高，風險：中]
**概念：** 在現有 CE loss 基礎上加 pairwise ranking loss（marginalized over cross-section）

```python
# 新增 pairwise IC loss
def ic_ranking_loss(logits_s1, token_out_s1, alpha=0.2):
    """Penalize pairs where predicted rank ≠ actual return rank."""
    ...

total_loss = ce_loss + alpha * ic_ranking_loss(...)
```

**依據：** arXiv 2510.14156 顯示 pairwise loss 在 transformer 架構下的 top-K IC 改善顯著
**風險：** 需要截面 batch（同一時間點多股票），目前 `CachedTokenDataset` 按 sequence 採樣，需要改 DataLoader 結構

---

### N2. Label Horizon 實驗（h3 監督 h5 推理）[MEDIUM 信心] [影響：中，風險：中]
**概念：** arXiv 2602.03395 發現最佳訓練 label 可能是中間 horizon（信噪比最高點）
**實驗設計：** 用 h3 作為 primary loss horizon（取代均值），h5 作為 eval/early-stop
**風險：** 尚未在 TWSE 日線場景驗證，為探索性實驗

---

### N3. Horizon-Weighted Loss [MEDIUM 信心] [影響：中]
**概念：** 對不同 horizon 的 CE loss 加權，讓 h4/h5 權重更高（對齊換倉決策點）

```python
horizon_weights = torch.tensor([0.5, 0.7, 0.85, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# 乘在每個 token position 的 loss 上
```

---

## Moonshot（高潛力，高風險，需要大幅重構）

### S1. 完全切換 Listwise Ranking Loss [HIGH 影響，HIGH 風險]
**概念：** 拋棄 CE loss，完全用 ListMLE 或 ListFold 直接最佳化 top-K IC-IR
**風險：** 需要重構整個訓練 loop（batch 按截面組織），Kronos decoder 架構可能不相容

### S2. End-to-End IC-IR 梯度優化 [HIGH 影響，HIGH 風險]
**概念：** 讓 IC-IR 直接進入 backward pass（用 REINFORCE 或 differentiable ranking）
**風險：** 極高實作複雜度，梯度估計方差大

---

## 優先執行順序（Round 2）

```
M3 → M1 → M2 → M4 → M5
↑          ↑
（起點）  （指標+樣本）
```

| 步驟 | 改動 | 預期耗時 | 風險 |
|------|------|---------|------|
| M3 | config hf_revision + pretrained_predictor 改路徑 | 10 min | 低 |
| M1 | ic_validation.py 新增 ic_ir_h5 函數 | 1 hr | 低 |
| M2 | config ic_val_symbols=300, dates=20 | 5 min | 低 |
| M4 | config epochs=20, lr=5e-5; scheduler pct_start=0.05 | 30 min | 低 |
| M5 | train_predictor.py push log 修正 | 30 min | 低 |

**估計 Round 2 訓練時間（RTX Pro 6000）：** 20 epoch × ~30 min/epoch = ~10 小時

---

## 收斂條件複查

| 指標 | 目標 | 現況最佳（Round 0） |
|------|------|------------------|
| Sharpe | ≥ 1.5 | 1.19（差 26%）|
| Annual Return | > 15% | 40.45%（已達）|
| Max Drawdown | < 20% | 31.60%（超標 58%）|
| IC-IR@h5 | 最大化 | 尚未測量（Round 0 h1 IR=0.625）|

**主要瓶頸：Sharpe 提升（減少 Max DD）**。Round 0 的 Max DD 偏高（31.6%），推測是因為 IC 衰減率高（h1→h5 衰減 46%）導致換倉決策不穩定。M1+M2 改善 early stop → 選出 h5 IC 更高的 checkpoint → 換倉信號更穩 → Max DD 有望降低。
