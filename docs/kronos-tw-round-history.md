# Kronos TW 台股微調歷史紀錄

**評估基準：** `backtest_next_open v2`（open 信號 + open 執行），top_k=10，hold=5d  
**目標：** Sharpe ≥ 1.5，Ann > 15%，MaxDD < 20%

---

## Round 0 — 2026-06-20

**起點：** NeoQuasar/Kronos-base（pretrained）  
**Config：** `config_tw_daily_rtx6000.yaml`，top_k=20，hold_days=5，lookback=90  
**HF：** `j835111/kronos-tw-finetune@round-0`

**調整：** 首輪 baseline。直接在 Kronos-base 上以 Taiwan TWSE daily data fine-tune tokenizer + predictor。

### 回測結果

| 評估方式 | Sharpe | Ann | MaxDD |
|----------|--------|-----|-------|
| close/close（grid search，top_k=20） | 1.19 | 40% | 32% |
| close/close（grid search，top_k=10） | 1.92 | 86% | 26% |
| **open/open v2（可執行，top_k=10，hold=5d）** | **1.356** | **50%** | **35%** |

### 預測品質（eval_forecast，來自 `docs/finetune_tw_predictor_retrain_analysis.md`）

| 指標 | Pretrained | **Round 0** | 說明 |
|------|-----------|------------|------|
| val_loss（↓） | **2.997** | 3.644 | Round 0 val_loss 反而更高（模型更「大膽」） |
| IC@h1 | 0.0497 | 0.0413 | pretrained IC 較高 |
| IC@h5（換倉決策點）| 0.0268 | **0.0319** | h5 Round 0 反超，這是回測決勝點 |
| IC-IR@h1（↑） | 0.601 | **0.625** | 信號穩定性 Round 0 更高 |
| ic_positive_rate@h1 | 70.9% | **72.8%** | 多數日子 IC 為正，top-K 一致性更佳 |
| MAPE / MAPE_naive | 1.19× | 1.34× | Round 0 預測更分散（分化能力強） |

IC 衰減率：pretrained h1→h5 衰減 46%，Round 0 衰減 23%。**Round 0 在 5 天持倉週期的信號維持性更強，這是 Sharpe 差距的直接原因。**

### 關鍵發現

- Grid search 最佳（top_k=10, hold=3d, Sharpe 1.92）**無法執行**（alpha 來自收盤到次日開盤跳空，實際掛單無法捕捉）
- hold=3d 在所有 next_open 版本皆無效（v1: 0.35, v2: 0.05）
- **最佳可執行：top_k=10, hold=5d, open/open v2 → Sharpe 1.356, Ann 50%**

**參考資料：**
- `finetune_tw/outputs/tw_daily/backtest_returns_round0.json`
- `finetune_tw/outputs/next_open/backtest_returns_round0_next_open_v2.json`
- `docs/finetune_tw_predictor_retrain_analysis.md`

---

## Round 1 — 2026-06-21（失敗）

**起點：** NeoQuasar/Kronos-base（pretrained，**起點錯誤**）  
**Branch：** `fix/predictor-retrain`  
**調整：** 加入 IC-based early stopping，ic_val_symbols=150, ic_val_dates=8，lr=1e-5，epochs=6

### 訓練歷程

| Epoch | train_loss | val_loss | val_ic | 備注 |
|-------|-----------|---------|--------|------|
| 1 | 3.268 | 3.086 | -0.029 | |
| 2 | 3.069 | 3.116 | -0.022 | |
| 3 | 3.065 | 3.133 | -0.029 | |
| 4 | 3.086 | 3.140 | -0.017 | |
| 5 | 3.088 | 3.141 | -0.011 | |
| 6 | **3.030** | 3.141 | **-0.008** | ← best（最不壞） |

val_ic **全程為負**，early stop 只能選最不壞的 checkpoint。

### 回測結果

| 指標 | Round 1 | Round 0 |
|------|---------|---------|
| Sharpe | 0.15 | 1.19 |
| Ann | -0.13% | 40% |
| MaxDD | 35% | 32% |

**失敗原因（三個根本，來自 `docs/finetune_tw_predictor_retrain_analysis.md`）：**
1. **起點不一致**：pretrained predictor + Round 0 tokenizer，lr=1e-5 在 6 epoch 內無法重新對齊分佈
2. **IC 噪音壓制 early stop**：150×8=1200 樣本，σ(IC)≈0.08 >> IC 本身（0.00-0.05），SNR < 1
3. **指標不對齊**：val_ic 均值 ≠ 回測決策點（h5），h1-h4 噪音淹沒 h5 信號

**另一反直覺發現：** Round 1 val_loss（3.141）優於 Round 0（3.644），但回測遠差——更低的 CE loss 代表模型更保守（預測接近 naive），分化能力反而下降。

---

## Week 1 策略改進（不重訓，2026-06-26）

**Branch：** `feature/atr-vol-open-ic`

### M1 — ATR Position Sizing（`--use-atr-weights`）

用模型預測的 `(high-low)/close` 作為前向波動估計，position size ∝ 1/pred_ATR。  
**結果：** 無明顯改善，維持等權重。

### M3 — Volume 信心過濾器

排除預測 volume 落底 25% 的低流動性股票。  
**結果：** 無改善。git commit `ba13774` 直接移除此功能。

→ **結論：策略層面的改動（ATR sizing、volume filter）對 Sharpe/MaxDD 無明顯幫助。**

---

## Round 2 — 2026-06-22（略輸 Round 0）

**起點：** Round 0 predictor（`j835111/kronos-tw-finetune@round-0`）  
**調整（依 `autoresearch/improve-260622-0042/improvement-plan.md`）：**
- IC-IR@h5 early stopping（從 val_ic 均值改為 IC/σ(IC) at h5）
- 擴大驗證集：ic_val_symbols=300, ic_val_dates=20
- 從 Round 0 起點（非 pretrained）
- epochs=20, lr=5e-5, warmup cosine

**Best epoch：** 4，ic_ir_h5=0.4066  
**HF：** `j835111/kronos-tw-finetune@round-2`

| hold | Sharpe | Ann | MaxDD |
|------|--------|-----|-------|
| 5d | 1.14 | 38% | 36% |
| 10d | 0.53 | 13% | 33% |

**結果：** 輸 Round 0（1.356 → 1.14）。  
**分析：** val_loss 從 epoch 1 起持續上升（退化），Fine-tuned → fine-tune 邊際報酬遞減，IC 估計仍用 close-to-close 未對齊部署目標。

---

## Round 3 — 2026-06-28（大幅退步）

**起點：** Round 0 predictor  
**平台：** RunPod A40（從 MoLab 遷移）  
**調整（依 `autoresearch/improve-260626-1240/improvement-plan.md` Week 2-3）：**
- **M2**：IC early stopping 改為 open-to-open（`ic_validation.py` 全面修改，`realized_return = open[T+h+1]/open[T+1]-1`）
- **N2**：擴大驗證集 ic_val_symbols=500, ic_val_dates=40

### 訓練歷程（`finetune_tw/outputs/tw_daily/train_log_round3.csv`）

| Epoch | Train Loss | Val Loss | IC-IR@h5 | 備注 |
|-------|-----------|----------|----------|------|
| 1 | 2.367 | 3.318 | +0.0026 | |
| 2 | 2.388 | 3.364 | +0.0082 | |
| 3 | 2.319 | 3.383 | -0.0177 | |
| 4 | 2.423 | 3.404 | **+0.0227** | ← Best |
| 5 | 2.318 | 3.431 | +0.0030 | no_improve=1 |
| 6 | 2.290 | 3.453 | -0.0162 | no_improve=2 |
| 7 | 2.323 | 3.468 | +0.0036 | no_improve=3 |
| 8 | 2.305 | 3.475 | -0.0007 | no_improve=4 → **Early Stop** |

### 回測結果（open/open v2，top_k=10，hold=5d）

| 指標 | Round 3 | Round 0（baseline） |
|------|---------|-------------------|
| **Sharpe** | **0.50** | **1.356** |
| **Ann** | **20%** | **50%** |
| **MaxDD** | **41%** | **35%** |

**失敗分析：**
1. **Val loss 全程上升**（3.32 → 3.48）：模型持續退化，IC-IR early stop 只選到「退化最少」的 epoch
2. **IC-IR@h5 信號極弱**：best 僅 0.0227，遠低於 Round 2 的 0.4066——open-to-open IC 比 close-to-close 更難學到
3. **驗證集過大（500×40）**：每 epoch validation 耗時 60-90 分鐘，效率低（主因仍是模型退化）
4. **Round 0 已是局部最優**：持續 fine-tune 破壞已學到的台股特定排名模式

**Best model 存檔：**
- `finetune_tw/outputs/tw_daily/predictor/best_model_round3/`（本地）
- `finetune_tw/outputs/tw_daily/backtest_round3_next_open.png`

---

## 各輪 Sharpe 彙整（open/open v2，top_k=10，hold=5d）

| 版本 | Sharpe | Ann | MaxDD | 備注 |
|------|--------|-----|-------|------|
| Round 0 | **1.356** | 50% | 35% | 目前最佳可執行 |
| Round 1 | 0.15 | -0.1% | 35% | 起點錯誤（pretrained）|
| Round 2 | 1.14 | 38% | 36% | close IC，輸 Round 0 |
| Round 3 | 0.50 | 20% | 41% | open IC，大幅退步 |

---

## 結論與下一步

**Round 0 是目前唯一可執行的版本（Sharpe 1.356）。**

重訓（Round 1-3）及策略改動（ATR sizing、volume filter）均無法超越 Round 0：
- Kronos-base 在 TWSE daily data fine-tune 後已達局部最優
- 進一步 fine-tune 破壞泛化能力（forgetting）
- open-to-open IC 比 close-to-close IC 更難優化（噪音更高，SNR 更低）
- ATR sizing 和 volume filter 在策略層面也無法改善指標

**若要繼續提升，建議方向（按信心由高到低）：**
1. **N3（Label Horizon 掃描）**：嘗試 IC validation at h=3, h=7——最優訓練 label horizon 可能不是 h=5（arxiv:2602.03395, ICML 2026 Label Horizon Paradox）
2. **N1（Ranking Loss）**：加入 pairwise IC auxiliary loss，直接在訓練時優化排名（arxiv:2510.14156, CIKM 2025）
3. **從 pretrained 重新 fine-tune**（非從 Round 0）：避免局部最優問題，同時重訓 tokenizer + predictor，配合 open-to-open IC early stopping
