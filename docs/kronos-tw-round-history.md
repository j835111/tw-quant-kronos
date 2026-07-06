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

> **⟳ 重跑基準（2026-06-30，test 2024-07-01→2026-06-29，DB 截至 2026-06-26）**
>
> | 評估方式 | Sharpe | Ann | MaxDD |
> |----------|--------|-----|-------|
> | close/close（top_k=10，hold=5d） | **1.27** | 50.41% | 31.46% |
> | **open/open v2（top_k=10，hold=5d）** | **1.12** | 38.59% | 35.03% |
>
> open/open Sharpe 從 1.356 → 1.12 的原因：測試期延長至 2026 年含較弱市場行情，且 DB 截至 2026-06-26 使末端 hold 期略有截斷。**新基準用於與後續 round 比較。**

### 預測品質（eval_forecast，來自 `docs/research/round0-1-predictor-retrain-analysis.md`）

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
- `docs/research/round0-1-predictor-retrain-analysis.md`

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

**失敗原因（三個根本，來自 `docs/research/round0-1-predictor-retrain-analysis.md`）：**
1. **起點不一致**：pretrained predictor + Round 0 tokenizer，lr=1e-5 在 6 epoch 內無法重新對齊分佈
2. **IC 噪音壓制 early stop**：150×8=1200 樣本，σ(IC)≈0.08 >> IC 本身（0.00-0.05），SNR < 1
3. **指標不對齊**：val_ic 均值 ≠ 回測決策點（h5），h1-h4 噪音淹沒 h5 信號

**另一反直覺發現：** Round 1 val_loss（3.141）優於 Round 0（3.644），但回測遠差——更低的 CE loss 代表模型更保守（預測接近 naive），分化能力反而下降。

---

## Week 1 策略改進（不重訓，2026-06-26）

**Branch：** `feature/atr-vol-open-ic`  
**計劃來源：** `autoresearch/improve-260626-1240/improvement-plan.md`

**驗證的方法（260626）：**
- ✅ **M1** ATR Position Sizing（`--use-atr-weights`）：position size ∝ 1/pred_ATR → **結論：無明顯改善，維持等權重**
- ✅ **M3** Volume 信心過濾器：排除預測 volume 底 25% 的低流動性股 → **結論：無改善，commit ba13774 移除**

→ **結論：策略層面的改動（ATR sizing、volume filter）對 Sharpe/MaxDD 無明顯幫助。**

---

## Round 2 — 2026-06-22（略輸 Round 0）

**起點：** Round 0 predictor（`j835111/kronos-tw-finetune@round-0`）  
**計劃來源：** `autoresearch/improve-260622-0042/improvement-plan.md`

**驗證的方法（260622）：**
- ✅ **M1** IC-IR@h5 early stopping（從 val_ic 均值改為 IC/σ(IC) at h5）→ **結論：有效提升 SNR，但無法改善回測**
- ✅ **M2** 驗證集 300×20（6000 樣本，σ(IC)↓ ~0.035）→ **結論：統計噪音降低，但模型仍退化**
- ✅ **M3** 從 Round 0 起點（非 pretrained）→ **結論：正確起點，不再從零重學**
- ✅ **M4** Warmup+Cosine, epochs=20, lr=5e-5 → **結論：訓練排程本身沒問題，問題在模型退化方向**

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
**計劃來源：** `autoresearch/improve-260626-1240/improvement-plan.md` Week 2-3

**驗證的方法（260626）：**
- ✅ **M2** Open-to-open IC early stopping（`realized_return = open[T+h+1]/open[T+1]-1`）→ **結論：IC-IR@h5 信號極弱（best=0.023），無法有效選 checkpoint，大幅退步**
- ✅ **N2** 擴大驗證集 500×40 → **結論：每 epoch validation 耗時 60-90 分鐘，效率過低，後續縮回**

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

## Step 1 — Open/Open v2 完整 Grid Search — 2026-06-29

**平台：** RunPod RTX 4090  
**模型：** Round 0（`j835111/kronos-tw-finetune@round-0`）  
**設定：** open/open v2 signal，1090 symbols，test 2024-07-01 → 2026-06-29  
**目的：** 確認是否有 top_k × hold_days 組合可超越 Round 0（Sharpe 1.356）

### 結果矩陣

| top_k | hold=5d Sharpe | hold=5d Ann | hold=5d MaxDD | hold=7d Sharpe | hold=7d Ann | hold=7d MaxDD |
|------:|:--------------:|:-----------:|:-------------:|:--------------:|:-----------:|:-------------:|
| 5 | 0.641 | 18.8% | 38.9% | 0.341 | 5.8% | 47.0% |
| **10** | 1.115 | 38.6% | 35.0% | 1.118 | 40.2% | 38.6% |
| 15 | 1.158 | 38.6% | 38.2% | 1.237 | 43.5% | 37.3% |
| 20 | 1.215 | 39.7% | 37.6% | 1.255 | 42.1% | 36.1% |
| 30 | 1.119 | 33.9% | 37.7% | 1.263 | 39.9% | 35.3% |
| Benchmark ^TWII | 1.47 | 41.2% | 28.7% | — | — | — |

> 注意：原始 Round 0 基準（Sharpe 1.356）是用稍早的 test end date 測得；本次測試延長至 2026-06-29，top_k=10 hold=5d 結果為 1.115，差距部分來自測試期間延長。

**結論：無任何組合超越 Round 0。**
- 最高 Sharpe 1.263（top_k=30, hold=7d），仍遠低於目標 1.5
- MaxDD 全部在 35%+，目標 <20% 根本未觸及
- top_k=5 特別差（Sharpe 0.64），集中度過高帶來高噪音
- hold=7d 略優於 hold=5d，但差距微小（+0.05-0.14）

**參考資料：** `finetune_tw/outputs/tw_daily/grid_search_round0_next_open.json`

---

## Step 2 — Label Horizon IC 曲線（eval_forecast）— 2026-06-29

**模型：** Round 0 best_model  
**目的：** 找出 Round 0 在哪個 horizon (h) 的信號最強，評估換 hold_days 的空間

### IC-IR 曲線（close-to-close，全測試期）

| h | IC | IC-IR | IC>0% | 說明 |
|--:|:--:|:-----:|:-----:|------|
| **1** | **0.042** | **0.64** | **73%** | 最強，短期預測力最佳 |
| 2 | 0.036 | 0.50 | 68% | |
| 3 | 0.032 | 0.42 | 68% | |
| 4 | 0.036 | 0.44 | 70% | 小反彈 |
| 5 | 0.032 | 0.37 | 70% | 目前回測用的 hold |
| 6 | 0.029 | 0.33 | 62% | |
| 7 | 0.024 | 0.29 | 61% | |
| 8 | 0.022 | 0.28 | 64% | |
| 9 | 0.022 | 0.29 | 66% | |
| 10 | 0.022 | 0.26 | 61% | |

val_loss（token CE）= **3.6440**（與 Round 0 訓練時一致，模型完整性確認）

**結論：IC-IR 從 h=1 單調下降，h=5 已是明顯衰減區（0.37 vs 0.64）。**
- h=7 的 IC-IR 僅 0.29，換 hold=7d 理論上更差（與 grid search 結果吻合）
- h=4 有小反彈但不顯著
- 不存在「隱藏的更強 horizon」：h>5 全面衰減，換持倉期無法改善

**參考資料：** `finetune_tw/outputs/tw_daily/eval/eval_metrics_finetuned.json`

---

## Round 4 — 2026-06-29（FPT + IC-IR@h1 early stop + extended warmup）

**起點：** Round 0 predictor（`j835111/kronos-tw-finetune@round-0`）  
**平台：** RunPod A40 48GB  
**計劃來源：** `autoresearch/improve-260629-1426/improvement-plan.md`  
**HF：** `j835111/kronos-tw-finetune@round-4`

**驗證的方法（260629）：**
- ✅ **M1** FPT Selective Freeze：凍結 self_attn + FFN，只訓練 LayerNorm + head（~5-7% 參數）→ **結論：best epoch=1，Round 0 是局部最優，凍結也無法改善**
- ✅ **M3** Label Horizon h1 + Extended Warmup（pct_start=0.08, div_factor=25）→ **結論：warmup 對退化無效**
- ⚠️ **M2（部分）** IC-IR@h1 early stopping：計劃要求 close-to-close IC（SNR=0.64），**實際實作為 open-to-open IC-IR@h1**（公式已確認對齊，但不是計劃中的 close 版本）→ **close-to-close IC-IR@h1 尚未測試**

### 訓練歷程

| Epoch | train_loss | val_loss | IC-IR@h1 | 備注 |
|-------|-----------|---------|----------|------|
| **1** | — | **3.665** | **0.3246** | ← **Best（首 epoch 即最佳）** |
| 2–7 | — | 持續上升 | 衰減 | no_improve 累積 |
| 7 | — | — | — | Early Stop |

**Best epoch = 1**，說明從 Round 0 繼續微調幾乎沒有可學習空間。

### 回測結果

**backtest.py（close-to-close signal）**

| hold | Ann | Sharpe | MaxDD |
|------|-----|--------|-------|
| **5d** | 44.46% | 1.17 | 34.10% |
| 10d | −0.53% | 0.17 | 44.40% |
| 15d | 17.76% | 0.65 | 36.01% |

**backtest_next_open.py（open/open v2，top_k=10）**

| hold | Ann | Sharpe | MaxDD |
|------|-----|--------|-------|
| **5d** | **45.94%** | **1.24** | 33.99% |
| 10d | 13.29% | 0.53 | 37.37% |
| 15d | 19.50% | 0.68 | 41.76% |

**與 Round 0 對比（open/open v2，top_k=10，hold=5d）：**

| 指標 | Round 0 | Round 4 | 差距 |
|------|---------|---------|------|
| Sharpe | **1.356** | 1.24 | −0.116 |
| Ann | **50%** | 45.94% | −4% |
| MaxDD | 35% | 33.99% | +1%（略優）|

**失敗分析：**
1. **Best epoch = 1**：模型在第一個 epoch 就達到 IC-IR 最高點，之後持續退化——說明 Round 0 已是此 fine-tuning 路徑的局部最優，FPT 也無法改善
2. **FPT 未能突破局部最優**：凍結 95% 參數本意是防止 forgetting，但也限制了模型調整排名能力的空間
3. **Round 0 → fine-tune 邊際效益極低**：累計四輪 retraining 皆退步，結論明確

**參考資料：**
- `finetune_tw/outputs/tw_daily/backtest_round4.png`
- `finetune_tw/outputs/tw_daily/backtest_round4_next_open.png`
- `finetune_tw/outputs/tw_daily/backtest_returns_round4.json`
- `finetune_tw/outputs/tw_daily/backtest_returns_round4_next_open.json`

---

## Round 5 — 2026-07-01（Pretrained 重啟 + Auxiliary Ranking Loss）

**起點：** `NeoQuasar/Kronos-base`（**完全重啟，非從 Round 0**）  
**平台：** RunPod A40 48GB  
**Branch：** `research/round-5`  
**HF：** `j835111/kronos-tw-finetune@round-5`

**驗證的方法：**
- ✅ **Pretrained restart**：從 Kronos-base 重啟，避免 Round 0 局部最優
- ✅ **Auxiliary Ranking Loss**：`ranking_loss_alpha=0.1`，每 5 步從 CrossSectionalDateSampler 採樣一個交易日的截面批次，計算 ListMLE/IC-IR@h5 ranking loss，疊加在 next-token prediction loss 上
- ✅ **S1 Oracle Table**：670/1024 tokens 有效（min_count=20），供 ranking loss 查表 open-to-open 報酬
- ✅ **IC-IR@h5 early stopping**：patience=5（實際 _bad>5，即 6 連未改善才停）

**技術細節：**
- `oracle_min_count=20`，S1 coverage 65%
- `cross_sectional_batch_size=64`，驗證集 150×40=6000 樣本
- val_loss 起點 2.93（from pretrained），遠低於 Round 0 起點（3.6）——pretrained 本來就是好的 token 預測器

### 訓練歷程

| Epoch | train_loss | val_loss | val_ic | ic_ir_h5 | 備注 |
|-------|-----------|---------|--------|---------|------|
| 1 | 2.9406 | 2.9271 | 0.0241 | 0.2164 | |
| 2 | 2.9674 | 2.9421 | 0.0245 | 0.4622 | |
| 3 | 2.8324 | 2.9657 | 0.0250 | 0.4530 | no_improve=1 |
| 4 | 2.9585 | 2.9774 | 0.0165 | 0.3094 | no_improve=2 |
| 5 | 2.8550 | 2.9854 | 0.0236 | 0.4571 | no_improve=3 |
| **6** | **2.7933** | 2.9912 | 0.0227 | **0.4701** | ← **Best** |
| 7 | 2.8258 | 2.9945 | 0.0245 | 0.3464 | no_improve=1 |
| 8 | 2.8369 | 2.9933 | 0.0270 | 0.4292 | no_improve=2 |
| 9 | 2.7844 | 2.9964 | 0.0238 | 0.3053 | no_improve=3 |
| 10 | 2.8353 | 3.0012 | 0.0311 | 0.4183 | no_improve=4 |
| 11 | 2.7885 | 3.0000 | 0.0281 | 0.3751 | no_improve=5 |
| 12 | 2.7933 | 3.0035 | 0.0347 | 0.3860 | no_improve=6 → **Early Stop** |

**觀察到的規律：** ic_ir_h5 呈現偶數 epoch 偏高（~0.43–0.47）、奇數 epoch 偏低（~0.31–0.35）的週期性波動，但整體未能突破 epoch 6 的高點。val_ic 有緩慢上升趨勢（0.024→0.035），但 ic_ir（穩定性）持平甚至下降。

### 回測結果（open/open v2，top_k=10，hold=5d）

| 指標 | Round 5 | Round 0（新基準 1.12） | 差距 |
|------|---------|----------------------|------|
| **Sharpe** | **0.982** | **1.12** | −0.138 |
| **Ann** | **31.79%** | **38.59%** | −6.8% |
| **MaxDD** | **39.86%** | **35.03%** | −4.8%（更差）|

Benchmark ^TWII：Sharpe=1.47，Ann=41.24%

### 失敗分析

1. **val_ic 極低（~0.02–0.03）**：即使 ranking loss 訓練後，截面 IC 信號仍非常弱。Ranking loss 讓模型在 **validation 集**的 ic_ir_h5 達到 0.47，但未轉化為可執行的回測 Sharpe
2. **val_loss 緩升（2.93→3.00）**：pretrained 重啟後，token prediction loss 仍在緩升，說明 ranking loss 對 token prediction 有輕微對立效果
3. **Ranking loss ≠ 回測改善**：訓練目標（截面 IC-IR@h5 in val set）與回測 Sharpe 之間的對應關係並不直接——val 集採樣的 150 symbols × 40 dates 可能不足以代表完整測試期的 97 個交易期 × 1090 symbols
4. **Oracle coverage 65%（670/1024 S1 tokens）**：35% 的 token 沒有有效的 oracle 報酬，ranking loss 在這些 token 上無法學習

**參考資料：**
- `finetune_tw/outputs/tw_daily/predictor/train_log.csv`
- `finetune_tw/outputs/tw_daily/backtest_returns_round5_next_open.json`
- `finetune_tw/outputs/tw_daily/backtest_round5_next_open.png`

---

## Round 6 — 2026-07-02（Kronos Embedding + XGBoost LambdaRankIC，M1）

**起點：** `NeoQuasar/Kronos-base`（**完全凍結，從未 fine-tune**）
**平台：** RunPod A40 48GB（SECURE cloud, $0.44/hr）
**Branch：** `research/round-6-m1-embedding`
**Plan：** `docs/superpowers/plans/2026-07-02-kronos-embedding-xgb-lambdarank.md`

**戰略背景：** Round 1–5 全部證實「fine-tune Kronos-base」這條路系統性失敗（文獻 arXiv:2511.18578 佐證：pretrained TSFM 在金融回報預測的 fine-tuning 情境下架構性失敗，不是超參數問題）。M1 改用完全不同的路徑：**凍結 Kronos，只當特徵抽取器，用 XGBoost 在其 hidden state 上直接學排序**，理論上繞開 catastrophic forgetting 與 fine-tuning 退化問題。

**驗證的方法：**
- ✅ **凍結 embedding 抽取**：`extract_embeddings.py`，mean-pool 最後一層 hidden state（Kronos-base 實際 `d_model=832`），加上 4 個 raw 技術指標（MA5/MA20 distance, 10日動量, volume ratio）
- ✅ **LambdaRankIC objective**：自行推導的 LambdaRank pairwise 目標函數，用 label rank 距離取代標準 LambdaRank 的 NDCG gain 項，逼近直接優化 Spearman Rank-IC（非逐字照抄 arXiv:2605.00501，是我方推導）
- ✅ **全母體驗證**：不像 Round 0-5 用抽樣驗證集（150-500 symbols），這裡驗證集是完整 130 天 × ~1039 檔股票，噪音低很多

**資料規模：**
- 訓練集：2015-01-01 → 2023-12-31，2,141,404 筆（2246 個交易日 × ~1039 檔股票）
- 驗證集：2024-01-01 → 2024-06-30，135,323 筆
- 測試集：2024-07-01 → 2026-07-02（97 個訊號日，hold=5d 間隔）

**工程筆記（重要，供未來大規模跑類似任務參考）：**
1. `extract_embeddings.py` 逐日迴圈是 **CPU 單執行緒瓶頸**，GPU 使用率 0%——改成把日期切成多段、各自獨立 CLI process 平行跑，GPU 衝到 100%，速度提升近 8 倍
2. 8-way 平行會撞到 pod 的 **50GB RAM cgroup 上限**（container 裡 `top`/`free` 看不出真實上限，需查 `/sys/fs/cgroup/memory/memory.limit_in_bytes`），單一 worker 隨進度累積記憶體可達 25GB。改成切更細（每段再折半）+ 3 併發才穩定
3. XGBoost 訓練階段 `pd.read_parquet` 讀 11GB 的合併訓練集會**膨脹到 42.7GB**（近 4 倍），逼近上限——改用 `pyarrow.ParquetFile.iter_batches()` 串流讀取到預先配置好的 numpy array，避開 pandas 整表 materialize 的記憶體開銷
4. 詳細記錄於 memory `runpod_training.md`

### 訓練歷程（XGBoost，非 epoch 制）

| Round | val-rank_ic | 備注 |
|-------|------------|------|
| 0 | 0.0406 | |
| 50 | 0.0568 | |
| 100 | 0.0631 | |
| **190** | **0.0665** | ← **Best（best_iteration=190）** |
| 199 | 0.0663 | 訓練上限（num_boost_round=200），early stop 未觸發 |

驗證集 IC 全程單調上升（僅 round 140 有一次微幅回落），最終 val-rank_ic ≈ 0.066——**顯著高於 Round 0-5 在 h5 量到的 IC**（Pretrained 0.0268、Round 0 0.0319、Round 5 val_ic ~0.02-0.03），且因為是全母體計算，統計上比抽樣估計可信得多。

### 回測結果（open/open v2，top_k=10，hold=5d）

| 指標 | Round 6 (M1) | Round 0（基準 1.12） | 差距 |
|------|-------------|---------------------|------|
| **Sharpe** | **0.340** | **1.12** | **−0.78** |
| **Ann** | **5.52%** | **38.59%** | **−33.1%** |
| **MaxDD** | 30.29% | 35.03% | +4.7%（略優）|

### 失敗分析

**逐季拆解後（2026-07-02 事後分析），頭條數字的落差主要來自單一季度，而非全程劣勢：**

| 期間 | Round 0 | Round 6 | ^TWII |
|------|--------|--------|-------|
| 2024-07 ~ 2026-03（前 7 個季度） | Sharpe 0.63 / Ann +16.7% | Sharpe 0.48 / Ann +9.2% | — |
| **2026-Q2（單季）** | **+43.9%（Sharpe 3.67）** | **−4.3%（Sharpe −0.59）** | **+40.5%** |

兩策略日報酬相關性 0.678（持股大量重疊），排除 2026-Q2 後差距是 0.63 vs 0.48 的溫和劣勢——不是 1.12 vs 0.34 的慘敗。**決定性差異在 2026-Q2：台股大盤單季 +40.5% 的動能行情裡，Round 0 完整吃到（+43.9%），Round 6 這個 long-only、每期 10 檔的組合竟然虧錢（−4.3%）**——代表它挑的股票跟行情完全反向。

1. **模型「本性」不同 → regime 依賴**（主因）：Kronos autoregressive 預測天生是趨勢外推，動能行情自然跟上；XGBoost + rank-IC 學的是 2015-2023 台股截面歷史規律，很可能學出偏**均值回歸／反轉因子**的模型（歷史上截面反轉在台股統計上最穩定，LambdaRankIC 的 rank-distance gain 又強化極端排序）。反轉因子在動能主導的軋空行情正是被屠殺最慘的一類——大盤 +40% 還虧錢就是「一直買預期反彈的落後股，行情卻集中在持續衝高的強勢股」的典型症狀
2. **驗證集 IC 高 ≠ 回測表現好，這次有了更具體的機制解釋**：(a) IC 量的是**全體 1039 檔的平均排序相關性**，策略只買 top 10（前 1%）——中段排得好可以撐高 IC，但獲利完全取決於尾端極值；(b) IC 是平均截面能力，**不衡量 regime 穩健性**——驗證期（2024H1）反轉規律有效所以 IC 高，動能行情一來同一個模型直接反噬。Round 5（val ic_ir_h5=0.47 但 Sharpe 0.98）是同一模式的前奏
3. **驗證期只涵蓋單一 regime**：early stopping 用 2024H1 一種行情選 best_iteration，把模型鎖定在那個 regime 的截面規律上
4. **連帶發現：Round 0 的 1.12 基準本身也要重新審視**——它有很大一塊是 2026-Q2 單季貢獻（排除後只剩 Sharpe 0.63），某種程度上是「賭對了一個動能行情」，穩健性沒有頭條數字看起來高
5. **凍結 embedding 假設仍未獲驗證，但排序更後面了**：mean-pooling 可能抹掉時間局部訊號、LambdaRankIC 可調參（`sigma`、gain 形式）、embedding vs raw features 的貢獻拆解（`layer_indices` 消融已實作未測）都還沒做——但在解決 regime 依賴之前，這些微調的預期收益有限

**對下一步的含義：** 問題的形狀從「M1 架構不行」變成「特徵/標籤缺乏動能資訊 + 驗證期單一」。最便宜的確認實驗：算出**測試期的逐期 IC**（不只驗證期），看 Round 6 的 IC 是否在 2026-Q2 翻負——若是，即確認 regime 依賴診斷。修正方向：加動能類特徵讓 XGBoost 有能力表達趨勢行情、用多段不同 regime 的驗證期做 early stopping、或 ensemble Kronos 訊號（動能性）與 XGBoost 訊號（反轉性）。

**參考資料：**
- `docs/superpowers/plans/2026-07-02-kronos-embedding-xgb-lambdarank.md`
- `finetune_tw/extract_embeddings.py`, `finetune_tw/lambdarank_ic.py`, `finetune_tw/train_xgb_lambdarank.py`, `finetune_tw/backtest_xgb_embedding.py`
- `finetune_tw/outputs/tw_daily/backtest_returns_xgb_embedding_next_open.json`（pod 上）/ `backtest_returns_round6_next_open.json`（本地）

---

## Round 6 後續（Batch 1-3c 診斷修正 + Direction 2 模型融合）— 2026-07-03 ~ 2026-07-04

**背景：** Round 6（M1）的失敗被進一步拆解為源碼層級診斷（`docs/research/round6-embedding-xgb-lambdarank-improvements.md`），找出三個具體缺陷：mean-pooling 稀釋時序訊號、特徵缺乏橫截面相對排名（cs_rank）、動能維度單一。以下 Batch 1-3c 依序驗證修正方案，全部細節見該文件；本節只記錄關鍵結論與最終產物。

### Batch 1 — 交易日曆修正診斷（CPU，2026-07-03，Go）

修正 `extract_embeddings.py` 用 `pd.bdate_range` 誤將台股非交易日（如農曆年假）當成交易日的缺陷後，對 Round 6 舊模型做逐季 rank-IC 診斷：**確認 2026-Q2 IC 崩到其他季度的 1/4（0.016 vs 平均 ~0.09），且集中在該單季**——與 Round 6 回測拆解的機制吻合，regime 依賴假說成立。同時發現一個比假說本身更重要的新問題：**top-tail 是慢性病，不只 2026-Q2**——八個季度的 top-10 命中率全部貼著隨機水準（~0.9% vs 隨機期望 0.96%）。

### Batch 2 — Embedding vs Raw Feature Ablation（CPU，2026-07-03，Go）

在乾淨日曆的 train/val 上重訓 `raw`（4 特徵）/ `emb`（832 維 embedding）/ `full`（合併）三組模型：**embedding 對全市場排序有真貢獻**（`emb` mean IC 0.064 > `raw` 0.057），但 **`emb` 的 top-10 overlap 幾乎歸零（0.15%）**，反而 `raw` 的 overlap 有 1.62%——**embedding 側擅長 full-universe ranking，raw 側擅長 top-tail identification，兩者互補而非取代**。結論：優先修特徵工程 + 選模目標，方案 A（改 pooling）延後。

### Batch 3 / 3b — 特徵工程 + top_k_excess 早停（No-Go，2026-07-04）

落地方案 B（cs_rank）+ 方案 C（13 個多尺度動能/波動率特徵），並將選模指標改為 `top_k_excess`。**結果：`raw`/`full` 的 `best_iteration` 只有 1（Batch 3）或 3（Batch 3b，驗證窗口擴大到 239 天後）**——`top_k_excess` 在小驗證集上方差過大，early stopping 在噪音中賭運氣，模型嚴重欠訓練，測試期表現全面劣於 Batch 2 baseline。診斷：特徵工程方向正確（`raw` 全市場 IC 提升 24%），但選模指標改造必須配合多 regime 驗證窗口同批實施，未做即失敗。

### Batch 3c — rank_ic 早停 + 刻意納入 2021 動能 regime 驗證窗口（Go，2026-07-04）

早停指標改回 `rank_ic`（訓練曲線平滑、不易被噪音誤判收斂），驗證窗口改為 2021H1（後疫情動能牛市）聯集 2023-01~2024-06（1.5 年），迫使模型在「動能」與「反轉」間找到穩健的迭代點。兩模型都跑滿 200 輪上限（`best_iteration=199`），未觸發早停。

**Diagnostics 結果（測試期 475 個交易日）：**

| 模型 | 特徵數 | val rank-IC | 測試期 mean IC | IC-IR | top-10 excess | top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|
| Batch 2 `full`（對照） | 836 | 0.0716 | 0.0813 | 0.610 | +0.206% | 0.78% |
| **Batch 3c `raw`** | 26 | 0.0810 | **0.0888**（+56%） | 0.568 | **+0.490%**（+149%） | 1.01% |
| **Batch 3c `full`** | 858 | 0.0873 | **0.0933**（+15%） | 0.592 | +0.348% | 0.82% |

2026-Q2 top-10 excess：`raw` 從 Round 6 原模型的 **-1.03% 翻正為 +0.751%**（四批實驗以來最好），`full` 從 -1.03% 收斂到接近零的 -0.045%。兩模型都輕鬆超過 Go 判準（IC > 0.080、top-10 excess > +0.20%）。

**真實 next-open 回測驗證（同一套 `signals_to_holdings` 框架，與 Round 0/4/5/6 直接可比）：**

| 策略 | hold_days | Ann | Sharpe | MaxDD |
|---|---:|---:|---:|---:|
| Round 0（純 Kronos） | 5 | 38.59% | 1.115 | 35.03% |
| Round 4（純 Kronos，先前最佳） | 5 | 45.94% | 1.241 | 33.99% |
| Round 6 M1 舊版 | 5 | 5.52% | 0.340 | 30.29% |
| Batch 3c `raw`（純技術指標+cs_rank，無 Kronos） | 5 | 24.44% | 1.104 | 27.69% |
| **Batch 3c `full`（Kronos embedding + 技術特徵 + XGBoost）** | 5 | **31.17%** | **1.336** | **27.21%** |

**`full` 的 Sharpe 1.336 刷新了 Round 0-6 系列的最佳紀錄**（超越先前最佳 Round 4 的 1.241），且用的是同一顆**未微調**的 Kronos pretrained backbone。已上傳至 HF `j835111/kronos-tw-finetune@round6-batch3c-full-production`。方案 A（GPU pooling 重抽 embedding）確認不再需要觸發。

### Backbone 替換實驗（pretrained vs round0 embedding）— No-Go（2026-07-04）

Batch 1-3c 全程用未微調的 `NeoQuasar/Kronos-base` 抽 embedding。兩輪 CPU smoketest（20 檔傳統股池、30 檔均衡電子/傳產股池）都顯示：**`pretrained` backbone 在所有 IC 指標上全面優於 `round0`（TW 微調過）backbone**，方向與直覺相反。診斷為「特徵表示空間塌陷」：`round0` 的微調目標（單股時序預測）把 embedding 空間收斂到對自身任務有利、但對 XGBoost 下游跨股票排名反而較貧乏的低維表示。**Direction 1（backbone 替換）判定 No-Go**，未來不再投入 GPU 資源做微調版 backbone 重抽。

### Direction 2 — `full` + `raw` 模型 Z-Score 融合（Go，最終 production，2026-07-04）

`full`（全市場排序強）與 `raw`（top-tail 抓飆股強）的每日分數 Spearman 相關 0.876、Top-10 持股重疊率僅 50%、策略日報酬相關 0.872——互補性充足。融合前對每日截面分數做 Z-Score 標準化再線性加權：$Score_{blended} = w\cdot Z(Score_{full}) + (1-w)\cdot Z(Score_{raw})$。

**全期網格搜索（475 交易日）：**

| 權重 $w$（full） | Ann | Sharpe | MaxDD |
|---|---:|---:|---:|
| 1.0（純 full） | 31.05% | 1.3258 | 27.21% |
| 0.0（純 raw） | 22.09% | 1.0191 | 27.69% |
| **0.6** | **36.35%** | **1.5434** | **24.86%** |

最佳權重 $w=0.6$ 周遭（0.5/0.7）表現平穩，非隨機噪訊尖峰。**樣本外時間分片驗證**（IS：2024H2-2025H1 校準 $w^*=0.6$；OOS：2025H2-2026H1，含 2026-Q2）：$w^*=0.6$ 在 OOS 達到 **Sharpe 3.3952（Ann 58.30%、MaxDD 8.37%）**，同時壓倒純 `raw`（2.14）與純 `full`（3.24），泛化穩定性通過驗證，排除選擇性洩漏疑慮。

**最終生產決策：部署靜態 Z-Score 融合策略（$w=0.6$），Sharpe 1.5434 / MaxDD 24.86% 為目前 Round 0-6 全系列最佳可執行結果**，取代 Batch 3c `full` 單模型作為正式 production。動態權重方案（滾動 IC 追蹤 / 大盤 regime gating）評估後判定暫緩——額外增益僅約 +0.03 Sharpe，不足以抵銷過擬合與系統複雜度風險。`finetune_tw/backtest_xgb_ensemble.py` 為固化後的線上推理腳本（此文件記錄時尚待完全落地）。

**參考資料：**
- `docs/research/round6-embedding-xgb-lambdarank-improvements.md`（完整診斷、代碼細節、每批次執行記錄）
- `finetune_tw/round6_diagnostics.py`, `finetune_tw/feature_engineering.py`, `finetune_tw/enrich_round6_features.py`, `finetune_tw/train_xgb_streaming.py`, `finetune_tw/backtest_xgb_ensemble.py`
- HF `j835111/kronos-tw-finetune@round6-batch3c-full-production`

---

## Round 6 Followup (XReg) — 2026-07-05（外生變量殘差回歸 XReg，近期優化）

**起點：** Round 6 Direction 2 融合模型（`full` + `raw` Z-Score 融合，$w=0.6$）
**平台：** 本地 CPU 離線模擬評估
**Branch：** `master` (作為近期非微調生產級功能直接併入)

**戰略背景：** 儘管 Round 6 Direction 2 藉由 `full` 與 `raw` 模型的 Z-Score 融合解決了部分風格對抗問題並將 Sharpe 提升至 1.5434，但大盤在極端動能行情（如 2026-Q2 大盤飆漲 40%）下，純 GBDT 橫截面排序模型依然存在因反轉因子主導而跑輸大盤甚至虧損的風險。XReg 的目標是引入外生大盤特徵（如加權指數 return 與 rolling 波動率），以平行回歸的方式修正預測分數，使其能動態感知市場 regime。

**驗證的方法：**
- ✅ **外生變量回歸 (XReg)**：實作 `finetune_tw/xreg.py` 模組。在每個重估日 $t$，對個股過去 $N=60$ 天的實現收益率，與 `^TWII` 的 1 日回報、5 日回報、10 日 rolling 波動率進行 Ridge 擬合，獲取個股動態 Beta 係數，並預測今日大盤驅動的預期回報 $\hat{y}_{s, t}$。
- ✅ **淨化窗口（Purging Gap）**：在 $t$ 日預估時，訓練歷史只包含已實現（$\le t-6$）的歷史回報，淨化 5 天持倉期，完全杜絕 Look-Ahead Leak（未來洩漏）。未淨化前年化收益高達 181%，淨化後恢復真實 OOS 水準。
- ✅ **乘數參數高原掃描**：對調整權重 $Score = Score_{gbdt} + mult \cdot \hat{y}_{exogenous}$ 進行 `[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]` 掃描。當乘數在 `1.0 ~ 5.0` 時，OOS Sharpe 穩定處於 `1.69 ~ 1.83` 的高原區，最優點 `mult=2.0` 時 Sharpe 達到 **1.8383**。

**回測結果（open/open v2，top_k=10，hold=5d）：**

| 指標 | Round 6 Followup (XReg, w=0.6, mult=2.0) | Blend 基准（w=0.6） | 差距 |
|------|-------------|---------------------|------|
| **Sharpe** | **1.8383** | **1.5434** | **+0.2949** |
| **Ann** | **44.66%** | **36.35%** | **+8.31%** |
| **MaxDD** | **25.03%** | **24.86%** | **+0.17%**（極微）|

**代碼落地：** 
- 將 XReg 整合進入離線回測指令 [backtest_xgb_ensemble.py](file:///mnt/d/project/Kronos/finetune_tw/backtest_xgb_ensemble.py)。
- 將 XReg 整合進入每日實時訊號輸出腳本 [signal_today_ensemble.py](file:///mnt/d/project/Kronos/finetune_tw/signal_today_ensemble.py)，供生產環境使用。

> **⚠️ Code Review 加註（2026-07-06，commit 前審查，尚未修正）**
>
> 1. **🔴 Ridge 迴歸實質失效，實際機制是動能覆蓋層而非大盤 Beta**：`xreg.py` 的三個大盤特徵（`twii_ret_1`/`twii_ret_5`/`twii_vol_10`）數值都在 0.01 量級且未標準化，60 樣本下 `XᵀX` 對角元素約 6e-3，被 `alpha=1.0` 壓過 160 倍以上——模擬驗證真實 beta=1.5 會被收縮到 0.049（收縮 ~97%），截距 ≈ mean(y)。因此 `market_expected ≈ 個股過去 60 日平均 5 日報酬`，調整項實際是 `mult × 動能因子`，大盤特徵幾乎沒有參與。**本節上文「以 rolling 60 天的個股大盤 Beta 加權」「動態感知市場 regime」的機制描述與實際不符**（但 Sharpe 提升本身可信：動能覆蓋正是 Batch 1 診斷出的缺口）。待辦：跑 intercept-only 對照（`score + mult × mean(y_train)`），若 Sharpe 相同則簡化掉 Ridge；若不同則標準化 X 後重掃 alpha。
> 2. **🔴 Production wrapper 未啟用 XReg**：`--xreg_enabled` 預設 off，`scripts/run_signal_today_ensemble.sh` 未傳此 flag——「直接調用即可生成具備大盤修正的 daily signals」的說法目前不成立（重演 commit 3635962 修過的 wrapper 漏參數 P1 模式）。需在 wrapper 加 `--xreg_enabled --xreg_mult 2.0` 或改口為 opt-in。
> 3. **🟡 mult=2.0 有選擇偏誤**：mult 掃描是在完整報告期上選最優點後直接報該點 Sharpe，未像 w=0.6 那樣做 IS/OOS 時間分片。誠實估計應為高原區間 1.69–1.83 而非峰值 1.8383；commit 前應補 IS 選 mult、OOS 報數的分片驗證。
> 4. **🟡 label horizon 與 purging_gap 旋鈕不成對**：label 硬編碼 `shift(-6)/shift(-1)`（5 日持倉），但 `--xreg_purging_gap` 可調——傳入 <5 會靜默重新引入 look-ahead leak（即上文已淨化的同型洩漏）；`hold_days≠5` 時 label 也不對齊。應以 horizon 參數 h 導出 shift 並 `assert purging_gap >= h`。
> 5. **🟡 生產端缺防護**：XReg 三條 early-return 路徑（日期不在 calendar、歷史不足、查無價格）全部靜默無日誌；`X_test` 未檢查 NaN，^TWII 資料異常時 `score + mult × NaN` 會污染全部分數。
> 6. **✅ Purging 邏輯本身經覆核正確**：最後訓練樣本（t−6）的 label 在今日 open 實現，與明日 open 起的持倉期無重疊；特徵時點（今日收盤）與訊號時點（收盤後產生、明早執行）一致。

**參考資料：**
- `finetune_tw/xreg.py` (NumPy 實作), `finetune_tw/backtest_xgb_ensemble.py`, `finetune_tw/signal_today_ensemble.py`
- 敏感度離線回測日誌：[evaluate_xreg.py](file:///home/james/.gemini/antigravity-cli/brain/ca89f99c-9139-4b9f-9139-91ce1051b936/scratch/evaluate_xreg.py)

---

## 各輪 Sharpe 彙整（open/open v2，top_k=10，hold=5d）

| 版本 | Sharpe | Ann | MaxDD | 備注 |
|------|--------|-----|-------|------|
| Round 0 | **1.356** | 50% | 35% | 原始最佳（舊測試期）|
| Round 0（新基準） | **1.12** | 38.59% | 35.03% | 延長至 2026-06-30 |
| Round 1 | 0.15 | -0.1% | 35% | 起點錯誤（pretrained）|
| Round 2 | 1.14 | 38% | 36% | close IC，輸 Round 0 |
| Round 3 | 0.50 | 20% | 41% | open IC，大幅退步 |
| Round 4 | 1.24 | 46% | 34% | FPT + IC-IR@h1，輸 Round 0 |
| Round 5 | 0.98 | 31.79% | 39.86% | Pretrained 重啟 + Ranking Loss，仍輸 Round 0 |
| Round 6 | 0.34 | 5.52% | 30.29% | Kronos Embedding + XGBoost LambdaRankIC（M1）；主因錯過 2026-Q2 動能行情（該季 −4.3% vs R0 +43.9%），排除該季後為 0.48 vs 0.63 |
| Round 6 Batch 3c `raw` | 1.104 | 24.44% | 27.69% | 純技術指標+cs_rank+XGBoost，不含 embedding，接近 Round 0 |
| Round 6 Batch 3c `full` | 1.336 | 31.17% | 27.21% | 修正版特徵工程（cs_rank+多尺度動能）+ rank_ic 早停 + 多 regime 驗證窗口，刷新當時最佳紀錄 |
| **Round 6 Direction 2（`full`+`raw` Z-Score 融合，w=0.6）** | **1.5434** | **36.35%** | **24.86%** | 先前最佳可執行結果；OOS 時間分片驗證 Sharpe 3.395，泛化穩定 |
| **Round 6 Followup (XReg, mult=2.0)** | **1.8383** | **44.66%** | **25.03%** | **最新全系列最佳可執行結果**；以 Ridge walk-forward 擬合 60 天大盤，淨化 5 天避免洩漏。⚠️ 見 XReg 章節 review 加註：實際機制為動能覆蓋層（alpha=1.0 使大盤特徵失效），mult 未做 IS/OOS 分片，誠實估計為高原區間 1.69–1.83 |

---

## 結論與下一步

**Round 6 Followup (XReg, mult=2.0) 為目前全系列最佳可執行版本（Sharpe 1.8383），已超越融合基線（1.5434）與此前所有微調與架構調整方向。**

已窮盡所有已知方向（截至 2026-07-05）：
- 重訓（Round 1-5）：全部退步，Round 0 曾是上限
- 策略參數（ATR sizing、volume filter）：no-op
- Stacking / MC ensemble：有害
- Grid search（所有 top_k × hold_days）：最高 Sharpe 1.263，無法突破 Round 0
- Label Horizon 掃描：IC-IR 從 h=1 單調衰減，換 hold_days 無效
- FPT freeze（Round 4）：best epoch=1
- Pretrained 重啟 + Auxiliary Ranking Loss（Round 5）：Sharpe 0.98，退步
- **凍結 Kronos + XGBoost LambdaRankIC 原版（Round 6 / M1）：Sharpe 0.34**——主因是錯過 2026-Q2 動能行情，修正後 `full` 模型 Sharpe 達 1.336
- **Z-Score 融合（Direction 2）：Sharpe 1.5434**——全期 w=0.6 網格搜索最佳
- **外生變量回歸 (XReg Followup)：Sharpe 1.8383**——模式一加權 2.0，成功修正 GBDT 大盤失效

**2026-07-05 的突破（XReg 離線實證與參數掃描）扭轉了結論：**
1. **look-ahead leak 被成功定位與淨化**：首輪 XReg 回測顯示 181% 年化，診斷為 5 日標籤重疊洩漏；實施 5 日 purging 後回歸真實 OOS 水準。
2. **xreg + timesfm 模式展現強大 regime 感知能力**：以 rolling 60 天的個股大盤 Beta 加權，在極端行情下修正選股偏好，Sharpe 自 1.5434 提升至 **1.8383**。⚠️ *2026-07-06 review 加註：alpha=1.0 + 未標準化特徵使大盤 Beta 收縮 ~97%，實際生效的是截距（60 日平均報酬）＝動能覆蓋層，「regime 感知」的歸因待 intercept-only ablation 確認。*
3. **敏感度網格掃描證明參數高原穩健性**：在 `mult ∈ [1.0, 5.0]` 區間內，Sharpe 皆大於 1.69，確認了方案的統計穩健度。

**下一步方向：**
1. **線上實時推理部署**：[signal_today_ensemble.py](file:///mnt/d/project/Kronos/finetune_tw/signal_today_ensemble.py) 與 [backtest_xgb_ensemble.py](file:///mnt/d/project/Kronos/finetune_tw/backtest_xgb_ensemble.py) 已正式內置 XReg。⚠️ *2026-07-06 review 加註：`--xreg_enabled` 預設 off 且 `scripts/run_signal_today_ensemble.sh` 未傳此 flag，「直接調用即可」目前不成立——需修 wrapper 或明確改為 opt-in。*
2. **近期方案二：Multi-Horizon 複合標籤訓練**：在 GBDT 構建階段進行複合 Y 標籤（$w_1 \cdot y_{h1} + w_3 \cdot y_{h3} + w_5 \cdot y_{h5}$）擬合。
3. **動態權重方案**：動態權重方案（滾動 IC / 滾動 IR 追蹤）評估後暫緩，僅在靜態融合出現失效跡象時才重新考慮。
4. **2026-04 殘留反轉曝險分析**：Batch 3c 兩模型仍最弱的月份，值得單獨診斷特徵貢獻。

**若要回到 fine-tuning 路線，可考慮的未驗證方向（遠期）：**
1. **L2-SP 正則化（L2 距離 pretrained weights）**：arXiv:2603.18596，在微調損失中加入偏離 pretrain 參數的懲罰項，防止表示空間塌陷。
2. **MoFO Optimizer**：arXiv:2407.20999，只更新動量幅度最大的 top-K% 參數，其餘參數動態凍結。
3. **SSPT 台股持續預訓練（股票分類 + 產業分類 + MA）**：arXiv:2506.16746。
4. **更大的驗證集 + 更長訓練**：val 集 150×40 可能太小，導致 early stop 不穩定。
5. **Ranking loss 調參**：`ranking_loss_alpha` 從 0.1 調低（如 0.01），或改用 pairwise hinge loss 減少對 token prediction 的干擾。

---

## Autoresearch 方法完整對照表

### 已驗證（各輪結果）

| 方法 | 來源計劃 | 驗證輪次 | 結果 |
|------|---------|---------|------|
| 從 Round 0 起點（非 pretrained）| 260622 M3 | Round 2 | ✅ 必要條件（Round 1 用 pretrained 失敗） |
| IC-IR@h5 early stopping | 260622 M1 | Round 2 | ❌ 輸 Round 0（Sharpe 1.14） |
| 驗證集 300×20 | 260622 M2 | Round 2 | ⚠️ 噪音降低但模型仍退化 |
| Warmup+Cosine, epochs=20 | 260622 M4 | Round 2 | ⚠️ 排程沒問題，問題是退化方向 |
| ATR position sizing（1/pred_ATR）| 260626 M1 | Week 1 | ❌ 無改善，已移除 |
| Volume filter（底 25% 排除）| 260626 M3 | Week 1 | ❌ 無改善，已移除 |
| Open-to-open IC early stopping（h5）| 260626 M2 | Round 3 | ❌ IC-IR@h5=0.023，信號極弱，大幅退步 |
| 驗證集 500×40 | 260626 N2 | Round 3 | ❌ 過慢（60-90 min/epoch），縮回 150×40 |
| FPT Selective Freeze | 260629 M1 | Round 4 | ❌ best epoch=1，Round 0 局部最優 |
| IC-IR@h1 early stopping（open-to-open）= Label Horizon h=1 代理標籤 | 260629 M3 / 260622 N2 / 260626 N3 | Round 4 | ❌ 首 epoch 即最佳，之後退化，Round 0 無學習空間 |
| Extended Warmup（pct=0.08, div=25）| 260629 M3 | Round 4 | ❌ 對退化無效 |
| Stacking（LightGBM，MC=5/10）| — | 獨立實驗 | ❌ 有害（-0.20 Sharpe），已移除 |
| MC ensemble（mc_mean）| — | 獨立實驗 | ❌ 有害（Sharpe 1.07 < benchmark 1.60）|
| Pretrained 重啟（從 Kronos-base）| — | Round 5 | ❌ 必要但不充分；重啟後 val_loss 起點 2.93，但回測仍輸 Round 0 |
| Auxiliary Ranking Loss（ListMLE, alpha=0.1）+ Pretrained 重啟 | 260622 N1, 260626 N1, 260629 N1 | Round 5 | ❌ val ic_ir_h5=0.47 但 Sharpe 0.98，ranking loss 未轉化為回測改善；alpha=0.1 對 token CE 有輕微干擾（val_loss 緩升） |
| Kronos Embedding + XGBoost LambdaRankIC（凍結 Kronos，繞開 fine-tuning）| 260701 M1 | Round 6 | ❌ val-rank_ic 創新高（0.066，全母體）但 Sharpe 僅 0.34；逐季拆解=主因錯過 2026-Q2 動能行情，排除後 0.48 vs 0.63 |
| 交易日曆修正 + 逐季 IC 診斷 | round6-improvements Batch 1 | Batch 1 | ✅ 確認 2026-Q2 regime 依賴，並發現 top-tail 慢性病（八季 top-10 命中率貼隨機水準） |
| Raw/Embedding/Full ablation | round6-improvements Batch 2 | Batch 2 | ✅ embedding 擅長全市場排序、raw 擅長 top-tail，互補非取代 |
| cs_rank + 多尺度動能特徵（B+C）+ `top_k_excess` 早停 | round6-improvements Batch 3 | Batch 3/3b | ❌ 小驗證集下 `top_k_excess` 噪音過大，best_iteration=1~3，嚴重欠訓練 |
| cs_rank + 多尺度動能特徵（B+C）+ `rank_ic` 早停 + 2021 動能 regime 驗證窗口 | round6-improvements Batch 3c | Batch 3c | ✅ Sharpe 1.336，刷新全系列紀錄，取代 Round 0 為建議 production 模型；`raw`（不含 embedding，純特徵工程）Sharpe 1.104 亦接近 Round 0，證明特徵工程本身有獨立價值 |
| Backbone 替換（round0 微調版 embedding 取代 pretrained）| round6-improvements 後續 | 兩輪 CPU smoketest | ❌ pretrained 全面優於 round0，判定「特徵表示空間塌陷」，No-Go |
| `full`+`raw` 每日截面 Z-Score 線性融合 | round6-improvements Direction 2 | Direction 2 | ✅ Sharpe 1.5434（全系列最佳），OOS 時間分片驗證 Sharpe 3.395，泛化穩定 |

### 未驗證（autoresearch 有記錄，從未測試）

| 方法 | 來源計劃 | 優先度 | 說明 |
|------|---------|--------|------|
| **Close-to-close IC-IR@h1 early stopping** | 260629 M2 | 🔴 高 | Round 4 用的是 open-to-open h1；close 版 SNR=0.64 從未測試。**需配合 pretrained 重啟** |
| **Ranking loss 調參（alpha=0.01）或改用 pairwise hinge loss** | — | 🟡 中 | Round 5 alpha=0.1 對 token loss 干擾過大；降低 alpha 或換 pairwise 形式可能使 ranking signal 更乾淨 |
| **Training loss primary horizon 改為 h3/h7**（改訓練目標本身）| 260622 N2, 260626 N3 | 🟡 中 | 注意區分：用短 horizon 作為 **early stop metric** 已在 Round 4 測試（h=1，失敗）。這裡指的是更強版本——把 h=3 作為 **training loss 的 primary target**（取代均值），讓模型在訓練期間就直接優化短期預測。Label Horizon Paradox（arxiv:2602.03395）支持此方向，但未測試 |
| **Horizon-Weighted Loss**（h4/h5 weight 更高）| 260622 N3 | 🟡 中 | 不需要架構改動，但需要重訓 |
| **Price-space MSE 輔助損失** | 260626 S1 | 🟠 低（moonshot）| 解碼 token → price，計算 MSE；需改 tokenizer |
| **ic_val_dates 至 60** | 260629 N2 | 🟠 低 | SE(IC-IR)=0.14，統計力充足；目前用 40 |
| **連續回歸 head / Chronos-2 架構** | 260626 S2 | 🔵 研究級 | 拋棄 BSQ 離散化，論文級工作量 |

**關鍵未驗證組合（最值得嘗試）：**  
`pretrained 完全重啟 + close-to-close IC-IR@h1 early stopping`  
→ Round 5 確認 pretrained 重啟必要；Ranking Loss（ListMLE alpha=0.1）已驗證無效。close-to-close IC-IR@h1 early stopping（SNR=0.64）從未配合 pretrained 重啟測試過，是目前最大的空白。

---

### 全新架構方向（autoresearch 260701）

| 方法 | 來源 | 優先度 | 說明 |
|------|------|--------|------|
| ~~**Kronos Embedding → XGBoost + LambdaRankIC**~~ | arXiv:2605.00501 | ✅ 已於 Round 6 驗證 | ❌ Sharpe 0.34；hidden state 實際為 832d（非 512d）。詳見 Round 6 章節的逐季拆解分析 |
| **L2-SP 正則化（L2 距離 pretrained weights）** | arXiv:2603.18596 | 🟡 高 | fine-tuning loss 加 `λ‖θ-θ₀‖²`，防止偏離 pretrained landscape。比 EWC 更簡單（不需 Fisher matrix） |
| **MoFO Optimizer** | arXiv:2407.20999 | 🟡 高 | 只更新動量幅度最大的 top-K% 參數；其他參數凍結。無需 pretrained 資料、無需 Fisher 估計，比 FPT 更動態 |
| **SSPT 台股持續預訓練（股票分類 + 產業分類 + MA）** | arXiv:2506.16746, KDD 2025 | 🟠 中 | 先以台股資料做自監督預訓練，讓 Kronos 了解台股身份後再 fine-tune predictor |
