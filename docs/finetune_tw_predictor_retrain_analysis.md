# finetune_tw Predictor Retrain — 實驗分析報告

**分支：** `fix/predictor-retrain`
**日期：** 2026-06-21
**測試期間：** 2024-07-01 → 2026-06-21

---

## 背景

本分支的目標是對 finetune_tw 台股預測模型進行系統性重訓，並評估不同訓練策略的效果。實驗比較了三個模型：

1. **Pretrained**：NeoQuasar/Kronos-base，零樣本，未做任何台股 fine-tuning
2. **Round 0**：第一輪 fine-tuning（20 epoch，lr=4e-4，CE loss，跑滿 epoch）
3. **Round 1**：本分支重訓（6 epoch，lr=1e-5，IC-based early stopping）

---

## 實驗設定

### 回測策略（共用）

| 參數 | 值 |
|------|-----|
| 股票池 | TWSE 全市場 ~1090 支 |
| 測試起點 | 2024-07-01 |
| lookback_window | 90 交易日 |
| pred_len | 10 天 |
| hold_days | 5 天換倉 |
| top_k | 20（每次持有前 20 名） |
| 基準 | ^TWII（台灣加權指數）|

### eval_forecast 設定

| 參數 | 值 |
|------|-----|
| 評估期 | val_end_date 以後（2024-07-01 起） |
| 樣本數 | ~109k–110k（每 horizon） |
| IC 定義 | 截面 Spearman rank correlation（每交易日計算後取平均）|

### Round 1 訓練設定

| 參數 | 值 |
|------|-----|
| 起點模型 | NeoQuasar/Kronos-base（預訓練，非 Round 0）|
| lr | 1e-5 |
| epochs | 最多 6（early stop patience=2）|
| early stop 指標 | val_ic（截面 IC 均值）|
| ic_val_symbols | 150 |
| ic_val_dates | 8 |

---

## 量化結果

### 回測表現

| 模型 | Annual Return | Sharpe | Max DD |
|------|--------------|--------|--------|
| ^TWII（基準）| 43.32% | 1.536 | 26.71% |
| Pretrained | -3.68% | 0.03 | 35.87% |
| Round 1 Retrain | -0.13% | 0.15 | 35.21% |
| **Round 0 Fine-tune** | **40.45%** | **1.19** | 31.60% |

### eval_forecast（預測品質）

| 指標 | Pretrained | Round 0 | Round 1 |
|------|-----------|---------|---------|
| val_loss（CE，↓）| **2.997** | 3.644 | 3.141 |
| IC@h1（↑）| **0.0497** | 0.0413 | 0.0241 |
| IC@h5（↑，換倉決策點）| 0.0268 | **0.0319** | 0.0108 |
| IC-IR@h1（↑）| 0.601 | **0.625** | 0.290 |
| ic_positive_rate@h1（↑）| 0.709 | **0.728** | 0.534 |
| direction@h1（↑）| **52.98%** | 51.11% | 49.8% |
| MAPE@h1 / MAPE_naive | 1.19× | 1.34× | 2.01× |

### Round 1 訓練歷程

| Epoch | train_loss | val_loss | val_ic |
|-------|-----------|---------|--------|
| 1 | 3.2682 | 3.0858 | -0.0292 |
| 2 | 3.0691 | 3.1162 | -0.0218 |
| 3 | 3.0648 | 3.1330 | -0.0286 |
| 4 | 3.0859 | 3.1397 | -0.0165 |
| 5 | 3.0881 | 3.1410 | -0.0114 |
| **6** | **3.0298** | **3.1414** | **-0.0083** ← best |

val_ic 全程為負，best checkpoint = Epoch 6（最不壞）。

---

## 核心發現

### 1. IC-Backtest 脫鉤

**截面 IC 最高的模型，回測表現最差。**

Pretrained IC@h1=0.0497 > Round 0 IC@h1=0.0413，但 Pretrained Sharpe=0.03 vs Round 0 Sharpe=1.19。

原因：截面 IC 測量的是「全市場 1090 支的排名相關性」，而回測 top-K 策略只在乎「前 20 名」的排名品質。兩者評估的不是同一件事。

### 2. IC-IR 和 ic_positive_rate 比 IC 均值更能預測回測效果

| 指標 | 說明 | Round 0 優勢 |
|------|------|------------|
| IC-IR = IC / σ(IC) | 信號穩定性 | 0.625 > Pretrained 0.601 |
| ic_positive_rate | 多少交易日 IC 為正 | 72.8% > Pretrained 70.9% |

Round 0 信號的均值雖低，但**波動更小、正 IC 日更多**。對 top-K 策略，信號一致性（不出現壞日子）比均值高更重要。

### 3. h5 horizon 是回測決策點，Round 0 在此反超

回測用 h5 預測排名（5 天後），不是 h1。

| Horizon | Pretrained IC | Round 0 IC |
|---------|--------------|-----------|
| h1 | **0.0497** | 0.0413 |
| h2 | **0.0469** | 0.0362 |
| h3 | **0.0437** | 0.0322 |
| h4 | **0.0415** | 0.0345 |
| **h5** | 0.0268 | **0.0319** ← 反超 |

Pretrained 的 IC 衰減率：h1→h5 衰減 46%。Round 0 衰減率：h1→h5 衰減 23%。**Round 0 在 5 天決策點的信號強度高於 pretrained**，這是回測差距的直接原因。

### 4. Fine-tuning 讓模型學到台股特有的動量/趨勢模式

Round 0 的 val_loss（3.644）比 pretrained（2.997）更差，代表它的 token-level 預測精度下降，但它學到了對 top-K 回測有利的**台股特定排名模式**：

- 更大膽的預測（MAPE 1.34× vs pretrained 1.19×），信號分化能力更強
- 長期 IC 衰減更慢，持倉週期內信號更穩定
- IC-IR 提升，說明排名在時序上更一致

推測 Round 0 學到了**台股的動量效應**（momentum effect），即近期表現強的股票預測繼續強勢。此效應在截面 IC 中不易捕捉（因為它是 unconditional），但對 top-K 策略有高度實用性。

### 5. Round 1 失敗的機制

Round 1 設計上採用 IC-based early stopping，期望讓模型直接優化截面排名能力。但訓練全程 val_ic 為負，暴露了以下問題：

1. **起點問題**：Round 1 從 pretrained（非 Round 0）出發，lr=1e-5 太保守，6 個 epoch 內不足以讓模型學到有效的台股排名信號。
2. **IC 估計噪音**：每次 val_ic 評估用 150 symbols × 8 dates = 僅 1200 個樣本，σ(IC)≈0.08，遠大於 IC 本身（0.00~0.05），信噪比太低。
3. **Early stop 的困境**：val_ic 全程為負表示模型從未學到正向信號；early stop 只能選出「最不壞」的 checkpoint，而非真正好的模型。
4. **val_loss vs backtest 的悖論**：Round 1 val_loss（3.141）優於 Round 0（3.644），但回測遠差。更低的 CE loss 反而代表模型**越保守**（預測越接近 naive no-change），分化能力越弱。

---

## 設計建議（下輪實驗）

### Early Stopping 指標

不要用 val_ic（截面 IC 均值），改用 **IC-IR@h5**：

```python
# 推薦
stopper = EarlyStopper(patience=3, mode="max", metric="ic_ir_h5")

# 現有（不推薦）
stopper = EarlyStopper(patience=2, mode="max", metric="val_ic")
```

理由：IC-IR 對噪音更魯棒，h5 對齊回測決策點。

### 訓練超參數

| 參數 | Round 1（失敗）| 建議（下輪）|
|------|--------------|-----------|
| 起點 | Pretrained Kronos-base | Round 0（更接近目標分佈）|
| lr | 1e-5 | 1e-5 ~ 5e-5（含 warmup + cosine decay）|
| epochs | 6 | 15~20（配合 early stop）|
| early stop 指標 | val_ic | IC-IR@h5 |
| ic_val_symbols | 150 | 300（降低 IC 估計噪音）|
| ic_val_dates | 8 | 20（更多日期，IC 估計更穩定）|

### 避免重蹈覆轍

1. **訓練完立刻 push 到 HF**：`push_best_model()` 已在 `hf_utils.py` 實作，確保 YAML 設定了 `hf_repo`
2. **監控 IC-IR 而非 IC**：IC 均值在小樣本下噪音太大
3. **不要從 pretrained 直接 fine-tune predictor**（除非 tokenizer 也同步 fine-tune）：Round 0 同時 fine-tune 了 tokenizer，Round 1 沿用 Round 0 tokenizer 但從 pretrained predictor 出發，這種不一致可能加劇了訓練困難

---

## 模型權重存放

| 模型 | 位置 |
|------|------|
| Round 0 tokenizer | `j835111/kronos-tw-finetune@round-0` → `tokenizer/best_model/` |
| Round 0 predictor | `j835111/kronos-tw-finetune@round-0` → `predictor/best_model/` |
| Round 1 predictor | 已遺失（molab sandbox 重啟前未下載）|
| Pretrained | `NeoQuasar/Kronos-base`、`NeoQuasar/Kronos-Tokenizer-base` |

---

## 基礎設施改動（本分支）

本分支除模型實驗外，也完成了以下工程改進：

1. **HF Hub 版控**（`finetune_tw/hf_utils.py`）：`push_best_model` / `restore_best_model` / `resolve_src`，訓練完自動 push，避免權重遺失
2. **config 路徑重構**（`finetune_tw/configs/config_tw_daily_molab.yaml`）：data 和 outputs 改為 `/marimo/Kronos/finetune_tw/` 相對路徑，不再依賴 `/mnt/first/`
3. **molab 工作流簡化**：移除 rclone/Google Drive，改用 HF Hub + ephemeral `/marimo/Kronos/` 儲存
4. **IC early stopping**（`finetune_tw/train_predictor.py`）：`EarlyStopper`、val_ic 評估、train_log.csv 記錄

---

*本分析基於 test_period=2024-07-01 ~ 2026-06-21 的台股資料，eval_forecast 使用 val_end_date 以後的樣本，backtest top-K=20、hold_days=5。*
