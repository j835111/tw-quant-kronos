# Design — finetune_tw Predictor 重練修復

**日期：** 2026-06-20
**狀態：** 已通過設計審查，待寫實作計畫
**前置診斷：** Round 0 模型體檢（`finetune_tw/eval_forecast.py`、`eval_tokenizer.py`）

## 背景與問題

Round 0 微調模型回測輸給 ^TWII 大盤（Sharpe 1.19 vs 1.54、Max DD 31.6% vs 26.7%、年化 40.45% vs 43.32%）。脫鉤持倉的純預測體檢證實根因不在策略參數，而在**模型預測力**：

- 點預測 MAPE 在每個 horizon 都**輸給 naive 不變基準**。
- 方向命中率 ~50%（擲銅板）。
- 唯一可用的是極弱的橫斷面 IC（~0.04）。

對照「未微調 Kronos-base」baseline 後發現：**predictor 微調是負貢獻**——baseline 在方向命中率（每個 horizon ~53% vs ~50%）與短 horizon IC/MAPE 全面較佳。

進一步 tokenizer 體檢證實：**tokenizer 微調是成功的**（重建 MSE 0.0069 vs baseline 0.0113，codebook 無塌縮、使用率更高）。問題純粹在 predictor。

| 體檢項目 | 微調後 | 未微調 baseline | 判讀 |
|----------|--------|----------------|------|
| 方向命中% (h1) | 51.1 | 53.0 | 微調更差 |
| IC (h1) | 0.041 | 0.050 | 微調更差 |
| tokenizer 重建 MSE | 0.00693 | 0.01128 | 微調更好 |

## 目標

重練 predictor，使其在 test set（2024-07-01 起）的 price-space 指標上**超越未微調 baseline**。可證偽：打不贏 baseline 即判定微調路線失敗，回退「直接用 pretrained + 改策略」。

## 失敗根因（修復標的）

1. **災難性遺忘**：`predictor_lr=4e-5` × 20 epochs 在僅 1090 檔台股、~9 年日線（對 102M 參數偏小）上過度訓練，洗掉預訓練泛化能力。
2. **選模指標錯誤**：用 token-CE val_loss 挑 best_model，但 token-CE 與 price-space 技能不一致。
3. **（待驗證）跨 sandbox resume**：optimizer/scheduler 狀態接續是否正確。
4. **log 同步漏洞**：`_gdrive_sync` 只同步 `best_model/`，`train_log.csv` 從未上傳，導致訓練曲線永久遺失。

## 設計

### 元件 1：Pipeline 簡化 — 凍結 tokenizer，只重練 predictor

- 不重跑 tokenizer 訓練。直接沿用既有 `outputs/tw_daily/tokenizer/best_model`（已驗證良好）。
- predictor 從 `NeoQuasar/Kronos-base` 預訓練權重初始化。
- 介面：`train_predictor.py` 已支援凍結 tokenizer（`requires_grad_(False)`），無須改動。

### 元件 2：對抗災難性遺忘的超參

| 參數 | 舊值 | 新值 |
|------|------|------|
| `predictor_lr` (= OneCycleLR max_lr) | 4e-5 | **1e-5** |
| `basemodel_epochs`（上限） | 20 | **6** |
| early-stop patience | 無 | **2** |
| batch_size / amp | 256 / bf16 | 不變 |

- （可選，plan 中標記為實驗開關）凍結 predictor 前 N 層 transformer，只微調上層 + head。

### 元件 3：price-space 早停 / 選模（修根因）

- 新增輕量驗證器 `_validate_predictor_ic()`：
  - 固定驗證宇宙：deterministic 取 ~150 檔。
  - 驗證日期：val 期間（2023-12-31 → 2024-06-30，**不可與 test 重疊**）內 ~8 個等距日期。
  - 每 epoch 結束跑一次，對每個（symbol, date）autoregressive 預測 `pred_len`，算 **val IC（h1–5 平均的 Spearman 橫斷面相關）**。
  - 成本約 +1–2 分/epoch。
- **best_model 改以 val IC 最大挑選**（取代 token-CE）。token-CE 仍記錄供觀察。
- `train_log.csv` 欄位擴充：`epoch,step,train_loss,val_loss,val_ic`。

### 元件 4：成功判準（驗證階段）

- 重練後執行 `python -m finetune_tw.eval_forecast --config <cfg>`（finetuned）與既有 `--baseline` 結果對照。
- **通過條件：** 方向命中率與 IC 在 h1–5 至少持平、理想為超越 baseline（baseline 參考：方向 ~53%、IC@h1 0.050）。
- **失敗處置：** 打不贏 baseline → 記錄結論，回退路線 1。保留舊 finetuned best_model 與 baseline JSON 作對照（不覆寫）。

### 元件 5：工程修補

- **log 同步**：`_gdrive_sync` 擴大同步範圍至含 `train_log.csv` 與 price-eval log（或單獨上傳），避免再次遺失訓練曲線。
- **resume 驗證**：在 plan 中明確加入「重啟後確認 scheduler/optimizer 狀態正確接續」的驗證步驟。RTX6000 ~56 分/epoch × 6 ≈ 5.6h > sandbox 3-4h 壽命，故跨 sandbox resume 是**必要組件**，須驗證可靠。

## 不在範圍（YAGNI）

- 不調整回測策略參數（top_k/hold_days/風控）——本輪聚焦模型預測力。
- 不改變預測目標（仍預測絕對 OHLCV 路徑）。
- 不做超參 sweep——先單次謹慎重練驗證方向。

## 風險

- **Compute 硬限制**：多 epoch 必然跨 sandbox，依賴 resume 可靠性；或需付費/常駐環境。
- **題目本質難**：即使修好訓練，台股日線預測上限可能就很低（baseline 方向也僅 ~53%）。成功判準設為「超越 baseline」而非絕對門檻，正是為此。
- **val IC 早停雜訊**：~150 檔 × 8 日期的 IC 估計有變異，patience=2 可能過早停。plan 需考慮 IC 平滑或加大驗證宇宙。

## 產出物

- `finetune_tw/train_predictor.py`：加 price-space 早停 / 選模、log 欄位、gdrive sync 修補。
- `finetune_tw/configs/`：新增重練 config（lr 1e-5、epochs 6）。
- 驗證：`eval_forecast.py` finetuned vs baseline 對照表。
- 既有 `eval_forecast.py` / `eval_tokenizer.py` 為可重用體檢工具（已建）。
