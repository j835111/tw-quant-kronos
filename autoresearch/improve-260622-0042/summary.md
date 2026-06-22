# Improvement Research Summary
**專案：** finetune_tw 台股預測模型改善
**日期：** 2026-06-22 | **研究輪次：** 15 | **類別覆蓋：** 5/5 | **狀態：** SATURATED

---

## 研究統計

| 項目 | 數量 |
|------|------|
| 總研究輪次 | 15 |
| 新洞見 | 10 |
| 延伸洞見 | 2 |
| 重複（跳過）| 0 |
| 覆蓋類別 | 5/5 |
| 飽和窗口 | 3（最後 3 輪新洞見 < 2）|
| 生成 PRD | 3 |

---

## 核心診斷（三個根本原因）

```
Round 1 失敗的根本原因鏈：
┌─────────────────────────────────────────────────────────┐
│ 1. 起點錯誤 → 訓練不一致                                  │
│    pretrained predictor + Round 0 tokenizer             │
│    → 前者從未見過後者的碼簿分佈                            │
│    → 6 epoch 內無法建立有效映射                           │
│                                                         │
│ 2. 指標噪音 → Early Stop 失效                             │
│    150×8 = 1200 樣本，SNR < 1                            │
│    → Stopper 選「最不壞」而非「真正好的」                   │
│                                                         │
│ 3. 指標不對齊 → 即使有信號也選不到                          │
│    val_ic（全 horizon 均值）≠ 回測決策點（h5）              │
│    → 即使 h5 信號存在，也被 h1~h4 噪音淹沒                  │
└─────────────────────────────────────────────────────────┘
```

---

## 改善方案優先矩陣

```
影響
 高 │ M3(起點)  M1(IC-IR)  M4(LR)
    │    *         *          *
    │
 中 │          M2(樣本)  M5(push)  N1(ranking loss)
    │              *        *           *
    │
 低 │                         N2(horizon)  N3(weighted)
    │                              *            *
    └─────────────────────────────────────────
      低         中         高          極高
                           風險
```

**Round 2（本輪）：** M3 → M1 → M2 → M4 → M5（全為低風險高影響）
**Round 3（下輪）：** N1 pairwise ranking loss（中風險高影響）
**探索性：** N2 Label Horizon Paradox、S1 Listwise Loss

---

## 最重要的新洞見（本研究發現）

### 1. Label Horizon Paradox（arXiv 2602.03395，2026-02）
訓練監督 label 的 horizon 不必等於推理 target horizon。最佳監督信號往往在「中間 horizon」（h3），因為：
- 近 horizon（h1）：signal 小，但 noise 也小
- 遠 horizon（h5）：signal 大，但 noise 也增大
- 中間（h3）：信噪比最高

**本專案影響：** Round 3 可嘗試 h3 loss + h5 IC-IR early stop，可能進一步提升 h5 排名品質。

### 2. arXiv 2510.14156（CIKM 2025）Loss Function Benchmark
S&P500 日線，Transformer 架構，系統比較：pointwise CE < pairwise ranking < listwise ranking
- Listwise loss 在 IC、ICIR、portfolio return 三指標均顯著優於 CE loss
- 建議 Round 3 嘗試加入 pairwise ranking loss 作為輔助（不需要全替換 CE）

### 3. 台灣市場短期動量弱
TWSE 1-5 天動量 IC 弱（驗證了 pretrained h1→h5 衰減 46% 的合理性），但 5-20 天動量明顯。Round 0 fine-tuning 縮短了衰減（23%），說明模型確實學到了台股的動量維持特性。

---

## Round 2 執行清單（可直接按此執行）

```bash
# Step 1: 確認 Round 0 predictor 在 HF 上
hf ls j835111/kronos-tw-finetune --revision round-0

# Step 2: 程式碼改動（3 個檔案）
# A. finetune_tw/ic_validation.py: 新增 validate_predictor_ic_ir()
# B. finetune_tw/train_predictor.py: 改用 ic_ir_h5 early stop + 更新 log
# C. finetune_tw/config.py: 新增 hf_revision, hf_revision_out 欄位

# Step 3: Config 改動
# finetune_tw/configs/config_tw_daily_rtx6000.yaml:
#   pretrained_predictor: "j835111/kronos-tw-finetune"
#   hf_revision: "round-0"
#   basemodel_epochs: 20
#   predictor_lr: 0.00005
#   ic_val_symbols: 300
#   ic_val_dates: 20
#   early_stop_patience: 3

# Step 4: 在 MoLab 訓練（約 10 小時）
python -m finetune_tw.train_predictor --config finetune_tw/configs/config_tw_daily_rtx6000.yaml

# Step 5: 回測 + 對比 Round 0
python -m finetune_tw.backtest --config finetune_tw/configs/config_tw_daily_rtx6000.yaml
python -m finetune_tw.eval_forecast --config finetune_tw/configs/config_tw_daily_rtx6000.yaml
```

---

## 收斂預測

| 指標 | Round 0（已知）| Round 2（預測）| 目標 |
|------|--------------|--------------|------|
| Sharpe | 1.19 | 1.2~1.4 | ≥ 1.5 |
| Annual Return | 40.45% | 35~45% | > 15% |
| Max Drawdown | 31.60% | 25~30% | < 20% |
| IC-IR@h5 | ~0.4（推估）| 0.5~0.7 | 最大化 |

**保守預估：** Round 2 能縮小 Round 0 vs 基準（Sharpe 1.19 vs 1.54）的差距約 30~50%，達到 Sharpe ~1.3~1.4。要達到 Sharpe ≥ 1.5，可能需要 Round 3 引入 ranking loss。
