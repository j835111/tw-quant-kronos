# PRD: IC 驗證樣本擴大 + 多指標監控

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## 問題陳述

現有 IC 驗證在每 epoch 末執行，但樣本量嚴重不足（150 symbols × 8 dates = 1200 pairs），導致 σ(IC) ≈ 0.08，而 IC 信號本身只有 0.00~0.05，SNR < 1。

根據統計推導（Central Limit Theorem）：
- σ(IC_mean) = σ(IC_daily) / √N_dates ≈ 0.08 / √8 ≈ 0.028
- 要達到 95% 顯著性（z = 1.96），需要 IC_mean > 1.96 × 0.028 ≈ 0.055
- 但 Round 0 IC@h5 = 0.0319，遠低於統計顯著閾值

→ 現有設定下，even Round 0 的最佳 checkpoint 都*無法被統計顯著地偵測到*。

**改到 300 symbols × 20 dates = 6000 pairs：**
- σ(IC_mean) ≈ 0.08 / √20 ≈ 0.018
- 顯著性閾值降至 ≈ 0.035，Round 0 IC@h5 = 0.0319 接近可偵測範圍
- IC-IR = mean / σ 的估計也更穩定

---

## 使用者故事

1. 作為研究員，我需要每個 epoch 末的 IC 估計有足夠統計功效（power），讓 early stopper 可以真正分辨好/壞 checkpoint。
2. 作為訓練腳本，我需要 train_log.csv 記錄 `val_ic`、`ic_ir_h5`、`val_loss`，讓我在訓練後可以繪製學習曲線並選出最佳 epoch。

---

## 需求

### Config 改動（MoSCoW）

**Must：**
```yaml
ic_val_symbols: 300    # 150 → 300
ic_val_dates: 20       # 8 → 20
early_stop_patience: 3 # 2 → 3（樣本增加後允許更多容忍）
val_ic_horizons: 5     # 維持，但 early stop 只看 h5
```

**Should：**
```yaml
# ic_val_dates 跨整個 val period（2024-01-01 ~ 2024-06-30）均勻採樣
# 現有 pick_val_dates() 已做 linspace，只需改 n=20 即可
```

**Could：**
- 對 ic_val_symbols 改為分層採樣（按市值分層，確保大/中/小型股各佔適當比例）

**Won't（本輪）：**
- 不用全 universe（~1090 支）做 IC 驗證（太慢）

### `train_log.csv` 欄位擴充

```
epoch, step, train_loss, val_loss, val_ic, ic_ir_h5
```

其中：
- `val_ic`：全 horizon 均值（h1~h5 average，保留參考）
- `ic_ir_h5`：h5 IC-IR（作為 early stop 主指標）

### 執行時間估算（RTX Pro 6000）

| 設定 | symbols | dates | pairs | 每次推論批次 | 估計時間 |
|------|---------|-------|-------|-----------|--------|
| 現況（Round 1）| 150 | 8 | 1200 | 64 | ~1 min |
| 建議（Round 2）| 300 | 20 | 6000 | 64 | ~4 min |

20 epoch × 4 min = 80 min 總驗證開銷。可接受（vs 訓練時間 ~10 小時）。

---

## 技術方案

### `ic_validation.py` 無需改動

`pick_val_universe(all_syms, cfg.ic_val_symbols, cfg.seed)` 和 `pick_val_dates(start, end, cfg.ic_val_dates)` 都已支援任意 n，只需改 config。

### 新增 `validate_predictor_ic_ir`（見 prd-ic-ir-early-stopping.md）

此 PRD 的實作是 ic-ir-early-stopping PRD 的前提（需要 ic_ir_h5 metric）。

### 可選：分層採樣（改進 ic_validation.py）

```python
def pick_val_universe_stratified(symbols, n, seed, db_path):
    """按市值分三層（大/中/小），各取 n//3 支。"""
    # 需要從 db 讀最新成交量作為 market cap proxy
    ...
```

---

## 驗收條件

- [ ] config 更新後，每 epoch IC 驗證用 6000 pairs
- [ ] train_log.csv 包含 `ic_ir_h5` 欄位
- [ ] Round 2 訓練過程中，`ic_ir_h5` 至少在 3 個 epoch 為正值（代表有效信號）
- [ ] early stop 在 patience=3 下選出的 best epoch ≠ 最後一個 epoch

---

## 風險

| 風險 | 機率 | 緩解 |
|------|------|------|
| 每次驗證 4 min 過長 | 低（RTX Pro 6000 足夠快）| 若過慢可縮回 ic_val_dates=15 |
| 300 symbols 中部分股票資料不足 | 中 | `build_ctx_fn` 已有 `if len(df) < lookback_window: return None` 保護 |
| 跨 20 個 val dates 但資料只有 6 個月 | 低 | 2024-01-01 ~ 2024-06-30 有約 120 交易日，均勻取 20 個足夠 |

---

## DECISION NEEDED

1. **分層採樣**：是否值得在 Round 2 實作市值分層？若 Round 2 時間緊，建議 Round 3 再做。
2. **ic_val_dates 的時間分佈**：均勻 linspace 還是後半段加密（近期資料更重要）？建議均勻（避免偏差），可在 Round 3 實驗加權。

## Open Questions

- 若 ic_ir_h5 在 Round 2 仍然全程為負（即使 300×20 樣本），這代表什麼？
  - 若 Round 0 predictor 起點已有 IC@h5=0.0319，但驗證 IC-IR 為負，代表日際 IC 波動太大（σ > mean）
  - 此時應考慮：增加 ic_val_dates 到 30+，或用 rolling 4-week IC 均值作指標
