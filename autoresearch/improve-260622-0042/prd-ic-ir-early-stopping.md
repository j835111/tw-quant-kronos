# PRD: IC-IR@h5 Early Stopping（取代 val_ic 均值）

> Auto-generated from research findings. DECISION NEEDED items require your judgment.

---

## 問題陳述

`EarlyStopper` 目前用 `val_ic`（所有 horizon 的截面 IC 均值）作為停止指標。在 150 symbols × 8 dates = 1200 樣本的設定下，IC 的標準差 σ≈0.08 遠大於信號本身（0.00~0.05），early stopper 無法分辨「真正進步」和「噪音波動」，Round 1 因此全程 val_ic 為負、選出「最不壞」的 checkpoint（Epoch 6）而非真正好的模型。

核心洞見（來自實驗資料）：
- 回測用 h5 決策，不是 h1。Round 0 在 h5 IC 反超 Pretrained（0.0319 vs 0.0268），這是 Sharpe 差距的直接原因。
- IC-IR（IC / σ(IC)）量化信號**穩定性**，Round 0 IC-IR@h1=0.625 > Pretrained 0.601，且正 IC 日比例更高（72.8% vs 70.9%）。IC-IR 比 IC 均值更能預測回測效果。

---

## 使用者故事

1. 作為訓練腳本，我需要每個 epoch 末計算 `ic_ir_h5`，以便 early stopper 選出 h5 信號*最一致*的 checkpoint，而不是 h5 IC 均值最高的 checkpoint。
2. 作為研究員，我需要 `train_log.csv` 記錄 `ic_ir_h5` 欄位，以便事後分析哪個 epoch 信號最穩定。

---

## 需求

### 功能需求（MoSCoW）

**Must：**
- `ic_validation.py` 新增 `validate_predictor_ic_ir(target_horizon=5)` 函式，回傳 float（IC-IR at h5）
- 計算方式：對所有 val_dates，分別計算 h5 截面 rank IC → 取 mean / std（不是全 horizon 均值）
- `train_predictor.py` 呼叫新函式，以 ic_ir_h5 作為 `stopper.update()` 的輸入
- `train_log.csv` 新增 `ic_ir_h5` 欄位

**Should：**
- 同時保留 `val_ic`（全 horizon 均值）作為參考記錄，不作為 early stop 指標

**Could：**
- 在 log 中也記錄 `ic_ir_h1`（h1 的 IC-IR）供事後對比分析

**Won't（本輪）：**
- 不嘗試 bi-level horizon optimization（N2 moonshot）

### 非功能需求
- 新函式在 RTX Pro 6000 每次呼叫 < 3 分鐘（300 symbols × 20 dates，batch_size=64）
- 向後相容：舊 config 若無 `ic_ir_h5` 欄位，仍可用 `val_ic` fallback

---

## 技術方案

### 修改 `ic_validation.py`

```python
def validate_predictor_ic_ir(
    predict_batch_fn,
    actual_lookup,
    val_universe,
    val_dates,
    cfg,
    build_ctx_fn,
    batch_size: int = 64,
    target_horizon: int = 5,
) -> float:
    """IC-IR at target_horizon = mean_IC / std_IC across val_dates."""
    per_date_ic = []

    for date in val_dates:
        # build context, predict, compute rank_ic at target_horizon
        # (複用現有 validate_predictor_ic 的 inner loop 邏輯)
        ...
        ic_this_date = rank_ic(pred_returns_h, actual_returns_h)
        if np.isfinite(ic_this_date):
            per_date_ic.append(ic_this_date)

    if len(per_date_ic) < 3:
        return float("nan")
    arr = np.array(per_date_ic)
    return float(arr.mean() / (arr.std() + 1e-8))
```

### 修改 `train_predictor.py`

```python
# 現有（移除）
val_ic = validate_predictor_ic(...)

# 新增
val_ic    = validate_predictor_ic(...)      # 保留作記錄
ic_ir_h5  = validate_predictor_ic_ir(..., target_horizon=5)

# log 欄位更新
log_path.write_text("epoch,step,train_loss,val_loss,val_ic,ic_ir_h5\n")
f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f},{val_ic:.4f},{ic_ir_h5:.4f}\n")

# early stop 改用 ic_ir_h5
is_best, should_stop = stopper.update(ic_ir_h5)
```

---

## 驗收條件

- [ ] `ic_validation.py` 有 `validate_predictor_ic_ir` 函式，有 unit test（mock predict fn，已知 IC 序列 → 驗證回傳值正確）
- [ ] `train_log.csv` 有 `ic_ir_h5` 欄位
- [ ] Round 2 訓練 log 顯示 `ic_ir_h5` 在某個 epoch 為正（代表找到有效 checkpoint）
- [ ] best checkpoint 不再是「最後一個」epoch，而是某個中間 epoch（代表 early stop 真正起效）

---

## 風險

| 風險 | 機率 | 緩解 |
|------|------|------|
| ic_ir_h5 全程為 NaN（樣本不足）| 低（增加 300×20）| 加 fallback to val_ic |
| IC-IR 在早期 epoch 噪音仍大 | 中 | patience=3（比現在的 2 多一輪容忍）|
| target_horizon=5 不夠樣本（資料截止）| 低 | val_end_date 到 2024-06-30，h5 有足夠實際股價 |

---

## 成功指標

- val_ic 信噪比（mean/σ）提升 ≥ 2×（300×20 vs 150×8）
- Round 2 best checkpoint early stop 在 epoch 5-15 之間（非最後）
- Round 2 回測 IC-IR@h5 > Round 0 的 IC-IR@h5（待測量）

---

## DECISION NEEDED

1. `target_horizon`：固定 h5（=5），還是與 `hold_days` config 聯動（`cfg.hold_days`）？建議後者，讓 config 驅動。
2. Fallback 行為：若 `ic_ir_h5` 全程 NaN，是否 fallback 到 `val_ic`？或直接中止訓練？建議 fallback + warning。

## Open Questions

- Round 2 訓練完後，是否需要重跑 `eval_forecast.py` 對比 IC-IR@h1/h5 的完整分佈？（建議是）
