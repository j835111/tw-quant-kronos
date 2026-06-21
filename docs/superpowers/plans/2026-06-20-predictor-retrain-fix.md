# Predictor 重練修復 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重練 finetune_tw predictor，改用 price-space（IC）早停選模、降低 LR 對抗災難性遺忘，使其在 test set 的方向命中率與 IC 上超越未微調 Kronos-base baseline。

**Architecture:** 凍結既有（已驗證良好）微調 tokenizer，只重練 predictor。新增一個 price-space 驗證模組（`ic_validation.py`）提供純函式（rank IC、驗證宇宙/日期取樣、EarlyStopper）與一個模型驅動的 val-IC 驗證器；`train_predictor.py` 改以 val IC 挑 best_model 並早停。驗證沿用既有 `eval_forecast.py` 與 baseline 對照。

**Tech Stack:** PyTorch、pandas/numpy、pytest、既有 Kronos model/、SQLite（finetune_tw/db.py）、molab RTX Pro 6000（訓練）、rclone（gdrive 備份）。

## Global Constraints

- 不重練 tokenizer：沿用 `finetune_tw/outputs/tw_daily/tokenizer/best_model`（凍結）。
- 不覆寫既有 finetuned best_model 與 baseline eval JSON（保留作對照）。
- 純函式測試遵循既有風格：`tests/finetune_tw/`，pandas/numpy，無 GPU、無網路。
- 早停指標 = val IC（h1–5 平均 Spearman 橫斷面相關），mode=max。
- val 期間 = `train_end_date`→`val_end_date`（2023-12-31→2024-06-30），**不可與 test（2024-07-01 起）重疊**。
- 成功判準：finetuned 在 h1–5 的方向命中率與 IC 至少持平、理想超越 baseline（baseline 參考：方向 ~53%、IC@h1 0.050）。
- AMP：RTX 用 bf16（`config_tw_daily_rtx6000.yaml` 慣例）。
- commit 訊息結尾保留專案既有 Co-Authored-By / Claude-Session 行（見既有 commit）。

---

### Task 1: Config 新增重練參數 + 重練 config 檔

**Files:**
- Modify: `finetune_tw/config.py:27`（在 `seed` 附近 Training 區塊加欄位）
- Create: `finetune_tw/configs/config_tw_daily_retrain.yaml`
- Test: `tests/finetune_tw/test_config_retrain.py`

**Interfaces:**
- Produces: `Config` 新欄位 `early_stop_patience: int`、`ic_val_symbols: int`、`ic_val_dates: int`、`val_ic_horizons: int`；皆有預設值，`Config.from_yaml` 可載入。

- [ ] **Step 1: 寫失敗測試**

```python
# tests/finetune_tw/test_config_retrain.py
from finetune_tw.config import Config


def test_config_defaults_have_ic_fields():
    cfg = Config()
    assert cfg.early_stop_patience == 2
    assert cfg.ic_val_symbols == 150
    assert cfg.ic_val_dates == 8
    assert cfg.val_ic_horizons == 5


def test_retrain_yaml_loads(tmp_path):
    cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily_retrain.yaml")
    assert cfg.predictor_lr == 1e-5
    assert cfg.basemodel_epochs == 6
    assert cfg.early_stop_patience == 2
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_config_retrain.py -v`
Expected: FAIL（`AttributeError: 'Config' has no attribute 'early_stop_patience'` 及找不到 yaml）

- [ ] **Step 3: 加 Config 欄位**

在 `finetune_tw/config.py` 的 Training 區塊（`seed: int = 42` 之後）加入：

```python
    # Early stopping / price-space validation
    early_stop_patience: int = 2
    ic_val_symbols: int = 150
    ic_val_dates: int = 8
    val_ic_horizons: int = 5
```

- [ ] **Step 4: 建重練 config**

```yaml
# finetune_tw/configs/config_tw_daily_retrain.yaml
# Predictor retrain — frozen fine-tuned tokenizer, low LR, price-space early stop
db_path: "finetune_tw/data/tw_stocks.db"
lookback_window: 90
predict_window: 10
max_context: 512
clip: 5.0
train_end_date: "2023-12-31"
val_end_date: "2024-06-30"

tokenizer_epochs: 30
basemodel_epochs: 6
batch_size: 256
save_steps: 500
log_interval: 100
tokenizer_lr: 0.0002
predictor_lr: 0.00001
adam_beta1: 0.9
adam_beta2: 0.95
adam_weight_decay: 0.1
num_workers: 4
amp_dtype: "bf16"
seed: 42

early_stop_patience: 2
ic_val_symbols: 150
ic_val_dates: 8
val_ic_horizons: 5

pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
pretrained_predictor: "NeoQuasar/Kronos-base"
exp_name: "tw_daily"
output_dir: "finetune_tw/outputs"

top_k: 20
hold_days: 5
pred_len: 10
test_start_date: "2024-07-01"
benchmark_symbol: "^TWII"
```

- [ ] **Step 5: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_config_retrain.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/config.py finetune_tw/configs/config_tw_daily_retrain.yaml tests/finetune_tw/test_config_retrain.py
git commit -m "feat(finetune_tw): add retrain config (lr 1e-5, epochs 6, IC early-stop params)"
```

---

### Task 2: ic_validation.py — 純函式（rank IC、取樣、平均 IC）

**Files:**
- Create: `finetune_tw/ic_validation.py`
- Modify: `finetune_tw/eval_forecast.py:38-50`（改用共用 `rank_ic`，DRY）
- Test: `tests/finetune_tw/test_ic_validation.py`

**Interfaces:**
- Produces:
  - `rank_ic(pred, actual) -> float`（Spearman = ranks 的 Pearson；<3 有效點或零變異回 nan）
  - `mean_cross_sectional_ic(per_group: dict) -> float`（`per_group` = `{key: (pred_seq, actual_seq)}`，逐 group 算 rank_ic 後取有效值平均）
  - `pick_val_universe(symbols: list[str], n: int, seed: int = 42) -> list[str]`（deterministic）
  - `pick_val_dates(start: str, end: str, n: int) -> list[pd.Timestamp]`（等距 business days）

- [ ] **Step 1: 寫失敗測試**

```python
# tests/finetune_tw/test_ic_validation.py
import numpy as np
import pandas as pd
import pytest
from finetune_tw.ic_validation import (
    rank_ic, mean_cross_sectional_ic, pick_val_universe, pick_val_dates,
)


def test_rank_ic_perfect_positive():
    assert rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_rank_ic_perfect_negative():
    assert rank_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_rank_ic_too_few_points_is_nan():
    assert np.isnan(rank_ic([1, 2], [3, 4]))


def test_rank_ic_zero_variance_is_nan():
    assert np.isnan(rank_ic([1, 1, 1, 1], [1, 2, 3, 4]))


def test_mean_cross_sectional_ic_averages_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),   # ic = 1.0
        "d2": ([1, 2, 3, 4], [40, 30, 20, 10]),   # ic = -1.0
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(0.0)


def test_mean_cross_sectional_ic_skips_nan_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),   # 1.0
        "d2": ([1, 1], [2, 3]),                    # nan (skipped)
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(1.0)


def test_pick_val_universe_deterministic_and_sized():
    syms = [f"{i:04d}" for i in range(1000)]
    a = pick_val_universe(syms, 150, seed=42)
    b = pick_val_universe(syms, 150, seed=42)
    assert a == b
    assert len(a) == 150
    assert len(set(a)) == 150


def test_pick_val_universe_returns_all_if_small():
    syms = ["A", "B", "C"]
    assert pick_val_universe(syms, 150) == ["A", "B", "C"]


def test_pick_val_dates_count_and_bounds():
    dates = pick_val_dates("2024-01-01", "2024-06-30", 8)
    assert len(dates) <= 8
    assert dates == sorted(dates)
    assert dates[0] >= pd.Timestamp("2024-01-01")
    assert dates[-1] <= pd.Timestamp("2024-06-30")
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_ic_validation.py -v`
Expected: FAIL（`ModuleNotFoundError: finetune_tw.ic_validation`）

- [ ] **Step 3: 實作 ic_validation.py 純函式**

```python
# finetune_tw/ic_validation.py
"""Price-space validation helpers (pure functions + model-driven IC validator)
for selecting predictor checkpoints by forecast skill instead of token CE.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rank_ic(pred, actual) -> float:
    """Spearman rank correlation = Pearson on ranks. No scipy dependency."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    m = np.isfinite(pred) & np.isfinite(actual)
    if m.sum() < 3:
        return float("nan")
    pr = pd.Series(pred[m]).rank().values
    ar = pd.Series(actual[m]).rank().values
    if pr.std() < 1e-9 or ar.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(pr, ar)[0, 1])


def mean_cross_sectional_ic(per_group: dict) -> float:
    """per_group: {key: (pred_seq, actual_seq)} -> mean of finite per-group rank_ic."""
    ics = [rank_ic(p, a) for (p, a) in per_group.values()]
    ics = [x for x in ics if np.isfinite(x)]
    return float(np.mean(ics)) if ics else float("nan")


def pick_val_universe(symbols, n: int, seed: int = 42) -> list:
    """Deterministic subset of symbols for cheap per-epoch validation."""
    syms = sorted(symbols)
    if len(syms) <= n:
        return syms
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(syms), size=n, replace=False)
    return [syms[i] for i in sorted(idx)]


def pick_val_dates(start: str, end: str, n: int) -> list:
    """Evenly spaced business days across [start, end]."""
    bdays = pd.bdate_range(start, end)
    if len(bdays) <= n:
        return list(bdays)
    pos = np.linspace(0, len(bdays) - 1, n).round().astype(int)
    return [bdays[i] for i in sorted(set(pos.tolist()))]
```

- [ ] **Step 4: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_ic_validation.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: DRY — eval_forecast 改用共用 rank_ic**

在 `finetune_tw/eval_forecast.py` 把私有 `_rank_ic` 的定義（約 `:38-50`）刪除，並在檔頭 import：

```python
from finetune_tw.ic_validation import rank_ic as _rank_ic
```

- [ ] **Step 6: 跑既有 eval 相關測試確認沒壞**

Run: `pytest tests/finetune_tw/ -k "ic or eval or backtest" -v`
Expected: PASS（無 import 錯誤）

- [ ] **Step 7: Commit**

```bash
git add finetune_tw/ic_validation.py finetune_tw/eval_forecast.py tests/finetune_tw/test_ic_validation.py
git commit -m "feat(finetune_tw): add ic_validation pure helpers; reuse rank_ic in eval_forecast"
```

---

### Task 3: EarlyStopper

**Files:**
- Modify: `finetune_tw/ic_validation.py`（append class）
- Test: `tests/finetune_tw/test_ic_validation.py`（append）

**Interfaces:**
- Produces: `EarlyStopper(patience: int = 2, mode: str = "max")`，方法 `update(value) -> (is_best: bool, should_stop: bool)`；屬性 `best`。nan/None 視為未改善並累計 patience。

- [ ] **Step 1: 寫失敗測試（append）**

```python
# append to tests/finetune_tw/test_ic_validation.py
from finetune_tw.ic_validation import EarlyStopper


def test_early_stopper_first_value_is_best():
    es = EarlyStopper(patience=2, mode="max")
    is_best, stop = es.update(0.1)
    assert is_best and not stop
    assert es.best == 0.1


def test_early_stopper_improvement_resets_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.1)
    is_best, stop = es.update(0.2)
    assert is_best and not stop


def test_early_stopper_stops_after_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.3)                 # best
    assert es.update(0.2) == (False, False)   # bad 1
    assert es.update(0.1) == (False, False)   # bad 2
    assert es.update(0.1) == (False, True)    # bad 3 > patience -> stop


def test_early_stopper_nan_counts_as_no_improvement():
    es = EarlyStopper(patience=1, mode="max")
    es.update(0.3)
    assert es.update(float("nan")) == (False, False)  # bad 1
    assert es.update(float("nan")) == (False, True)   # bad 2 > patience
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_ic_validation.py -k early_stopper -v`
Expected: FAIL（`ImportError: cannot import name 'EarlyStopper'`）

- [ ] **Step 3: 實作 EarlyStopper（append 到 ic_validation.py）**

```python
import math


class EarlyStopper:
    """Track best metric; signal when to stop after `patience` non-improving epochs."""

    def __init__(self, patience: int = 2, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self._bad = 0

    def update(self, value):
        """Return (is_best, should_stop)."""
        improved = (
            value is not None
            and isinstance(value, (int, float))
            and not math.isnan(value)
            and (self.best is None
                 or (value > self.best if self.mode == "max" else value < self.best))
        )
        if improved:
            self.best = value
            self._bad = 0
            return True, False
        self._bad += 1
        return False, self._bad > self.patience
```

- [ ] **Step 4: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_ic_validation.py -v`
Expected: PASS（全部 passed）

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/ic_validation.py tests/finetune_tw/test_ic_validation.py
git commit -m "feat(finetune_tw): add EarlyStopper for price-space early stopping"
```

---

### Task 4: 模型驅動 val-IC 驗證器（可注入 predict_fn，可測）

**Files:**
- Modify: `finetune_tw/ic_validation.py`（append `validate_predictor_ic`）
- Test: `tests/finetune_tw/test_validate_predictor_ic.py`

**Interfaces:**
- Consumes: `rank_ic` / `mean_cross_sectional_ic`（Task 2）
- Produces:
  `validate_predictor_ic(predict_batch_fn, actual_lookup, val_universe, val_dates, cfg, build_ctx_fn) -> float`
  - `build_ctx_fn(sym, rebal_date) -> (ctx_df, x_ts, y_ts, ctx_last_date, ctx_close)`；回 None 表示資料不足跳過
  - `predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len) -> list[pd.DataFrame]`（與 `KronosPredictor.predict_batch` 的子集簽名相容）
  - `actual_lookup(sym, ctx_last_date, n) -> np.ndarray`（回 ctx 之後 n 個交易日的實際 close；不足回較短陣列）
  - 回傳：h1..`val_ic_horizons` 的橫斷面 rank IC 平均

- [ ] **Step 1: 寫失敗測試（用 fake predictor，不碰 GPU）**

```python
# tests/finetune_tw/test_validate_predictor_ic.py
import numpy as np
import pandas as pd
from finetune_tw.ic_validation import validate_predictor_ic


class _Cfg:
    pred_len = 5
    val_ic_horizons = 5


def test_validate_predictor_ic_perfect_skill_returns_high_ic():
    # Fake predictor that returns the true future path -> IC should be ~1.0
    actual_paths = {
        "A": [101, 102, 103, 104, 105],
        "B": [100, 99, 98, 97, 96],
        "C": [100, 100.5, 101, 101.5, 102],
        "D": [100, 99.5, 99, 98.5, 98],
    }
    order = ["A", "B", "C", "D"]
    ctx_close = 100.0

    def build_ctx(sym, date):
        df = pd.DataFrame({"open": [100.0]*3, "high": [100.0]*3, "low": [100.0]*3,
                           "close": [100.0, 100.0, ctx_close], "volume": [1.0]*3, "amount": [1.0]*3})
        return (df, pd.Series(pd.bdate_range("2024-01-01", periods=3)),
                pd.Series(pd.bdate_range(date, periods=5)), pd.Timestamp(date), ctx_close)

    def predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, _order=order, _ap=actual_paths):
        # df_list arrives in the order validate_predictor_ic enumerates val_universe
        res = []
        for i in range(len(df_list)):
            sym = _order[i % len(_order)]
            res.append(pd.DataFrame({"close": _ap[sym][:pred_len]}))
        return res

    def actual_lookup(sym, ctx_last_date, n):
        return np.array(actual_paths[sym][:n], dtype=float)

    ic = validate_predictor_ic(predict_batch, actual_lookup, order,
                               [pd.Timestamp("2024-03-01")], _Cfg(), build_ctx)
    assert ic > 0.9
```

> 註：`predict_batch` 依 `validate_predictor_ic` 列舉 `val_universe` 的順序回傳，與真實 `KronosPredictor.predict_batch`（依輸入 df_list 順序回傳）一致。

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_validate_predictor_ic.py -v`
Expected: FAIL（`ImportError: cannot import name 'validate_predictor_ic'`）

- [ ] **Step 3: 實作 validate_predictor_ic（append 到 ic_validation.py）**

```python
def validate_predictor_ic(predict_batch_fn, actual_lookup, val_universe, val_dates,
                          cfg, build_ctx_fn, batch_size: int = 64) -> float:
    """Mean cross-sectional rank IC over h1..cfg.val_ic_horizons on a val subset.

    Pure orchestration over injected callables so it is testable without a model.
    """
    L = cfg.pred_len
    H = min(cfg.val_ic_horizons, L)
    per_group: dict = {}  # (date, h) -> (pred_returns, actual_returns)
    for date in val_dates:
        syms, dfs, xts, yts, last_dates, ctx_closes = [], [], [], [], [], []
        for sym in val_universe:
            built = build_ctx_fn(sym, date)
            if built is None:
                continue
            ctx_df, x_ts, y_ts, last_date, ctx_close = built
            syms.append(sym); dfs.append(ctx_df); xts.append(x_ts); yts.append(y_ts)
            last_dates.append(last_date); ctx_closes.append(ctx_close)
        # batched predict
        rows = []  # (sym, pred_close[np], ctx_close, last_date)
        for b in range(0, len(syms), batch_size):
            preds = predict_batch_fn(dfs[b:b+batch_size], xts[b:b+batch_size],
                                     yts[b:b+batch_size], L)
            for k, pred in enumerate(preds):
                if pred is None or len(pred) < L:
                    continue
                j = b + k
                rows.append((syms[j], pred["close"].values.astype(float),
                             ctx_closes[j], last_dates[j]))
        for h in range(H):
            preds_h, acts_h = [], []
            for sym, pclose, cclose, last_date in rows:
                act = actual_lookup(sym, last_date, L)
                if len(act) <= h:
                    continue
                preds_h.append(pclose[h] / cclose - 1.0)
                acts_h.append(act[h] / cclose - 1.0)
            if len(preds_h) >= 3:
                per_group[(date, h)] = (preds_h, acts_h)
    return mean_cross_sectional_ic(per_group)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_validate_predictor_ic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/ic_validation.py tests/finetune_tw/test_validate_predictor_ic.py
git commit -m "feat(finetune_tw): add injectable model-driven val-IC validator"
```

---

### Task 5: train_predictor 改用 IC 早停選模 + log 欄位 + gdrive log 同步

**Files:**
- Modify: `finetune_tw/train_predictor.py:21-30`（`_gdrive_sync` 旁新增 log 同步）
- Modify: `finetune_tw/train_predictor.py:88-92`（建 val_universe / val_dates）
- Modify: `finetune_tw/train_predictor.py:110-162`（log header、選模、早停）
- Test: 既有 `tests/finetune_tw/test_train_predictor.py`（append 一個非 GPU 的 helper 測試）

**Interfaces:**
- Consumes: `validate_predictor_ic`、`pick_val_universe`、`pick_val_dates`、`EarlyStopper`（Tasks 2–4）
- Produces: `_build_ctx_for_date(cfg, sym, rebal_date)`、`_actual_close_lookup(cfg, sym, ...)`、`_make_predict_batch_fn(predictor, ...)`（供 `validate_predictor_ic` 注入）；`train_log.csv` 多一欄 `val_ic`；best_model 以 val IC 最大挑選並早停。

- [ ] **Step 1: 寫失敗測試（純 helper，無 GPU）**

`_build_ctx_for_date` 用合成 DB 驗證 context 形狀與 ctx_close。

```python
# append to tests/finetune_tw/test_train_predictor.py
import pandas as pd
from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.train_predictor import _build_ctx_for_date


def test_build_ctx_for_date_shapes(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    dates = pd.bdate_range("2023-06-01", periods=200)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100.0 + i*0.1 for i in range(200)],
        "volume": 1000.0, "amount": 1e5,
    })
    upsert_prices(db, "9999", df)
    cfg = Config(db_path=db, lookback_window=90, pred_len=10)
    built = _build_ctx_for_date(cfg, "9999", pd.Timestamp("2024-01-15"))
    assert built is not None
    ctx_df, x_ts, y_ts, last_date, ctx_close = built
    assert len(ctx_df) == 90
    assert list(ctx_df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert len(y_ts) == cfg.pred_len
    assert ctx_close == ctx_df["close"].iloc[-1]


def test_build_ctx_for_date_insufficient_returns_none(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    dates = pd.bdate_range("2023-12-01", periods=10)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 1000.0, "amount": 1e5,
    })
    upsert_prices(db, "9999", df)
    cfg = Config(db_path=db, lookback_window=90, pred_len=10)
    assert _build_ctx_for_date(cfg, "9999", pd.Timestamp("2024-01-15")) is None
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_train_predictor.py -k build_ctx -v`
Expected: FAIL（`ImportError: cannot import name '_build_ctx_for_date'`）

- [ ] **Step 3: 加 context/lookup/predict 注入 helper**

在 `finetune_tw/train_predictor.py` 檔頭 import：

```python
import pandas as pd
from finetune_tw.db import query_symbol, list_symbols
from finetune_tw.ic_validation import (
    validate_predictor_ic, pick_val_universe, pick_val_dates, EarlyStopper,
)
```

新增 helper（放在 `run_training` 之前）：

```python
def _build_ctx_for_date(cfg, sym, rebal_date):
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
    if len(df) < cfg.lookback_window:
        return None
    ctx = df.iloc[-cfg.lookback_window:]
    ctx_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    if ctx_df.isnull().any().any():
        return None
    x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
    y_ts = pd.Series(pd.date_range(rebal_date, periods=cfg.pred_len, freq="B"))
    return ctx_df, x_ts, y_ts, x_ts.iloc[-1], float(ctx_df["close"].iloc[-1])


def _actual_close_lookup(cfg, cache, sym, ctx_last_date, n):
    ser = cache.get(sym)
    if ser is None:
        return np.array([], dtype=float)
    pos = ser.index.searchsorted(ctx_last_date, side="right")
    return ser.iloc[pos:pos + n].values.astype(float)


def _make_predict_batch_fn(predictor):
    def fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        with torch.no_grad():
            return predictor.predict_batch(
                df_list=df_list, x_timestamp_list=x_timestamp_list,
                y_timestamp_list=y_timestamp_list, pred_len=pred_len,
                T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False)
    return fn
```

加 `import numpy as np` 於檔頭（若未有）。

- [ ] **Step 4: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_train_predictor.py -k build_ctx -v`
Expected: PASS

- [ ] **Step 5: 改訓練迴圈為 IC 早停選模**

A. log header（`finetune_tw/train_predictor.py:110-112`）改為：

```python
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss,val_ic\n")
```

B. 在 `for epoch ...` 迴圈前（約 `:113`）建立驗證器需要的 predictor 與 val 取樣：

```python
    from model import KronosPredictor
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    all_syms = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    val_universe = pick_val_universe(all_syms, cfg.ic_val_symbols, cfg.seed)
    val_dates = pick_val_dates(cfg.train_end_date, cfg.val_end_date, cfg.ic_val_dates)
    # preload actual close for val universe over the val window (+buffer)
    _buf = (pd.Timestamp(cfg.train_end_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    actual_cache = {}
    for s in val_universe:
        _df = query_symbol(cfg.db_path, s, start=_buf, end=cfg.val_end_date)
        if len(_df):
            actual_cache[s] = pd.Series(_df["close"].values, index=pd.DatetimeIndex(_df["date"]))
    stopper = EarlyStopper(patience=cfg.early_stop_patience, mode="max")
```

C. 把舊的 `best_val_loss` 選模區塊（`:154-162`）替換為：

```python
        val_loss = _validate_predictor(model, tokenizer, val_loader, device, amp_enabled, amp_dtype)
        model.eval()
        val_ic = validate_predictor_ic(
            _make_predict_batch_fn(predictor),
            lambda sym, last, n: _actual_close_lookup(cfg, actual_cache, sym, last, n),
            val_universe, val_dates, cfg, lambda s, d: _build_ctx_for_date(cfg, s, d),
        )
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f},{val_ic:.4f}\n")

        is_best, should_stop = stopper.update(val_ic)
        if is_best:
            model.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_ic={val_ic:.4f} (val_loss={val_loss:.4f}), saved.")
            _gdrive_sync(save_dir / "best_model", remote=remote_root)
            _gdrive_sync_logs(log_path, remote_root)
        if should_stop:
            print(f"  -> early stop at epoch {epoch+1} (best val_ic={stopper.best:.4f})")
            break
```

D. 刪除原本 `best_val_loss = float("inf")`（`:109`）。

- [ ] **Step 6: 加 gdrive log 同步 helper**

在 `_gdrive_sync`（`:21-30`）之後新增：

```python
def _gdrive_sync_logs(log_path: Path, remote: str) -> None:
    """Upload train_log.csv to Drive (fixes the lost-log gap)."""
    if shutil.which("rclone") is None or not log_path.exists():
        return
    subprocess.Popen(
        ["rclone", "copy", str(log_path), remote],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
```

- [ ] **Step 7: 跑全 finetune_tw 測試確認沒壞**

Run: `pytest tests/finetune_tw/ -v`
Expected: PASS（含新測試；無 import 錯誤）

- [ ] **Step 8: Commit**

```bash
git add finetune_tw/train_predictor.py tests/finetune_tw/test_train_predictor.py
git commit -m "feat(finetune_tw): select predictor best_model by val IC + early stop + sync train_log"
```

---

### Task 6: 本機 smoke — 訓練迴圈能跑、會寫 val_ic、能選模（CPU 小資料）

**Files:**
- Test: `tests/finetune_tw/test_train_predictor_smoke.py`

**Interfaces:**
- Consumes: `run_training`（Task 5 後）
- Produces: 無（驗證任務）

> 用既有合成 DB 模式 + `max_steps` 早退，確認端到端不爆、且 `train_log.csv` 有 `val_ic` 欄。若機器無法載入 Kronos 權重（需網路），以 `pytest.mark.skipif` 跳過——此 task 主要在 molab 執行。

- [ ] **Step 1: 寫 smoke 測試**

```python
# tests/finetune_tw/test_train_predictor_smoke.py
import os
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GPU_SMOKE") != "1",
    reason="needs model download + heavy compute; run on molab with RUN_GPU_SMOKE=1",
)


def test_training_writes_val_ic(tmp_path):
    from finetune_tw.config import Config
    from finetune_tw.db import init_db, upsert_prices
    from finetune_tw import train_predictor

    db = str(tmp_path / "t.db")
    init_db(db)
    dates = pd.bdate_range("2015-01-01", "2024-06-30")
    for sym in [f"{i:04d}" for i in range(10)] + ["^TWII"]:
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": [100.0 + (j % 50) * 0.2 for j in range(len(dates))],
            "volume": 1000.0, "amount": 1e5,
        })
        upsert_prices(db, sym, df)
    cfg = Config(db_path=db, output_dir=str(tmp_path / "out"),
                 batch_size=4, basemodel_epochs=1, ic_val_symbols=5, ic_val_dates=2)
    # requires a tokenizer best_model; copy/point to one in molab env beforehand
    train_predictor.run_training(cfg, max_steps=20)
    log = (tmp_path / "out" / cfg.exp_name / "predictor" / "train_log.csv").read_text()
    assert "val_ic" in log.splitlines()[0]
```

- [ ] **Step 2: 在 molab 執行 smoke**

Run（molab session 內，已有 tokenizer best_model）：
`cd /marimo/Kronos && RUN_GPU_SMOKE=1 pytest tests/finetune_tw/test_train_predictor_smoke.py -v`
Expected: PASS（log 首列含 `val_ic`）

- [ ] **Step 3: Commit**

```bash
git add tests/finetune_tw/test_train_predictor_smoke.py
git commit -m "test(finetune_tw): gpu smoke for val-IC training loop (opt-in)"
```

---

### Task 7: molab 全量重練 + resume 驗證

**Files:** 無（執行任務，使用 `config_tw_daily_retrain.yaml`）

**Interfaces:**
- Consumes: Tasks 1–5 的程式 + 既有凍結 tokenizer best_model
- Produces: 新 `predictor/best_model`（以 val IC 選出）、`train_log.csv`（含 val_ic 曲線）

- [ ] **Step 1: 同步程式到 molab**

在 molab session 內 `git pull`（或注入更新檔）至 `/marimo/Kronos`，確認 `config_tw_daily_retrain.yaml`、`ic_validation.py`、改過的 `train_predictor.py` 都在。

- [ ] **Step 2: 確認凍結 tokenizer 就位**

Run: `ls /marimo/Kronos/finetune_tw/outputs/tw_daily/tokenizer/best_model/model.safetensors`
Expected: 檔案存在（沿用既有；若缺，從 gdrive `rclone copy` 回來）

- [ ] **Step 3: 啟動重練（背景 + log）**

Run:
```bash
cd /marimo/Kronos && nohup python -u -m finetune_tw.train_predictor \
  --config finetune_tw/configs/config_tw_daily_retrain.yaml \
  > /marimo/kronos_data/retrain.log 2>&1 &
```
Expected: log 出現 `[epoch 1 ...]` 與每 epoch 結尾 `new best val_ic=...`

- [ ] **Step 4: 驗證 resume（跨 sandbox 必要）**

殺掉 process 後重跑同指令，確認 log 出現 `Resumed from ckpt-... (step N)` 且 step 從上次續接、scheduler LR 未跳回起點。
Expected: `Resumed from ...`，且後續 `[epoch ...]` step 連續遞增。

- [ ] **Step 5: 等收斂或早停**

監看 `retrain.log` 直到出現 `early stop` 或跑滿 `basemodel_epochs=6`。觀察 `train_log.csv` 的 `val_ic` 是否隨 epoch 上升。

- [ ] **Step 6: 取回權重與 log**

Run（本機）:
```bash
LOCAL=finetune_tw/outputs/tw_daily/predictor
rclone copy "gdrive:Kronos/outputs/tw_daily/predictor/best_model" "$LOCAL/best_model_retrain"
rclone copy "gdrive:Kronos/outputs/tw_daily/predictor/train_log.csv" "$LOCAL/"
```
Expected: 取得 retrain best_model 與含 val_ic 的 train_log.csv（注意：放到 `best_model_retrain` 不覆寫舊 finetuned best_model）。

---

### Task 8: 驗證 — finetuned(retrain) vs baseline 對照，判定成敗

**Files:** 無（執行 + 記錄任務，使用 `eval_forecast.py`）

**Interfaces:**
- Consumes: retrain best_model（Task 7）、既有 baseline eval JSON
- Produces: `eval_metrics_finetuned.json` 對照結論；memory 與回測歷史更新

- [ ] **Step 1: 跑 retrain 模型的 forecast eval**

Run（molab 或本機 GPU，best_model 指向 retrain 權重）:
```bash
python -m finetune_tw.eval_forecast --config finetune_tw/configs/config_tw_daily_retrain.yaml
```
Expected: 產生 `outputs/tw_daily/eval/eval_metrics_finetuned.json` 與每 horizon 的 dir%/IC/MAPE。

- [ ] **Step 2: 對照 baseline 判定**

把 retrain 的 h1–5 方向命中率與 IC 跟既有 `eval_metrics_baseline.json`（方向 ~53%、IC@h1 0.050）並排。
- 通過：retrain 在 h1–5 至少持平、理想超越 baseline → 微調修復成功。
- 失敗：仍輸 baseline → 記錄結論，回退路線 1（直接用 pretrained + 改策略）。

- [ ] **Step 3: 更新 memory 與回測歷史**

更新 `memory/forecast_eval.md`（加 retrain 對照列）、`memory/backtest_results.md`，並在 CLAUDE.md 「回測歷史」表加一輪（若有跑回測）。

- [ ] **Step 4: Commit**

```bash
git add finetune_tw/outputs/tw_daily/eval/eval_metrics_finetuned.json CLAUDE.md
git commit -m "chore(finetune_tw): record retrain forecast eval vs baseline"
```

---

## Self-Review

**Spec coverage：**
- 元件1（凍結 tokenizer，只重練 predictor）→ Task 1 config（不含 tokenizer stage）+ Task 7 Step 2。✓
- 元件2（降 LR / 減 epochs）→ Task 1 yaml（lr 1e-5、epochs 6）。✓
- 元件3（price-space 早停/選模）→ Tasks 2–5。✓
- 元件4（成功判準）→ Task 8。✓
- 元件5（log 同步 + resume 驗證）→ Task 5 Step 6（`_gdrive_sync_logs`）+ Task 7 Step 4。✓
- （可選）凍結前 N 層：spec 標為可選實驗開關，本計畫 YAGNI 不納入；如需，於 Task 5 加 `requires_grad_` 區塊。已知取捨，非遺漏。

**Placeholder scan：** 無 TBD/TODO；所有 code step 含完整程式碼。Tasks 6–8 為執行/驗證任務，提供確切指令與期望輸出。

**Type consistency：** `rank_ic`/`mean_cross_sectional_ic`/`EarlyStopper.update`→`(is_best, should_stop)`/`validate_predictor_ic` 簽名在 Tasks 2–5 一致；`_build_ctx_for_date` 回傳 5-tuple `(ctx_df, x_ts, y_ts, last_date, ctx_close)` 在定義（Task 5）與消費（Task 4 注入介面、Task 5C）一致。✓
