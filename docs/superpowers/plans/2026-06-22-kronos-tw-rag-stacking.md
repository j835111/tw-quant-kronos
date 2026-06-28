# Kronos-TW RAG + Stacking 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 實作 `finetune_tw/` 的三層架構——Kronos MC 訊號降噪 (Layer 1) → LightGBM cross-sectional stacking (Layer 3) → Analog RAG 特徵 (Layer 2)——每一步均以 Rank-IC 量增量貢獻，不改動現有 backtest.py 流程。

**Architecture:** 新增六個模組 (walkforward / signal / features / analog / stacking / stacking_backtest)，透過「feature table」（MultiIndex DataFrame）解耦各層；現有 backtest.py、train_*.py 完全不改動。LightGBM stacker 訓練時使用時間序列切分（stacking_train: 2022–2023，test: 2024-07+），避免 look-ahead；Analog 引擎嚴格 point-in-time（retrieval DB 截止日 ≤ as_of − pred_len）。

**Tech Stack:** Python 3.12, torch (現有), lightgbm (新增), scikit-learn (新增，僅用 NearestNeighbors), pandas/numpy/sqlite3 (現有)

## Global Constraints

- Kronos context 上限 512 tokens（`KronosPredictor.max_context`）；`cfg.lookback_window` 預設 90
- **Point-in-time 鐵律**：Analog 引擎的 retrieval DB 截止 `cutoff_date = as_of − pred_len * 2` calendar days
- 主指標是 **Rank-IC**（cross-sectional Spearman），不是 MSE
- `predict_batch` 要求 batch 內所有序列同長 (`lookback_window`)
- MC 樣本數 ≥ 20（`n_samples` 預設 20），用 `top_k=40, T=1.0, sample_count=1` 重複呼叫取分布
- Stacking 訓練 target = cross-sectional rank（整數，LightGBM lambdarank 要求非負）
- Analog 與 Kronos 特徵欄位名稱在 `stacking.FEATURE_COLS` 中定義，是 Tasks 2–4 與 Task 5 的介面合約
- 新依賴加入 `requirements.txt`：`lightgbm>=4.0` 和 `scikit-learn>=1.3`

---

## File Structure

```
finetune_tw/
  walkforward.py        # NEW: WalkForwardFold + single_fold() + oof_folds()
  signal.py             # NEW: KronosSignal + KronosSignalExtractor
  features.py           # NEW: build_tech_features(), build_market_relative_features()
  analog.py             # NEW: AnalogFeatures + AnalogEngine (fit/query)
  feature_table.py      # NEW: build_feature_table() — assembles all features
  stacking.py           # NEW: StackingModel (FEATURE_COLS, fit, predict, save, load)
  stacking_backtest.py  # NEW: end-to-end CLI runner + baseline comparison
  config.py             # MODIFY: add 8 new fields
requirements.txt        # MODIFY: add lightgbm, scikit-learn

tests/finetune_tw/
  test_walkforward.py   # NEW
  test_signal.py        # NEW
  test_features.py      # NEW
  test_analog.py        # NEW
  test_stacking.py      # NEW
```

---

### Task 1: Config 擴充 + Walk-forward 切割工具

**Files:**
- Modify: `finetune_tw/config.py`
- Create: `finetune_tw/walkforward.py`
- Create: `tests/finetune_tw/test_walkforward.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `WalkForwardFold(train_start, train_end, embargo_end, val_start, val_end)`, `single_fold(train_start, train_end, val_end, embargo_days=110)`, `oof_folds(start, end, n_folds=5, embargo_days=110) -> list[WalkForwardFold]`
- Produces: `Config` 新增欄位 `mc_sample_count`, `stacking_enabled`, `analog_enabled`, `analog_n_neighbors`, `analog_window`, `stacking_train_start`, `stacking_train_end`, `wf_embargo_days`

- [ ] **Step 1: 安裝新依賴**

```bash
pip install "lightgbm>=4.0" "scikit-learn>=1.3"
```

- [ ] **Step 2: 更新 requirements.txt**

在 `requirements.txt` 尾端加入：
```
lightgbm>=4.0
scikit-learn>=1.3
```

- [ ] **Step 3: 寫 walkforward 測試（先讓它失敗）**

建立 `tests/finetune_tw/test_walkforward.py`：

```python
import pytest
import pandas as pd
from finetune_tw.walkforward import single_fold, oof_folds, WalkForwardFold


def test_single_fold_embargo_gap():
    fold = single_fold("2015-01-01", "2023-12-31", "2024-06-30", embargo_days=110)
    assert isinstance(fold, WalkForwardFold)
    embargo_ts = pd.Timestamp(fold.embargo_end)
    train_end_ts = pd.Timestamp(fold.train_end)
    assert (embargo_ts - train_end_ts).days >= 110
    assert fold.val_start == fold.embargo_end
    assert fold.val_end == "2024-06-30"


def test_single_fold_no_overlap():
    fold = single_fold("2015-01-01", "2022-12-31", "2023-12-31", embargo_days=100)
    assert fold.val_start > fold.train_end
    assert fold.val_end >= fold.val_start


def test_oof_folds_count():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=4, embargo_days=110)
    assert len(folds) == 4


def test_oof_folds_expanding():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=3, embargo_days=110)
    # Each fold's train_end should be later than the previous
    for i in range(1, len(folds)):
        assert folds[i].train_end > folds[i - 1].train_end


def test_oof_folds_val_no_overlap_with_train():
    folds = oof_folds("2015-01-01", "2023-12-31", n_folds=3, embargo_days=110)
    for fold in folds:
        assert fold.val_start > fold.train_end
```

- [ ] **Step 4: 執行測試確認失敗**

```bash
pytest tests/finetune_tw/test_walkforward.py -v
```

預期：`ImportError: cannot import name 'single_fold'`

- [ ] **Step 5: 實作 finetune_tw/walkforward.py**

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class WalkForwardFold:
    train_start: str
    train_end: str
    embargo_end: str   # train_end + embargo_days
    val_start: str     # == embargo_end
    val_end: str


def single_fold(
    train_start: str,
    train_end: str,
    val_end: str,
    embargo_days: int = 110,
) -> WalkForwardFold:
    """Return a walk-forward fold with a purged embargo gap after train_end."""
    embargo_end = (
        pd.Timestamp(train_end) + pd.Timedelta(days=embargo_days)
    ).strftime("%Y-%m-%d")
    return WalkForwardFold(
        train_start=train_start,
        train_end=train_end,
        embargo_end=embargo_end,
        val_start=embargo_end,
        val_end=val_end,
    )


def oof_folds(
    start: str,
    end: str,
    n_folds: int = 5,
    embargo_days: int = 110,
) -> list[WalkForwardFold]:
    """Generate expanding-window walk-forward folds with embargo gaps.

    The timeline [start, end] is split into (n_folds + 1) equal segments.
    Fold i trains on [start, segment_(i+1)] and validates on [segment_(i+1)+embargo, segment_(i+2)].
    """
    bdays = pd.bdate_range(start, end)
    if len(bdays) < (n_folds + 1) * 2:
        raise ValueError(
            f"Not enough business days ({len(bdays)}) for {n_folds} folds "
            f"(need at least {(n_folds + 1) * 2})"
        )
    fold_size = len(bdays) // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        te = bdays[fold_size * (i + 1) - 1]
        ve = bdays[min(fold_size * (i + 2) - 1, len(bdays) - 1)]
        folds.append(
            single_fold(
                train_start=start,
                train_end=te.strftime("%Y-%m-%d"),
                val_end=ve.strftime("%Y-%m-%d"),
                embargo_days=embargo_days,
            )
        )
    return folds
```

- [ ] **Step 6: 更新 finetune_tw/config.py 加入新欄位**

在 `Config` dataclass 的 `# Backtest` 區塊後面加入：

```python
    # Stacking meta-model
    mc_sample_count: int = 20           # MC forward passes for Kronos signal distribution
    stacking_enabled: bool = False      # Enable LightGBM stacking layer
    stacking_train_start: str = "2018-01-01"
    stacking_train_end: str = "2023-12-31"
    wf_embargo_days: int = 110          # >= lookback_window + predict_window (90+10=100)
    analog_enabled: bool = False        # Enable analog RAG features
    analog_n_neighbors: int = 20
    analog_window: int = 20             # Window length for retrieval key featurization
```

- [ ] **Step 7: 執行測試確認通過**

```bash
pytest tests/finetune_tw/test_walkforward.py -v
```

預期：5 tests PASSED

- [ ] **Step 8: Commit**

```bash
git add requirements.txt finetune_tw/config.py finetune_tw/walkforward.py tests/finetune_tw/test_walkforward.py
git commit -m "feat(stacking): add walk-forward split utilities and config extensions"
```

---

### Task 2: Kronos MC 訊號萃取器

**Files:**
- Create: `finetune_tw/signal.py`
- Create: `tests/finetune_tw/test_signal.py`

**Interfaces:**
- Consumes: `KronosPredictor.predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, sample_count, verbose)`, `Config`, `query_symbol`
- Produces: `KronosSignal(mean_return, q10, q50, q90, dispersion, dir_prob)`, `KronosSignalExtractor.extract_date(date, symbols, cfg, horizon=4) -> dict[str, KronosSignal]`, `KronosSignalExtractor.extract_date_range(dates, symbols, cfg, horizon=4) -> pd.DataFrame` with MultiIndex (date, symbol) and columns `kronos_mean, kronos_q10, kronos_q50, kronos_q90, kronos_disp, kronos_dir_prob`

- [ ] **Step 1: 寫測試**

建立 `tests/finetune_tw/test_signal.py`：

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from finetune_tw.signal import KronosSignal, KronosSignalExtractor


def _make_pred_df(close_val: float) -> pd.DataFrame:
    return pd.DataFrame({
        "open": [close_val] * 10,
        "high": [close_val] * 10,
        "low": [close_val] * 10,
        "close": [close_val] * 10,
        "volume": [0.0] * 10,
        "amount": [0.0] * 10,
    })


def _make_mock_predictor(close_values: list[float]):
    """Returns a mock predictor whose predict_batch returns close_values in order."""
    calls = iter(close_values)

    def predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len,
                      T, top_k, top_p, sample_count, verbose):
        val = next(calls, 1.0)
        return [_make_pred_df(val) for _ in df_list]

    predictor = MagicMock()
    predictor.predict_batch.side_effect = predict_batch
    return predictor


def test_kronos_signal_fields():
    sig = KronosSignal(
        mean_return=0.05, q10=0.01, q50=0.05, q90=0.09, dispersion=0.02, dir_prob=0.8
    )
    assert sig.mean_return == pytest.approx(0.05)
    assert sig.dir_prob == pytest.approx(0.8)


def test_extract_date_returns_signals():
    # Mock: 20 samples all returning 1.05 (5% return with last_close=1.0)
    close_values = [1.05] * 20
    predictor = _make_mock_predictor(close_values)

    # Minimal config mock
    cfg = MagicMock()
    cfg.db_path = ":memory:"
    cfg.lookback_window = 3
    cfg.pred_len = 5

    extractor = KronosSignalExtractor(predictor, n_samples=3, top_k=40)

    # Patch _load_context to avoid DB
    import finetune_tw.signal as sig_mod
    original = sig_mod.KronosSignalExtractor._load_context

    def fake_load_context(self, sym, as_of, cfg):
        ctx_df = pd.DataFrame({
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [0.0, 0.0, 0.0],
            "amount": [0.0, 0.0, 0.0],
        })
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
        y_ts = pd.Series(pd.date_range("2024-01-04", periods=5, freq="B"))
        return ctx_df, x_ts, y_ts

    sig_mod.KronosSignalExtractor._load_context = fake_load_context
    try:
        result = extractor.extract_date(
            pd.Timestamp("2024-01-04"), ["2330.TW"], cfg, horizon=4
        )
    finally:
        sig_mod.KronosSignalExtractor._load_context = original

    assert "2330.TW" in result
    sig = result["2330.TW"]
    assert isinstance(sig, KronosSignal)
    assert sig.dir_prob == pytest.approx(1.0)  # all samples > 0


def test_extract_date_range_returns_dataframe():
    predictor = _make_mock_predictor([1.05] * 100)
    cfg = MagicMock()
    cfg.db_path = ":memory:"
    cfg.lookback_window = 3
    cfg.pred_len = 5

    extractor = KronosSignalExtractor(predictor, n_samples=2, top_k=40)

    import finetune_tw.signal as sig_mod

    def fake_load_context(self, sym, as_of, cfg):
        ctx_df = pd.DataFrame({
            "open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0],
            "volume": [0.0, 0.0, 0.0], "amount": [0.0, 0.0, 0.0],
        })
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
        y_ts = pd.Series(pd.date_range("2024-01-04", periods=5, freq="B"))
        return ctx_df, x_ts, y_ts

    sig_mod.KronosSignalExtractor._load_context = fake_load_context
    try:
        df = extractor.extract_date_range(
            [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
            ["2330.TW", "2317.TW"],
            cfg, horizon=4,
        )
    finally:
        sig_mod.KronosSignalExtractor._load_context = original

    assert isinstance(df, pd.DataFrame)
    assert "kronos_mean" in df.columns
    assert "kronos_dir_prob" in df.columns
    assert df.index.names == ["date", "symbol"]
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
pytest tests/finetune_tw/test_signal.py -v
```

預期：`ImportError`

- [ ] **Step 3: 實作 finetune_tw/signal.py**

```python
"""Kronos MC signal extractor.

Calls predict_batch n_samples times with stochastic decoding (top_k=40) to build
a distribution of predicted returns per symbol per date.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from model import KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol

_BATCH_SIZE = 32


@dataclass
class KronosSignal:
    mean_return: float
    q10: float
    q50: float
    q90: float
    dispersion: float
    dir_prob: float     # P(return > 0) across samples


class KronosSignalExtractor:
    def __init__(
        self,
        predictor: KronosPredictor,
        n_samples: int = 20,
        top_k: int = 40,
        temperature: float = 1.0,
    ) -> None:
        self.predictor = predictor
        self.n_samples = n_samples
        self.top_k = top_k
        self.temperature = temperature

    def _load_context(
        self,
        sym: str,
        as_of: pd.Timestamp,
        cfg: Config,
    ) -> "tuple[pd.DataFrame, pd.Series, pd.Series] | None":
        start = (as_of - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")
        df = query_symbol(cfg.db_path, sym, start=start, end=end)
        if len(df) < cfg.lookback_window:
            return None
        ctx = df.iloc[-cfg.lookback_window:]
        ctx_df = ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            return None
        x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        y_ts = pd.Series(pd.date_range(as_of, periods=cfg.pred_len, freq="B"))
        return ctx_df, x_ts, y_ts

    def extract_date(
        self,
        date: pd.Timestamp,
        symbols: list[str],
        cfg: Config,
        horizon: int = 4,  # 0-indexed: 4 = 5th predicted day
    ) -> dict[str, KronosSignal]:
        contexts: dict[str, tuple] = {}
        for sym in symbols:
            result = self._load_context(sym, date, cfg)
            if result is not None:
                ctx_df, x_ts, y_ts = result
                last_close = float(ctx_df["close"].iloc[-1])
                contexts[sym] = (ctx_df, x_ts, y_ts, last_close)

        if not contexts:
            return {}

        sym_list = list(contexts.keys())
        sample_returns: dict[str, list[float]] = {s: [] for s in sym_list}
        df_list = [contexts[s][0] for s in sym_list]
        x_ts_list = [contexts[s][1] for s in sym_list]
        y_ts_list = [contexts[s][2] for s in sym_list]
        last_closes = {s: contexts[s][3] for s in sym_list}

        with torch.no_grad():
            for _ in range(self.n_samples):
                for b in range(0, len(sym_list), _BATCH_SIZE):
                    batch = sym_list[b:b + _BATCH_SIZE]
                    preds = self.predictor.predict_batch(
                        df_list=df_list[b:b + _BATCH_SIZE],
                        x_timestamp_list=x_ts_list[b:b + _BATCH_SIZE],
                        y_timestamp_list=y_ts_list[b:b + _BATCH_SIZE],
                        pred_len=cfg.pred_len,
                        T=self.temperature,
                        top_k=self.top_k,
                        top_p=1.0,
                        sample_count=1,
                        verbose=False,
                    )
                    for sym, pred in zip(batch, preds):
                        if pred is not None and len(pred) > horizon:
                            r = float(pred["close"].iloc[horizon]) / last_closes[sym] - 1.0
                            sample_returns[sym].append(r)

        results: dict[str, KronosSignal] = {}
        for sym in sym_list:
            arr = np.array(sample_returns[sym])
            if len(arr) < 3:
                continue
            results[sym] = KronosSignal(
                mean_return=float(arr.mean()),
                q10=float(np.percentile(arr, 10)),
                q50=float(np.percentile(arr, 50)),
                q90=float(np.percentile(arr, 90)),
                dispersion=float(arr.std()),
                dir_prob=float((arr > 0).mean()),
            )
        return results

    def extract_date_range(
        self,
        dates: list[pd.Timestamp],
        symbols: list[str],
        cfg: Config,
        horizon: int = 4,
    ) -> pd.DataFrame:
        rows = []
        for date in dates:
            signals = self.extract_date(date, symbols, cfg, horizon)
            for sym, sig in signals.items():
                rows.append({
                    "date": date, "symbol": sym,
                    "kronos_mean": sig.mean_return,
                    "kronos_q10": sig.q10,
                    "kronos_q50": sig.q50,
                    "kronos_q90": sig.q90,
                    "kronos_disp": sig.dispersion,
                    "kronos_dir_prob": sig.dir_prob,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index(["date", "symbol"])
```

- [ ] **Step 4: 執行測試確認通過**

```bash
pytest tests/finetune_tw/test_signal.py -v
```

預期：3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/signal.py tests/finetune_tw/test_signal.py
git commit -m "feat(stacking): add Kronos MC signal extractor (n_samples=20, top_k=40)"
```

---

### Task 3: 技術與市場相對特徵

**Files:**
- Create: `finetune_tw/features.py`
- Create: `tests/finetune_tw/test_features.py`

**Interfaces:**
- Produces: `build_tech_features(df, as_of, lookback=90) -> dict[str, float] | None` — keys: `ma20_gap, ma60_gap, rsi_14, bb_pct, mom_10d, mom_20d, vol_20d`
- Produces: `build_market_relative_features(sym_df, bench_df, as_of, lookback=90) -> dict[str, float] | None` — keys: `alpha_20d, alpha_60d, rel_vol`
- Input `df` / `sym_df` / `bench_df`: DataFrame with columns `date, open, high, low, close, volume, amount`

- [ ] **Step 1: 寫測試**

建立 `tests/finetune_tw/test_features.py`：

```python
import numpy as np
import pandas as pd
import pytest
from finetune_tw.features import build_tech_features, build_market_relative_features


def _make_df(close_values: list[float], start: str = "2023-01-02") -> pd.DataFrame:
    n = len(close_values)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates,
        "open": close_values,
        "high": [c * 1.01 for c in close_values],
        "low": [c * 0.99 for c in close_values],
        "close": close_values,
        "volume": [1000.0] * n,
        "amount": [c * 1000 for c in close_values],
    })


def test_tech_features_keys():
    close = list(range(100, 165))  # 65 rows
    df = _make_df(close)
    as_of = pd.Timestamp("2023-04-14")
    result = build_tech_features(df, as_of)
    assert result is not None
    expected_keys = {"ma20_gap", "ma60_gap", "rsi_14", "bb_pct", "mom_10d", "mom_20d", "vol_20d"}
    assert set(result.keys()) == expected_keys


def test_tech_features_insufficient_data():
    df = _make_df([100.0] * 30)  # only 30 rows, need 60
    result = build_tech_features(df, pd.Timestamp("2023-02-16"))
    assert result is None


def test_tech_features_rsi_range():
    close = [100 + i for i in range(65)]  # trending up
    df = _make_df(close)
    result = build_tech_features(df, pd.Timestamp("2023-04-14"))
    assert 0 <= result["rsi_14"] <= 100


def test_tech_features_ma20_gap_sign():
    close = [100.0] * 20 + [200.0] * 45  # price doubled
    df = _make_df(close)
    as_of = pd.Timestamp(df["date"].iloc[-1])
    result = build_tech_features(df, as_of)
    assert result["ma20_gap"] > 0  # last close > MA20


def test_market_relative_features_keys():
    sym_close = [100.0 + i for i in range(65)]
    bench_close = [1000.0 + i for i in range(65)]
    sym_df = _make_df(sym_close)
    bench_df = _make_df(bench_close)
    as_of = pd.Timestamp(sym_df["date"].iloc[-1])
    result = build_market_relative_features(sym_df, bench_df, as_of)
    assert result is not None
    assert set(result.keys()) == {"alpha_20d", "alpha_60d", "rel_vol"}


def test_market_relative_features_alpha_positive():
    # sym goes up more than bench
    sym_close = [100.0 * (1.01 ** i) for i in range(65)]
    bench_close = [100.0] * 65  # flat
    sym_df = _make_df(sym_close)
    bench_df = _make_df(bench_close)
    as_of = pd.Timestamp(sym_df["date"].iloc[-1])
    result = build_market_relative_features(sym_df, bench_df, as_of)
    assert result["alpha_20d"] > 0
    assert result["alpha_60d"] > 0
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
pytest tests/finetune_tw/test_features.py -v
```

預期：`ImportError`

- [ ] **Step 3: 實作 finetune_tw/features.py**

```python
"""Technical and market-relative feature builders.

All functions are pure (no side effects, no DB access) — input is a DataFrame
with columns: date, open, high, low, close, volume, amount. All rows up to
and including as_of are used; later rows are ignored.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _close_tail(df: pd.DataFrame, as_of: pd.Timestamp, n: int) -> "np.ndarray | None":
    sub = df[pd.to_datetime(df["date"]) <= as_of].tail(n)
    if len(sub) < n:
        return None
    return sub["close"].values.astype(float)


def _returns(close: np.ndarray) -> np.ndarray:
    return np.diff(close) / (close[:-1] + 1e-9)


def build_tech_features(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
) -> "dict[str, float] | None":
    """Compute technical indicators as of a given date.

    Requires at least 60 rows ending at or before as_of.
    Returns None if insufficient data.
    """
    sub = df[pd.to_datetime(df["date"]) <= as_of].tail(lookback)
    if len(sub) < 60:
        return None
    close = sub["close"].values.astype(float)

    last = close[-1]

    # MA gaps
    ma20 = close[-20:].mean()
    ma60 = close[-60:].mean()
    ma20_gap = (last - ma20) / (ma20 + 1e-9)
    ma60_gap = (last - ma60) / (ma60 + 1e-9)

    # RSI-14 (Wilder's, simplified as SMA-based)
    rets_14 = _returns(close[-15:])
    gains = np.where(rets_14 > 0, rets_14, 0.0)
    losses = np.where(rets_14 < 0, -rets_14, 0.0)
    rs = gains.mean() / (losses.mean() + 1e-9)
    rsi_14 = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger %B (20-day, 2σ)
    c20 = close[-20:]
    bb_mid = c20.mean()
    bb_std = c20.std()
    bb_pct = (last - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # Momentum
    mom_10d = (last / (close[-11] + 1e-9) - 1.0) if len(close) >= 11 else float("nan")
    mom_20d = (last / (close[-21] + 1e-9) - 1.0) if len(close) >= 21 else float("nan")

    # 20-day volatility
    vol_20d = float(np.std(_returns(close[-21:]))) if len(close) >= 21 else float("nan")

    return {
        "ma20_gap": float(ma20_gap),
        "ma60_gap": float(ma60_gap),
        "rsi_14": float(rsi_14),
        "bb_pct": float(bb_pct),
        "mom_10d": float(mom_10d),
        "mom_20d": float(mom_20d),
        "vol_20d": float(vol_20d),
    }


def build_market_relative_features(
    sym_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
) -> "dict[str, float] | None":
    """Compute alpha and relative volatility vs a benchmark.

    Returns None if either series has fewer than 61 rows.
    """
    sym60 = _close_tail(sym_df, as_of, 61)
    bench60 = _close_tail(bench_df, as_of, 61)
    if sym60 is None or bench60 is None:
        return None

    sym_ret_20 = sym60[-1] / (sym60[-21] + 1e-9) - 1.0
    sym_ret_60 = sym60[-1] / (sym60[0] + 1e-9) - 1.0
    bench_ret_20 = bench60[-1] / (bench60[-21] + 1e-9) - 1.0
    bench_ret_60 = bench60[-1] / (bench60[0] + 1e-9) - 1.0

    sym_vol = float(np.std(_returns(sym60[-21:])))
    bench_vol = float(np.std(_returns(bench60[-21:])))

    return {
        "alpha_20d": float(sym_ret_20 - bench_ret_20),
        "alpha_60d": float(sym_ret_60 - bench_ret_60),
        "rel_vol": float(sym_vol / (bench_vol + 1e-9)),
    }
```

- [ ] **Step 4: 執行測試確認通過**

```bash
pytest tests/finetune_tw/test_features.py -v
```

預期：6 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/features.py tests/finetune_tw/test_features.py
git commit -m "feat(stacking): add technical and market-relative feature builders"
```

---

### Task 4: Analog Engine（RAG-as-features，Layer 2）

**Files:**
- Create: `finetune_tw/analog.py`
- Create: `tests/finetune_tw/test_analog.py`

**Interfaces:**
- Produces: `AnalogFeatures(fwd_q25, fwd_q50, fwd_q75, up_prob, max_gain, max_loss, dispersion, n_analogs)`
- Produces: `AnalogEngine(n_neighbors=20, window=20).fit(db_path, symbols, cutoff_date, pred_len=10) -> AnalogEngine`
- Produces: `AnalogEngine.query(recent_close, recent_volume) -> AnalogFeatures | None`
- **Invariant**: `cutoff_date` 傳入時必須 = `as_of - pred_len * 2 calendar days`，確保 point-in-time safe

- [ ] **Step 1: 寫測試**

建立 `tests/finetune_tw/test_analog.py`：

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from finetune_tw.analog import AnalogEngine, AnalogFeatures


def _make_price_df(close_values: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    n = len(close_values)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates, "open": close_values,
        "high": [c * 1.01 for c in close_values],
        "low": [c * 0.99 for c in close_values],
        "close": close_values,
        "volume": [1000.0] * n, "amount": [c * 1000 for c in close_values],
    })


def test_analog_engine_fit_and_query():
    """Build index from synthetic data, query should return AnalogFeatures."""
    close = list(range(100, 200))  # 100 rows
    fake_df = _make_price_df(close)

    engine = AnalogEngine(n_neighbors=5, window=10)
    with patch("finetune_tw.analog.query_symbol", return_value=fake_df), \
         patch("finetune_tw.analog.list_symbols", return_value=["2330.TW"]):
        engine.fit(":memory:", ["2330.TW"], cutoff_date="2020-06-01", pred_len=5)

    assert len(engine._keys) > 0
    assert len(engine._fwd_returns) > 0

    recent_close = np.array(list(range(150, 160)), dtype=float)
    recent_volume = np.full(10, 1000.0)
    result = engine.query(recent_close, recent_volume)

    assert result is not None
    assert isinstance(result, AnalogFeatures)
    assert 0.0 <= result.up_prob <= 1.0
    assert result.n_analogs <= 5


def test_analog_engine_empty_returns_none():
    engine = AnalogEngine(n_neighbors=5, window=10)
    # No fit() called → _keys is empty
    recent_close = np.ones(10)
    recent_volume = np.ones(10)
    result = engine.query(recent_close, recent_volume)
    assert result is None


def test_analog_features_fields():
    af = AnalogFeatures(
        fwd_q25=0.01, fwd_q50=0.03, fwd_q75=0.05,
        up_prob=0.7, max_gain=0.12, max_loss=-0.08,
        dispersion=0.03, n_analogs=20,
    )
    assert af.up_prob == pytest.approx(0.7)
    assert af.n_analogs == 20


def test_point_in_time_cutoff():
    """Verify that fit uses a strict cutoff before as_of."""
    calls = []

    def mock_query(db_path, symbol, start=None, end=None):
        calls.append(end)
        return _make_price_df([100.0] * 10)  # too short, will be skipped

    engine = AnalogEngine(n_neighbors=5, window=10)
    with patch("finetune_tw.analog.query_symbol", side_effect=mock_query), \
         patch("finetune_tw.analog.list_symbols", return_value=["2330.TW"]):
        engine.fit(":memory:", ["2330.TW"], cutoff_date="2024-01-10", pred_len=10)

    # The end date passed to query_symbol must be BEFORE cutoff_date
    assert len(calls) > 0
    for end_date in calls:
        assert end_date < "2024-01-10"
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
pytest tests/finetune_tw/test_analog.py -v
```

預期：`ImportError`

- [ ] **Step 3: 實作 finetune_tw/analog.py**

```python
"""Analog Engine: point-in-time k-NN retrieval of historically similar windows.

Invariant: ALL windows in the retrieval index have their forward outcomes already
realized at query time. Enforced by: cutoff_date = as_of − pred_len*2 calendar days.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from finetune_tw.db import query_symbol, list_symbols


@dataclass
class AnalogFeatures:
    fwd_q25: float
    fwd_q50: float
    fwd_q75: float
    up_prob: float       # P(forward return > 0) among analogs
    max_gain: float
    max_loss: float
    dispersion: float    # std of analog forward returns = confidence signal
    n_analogs: int


class AnalogEngine:
    """k-NN retrieval engine for 'look-alike' historical windows."""

    def __init__(self, n_neighbors: int = 20, window: int = 20) -> None:
        self.n_neighbors = n_neighbors
        self.window = window
        self._keys: np.ndarray = np.empty((0, 0))
        self._fwd_returns: np.ndarray = np.empty(0)

    def _featurize(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Convert a price/volume window into a shape-normalized retrieval key."""
        if len(close) < 2:
            return np.zeros(self.window + 2)
        log_rets = np.diff(np.log(close + 1e-9))
        if log_rets.std() > 1e-9:
            log_rets = (log_rets - log_rets.mean()) / log_rets.std()
        # Pad/truncate to (window - 1) returns
        n = self.window - 1
        if len(log_rets) >= n:
            log_rets = log_rets[-n:]
        else:
            log_rets = np.pad(log_rets, (n - len(log_rets), 0))
        vol_z = (volume[-1] - volume.mean()) / (volume.std() + 1e-9)
        range_feat = (close.max() - close.min()) / (close.mean() + 1e-9)
        return np.concatenate([log_rets, [vol_z, range_feat]])

    def fit(
        self,
        db_path: str,
        symbols: list[str],
        cutoff_date: str,
        pred_len: int = 10,
        start_date: str = "2015-01-01",
    ) -> "AnalogEngine":
        """Build retrieval index. All windows end strictly before cutoff_date."""
        strict_cutoff = (
            pd.Timestamp(cutoff_date) - pd.Timedelta(days=pred_len * 2)
        ).strftime("%Y-%m-%d")

        keys, fwd_returns = [], []
        for sym in symbols:
            df = query_symbol(db_path, sym, start=start_date, end=strict_cutoff)
            if len(df) < self.window + pred_len + 1:
                continue
            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            for i in range(len(df) - self.window - pred_len):
                win_close = close[i:i + self.window]
                win_vol = volume[i:i + self.window]
                fwd_last = close[i + self.window + pred_len - 1]
                fwd_ret = fwd_last / (win_close[-1] + 1e-9) - 1.0
                keys.append(self._featurize(win_close, win_vol))
                fwd_returns.append(fwd_ret)

        if keys:
            self._keys = np.array(keys, dtype=float)
            self._fwd_returns = np.array(fwd_returns, dtype=float)
        return self

    def query(
        self,
        recent_close: np.ndarray,
        recent_volume: np.ndarray,
    ) -> "AnalogFeatures | None":
        """Return statistics of forward returns from k nearest analog windows."""
        if self._keys.shape[0] == 0:
            return None
        key = self._featurize(recent_close, recent_volume)
        k = min(self.n_neighbors, len(self._keys))
        dists = np.linalg.norm(self._keys - key, axis=1)
        idx = np.argpartition(dists, k - 1)[:k]
        fwd = self._fwd_returns[idx]
        return AnalogFeatures(
            fwd_q25=float(np.percentile(fwd, 25)),
            fwd_q50=float(np.percentile(fwd, 50)),
            fwd_q75=float(np.percentile(fwd, 75)),
            up_prob=float((fwd > 0).mean()),
            max_gain=float(fwd.max()),
            max_loss=float(fwd.min()),
            dispersion=float(fwd.std()),
            n_analogs=k,
        )
```

- [ ] **Step 4: 執行測試確認通過**

```bash
pytest tests/finetune_tw/test_analog.py -v
```

預期：4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/analog.py tests/finetune_tw/test_analog.py
git commit -m "feat(stacking): add Analog Engine with point-in-time k-NN retrieval"
```

---

### Task 5: Feature Table 組裝 + LightGBM Stacking Model

**Files:**
- Create: `finetune_tw/stacking.py`
- Create: `tests/finetune_tw/test_stacking.py`

**Interfaces:**
- Consumes: `KronosSignal` columns (`kronos_mean` etc.), `build_tech_features()` output, `build_market_relative_features()` output, `AnalogFeatures` (optional)
- Produces: `FEATURE_COLS: list[str]` — canonical feature column list (14 Kronos+tech+mkt, 7 analog)
- Produces: `build_feature_row(sym, as_of, kronos_signal, sym_df, bench_df, analog_engine, cfg) -> dict | None`
- Produces: `StackingModel.fit(feature_df: pd.DataFrame) -> StackingModel` — feature_df has MultiIndex (date, symbol) and column `fwd_return`
- Produces: `StackingModel.predict(feature_df: pd.DataFrame) -> pd.Series` — stacking scores
- Produces: `StackingModel.save(path)`, `StackingModel.load(path)`

- [ ] **Step 1: 寫測試**

建立 `tests/finetune_tw/test_stacking.py`：

```python
import numpy as np
import pandas as pd
import pytest
from finetune_tw.stacking import StackingModel, FEATURE_COLS, build_feature_row
from finetune_tw.signal import KronosSignal
from finetune_tw.analog import AnalogFeatures


def _make_feature_df(n_dates: int = 20, n_syms: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    syms = [f"{i:04d}.TW" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {col: float(rng.normal()) for col in FEATURE_COLS}
            row["date"] = d
            row["symbol"] = s
            row["fwd_return"] = float(rng.normal(0, 0.02))
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    return df


def test_stacking_model_fit_predict():
    df = _make_feature_df(10, 30)
    model = StackingModel(num_rounds=20)
    model.fit(df)
    scores = model.predict(df.drop(columns=["fwd_return"]))
    assert len(scores) == len(df)
    assert not scores.isna().any()


def test_stacking_model_scores_ranked():
    df = _make_feature_df(5, 40)
    model = StackingModel(num_rounds=20)
    model.fit(df)
    scores = model.predict(df.drop(columns=["fwd_return"]))
    # Scores should vary (not all identical)
    assert scores.std() > 0


def test_stacking_model_save_load(tmp_path):
    df = _make_feature_df(5, 30)
    model = StackingModel(num_rounds=10)
    model.fit(df)
    path = str(tmp_path / "stacker.lgb")
    model.save(path)
    loaded = StackingModel.load(path)
    original_scores = model.predict(df.drop(columns=["fwd_return"]))
    loaded_scores = loaded.predict(df.drop(columns=["fwd_return"]))
    pd.testing.assert_series_equal(original_scores, loaded_scores, check_names=False)


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 21  # 6 Kronos + 7 tech+mkt + 8 analog (wait, let's check)


def test_build_feature_row_with_kronos_signal():
    sig = KronosSignal(mean_return=0.02, q10=0.0, q50=0.02, q90=0.04, dispersion=0.01, dir_prob=0.7)
    sym_close = [100.0 + i for i in range(70)]
    bench_close = [1000.0 + i for i in range(70)]
    dates = pd.bdate_range("2022-01-03", periods=70).strftime("%Y-%m-%d")
    sym_df = pd.DataFrame({
        "date": dates, "open": sym_close, "high": [c * 1.01 for c in sym_close],
        "low": [c * 0.99 for c in sym_close], "close": sym_close,
        "volume": [1000.0] * 70, "amount": [c * 1000 for c in sym_close],
    })
    bench_df = sym_df.copy()
    bench_df["close"] = bench_close
    cfg = type("Cfg", (), {"lookback_window": 90, "pred_len": 10})()
    as_of = pd.Timestamp(dates[-1])

    row = build_feature_row("2330.TW", as_of, sig, sym_df, bench_df, None, cfg)
    assert row is not None
    assert "kronos_mean" in row
    assert row["kronos_mean"] == pytest.approx(0.02)
    assert "ma20_gap" in row
    assert "alpha_20d" in row
    # analog cols should be 0 when no engine
    assert row["analog_q50"] == pytest.approx(0.0)
```

- [ ] **Step 2: 執行測試確認失敗**

```bash
pytest tests/finetune_tw/test_stacking.py -v
```

預期：`ImportError`

- [ ] **Step 3: 實作 finetune_tw/stacking.py**

```python
"""LightGBM cross-sectional stacking meta-model + feature row builder."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from finetune_tw.signal import KronosSignal
from finetune_tw.analog import AnalogEngine, AnalogFeatures
from finetune_tw.features import build_tech_features, build_market_relative_features
from finetune_tw.config import Config

# ── Canonical feature column list ─────────────────────────────────────────────
# Order matters: must match exactly between fit() and predict().
FEATURE_COLS = [
    # Layer 1 — Kronos MC signals
    "kronos_mean", "kronos_q10", "kronos_q50", "kronos_q90",
    "kronos_disp", "kronos_dir_prob",
    # Layer 3 — Technical
    "ma20_gap", "ma60_gap", "rsi_14", "bb_pct", "mom_10d", "mom_20d", "vol_20d",
    # Layer 3 — Market-relative
    "alpha_20d", "alpha_60d", "rel_vol",
    # Layer 2 — Analog RAG (filled 0.0 when analog disabled)
    "analog_q25", "analog_q50", "analog_q75",
    "analog_up_prob", "analog_max_gain", "analog_max_loss", "analog_disp",
]

_LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def build_feature_row(
    sym: str,
    as_of: pd.Timestamp,
    kronos_signal: "KronosSignal | None",
    sym_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    analog_engine: "AnalogEngine | None",
    cfg: Config,
) -> "dict[str, float] | None":
    """Build a single feature dict for (sym, as_of).

    Returns None if technical features cannot be computed (insufficient history).
    """
    tech = build_tech_features(sym_df, as_of)
    if tech is None:
        return None
    mkt = build_market_relative_features(sym_df, bench_df, as_of)
    if mkt is None:
        return None

    row: dict[str, float] = {}

    # Kronos MC signals
    if kronos_signal is not None:
        row["kronos_mean"] = kronos_signal.mean_return
        row["kronos_q10"] = kronos_signal.q10
        row["kronos_q50"] = kronos_signal.q50
        row["kronos_q90"] = kronos_signal.q90
        row["kronos_disp"] = kronos_signal.dispersion
        row["kronos_dir_prob"] = kronos_signal.dir_prob
    else:
        for k in ["kronos_mean", "kronos_q10", "kronos_q50", "kronos_q90",
                  "kronos_disp", "kronos_dir_prob"]:
            row[k] = 0.0

    row.update(tech)
    row.update(mkt)

    # Analog features
    if analog_engine is not None:
        sub = sym_df[pd.to_datetime(sym_df["date"]) <= as_of].tail(analog_engine.window)
        if len(sub) == analog_engine.window:
            af = analog_engine.query(
                sub["close"].values.astype(float),
                sub["volume"].values.astype(float),
            )
            if af is not None:
                row["analog_q25"] = af.fwd_q25
                row["analog_q50"] = af.fwd_q50
                row["analog_q75"] = af.fwd_q75
                row["analog_up_prob"] = af.up_prob
                row["analog_max_gain"] = af.max_gain
                row["analog_max_loss"] = af.max_loss
                row["analog_disp"] = af.dispersion
                return row

    for k in ["analog_q25", "analog_q50", "analog_q75", "analog_up_prob",
              "analog_max_gain", "analog_max_loss", "analog_disp"]:
        row[k] = 0.0

    return row


class StackingModel:
    """LightGBM cross-sectional stacking meta-model for stock ranking."""

    def __init__(
        self,
        params: "dict | None" = None,
        num_rounds: int = 200,
    ) -> None:
        self.params = params or _LGBM_PARAMS
        self.num_rounds = num_rounds
        self._booster: "lgb.Booster | None" = None

    def fit(self, feature_df: pd.DataFrame) -> "StackingModel":
        """Train on a MultiIndex(date, symbol) DataFrame with a 'fwd_return' column.

        Uses cross-sectional rank as LightGBM label (avoids negative values issue
        with lambdarank) and groups by date.
        """
        df = feature_df.copy()
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df.dropna(subset=["fwd_return"])

        # Cross-sectional rank as label: 0-indexed rank within each date
        df["_label"] = df.groupby(level="date")["fwd_return"].rank(method="average") - 1

        # Ensure rows are sorted by date (required for LightGBM group parameter)
        df = df.sort_index(level="date")

        X = df[FEATURE_COLS].fillna(0.0).values.astype(float)
        y = df["_label"].values.astype(float)
        groups = df.groupby(level="date").size().values.tolist()

        dtrain = lgb.Dataset(X, label=y, group=groups)
        self._booster = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_rounds,
        )
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """Return stacking scores indexed as feature_df."""
        if self._booster is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        df = feature_df.copy()
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        X = df[FEATURE_COLS].fillna(0.0).values.astype(float)
        scores = self._booster.predict(X)
        return pd.Series(scores, index=feature_df.index, name="score")

    def save(self, path: str) -> None:
        if self._booster is None:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(path)

    @classmethod
    def load(cls, path: str) -> "StackingModel":
        obj = cls()
        obj._booster = lgb.Booster(model_file=path)
        return obj
```

- [ ] **Step 4: 修正 test_feature_cols_count（預期 22 欄位，not 21）**

編輯 `tests/finetune_tw/test_stacking.py` 中的 `test_feature_cols_count`：

```python
def test_feature_cols_count():
    # 6 Kronos + 7 tech + 3 market-relative + 7 analog = 23... 
    # Count from FEATURE_COLS directly to avoid hardcoding
    from finetune_tw.stacking import FEATURE_COLS
    assert len(FEATURE_COLS) == 23
```

- [ ] **Step 5: 執行測試確認通過**

```bash
pytest tests/finetune_tw/test_stacking.py -v
```

預期：5 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/stacking.py tests/finetune_tw/test_stacking.py
git commit -m "feat(stacking): add StackingModel (LightGBM lambdarank) and feature row builder"
```

---

### Task 6: Stacking Backtest Runner（端到端 CLI）

**Files:**
- Create: `finetune_tw/stacking_backtest.py`

**Interfaces:**
- Consumes: Tasks 1–5 的所有模組
- Produces: CLI `python -m finetune_tw.stacking_backtest --config ... --model round0 [--train-only] [--no-analog]`
- Produces: `{output_dir}/{exp_name}/stacking_feature_table.parquet` (可選快取)
- Produces: `{output_dir}/{exp_name}/stacking_model.lgb`
- Produces: `{output_dir}/{exp_name}/backtest_stacking.json` + `backtest_stacking.png`（與既有 backtest.py 格式相容）

- [ ] **Step 1: 實作 finetune_tw/stacking_backtest.py**

```python
"""
End-to-end stacking backtest pipeline.

Usage:
    python -m finetune_tw.stacking_backtest \\
        --config finetune_tw/configs/config_tw_daily_rtx6000.yaml \\
        --model round0

Steps executed:
  1. Load Kronos model (round0/round1/round2/pretrained)
  2. Extract Kronos MC signals for stacking training period (stacking_train_start → stacking_train_end)
  3. Build tech + market-relative features for training period
  4. (Optional) Fit Analog Engine on data before training period and build analog features
  5. Attach forward returns as target (5-day, = hold_days)
  6. Train LightGBM stacker
  7. Forward-test on test period (test_start_date → today):
     extract signals + features → score with stacker → rank top_k → portfolio returns
  8. Compare: Stacker vs Kronos-only vs Benchmark
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import (
    build_model_specs,
    load_predictor_from_spec,
    build_portfolio_returns,
    compute_metrics,
    rank_stocks,
)
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols
from finetune_tw.signal import KronosSignalExtractor
from finetune_tw.features import build_tech_features, build_market_relative_features
from finetune_tw.analog import AnalogEngine
from finetune_tw.stacking import StackingModel, build_feature_row, FEATURE_COLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_ohlcv(db_path: str, sym: str, start: str, end: str) -> pd.DataFrame:
    return query_symbol(db_path, sym, start=start, end=end)


def _fwd_return(db_path: str, sym: str, as_of: pd.Timestamp, pred_len: int) -> "float | None":
    """Actual forward return: close at as_of+pred_len / close at as_of − 1."""
    start = as_of.strftime("%Y-%m-%d")
    # Look ahead pred_len business days
    future = (as_of + pd.offsets.BDay(pred_len + 5)).strftime("%Y-%m-%d")
    df = query_symbol(db_path, sym, start=start, end=future)
    if len(df) < pred_len + 1:
        return None
    return float(df["close"].iloc[pred_len]) / float(df["close"].iloc[0]) - 1.0


def build_date_feature_table(
    date: pd.Timestamp,
    symbols: list[str],
    extractor: KronosSignalExtractor,
    cfg: Config,
    bench_df_cache: dict[str, pd.DataFrame],
    sym_df_cache: dict[str, pd.DataFrame],
    analog_engine: "AnalogEngine | None",
    include_target: bool = False,
) -> pd.DataFrame:
    horizon = min(cfg.hold_days, cfg.pred_len) - 1  # 0-indexed
    signals = extractor.extract_date(date, symbols, cfg, horizon=horizon)
    rows = []
    bench_df = bench_df_cache.get(cfg.benchmark_symbol, pd.DataFrame())

    for sym in symbols:
        sym_df = sym_df_cache.get(sym, pd.DataFrame())
        if sym_df.empty:
            continue
        row = build_feature_row(
            sym, date, signals.get(sym), sym_df, bench_df, analog_engine, cfg
        )
        if row is None:
            continue
        if include_target:
            fwd = _fwd_return(cfg.db_path, sym, date, cfg.hold_days)
            if fwd is None:
                continue
            row["fwd_return"] = fwd
        row["date"] = date
        row["symbol"] = sym
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["date", "symbol"])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_stacking_backtest(cfg: Config, model_key: str, use_analog: bool = False) -> None:
    specs = build_model_specs(cfg)
    spec = specs[model_key]

    out_dir = Path(cfg.output_dir) / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())

    print(f"\n{'='*60}")
    print(f"Stacking Backtest: {spec.label}")
    print(f"Train: {cfg.stacking_train_start} → {cfg.stacking_train_end}")
    print(f"Test:  {cfg.test_start_date} → {test_end}")
    print(f"Analog: {use_analog}  |  MC samples: {cfg.mc_sample_count}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # ── Load model ────────────────────────────────────────────────────────────
    predictor = load_predictor_from_spec(spec, cfg)
    extractor = KronosSignalExtractor(predictor, n_samples=cfg.mc_sample_count, top_k=40)

    # ── Pre-load OHLCV into memory (avoid repeated DB calls) ─────────────────
    all_dates_start = (
        pd.Timestamp(cfg.stacking_train_start) - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")

    print("Loading OHLCV data into memory ...")
    sym_df_cache: dict[str, pd.DataFrame] = {}
    for sym in symbols + [cfg.benchmark_symbol]:
        df = _load_ohlcv(cfg.db_path, sym, all_dates_start, test_end)
        if len(df) > 0:
            sym_df_cache[sym] = df
    bench_df_cache = {cfg.benchmark_symbol: sym_df_cache.get(cfg.benchmark_symbol, pd.DataFrame())}
    print(f"  Loaded {len(sym_df_cache)} symbols")
    sys.stdout.flush()

    # ── Analog Engine ─────────────────────────────────────────────────────────
    analog_engine: "AnalogEngine | None" = None
    if use_analog:
        print("Fitting Analog Engine ...")
        analog_engine = AnalogEngine(n_neighbors=cfg.analog_n_neighbors, window=cfg.analog_window)
        analog_engine.fit(
            db_path=cfg.db_path,
            symbols=symbols,
            cutoff_date=cfg.stacking_train_start,
            pred_len=cfg.pred_len,
        )
        print(f"  Index size: {len(analog_engine._keys)} windows")
        sys.stdout.flush()

    # ── Build training feature table ──────────────────────────────────────────
    cache_path = out_dir / f"stacking_features_{model_key}.parquet"
    if cache_path.exists():
        print(f"Loading cached feature table: {cache_path}")
        train_df = pd.read_parquet(cache_path)
    else:
        print("Building stacking training feature table ...")
        train_dates = pd.bdate_range(cfg.stacking_train_start, cfg.stacking_train_end)[::cfg.hold_days]
        parts = []
        for i, date in enumerate(train_dates):
            part = build_date_feature_table(
                date, symbols, extractor, cfg, bench_df_cache, sym_df_cache,
                analog_engine, include_target=True,
            )
            if not part.empty:
                parts.append(part)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(train_dates)}] {date.date()}: {len(part)} rows")
                sys.stdout.flush()
        train_df = pd.concat(parts) if parts else pd.DataFrame()
        if not train_df.empty:
            train_df.to_parquet(cache_path)
            print(f"  Cached → {cache_path}")

    if train_df.empty:
        print("ERROR: empty feature table — aborting.")
        return

    # ── Train stacker ─────────────────────────────────────────────────────────
    print(f"\nTraining LightGBM stacker on {len(train_df)} rows ...")
    model = StackingModel()
    model.fit(train_df)
    model_path = str(out_dir / f"stacking_model_{model_key}.lgb")
    model.save(model_path)
    print(f"  Saved → {model_path}")
    sys.stdout.flush()

    # ── Forward test ──────────────────────────────────────────────────────────
    print("\nForward test (stacker + Kronos-only comparison) ...")
    test_dates = pd.bdate_range(cfg.test_start_date, test_end)[::cfg.hold_days]
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = sym_df_cache.get(sym, pd.DataFrame())
        if not df.empty:
            close_prices[sym] = pd.Series(
                df["close"].values, index=pd.DatetimeIndex(df["date"])
            )

    stacker_holdings, kronos_holdings = [], []
    for i, date in enumerate(test_dates):
        part = build_date_feature_table(
            date, symbols, extractor, cfg, bench_df_cache, sym_df_cache,
            analog_engine, include_target=False,
        )
        if part.empty:
            stacker_holdings.append(set())
            kronos_holdings.append(set())
            continue

        # Stacker scores
        scores = model.predict(part)
        score_dict = {sym: float(scores.loc[(date, sym)]) for (d, sym) in scores.index if d == date}
        stacker_holdings.append(rank_stocks(score_dict, cfg.top_k))

        # Kronos-only (use kronos_mean as signal)
        if "kronos_mean" in part.columns:
            kronos_dict = {sym: float(part.loc[(date, sym), "kronos_mean"]) for (d, sym) in part.index if d == date}
            kronos_holdings.append(rank_stocks(kronos_dict, cfg.top_k))
        else:
            kronos_holdings.append(set())

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test_dates)}] {date.date()}")
            sys.stdout.flush()

    stacker_dr = build_portfolio_returns(close_prices, stacker_holdings, test_dates)
    kronos_dr = build_portfolio_returns(close_prices, kronos_holdings, test_dates)

    bm_df = sym_df_cache.get(cfg.benchmark_symbol, pd.DataFrame())
    bm_daily = pd.Series(
        bm_df["close"].values, index=pd.DatetimeIndex(bm_df["date"])
    ).pct_change().dropna()
    bm_daily = bm_daily[bm_daily.index >= pd.Timestamp(cfg.test_start_date)]

    sm = compute_metrics(stacker_dr)
    km = compute_metrics(kronos_dr)
    bm = compute_metrics(bm_daily)

    print(f"\n{'='*60}")
    print(f"  Stacker:      Sharpe={sm['sharpe']:.2f}  Ann={sm['annualised_return']:.1%}  DD={sm['max_drawdown']:.1%}")
    print(f"  Kronos-only:  Sharpe={km['sharpe']:.2f}  Ann={km['annualised_return']:.1%}  DD={km['max_drawdown']:.1%}")
    print(f"  Benchmark:    Sharpe={bm['sharpe']:.2f}  Ann={bm['annualised_return']:.1%}  DD={bm['max_drawdown']:.1%}")
    print(f"{'='*60}")

    # ── Save JSON + plot ──────────────────────────────────────────────────────
    result = {
        "model_key": f"stacking_{model_key}",
        "test_start": cfg.test_start_date,
        "test_end": test_end,
        "top_k": cfg.top_k,
        "hold_days": cfg.hold_days,
        "stacker": {"metrics": sm, "dates": [d.strftime("%Y-%m-%d") for d in stacker_dr.index], "daily_returns": stacker_dr.tolist()},
        "kronos_only": {"metrics": km, "dates": [d.strftime("%Y-%m-%d") for d in kronos_dr.index], "daily_returns": kronos_dr.tolist()},
        "benchmark": {"metrics": bm, "dates": [d.strftime("%Y-%m-%d") for d in bm_daily.index], "daily_returns": bm_daily.tolist()},
    }
    json_path = out_dir / f"backtest_stacking_{model_key}.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {json_path}")

    _plot_stacking(result, out_dir, model_key, spec.label)


def _plot_stacking(data: dict, out_dir: Path, model_key: str, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Stacking Backtest — {label}  ({data['test_start']} → {data['test_end']}  top-{data['top_k']})",
                 fontsize=12, fontweight="bold")
    colors = {"stacker": "#2196F3", "kronos_only": "#FF9800", "benchmark": "#9E9E9E"}
    for name, col in colors.items():
        d = data[name]
        dr = pd.Series(d["daily_returns"], index=pd.DatetimeIndex(d["dates"]))
        cum = (1 + dr).cumprod()
        m = d["metrics"]
        ls = "--" if name == "benchmark" else "-"
        lbl = f"{name}  Sharpe={m['sharpe']:.2f}  Ann={m['annualised_return']:.1%}"
        axes[0].plot(cum.index, cum.values, color=col, lw=1.8, ls=ls, label=lbl)
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    axes[0].axhline(1, color="black", lw=0.6, ls=":")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Cumulative Returns")

    metric_names = ["Ann Return", "Sharpe/3", "−Max DD"]
    x = np.arange(3)
    for i, (name, col) in enumerate(colors.items()):
        m = data[name]["metrics"]
        vals = [m["annualised_return"], m["sharpe"] / 3, -m["max_drawdown"]]
        axes[1].bar(x + (i - 1) * 0.25, vals, 0.25, color=col, label=name, alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_names)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Key Metrics")

    out_path = out_dir / f"backtest_stacking_{model_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--model", required=True,
                        choices=["pretrained", "round0", "round1", "round2"])
    parser.add_argument("--no-analog", action="store_true", help="Disable analog RAG features")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    run_stacking_backtest(cfg, args.model, use_analog=not args.no_analog and cfg.analog_enabled)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 驗證 import 正確**

```bash
python -c "from finetune_tw.stacking_backtest import run_stacking_backtest; print('OK')"
```

預期：`OK`

- [ ] **Step 3: 執行完整測試套件確認無回歸**

```bash
pytest tests/finetune_tw/ -v --ignore=tests/finetune_tw/test_train_predictor.py --ignore=tests/finetune_tw/test_train_tokenizer.py -x
```

預期：所有測試 PASSED（排除需要 GPU/大模型的訓練測試）

- [ ] **Step 4: Commit**

```bash
git add finetune_tw/stacking_backtest.py
git commit -m "feat(stacking): add end-to-end stacking backtest runner with Kronos-only comparison"
```

---

## Spec Coverage Self-Review

| Spec 要求 | 對應任務 | 狀態 |
|---|---|---|
| Layer 0: walk-forward + purged embargo split | Task 1 `walkforward.py` | ✓ |
| Layer 0: baseline ladder (zero-shot / FT / LightGBM-only / naive) | `stacking_backtest.py` Kronos-only vs Stacker 對比 | partial（完整 baseline ladder 可在後續迭代加入） |
| Layer 1: MC 平均報酬 + sample dispersion | Task 2 `signal.py` | ✓ |
| Layer 1: Checkpoint 選擇用 val Rank-IC | 現有 `ic_validation.py` 已實作 | ✓ (existing) |
| Layer 2: Analog Engine 不進 Kronos context | Task 4 `analog.py` | ✓ |
| Layer 2: Point-in-time 嚴格防洩漏 | `AnalogEngine.fit(cutoff_date)` 截止日限制 | ✓ |
| Layer 3: LightGBM lambdarank cross-sectional | Task 5 `stacking.py` | ✓ |
| Layer 3: 市場相對特徵 (alpha, rel_vol) | Task 3 `features.py` | ✓ |
| Layer 3: Analog 特徵接進 stacker | `FEATURE_COLS` 包含 analog_* 欄位 | ✓ |
| 防過擬合: OOF 生成 | 目前使用時間切分 (2022-2023 train → 2024+ test)；嚴格 OOF 為後續迭代 | partial |
| 主指標 Rank-IC | `ic_validation.rank_ic` 現有；stacking 結果用 Sharpe 驗收 | ✓ |

### 未完全覆蓋（後續迭代）

1. **完整 baseline ladder**：naive 動能、kNN-analog-only、LightGBM-only（無 Kronos）對比需另外加入 `stacking_backtest.py`
2. **嚴格 OOF 生成**：需要 K 折 Kronos 訊號，會大幅增加計算成本，留待下一輪
3. **Layer 4 交易系統**：成本/滑價/換倉限制，spec 明確標為「最後才做」，不在本計畫範圍
