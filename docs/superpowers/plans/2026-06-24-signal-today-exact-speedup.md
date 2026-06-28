# Signal Today Exact Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 加速 `python -m finetune_tw.signal_today` 的單日選股推論，同時保持同一組 DB / config / predictor 輸入下的訊號結果完全不變。

**Architecture:** 保留現有模型、抽樣參數、`BATCH_SIZE=64` 與 CLI 介面，不碰任何可能改變數值路徑的設定。加速只來自兩件事：把 `signal_today` 的逐檔 SQLite 查詢改成單次 window 查詢，以及把 `KronosPredictor.predict_batch()` 內部的前處理拆成可重用的 prepared API，讓 `signal_today` 不再重複做 pandas/時間特徵/標準化工作。

**Tech Stack:** Python、pandas/numpy、PyTorch、SQLite、pytest、既有 `finetune_tw` / `model.kronos` 程式碼。

## Global Constraints

- 不能改變 `signal_today.py` 的 CLI 參數、標題文字、top-k 選股規則、持倉建議邏輯。
- 不能改變模型數值路徑：`T=1.0`、`top_k=1`、`top_p=1.0`、`sample_count=1`、`BATCH_SIZE=64` 必須維持不變。
- `predict_batch()` 的 public behavior 必須保持相容；新 prepared API 是額外抽出，不是替換外部介面。
- bulk DB 查詢後，`signal_today` 仍必須依照呼叫端提供的 `symbols` 順序建 batch，避免因 tie-break 順序不同造成選股集合偏移。
- 測試只用 temp SQLite DB、假的 predictor 或 monkeypatch；不依賴 GPU、網路、HF 權重。
- 驗證標準是「完全一致」，不是 `approx`：能用 `==` 的地方就用 `==`，陣列則用 `np.testing.assert_allclose(..., rtol=0, atol=0)`。

---

### Task 1: 新增單次 window DB 查詢 helper

**Files:**
- Modify: `finetune_tw/db.py`
- Modify: `tests/finetune_tw/test_db.py`

**Interfaces:**
- Produces: `query_symbols_window(db_path: str, symbols: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame`
- Return columns: `["symbol", "date", "open", "high", "low", "close", "volume", "amount"]`

- [ ] **Step 1: 先寫失敗測試**

```python
# tests/finetune_tw/test_db.py
from finetune_tw.db import (
    init_db,
    upsert_prices,
    query_symbol,
    query_symbols_window,
    list_symbols,
    get_last_date,
)


def test_query_symbols_window_filters_symbols_and_dates(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(6))
    upsert_prices(db, "2317.TW", _make_df(6))
    upsert_prices(db, "2454.TW", _make_df(6))

    df = query_symbols_window(
        db,
        ["2330.TW", "2454.TW"],
        start="2024-01-03",
        end="2024-01-05",
    )

    assert list(df.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert sorted(df["symbol"].unique().tolist()) == ["2330.TW", "2454.TW"]
    assert all(df["date"] >= "2024-01-03")
    assert all(df["date"] <= "2024-01-05")


def test_query_symbols_window_empty_symbols_returns_empty_frame(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)

    df = query_symbols_window(db, [], start="2024-01-01", end="2024-01-05")

    assert df.empty
    assert list(df.columns) == [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_db.py -k "query_symbols_window" -v`
Expected: FAIL（`ImportError` 或 `AttributeError: cannot import name 'query_symbols_window'`）

- [ ] **Step 3: 在 `finetune_tw/db.py` 實作 bulk helper**

```python
def query_symbols_window(
    db_path: str,
    symbols: list[str],
    start: str = None,
    end: str = None,
) -> pd.DataFrame:
    columns = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    if not symbols:
        return pd.DataFrame(columns=columns)

    placeholders = ",".join("?" for _ in symbols)
    q = (
        "SELECT symbol,date,open,high,low,close,volume,amount "
        f"FROM daily_prices WHERE symbol IN ({placeholders})"
    )
    params: list = list(symbols)
    if start:
        q += " AND date>=?"
        params.append(start)
    if end:
        q += " AND date<=?"
        params.append(end)
    q += " ORDER BY symbol, date"

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(q, conn, params=params)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `pytest tests/finetune_tw/test_db.py -k "query_symbols_window" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/db.py tests/finetune_tw/test_db.py
git commit -m "feat(db): add bulk symbol window query for signal inference"
```

---

### Task 2: 抽出 prepared batch inference API，保留 `predict_batch()` 行為

**Files:**
- Modify: `model/kronos.py`
- Create: `tests/finetune_tw/test_kronos_predictor_batch.py`

**Interfaces:**
- Produces:
  - `KronosPredictor.prepare_batch_inputs(...) -> tuple[x_batch, x_stamp_batch, y_stamp_batch, means, stds, y_index_list]`
  - `KronosPredictor.predict_prepared_batch(...) -> list[pd.DataFrame]`
- Preserves: `KronosPredictor.predict_batch(...)` signature and return values

- [ ] **Step 1: 先寫失敗測試**

```python
# tests/finetune_tw/test_kronos_predictor_batch.py
import numpy as np
import pandas as pd

from model.kronos import KronosPredictor


_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _make_predictor_stub() -> KronosPredictor:
    predictor = KronosPredictor.__new__(KronosPredictor)
    predictor.price_cols = ["open", "high", "low", "close"]
    predictor.vol_col = "volume"
    predictor.amt_vol = "amount"
    predictor.clip = 5
    return predictor


def _make_df(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0 + offset, 11.0 + offset, 12.0 + offset],
            "high": [11.0 + offset, 12.0 + offset, 13.0 + offset],
            "low": [9.0 + offset, 10.0 + offset, 11.0 + offset],
            "close": [10.5 + offset, 11.5 + offset, 12.5 + offset],
            "volume": [100.0, 110.0, 120.0],
            "amount": [1000.0, 1100.0, 1200.0],
        }
    )


def test_prepare_batch_inputs_returns_current_normalization():
    predictor = _make_predictor_stub()
    df_list = [_make_df(0.0), _make_df(5.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=3))
    y_ts = pd.Series(pd.bdate_range("2024-01-04", periods=2))

    x_batch, x_stamp_batch, y_stamp_batch, means, stds, y_index_list = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
    )

    expected_x0 = df_list[0][_PRICE_COLUMNS].values.astype(np.float32)
    expected_mean0 = expected_x0.mean(axis=0)
    expected_std0 = expected_x0.std(axis=0)
    expected_norm0 = np.clip((expected_x0 - expected_mean0) / (expected_std0 + 1e-5), -5, 5)

    np.testing.assert_allclose(x_batch[0], expected_norm0, rtol=0, atol=0)
    np.testing.assert_allclose(means[0], expected_mean0, rtol=0, atol=0)
    np.testing.assert_allclose(stds[0], expected_std0, rtol=0, atol=0)
    assert x_stamp_batch.shape == (2, 3, 5)
    assert y_stamp_batch.shape == (2, 2, 5)
    assert list(y_index_list[0]) == list(y_ts)


def test_predict_prepared_batch_matches_predict_batch():
    predictor = _make_predictor_stub()
    df_list = [_make_df(0.0), _make_df(5.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=3))
    y_ts = pd.Series(pd.bdate_range("2024-01-04", periods=2))

    generated = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 10.0, 20.0], [1.5, 2.5, 3.5, 4.5, 11.0, 21.0]],
            [[5.0, 6.0, 7.0, 8.0, 30.0, 40.0], [5.5, 6.5, 7.5, 8.5, 31.0, 41.0]],
        ],
        dtype=np.float32,
    )

    def fake_generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose, return_all_samples=False):
        assert pred_len == 2
        assert return_all_samples is False
        return generated

    predictor.generate = fake_generate

    prepared = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
    )

    direct = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    prepared_out = predictor.predict_prepared_batch(
        *prepared,
        pred_len=2,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )

    assert [df["close"].tolist() for df in direct] == [df["close"].tolist() for df in prepared_out]
    assert [df.index.tolist() for df in direct] == [df.index.tolist() for df in prepared_out]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_kronos_predictor_batch.py -v`
Expected: FAIL（`AttributeError: 'KronosPredictor' object has no attribute 'prepare_batch_inputs'`）

- [ ] **Step 3: 在 `model/kronos.py` 抽出 prepared API**

把 `predict_batch()` 目前 for-loop 內的驗證、`calc_time_stamps()`、mean/std、`np.clip()` 抽到 `prepare_batch_inputs()`，再新增 `predict_prepared_batch()` 專門做 `generate()` + denorm + `DataFrame` 封裝：

```python
    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        y_index_list = []
        seq_lens = []
        y_lens = []

        for i in range(len(df_list)):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]
            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)
            y_index_list.append(pd.Index(y_timestamp))
            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)
        return x_batch, x_stamp_batch, y_stamp_batch, means, stds, y_index_list


    def predict_prepared_batch(
        self,
        x_batch,
        x_stamp_batch,
        y_stamp_batch,
        means,
        stds,
        y_index_list,
        pred_len,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    ):
        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        for i in range(len(means)):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(
                preds_i,
                columns=self.price_cols + [self.vol_col, self.amt_vol],
                index=y_index_list[i],
            )
            pred_dfs.append(pred_df)
        return pred_dfs
```

然後把 `predict_batch()` 改成：

```python
        prepared = self.prepare_batch_inputs(
            df_list=df_list,
            x_timestamp_list=x_timestamp_list,
            y_timestamp_list=y_timestamp_list,
            pred_len=pred_len,
        )
        return self.predict_prepared_batch(
            *prepared,
            pred_len=pred_len,
            T=T,
            top_k=top_k,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
```

- [ ] **Step 4: 跑測試確認 prepared API 與原行為一致**

Run: `pytest tests/finetune_tw/test_kronos_predictor_batch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/kronos.py tests/finetune_tw/test_kronos_predictor_batch.py
git commit -m "refactor(kronos): extract prepared batch inference path"
```

---

### Task 3: 用 bulk query + prepared API 重寫 `signal_today` 單日推論

**Files:**
- Modify: `finetune_tw/signal_today.py`
- Create: `tests/finetune_tw/test_signal_today.py`

**Interfaces:**
- Preserves: `get_signals_for_date(...) -> dict[str, float]`
- Produces:
  - `_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]`
  - `_load_signal_contexts(cfg, rebal_date, hold_days, symbols) -> list[tuple[sym, ctx_df, x_ts, y_ts]]`

- [ ] **Step 1: 先寫失敗測試**

```python
# tests/finetune_tw/test_signal_today.py
import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, query_symbol, upsert_prices
from finetune_tw.signal_today import get_signals_for_date


_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _make_price_frame(start: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            "amount": [100000.0] * len(closes),
        }
    )


class _FakePredictor:
    def _predict_from_frames(self, df_list, y_timestamp_list, pred_len):
        out = []
        for df, y_ts in zip(df_list, y_timestamp_list):
            last_close = float(df["close"].iloc[-1])
            close_path = [last_close * (1.0 + 0.01 * (i + 1)) for i in range(pred_len)]
            out.append(
                pd.DataFrame(
                    {
                        "open": close_path,
                        "high": close_path,
                        "low": close_path,
                        "close": close_path,
                        "volume": [0.0] * pred_len,
                        "amount": [0.0] * pred_len,
                    },
                    index=y_ts,
                )
            )
        return out

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_k, top_p, sample_count, verbose):
        return self._predict_from_frames(df_list, y_timestamp_list, pred_len)

    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        means = [float(df["close"].iloc[-1]) for df in df_list]
        return df_list, x_timestamp_list, y_timestamp_list, means, means, y_timestamp_list

    def predict_prepared_batch(self, df_list, x_timestamp_list, y_timestamp_list, means, stds, y_index_list, pred_len, T, top_k, top_p, sample_count, verbose):
        return self._predict_from_frames(df_list, y_timestamp_list, pred_len)


class _PreparedOnlyPredictor(_FakePredictor):
    def __init__(self):
        self.prepared_called = False

    def predict_batch(self, *args, **kwargs):
        raise AssertionError("legacy predict_batch path should not be used")

    def prepare_batch_inputs(self, df_list, x_timestamp_list, y_timestamp_list, pred_len):
        self.prepared_called = True
        return super().prepare_batch_inputs(df_list, x_timestamp_list, y_timestamp_list, pred_len)


def _legacy_get_signals_for_date(predictor, cfg, rebal_date, hold_days, symbols):
    batch_syms, batch_dfs, batch_xts, batch_yts = [], [], [], []
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)).strftime("%Y-%m-%d")
    y_ts = pd.date_range(rebal_date, periods=hold_days, freq="B")

    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=lookback_start, end=rebal_str)
        if len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:]
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        batch_syms.append(sym)
        batch_dfs.append(ctx_df)
        batch_xts.append(pd.to_datetime(ctx["date"]).reset_index(drop=True))
        batch_yts.append(pd.Series(y_ts))

    signals = {}
    preds = predictor.predict_batch(
        df_list=batch_dfs,
        x_timestamp_list=batch_xts,
        y_timestamp_list=batch_yts,
        pred_len=hold_days,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    for sym, pred, ctx_df in zip(batch_syms, preds, batch_dfs):
        last_close = float(ctx_df["close"].iloc[-1])
        signals[sym] = float(pred["close"].iloc[hold_days - 1]) / last_close - 1.0
    return signals


def test_get_signals_for_date_matches_legacy_path_exactly(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))
    upsert_prices(db, "1301", _make_price_frame("2024-01-01", [30, 31]))  # insufficient lookback

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _FakePredictor()
    rebal_date = pd.Timestamp("2024-01-08")
    symbols = ["1101", "1216", "1301"]

    expected = _legacy_get_signals_for_date(predictor, cfg, rebal_date, 3, symbols)
    actual = get_signals_for_date(predictor, cfg, rebal_date, 3, symbols)

    assert actual == expected
    assert actual == {"1101": 0.03, "1216": 0.03}


def test_get_signals_for_date_prefers_prepared_batch_api(tmp_path):
    db = str(tmp_path / "tw.db")
    init_db(db)
    upsert_prices(db, "1101", _make_price_frame("2024-01-01", [10, 11, 12, 13, 14, 15]))
    upsert_prices(db, "1216", _make_price_frame("2024-01-01", [20, 21, 22, 23, 24, 25]))

    cfg = Config(db_path=db, lookback_window=4, hold_days=3, pred_len=3)
    predictor = _PreparedOnlyPredictor()

    actual = get_signals_for_date(
        predictor,
        cfg,
        pd.Timestamp("2024-01-08"),
        3,
        ["1101", "1216"],
    )

    assert predictor.prepared_called is True
    assert actual == {"1101": 0.03, "1216": 0.03}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `pytest tests/finetune_tw/test_signal_today.py -v`
Expected: FAIL（`test_get_signals_for_date_prefers_prepared_batch_api` 會因舊實作仍呼叫 `predict_batch()` 而丟出 `AssertionError`）

- [ ] **Step 3: 在 `finetune_tw/signal_today.py` 實作 bulk context loader**

先把 import 改成包含新 helper：

```python
from finetune_tw.db import list_symbols, get_last_date, query_symbols_window
```

在檔案頂部常數區加入：

```python
_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
```

新增 helper，注意 `rows.groupby("symbol")` 之後仍要回到 `for sym in symbols:`，保留外部 symbol 順序：

```python
def _load_signal_contexts(
    cfg: Config,
    rebal_date: pd.Timestamp,
    hold_days: int,
    symbols: list[str],
) -> list[tuple[str, pd.DataFrame, pd.Series, pd.Series]]:
    rebal_str = rebal_date.strftime("%Y-%m-%d")
    lookback_start = (
        rebal_date - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    y_ts = pd.Series(pd.date_range(rebal_date, periods=hold_days, freq="B"))

    rows = query_symbols_window(
        cfg.db_path,
        symbols,
        start=lookback_start,
        end=rebal_str,
    )
    if rows.empty:
        return []

    grouped = {sym: grp.reset_index(drop=True) for sym, grp in rows.groupby("symbol", sort=False)}
    contexts = []
    for sym in symbols:
        df = grouped.get(sym)
        if df is None or len(df) < cfg.lookback_window:
            continue
        ctx = df.iloc[-cfg.lookback_window:].reset_index(drop=True)
        ctx_df = ctx[_PRICE_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue
        x_ts = pd.to_datetime(ctx["date"]).reset_index(drop=True)
        contexts.append((sym, ctx_df, x_ts, y_ts.copy()))
    return contexts
```

- [ ] **Step 4: 讓 `get_signals_for_date()` 走 prepared API，但保留相容 fallback**

把現有逐檔 `query_symbol()` 迴圈改成：

```python
    contexts = _load_signal_contexts(cfg, rebal_date, hold_days, symbols)
    batch_syms = [sym for sym, _, _, _ in contexts]
    batch_dfs = [ctx_df for _, ctx_df, _, _ in contexts]
    batch_xts = [x_ts for _, _, x_ts, _ in contexts]
    batch_yts = [y_ts for _, _, _, y_ts in contexts]
```

把 batch inference 改成：

```python
        for b in range(0, len(batch_syms), BATCH_SIZE):
            df_slice = batch_dfs[b : b + BATCH_SIZE]
            xt_slice = batch_xts[b : b + BATCH_SIZE]
            yt_slice = batch_yts[b : b + BATCH_SIZE]

            if hasattr(predictor, "prepare_batch_inputs") and hasattr(predictor, "predict_prepared_batch"):
                prepared = predictor.prepare_batch_inputs(
                    df_list=df_slice,
                    x_timestamp_list=xt_slice,
                    y_timestamp_list=yt_slice,
                    pred_len=hold_days,
                )
                preds = predictor.predict_prepared_batch(
                    *prepared,
                    pred_len=hold_days,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
            else:
                preds = predictor.predict_batch(
                    df_list=df_slice,
                    x_timestamp_list=xt_slice,
                    y_timestamp_list=yt_slice,
                    pred_len=hold_days,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
```

保留原本的回報率計算式：

```python
                    ret = float(pred["close"].iloc[hold_days - 1]) / last_close - 1.0
```

- [ ] **Step 5: 跑 targeted regression tests**

Run: `pytest tests/finetune_tw/test_db.py tests/finetune_tw/test_kronos_predictor_batch.py tests/finetune_tw/test_signal_today.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/signal_today.py tests/finetune_tw/test_signal_today.py
git commit -m "perf(signal_today): bulk-load contexts and reuse prepared batch inference"
```

---

### Task 4: 最後驗證與人工 smoke check

**Files:**
- Modify: none
- Test: `tests/finetune_tw/test_db.py`, `tests/finetune_tw/test_kronos_predictor_batch.py`, `tests/finetune_tw/test_signal_today.py`

- [ ] **Step 1: 跑完整相關 pytest**

Run: `pytest tests/finetune_tw/test_db.py tests/finetune_tw/test_kronos_predictor_batch.py tests/finetune_tw/test_signal_today.py tests/finetune_tw/test_signal.py -v`
Expected: PASS

- [ ] **Step 2: 確認工作樹乾淨，避免遺漏未追蹤修改**

```bash
git status
```

Expected: `nothing to commit, working tree clean`
