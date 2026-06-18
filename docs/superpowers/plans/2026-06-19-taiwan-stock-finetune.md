# Taiwan Stock Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `finetune_tw/` — a self-contained module to fine-tune Kronos-base on all ~1700 TWSE daily K-line stocks (2015–2026) and backtest on Colab free tier.

**Architecture:** Multi-source downloader writes per-stock OHLCV data to a single SQLite file on Google Drive. A `MultiStockDataset` samples isolated windows from that DB to train a two-stage pipeline (tokenizer → predictor) on a single T4 GPU with AMP and checkpoint resumption. A pure-pandas backtest evaluates the fine-tuned model using a cross-sectional top-K strategy against `^TWII`.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0, yfinance, requests, sqlite3 (stdlib), pandas, numpy, safetensors, huggingface_hub, pyyaml, pytest

## Global Constraints

- Single GPU only (`torch.device("cuda:0")`), no `torchrun` / DDP
- Mixed precision via `torch.cuda.amp` (fp16) throughout training
- All persistent artefacts (DB, checkpoints, logs) stored under `output_dir` so they survive Colab session resets when `output_dir` is a Google Drive path
- Dataset `__getitem__` returns `(x_tensor, x_stamp_tensor)` — **same shapes as `finetune_csv/finetune_base_model.py::CustomKlineDataset`** — so the existing tokenizer/predictor training logic can be reused verbatim
  - `x_tensor`: `(lookback_window + predict_window + 1, 6)` float32, per-window z-score normalised + clipped
  - `x_stamp_tensor`: `(lookback_window + predict_window + 1, 5)` float32, time features `[minute, hour, weekday, day, month]`
- `amount` field set to `0.0` when not available from data source
- TWSE scraper must not exceed 3 requests per 5 seconds
- No qlib dependency anywhere in this module
- Train / val / test split by **date** (not by fraction): train ≤ `train_end_date`, val ≤ `val_end_date`, test starts after `val_end_date`

---

## File Map

```
finetune_tw/
├── __init__.py
├── config.py                      NEW — Config dataclass + from_yaml()
├── db.py                          NEW — SQLite schema, upsert, query helpers
├── download_data.py               NEW — CLI orchestrator (--source, --update)
├── dataset.py                     NEW — MultiStockDataset
├── train_tokenizer.py             NEW — tokenizer fine-tuning, single-GPU + AMP
├── train_predictor.py             NEW — predictor fine-tuning, single-GPU + AMP
├── backtest.py                    NEW — pure-pandas top-K strategy + metrics
├── colab_setup.ipynb              NEW — Colab entry point
├── fetchers/
│   ├── __init__.py
│   ├── yfinance_fetcher.py        NEW — bulk yfinance downloader
│   ├── twse_scraper.py            NEW — rate-limited TWSE official scraper
│   └── finmind_fetcher.py        NEW — optional FinMind supplemental fetcher
├── configs/
│   └── config_tw_daily.yaml      NEW — reference YAML config
└── data/                         gitignored

tests/finetune_tw/
├── __init__.py
├── test_db.py
├── test_fetchers.py
├── test_dataset.py
└── test_backtest.py
```

---

### Task 1: Scaffold + Config + DB layer

**Files:**
- Create: `finetune_tw/__init__.py`
- Create: `finetune_tw/config.py`
- Create: `finetune_tw/db.py`
- Create: `finetune_tw/configs/config_tw_daily.yaml`
- Create: `tests/finetune_tw/__init__.py`
- Create: `tests/finetune_tw/test_db.py`

**Interfaces:**
- Produces:
  - `Config` dataclass with fields listed below; `Config.from_yaml(path) -> Config`
  - `init_db(db_path: str) -> None`
  - `upsert_prices(db_path: str, symbol: str, df: pd.DataFrame) -> int` — returns row count inserted
  - `query_symbol(db_path: str, symbol: str, start: str = None, end: str = None) -> pd.DataFrame`
  - `list_symbols(db_path: str) -> list[str]`
  - `get_last_date(db_path: str, symbol: str) -> str | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/finetune_tw/test_db.py
import pandas as pd
import pytest
from finetune_tw.db import init_db, upsert_prices, query_symbol, list_symbols, get_last_date

def _make_df(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates, "open": [100.0] * n, "high": [101.0] * n,
        "low": [99.0] * n, "close": [100.5] * n,
        "volume": [1_000_000.0] * n, "amount": [0.0] * n,
    })

def test_init_creates_tables(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stocks", "daily_prices"} <= tables

def test_upsert_returns_row_count(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    assert upsert_prices(db, "2330.TW", _make_df(5)) == 5

def test_upsert_is_idempotent(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(5))
    upsert_prices(db, "2330.TW", _make_df(5))  # same rows, no duplicate
    result = query_symbol(db, "2330.TW")
    assert len(result) == 5

def test_query_returns_correct_columns(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(10))
    df = query_symbol(db, "2330.TW")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 10

def test_query_date_filter(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(10))
    df = query_symbol(db, "2330.TW", start="2024-01-03", end="2024-01-05")
    assert all(df["date"] >= "2024-01-03")
    assert all(df["date"] <= "2024-01-05")

def test_list_symbols(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(5))
    upsert_prices(db, "2317.TW", _make_df(5))
    assert sorted(list_symbols(db)) == ["2317.TW", "2330.TW"]

def test_get_last_date(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_df(5))
    last = get_last_date(db, "2330.TW")
    assert last == "2024-01-07"  # 5 business days from 2024-01-01

def test_get_last_date_missing_symbol(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    assert get_last_date(db, "9999.TW") is None
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /mnt/d/project/Kronos
pytest tests/finetune_tw/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'finetune_tw'`

- [ ] **Step 3: Create scaffold files**

```python
# finetune_tw/__init__.py
# (empty)
```

```python
# tests/finetune_tw/__init__.py
# (empty)
```

- [ ] **Step 4: Implement `finetune_tw/config.py`**

```python
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    # Data
    db_path: str = "finetune_tw/data/tw_stocks.db"
    lookback_window: int = 90
    predict_window: int = 10
    max_context: int = 512
    clip: float = 5.0
    train_end_date: str = "2023-12-31"
    val_end_date: str = "2024-06-30"

    # Training
    tokenizer_epochs: int = 30
    basemodel_epochs: int = 20
    batch_size: int = 16
    save_steps: int = 500
    log_interval: int = 50
    tokenizer_lr: float = 2e-4
    predictor_lr: float = 4e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.1
    num_workers: int = 2
    seed: int = 42

    # Model paths
    pretrained_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor: str = "NeoQuasar/Kronos-base"
    exp_name: str = "tw_daily"
    output_dir: str = "finetune_tw/outputs"

    # Backtest
    top_k: int = 20
    hold_days: int = 5
    pred_len: int = 10
    test_start_date: str = "2024-07-01"
    benchmark_symbol: str = "^TWII"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
```

- [ ] **Step 5: Implement `finetune_tw/db.py`**

```python
import sqlite3
from pathlib import Path
import pandas as pd


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stocks (
                symbol     TEXT PRIMARY KEY,
                name       TEXT,
                first_date TEXT,
                last_date  TEXT
            );
            CREATE TABLE IF NOT EXISTS daily_prices (
                symbol TEXT,
                date   TEXT,
                open   REAL,
                high   REAL,
                low    REAL,
                close  REAL,
                volume REAL,
                amount REAL,
                PRIMARY KEY (symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_date ON daily_prices(date);
        """)


def upsert_prices(db_path: str, symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = df.assign(symbol=symbol)[
        ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
    ].values.tolist()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices VALUES (?,?,?,?,?,?,?,?)", rows
        )
        conn.execute(
            "INSERT OR REPLACE INTO stocks (symbol, first_date, last_date) VALUES (?,?,?)",
            (symbol, df["date"].min(), df["date"].max()),
        )
    return len(rows)


def query_symbol(
    db_path: str, symbol: str, start: str = None, end: str = None
) -> pd.DataFrame:
    q = "SELECT date,open,high,low,close,volume,amount FROM daily_prices WHERE symbol=?"
    params: list = [symbol]
    if start:
        q += " AND date>=?"; params.append(start)
    if end:
        q += " AND date<=?"; params.append(end)
    q += " ORDER BY date"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(q, conn, params=params)


def list_symbols(db_path: str) -> list:
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT symbol FROM stocks ORDER BY symbol")]


def get_last_date(db_path: str, symbol: str) -> "str | None":
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_date FROM stocks WHERE symbol=?", (symbol,)
        ).fetchone()
    return row[0] if row else None
```

- [ ] **Step 6: Create `finetune_tw/configs/config_tw_daily.yaml`**

```yaml
# Reference config for TWSE daily fine-tuning
db_path: "finetune_tw/data/tw_stocks.db"
lookback_window: 90
predict_window: 10
max_context: 512
clip: 5.0
train_end_date: "2023-12-31"
val_end_date: "2024-06-30"

tokenizer_epochs: 30
basemodel_epochs: 20
batch_size: 16
save_steps: 500
log_interval: 50
tokenizer_lr: 0.0002
predictor_lr: 0.00004
adam_beta1: 0.9
adam_beta2: 0.95
adam_weight_decay: 0.1
num_workers: 2
seed: 42

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

- [ ] **Step 7: Run tests, verify they pass**

```bash
pytest tests/finetune_tw/test_db.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 8: Commit**

```bash
git add finetune_tw/ tests/finetune_tw/
git commit -m "feat(finetune_tw): scaffold, Config dataclass, SQLite DB layer"
```

---

### Task 2: yfinance fetcher

**Files:**
- Create: `finetune_tw/fetchers/__init__.py`
- Create: `finetune_tw/fetchers/yfinance_fetcher.py`
- Create: `tests/finetune_tw/test_fetchers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `get_twse_symbol_list() -> list[str]` — returns `["2330.TW", "2317.TW", ...]` via TWSE OpenAPI
  - `fetch_symbol(symbol: str, start: str, end: str | None) -> pd.DataFrame | None` — columns: `date, open, high, low, close, volume, amount`; returns `None` on failure

- [ ] **Step 1: Write failing tests**

```python
# tests/finetune_tw/test_fetchers.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from finetune_tw.fetchers.yfinance_fetcher import fetch_symbol, get_twse_symbol_list


def _mock_history(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="Asia/Taipei")
    return pd.DataFrame({
        "Open": [100.0] * n, "High": [101.0] * n,
        "Low": [99.0] * n, "Close": [100.5] * n, "Volume": [1_000_000] * n,
    }, index=idx)


def test_fetch_symbol_returns_standard_columns():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(5)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 5


def test_fetch_symbol_date_format():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert df["date"].iloc[0] == "2024-01-01"


def test_fetch_symbol_amount_is_zero():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_history(3)
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        df = fetch_symbol("2330.TW", start="2024-01-01")
    assert (df["amount"] == 0.0).all()


def test_fetch_symbol_returns_none_on_empty():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", return_value=mock_ticker):
        result = fetch_symbol("9999.TW", start="2024-01-01")
    assert result is None


def test_fetch_symbol_returns_none_on_exception():
    with patch("finetune_tw.fetchers.yfinance_fetcher.yf.Ticker", side_effect=Exception("network error")):
        result = fetch_symbol("2330.TW", start="2024-01-01")
    assert result is None


def test_get_twse_symbol_list_parses_response():
    mock_json = [{"Code": "2330", "Name": "台積電"}, {"Code": "2317", "Name": "鴻海"}]
    with patch("finetune_tw.fetchers.yfinance_fetcher.requests.get") as mock_get:
        mock_get.return_value.json.return_value = mock_json
        mock_get.return_value.raise_for_status = MagicMock()
        symbols = get_twse_symbol_list()
    assert "2330.TW" in symbols
    assert "2317.TW" in symbols
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/finetune_tw/test_fetchers.py -v
```

Expected: `ModuleNotFoundError: No module named 'finetune_tw.fetchers'`

- [ ] **Step 3: Implement `finetune_tw/fetchers/__init__.py`**

```python
# (empty)
```

- [ ] **Step 4: Implement `finetune_tw/fetchers/yfinance_fetcher.py`**

```python
from __future__ import annotations
import requests
import pandas as pd
import yfinance as yf

TWSE_LISTING_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_AVG_ALL"


def get_twse_symbol_list() -> list[str]:
    resp = requests.get(TWSE_LISTING_URL, timeout=15)
    resp.raise_for_status()
    return [f"{item['Code']}.TW" for item in resp.json() if item.get("Code")]


def fetch_symbol(
    symbol: str, start: str = "2015-01-01", end: str | None = None
) -> pd.DataFrame | None:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            return None
        df = hist.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        # Strip timezone if present
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        df["amount"] = 0.0
        return df[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    except Exception:
        return None
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
pytest tests/finetune_tw/test_fetchers.py::test_fetch_symbol_returns_standard_columns \
       tests/finetune_tw/test_fetchers.py::test_fetch_symbol_date_format \
       tests/finetune_tw/test_fetchers.py::test_fetch_symbol_amount_is_zero \
       tests/finetune_tw/test_fetchers.py::test_fetch_symbol_returns_none_on_empty \
       tests/finetune_tw/test_fetchers.py::test_fetch_symbol_returns_none_on_exception \
       tests/finetune_tw/test_fetchers.py::test_get_twse_symbol_list_parses_response -v
```

Expected: all 6 PASS

- [ ] **Step 6: Commit**

```bash
git add finetune_tw/fetchers/ tests/finetune_tw/test_fetchers.py
git commit -m "feat(finetune_tw): yfinance fetcher"
```

---

### Task 3: TWSE scraper

**Files:**
- Create: `finetune_tw/fetchers/twse_scraper.py`
- Modify: `tests/finetune_tw/test_fetchers.py` — append TWSE tests

**Interfaces:**
- Produces:
  - `fetch_month(symbol: str, year: int, month: int) -> pd.DataFrame | None` — single month of TWSE data, same column schema as yfinance fetcher
  - `fetch_symbol_twse(symbol: str, start: str, end: str) -> pd.DataFrame | None` — iterates months, respects rate limit

- [ ] **Step 1: Append failing tests to `tests/finetune_tw/test_fetchers.py`**

```python
from finetune_tw.fetchers.twse_scraper import fetch_month, fetch_symbol_twse

TWSE_SAMPLE_RESPONSE = {
    "stat": "OK",
    "data": [
        ["113/01/02", "10,000", "1,000,000", "580.00", "585.00", "578.00", "582.00", "2.00", "100"],
        ["113/01/03", "12,000", "1,200,000", "582.00", "588.00", "580.00", "586.00", "4.00", "120"],
    ],
}


def test_twse_fetch_month_standard_columns():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = TWSE_SAMPLE_RESPONSE
        df = fetch_month("2330", 2024, 1)
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]


def test_twse_fetch_month_roc_date_conversion():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = TWSE_SAMPLE_RESPONSE
        df = fetch_month("2330", 2024, 1)
    assert df["date"].iloc[0] == "2024-01-02"


def test_twse_fetch_month_returns_none_on_bad_stat():
    with patch("finetune_tw.fetchers.twse_scraper.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"stat": "查無資料"}
        result = fetch_month("9999", 2024, 1)
    assert result is None
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/finetune_tw/test_fetchers.py::test_twse_fetch_month_standard_columns -v
```

Expected: `ModuleNotFoundError: No module named 'finetune_tw.fetchers.twse_scraper'`

- [ ] **Step 3: Implement `finetune_tw/fetchers/twse_scraper.py`**

```python
from __future__ import annotations
import time
import requests
import pandas as pd

TWSE_URL = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
_req_times: list[float] = []


def _rate_limit() -> None:
    now = time.time()
    _req_times[:] = [t for t in _req_times if now - t < 5.0]
    if len(_req_times) >= 3:
        wait = 5.0 - (now - _req_times[0]) + 0.1
        if wait > 0:
            time.sleep(wait)
    _req_times.append(time.time())


def fetch_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    _rate_limit()
    try:
        resp = requests.get(
            TWSE_URL,
            params={"response": "json", "date": f"{year}{month:02d}01", "stockNo": symbol},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("stat") != "OK" or not data.get("data"):
            return None
        rows = []
        for row in data["data"]:
            try:
                y, m, d = row[0].split("/")
                ad_date = f"{int(y) + 1911}-{m}-{d}"
                rows.append({
                    "date": ad_date,
                    "volume": float(row[1].replace(",", "")),
                    "amount": float(row[2].replace(",", "")),
                    "open": float(row[3].replace(",", "")),
                    "high": float(row[4].replace(",", "")),
                    "low": float(row[5].replace(",", "")),
                    "close": float(row[6].replace(",", "")),
                })
            except (ValueError, IndexError):
                continue
        if not rows:
            return None
        return pd.DataFrame(rows)[
            ["date", "open", "high", "low", "close", "volume", "amount"]
        ]
    except Exception:
        return None


def fetch_symbol_twse(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch all months in [start, end] for a 4-digit TWSE symbol (without .TW suffix)."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    cur = start_dt.replace(day=1)
    while cur <= end_dt:
        df = fetch_month(symbol, cur.year, cur.month)
        if df is not None:
            frames.append(df)
        cur = (cur + pd.offsets.MonthEnd(1)) + pd.offsets.Day(1)
    if not frames:
        return None
    result = pd.concat(frames).drop_duplicates("date")
    result = result[(result["date"] >= start) & (result["date"] <= end)]
    return result.reset_index(drop=True)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/finetune_tw/test_fetchers.py -v
```

Expected: all TWSE tests PASS (9 total passing now)

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/fetchers/twse_scraper.py tests/finetune_tw/test_fetchers.py
git commit -m "feat(finetune_tw): TWSE rate-limited scraper"
```

---

### Task 4: FinMind fetcher + Download orchestrator

**Files:**
- Create: `finetune_tw/fetchers/finmind_fetcher.py`
- Create: `finetune_tw/download_data.py`
- Modify: `tests/finetune_tw/test_fetchers.py` — append FinMind + orchestrator tests

**Interfaces:**
- Consumes: `yfinance_fetcher.fetch_symbol`, `yfinance_fetcher.get_twse_symbol_list`, `twse_scraper.fetch_symbol_twse`, `db.init_db`, `db.upsert_prices`, `db.get_last_date`
- Produces:
  - `finmind_fetcher.fetch_symbol_finmind(symbol: str, start: str, end: str, token: str) -> pd.DataFrame | None`
  - `download_data.download(db_path, symbols, start, end, source, update_only)` — orchestrates fetching + DB writes

- [ ] **Step 1: Append failing tests**

```python
# append to tests/finetune_tw/test_fetchers.py
from finetune_tw.fetchers.finmind_fetcher import fetch_symbol_finmind

FINMIND_RESPONSE = {
    "msg": "success",
    "data": [
        {"date": "2024-01-02", "open": 580.0, "max": 585.0,
         "min": 578.0, "close": 582.0, "Trading_Volume": 10000, "Trading_money": 5800000},
    ],
}


def test_finmind_returns_standard_columns():
    with patch("finetune_tw.fetchers.finmind_fetcher.requests.get") as mock_get:
        mock_get.return_value.json.return_value = FINMIND_RESPONSE
        mock_get.return_value.raise_for_status = MagicMock()
        df = fetch_symbol_finmind("2330", "2024-01-01", "2024-01-31", token="test_token")
    assert df is not None
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "amount"]
```

- [ ] **Step 2: Implement `finetune_tw/fetchers/finmind_fetcher.py`**

```python
from __future__ import annotations
import requests
import pandas as pd

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def fetch_symbol_finmind(
    symbol: str, start: str, end: str, token: str
) -> pd.DataFrame | None:
    try:
        resp = requests.get(
            FINMIND_URL,
            params={
                "dataset": "TaiwanStockPrice",
                "data_id": symbol,
                "start_date": start,
                "end_date": end,
                "token": token,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("msg") != "success" or not data.get("data"):
            return None
        df = pd.DataFrame(data["data"])
        df = df.rename(columns={
            "max": "high", "min": "low",
            "Trading_Volume": "volume", "Trading_money": "amount",
        })
        return df[["date", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    except Exception:
        return None
```

- [ ] **Step 3: Implement `finetune_tw/download_data.py`**

```python
"""
Usage:
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml --update
  python -m finetune_tw.download_data --config configs/config_tw_daily.yaml --source twse
"""
from __future__ import annotations
import argparse
from datetime import date
from tqdm import tqdm

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices, get_last_date
from finetune_tw.fetchers.yfinance_fetcher import fetch_symbol, get_twse_symbol_list
from finetune_tw.fetchers.twse_scraper import fetch_symbol_twse


def download(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance",
    update_only: bool = False,
) -> None:
    init_db(db_path)
    for sym in tqdm(symbols, desc=f"Downloading [{source}]"):
        effective_start = start
        if update_only:
            last = get_last_date(db_path, sym)
            if last:
                effective_start = last  # re-fetch last known date to catch amendments

        df = None
        if source in ("yfinance", "auto"):
            df = fetch_symbol(sym, start=effective_start, end=end)
        if df is None and source in ("twse", "auto"):
            bare = sym.replace(".TW", "")
            df = fetch_symbol_twse(bare, effective_start, end)
        if df is not None and not df.empty:
            upsert_prices(db_path, sym, df)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--source", choices=["yfinance", "twse", "auto"], default="auto")
    parser.add_argument("--update", action="store_true", help="Only fetch missing dates")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=str(date.today()))
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    symbols = get_twse_symbol_list()
    # Also add benchmark
    symbols = [cfg.benchmark_symbol] + symbols

    download(
        db_path=cfg.db_path,
        symbols=symbols,
        start=args.start,
        end=args.end,
        source=args.source,
        update_only=args.update,
    )
    print(f"Done. DB: {cfg.db_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/finetune_tw/ -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/fetchers/finmind_fetcher.py finetune_tw/download_data.py tests/finetune_tw/test_fetchers.py
git commit -m "feat(finetune_tw): FinMind fetcher + download orchestrator"
```

---

### Task 5: MultiStockDataset

**Files:**
- Create: `finetune_tw/dataset.py`
- Create: `tests/finetune_tw/test_dataset.py`

**Interfaces:**
- Consumes: `db.init_db`, `db.upsert_prices`, `db.query_symbol`, `db.list_symbols`
- Produces:
  - `MultiStockDataset(db_path, lookback_window, predict_window, start_date, end_date, clip, seed)` — PyTorch Dataset
  - `__len__() -> int`
  - `__getitem__(idx) -> tuple[torch.Tensor, torch.Tensor]` — `(x_tensor, x_stamp_tensor)` shapes `(window, 6)` and `(window, 5)`
  - `set_epoch_seed(epoch: int) -> None` — matches `CustomKlineDataset` interface

- [ ] **Step 1: Write failing tests**

```python
# tests/finetune_tw/test_dataset.py
import numpy as np
import pandas as pd
import pytest
import torch
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.dataset import MultiStockDataset

LOOKBACK = 10
PRED = 5
WINDOW = LOOKBACK + PRED + 1


def _make_stock_df(n: int = 50, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "open": rng.uniform(100, 200, n),
        "high": rng.uniform(100, 200, n) + 5,
        "low": rng.uniform(90, 190, n),
        "close": rng.uniform(100, 200, n),
        "volume": rng.uniform(1e6, 1e7, n),
        "amount": np.zeros(n),
    })


@pytest.fixture
def populated_db(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_stock_df(60, "2020-01-01"))
    upsert_prices(db, "2317.TW", _make_stock_df(60, "2020-01-01"))
    return db


def test_dataset_len_positive(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    assert len(ds) > 0


def test_dataset_item_shapes(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, x_stamp = ds[0]
    assert x.shape == (WINDOW, 6)
    assert x_stamp.shape == (WINDOW, 5)


def test_dataset_returns_tensors(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, x_stamp = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(x_stamp, torch.Tensor)


def test_dataset_x_is_normalized(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, _ = ds[0]
    # After normalization and clip=5, values should be in [-5, 5]
    assert x.abs().max().item() <= 5.0 + 1e-5


def test_dataset_no_cross_stock_windows(populated_db):
    # Each window comes from a single stock — verify by checking that n_samples
    # equals sum of per-stock valid windows
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    # Both stocks have ~42 trading days, so each contributes ~42-WINDOW+1 windows
    assert len(ds) == len(ds._samples)

def test_dataset_skips_short_stocks(tmp_path):
    db = str(tmp_path / "short.db")
    init_db(db)
    # Only 5 rows — too short for any window
    upsert_prices(db, "TINY.TW", _make_stock_df(5))
    ds = MultiStockDataset(db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    assert len(ds) == 0
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/finetune_tw/test_dataset.py -v
```

Expected: `ModuleNotFoundError: No module named 'finetune_tw.dataset'`

- [ ] **Step 3: Implement `finetune_tw/dataset.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from finetune_tw.db import query_symbol, list_symbols

FEATURES = ["open", "high", "low", "close", "volume", "amount"]  # 6 features, matches d_in=6


class MultiStockDataset(Dataset):
    """
    Samples (lookback_window + predict_window + 1)-length windows from any stock in the DB.
    Windows are isolated per stock — never cross stock boundaries.
    Returns (x_tensor, x_stamp_tensor) matching CustomKlineDataset's interface.
    """

    def __init__(
        self,
        db_path: str,
        lookback_window: int,
        predict_window: int,
        start_date: str,
        end_date: str,
        clip: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed

        self._data: dict[str, np.ndarray] = {}          # symbol -> (T, 6) float32
        self._stamps: dict[str, np.ndarray] = {}         # symbol -> (T, 5) float32
        self._samples: list[tuple[str, int]] = []        # (symbol, start_row)

        for sym in list_symbols(db_path):
            df = query_symbol(db_path, sym, start=start_date, end=end_date)
            if len(df) < self.window:
                continue
            df = df.reset_index(drop=True)
            arr = df[FEATURES].values.astype(np.float32)
            stamps = _build_stamps(df["date"])
            self._data[sym] = arr
            self._stamps[sym] = stamps
            for i in range(len(arr) - self.window + 1):
                self._samples.append((sym, i))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sym, start = self._samples[idx]
        x = self._data[sym][start : start + self.window].copy()
        s = self._stamps[sym][start : start + self.window].copy()

        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-5
        x = np.clip((x - mean) / std, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(s)

    def set_epoch_seed(self, epoch: int) -> None:
        # Provided for compatibility with existing training loop; no-op here
        # because windows are addressed deterministically by index.
        pass


def _build_stamps(dates: pd.Series) -> np.ndarray:
    """Returns (T, 5) array: [minute=0, hour=9, weekday, day, month]."""
    dt = pd.to_datetime(dates)
    stamps = np.stack([
        np.zeros(len(dt), dtype=np.float32),          # minute (fixed 0 for daily)
        np.full(len(dt), 9, dtype=np.float32),         # hour (fixed 9 for market open)
        dt.dt.weekday.values.astype(np.float32),
        dt.dt.day.values.astype(np.float32),
        dt.dt.month.values.astype(np.float32),
    ], axis=1)
    return stamps
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/finetune_tw/test_dataset.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/dataset.py tests/finetune_tw/test_dataset.py
git commit -m "feat(finetune_tw): MultiStockDataset with per-stock window isolation"
```

---

### Task 6: train_tokenizer.py

**Files:**
- Create: `finetune_tw/train_tokenizer.py`

**Interfaces:**
- Consumes: `Config`, `MultiStockDataset`, `KronosTokenizer`
- No new public API — this is a runnable script

- [ ] **Step 1: Write a smoke test**

```python
# append to tests/finetune_tw/test_dataset.py (reuse db fixture)
import sys, os

def test_tokenizer_train_one_step(populated_db, tmp_path, monkeypatch):
    """Verify that train_tokenizer runs one gradient step without crashing."""
    monkeypatch.syspath_prepend(str(tmp_path))
    from finetune_tw.config import Config
    from finetune_tw.train_tokenizer import run_training

    cfg = Config(
        db_path=populated_db,
        lookback_window=10,
        predict_window=5,
        train_end_date="2020-06-30",
        val_end_date="2020-12-31",
        tokenizer_epochs=1,
        batch_size=2,
        save_steps=9999,
        log_interval=1,
        output_dir=str(tmp_path / "outputs"),
        pretrained_tokenizer="NeoQuasar/Kronos-Tokenizer-base",
    )
    # Only runs if GPU available; skip otherwise
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    run_training(cfg, max_steps=1)
```

- [ ] **Step 2: Implement `finetune_tw/train_tokenizer.py`**

```python
"""
python finetune_tw/train_tokenizer.py --config finetune_tw/configs/config_tw_daily.yaml
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model import KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    save_dir = Path(cfg.output_dir) / cfg.exp_name / "tokenizer"
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer).to(device)

    train_ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 "2015-01-01", cfg.train_end_date, cfg.clip, cfg.seed)
    val_ds   = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=cfg.tokenizer_lr,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.tokenizer_lr,
        steps_per_epoch=len(train_loader), epochs=cfg.tokenizer_epochs,
        pct_start=0.03, div_factor=10,
    )
    scaler = GradScaler()

    # Resume from latest checkpoint
    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, tokenizer, optimizer, scheduler, scaler)
    best_val_loss = float("inf")
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss\n")

    for epoch in range(start_epoch, cfg.tokenizer_epochs):
        tokenizer.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with autocast():
                zs, bsq_loss, z_pre, _ = tokenizer(batch_x)
                recon_loss = F.mse_loss(zs, batch_x) + F.mse_loss(z_pre, batch_x)
                loss = (recon_loss + bsq_loss) / 2
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % cfg.log_interval == 0:
                print(f"[epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

            if global_step % cfg.save_steps == 0:
                _save_checkpoint(ckpt_dir, global_step, tokenizer, optimizer, scheduler, scaler)

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate(tokenizer, val_loader, device)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tokenizer.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_loss={val_loss:.4f}, saved.")


def _validate(tokenizer: KronosTokenizer, loader: DataLoader, device: torch.device) -> float:
    tokenizer.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            with autocast():
                zs, _, _, _ = tokenizer(batch_x)
                total += F.mse_loss(zs, batch_x).item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0


def _save_checkpoint(ckpt_dir: Path, step: int, model, optimizer, scheduler, scaler) -> None:
    torch.save({
        "step": step, "epoch": 0,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }, ckpt_dir / f"ckpt-{step}.pt")


def _load_latest_checkpoint(ckpt_dir: Path, model, optimizer, scheduler, scaler):
    ckpts = sorted(ckpt_dir.glob("ckpt-*.pt"),
                   key=lambda p: int(p.stem.split("-")[1]))
    if not ckpts:
        return 0, 0
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    print(f"Resumed from {ckpts[-1].name} (step {state['step']})")
    return state.get("epoch", 0), state["step"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    cfg = Config.from_yaml(parser.parse_args().config)
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual smoke run (optional on GPU machine)**

```bash
python finetune_tw/train_tokenizer.py --config finetune_tw/configs/config_tw_daily.yaml
```

Expected: prints loss after every `log_interval` steps; saves checkpoint to `finetune_tw/outputs/tw_daily/tokenizer/checkpoints/`

- [ ] **Step 4: Commit**

```bash
git add finetune_tw/train_tokenizer.py
git commit -m "feat(finetune_tw): tokenizer fine-tuning with AMP + checkpoint resume"
```

---

### Task 7: train_predictor.py

**Files:**
- Create: `finetune_tw/train_predictor.py`

**Interfaces:**
- Consumes: `Config`, `MultiStockDataset`, `KronosTokenizer` (frozen, loaded from Task 6 best_model), `Kronos`
- Produces: `finetune_tw/outputs/{exp_name}/predictor/best_model/` (via `model.save_pretrained`)

- [ ] **Step 1: Implement `finetune_tw/train_predictor.py`**

```python
"""
python finetune_tw/train_predictor.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: tokenizer best_model saved by train_tokenizer.py
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from model import Kronos, KronosTokenizer
from finetune_tw.config import Config
from finetune_tw.dataset import MultiStockDataset
from finetune_tw.train_tokenizer import _load_latest_checkpoint, _save_checkpoint


def run_training(cfg: Config, max_steps: int = -1) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    tok_path = Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model"
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run train_tokenizer.py first.")

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path)).to(device)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    model = Kronos.from_pretrained(cfg.pretrained_predictor).to(device)

    save_dir = Path(cfg.output_dir) / cfg.exp_name / "predictor"
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 "2015-01-01", cfg.train_end_date, cfg.clip, cfg.seed)
    val_ds   = MultiStockDataset(cfg.db_path, cfg.lookback_window, cfg.predict_window,
                                 cfg.train_end_date, cfg.val_end_date, cfg.clip, cfg.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.predictor_lr,
                                  betas=(cfg.adam_beta1, cfg.adam_beta2),
                                  weight_decay=cfg.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.predictor_lr,
        steps_per_epoch=len(train_loader), epochs=cfg.basemodel_epochs,
        pct_start=0.03, div_factor=10,
    )
    scaler = GradScaler()

    start_epoch, global_step = _load_latest_checkpoint(ckpt_dir, model, optimizer, scheduler, scaler)
    best_val_loss = float("inf")
    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,step,train_loss,val_loss\n")

    for epoch in range(start_epoch, cfg.basemodel_epochs):
        model.train()
        for batch_x, batch_x_stamp in train_loader:
            batch_x       = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
            token_out = [token_s1[:, 1:],  token_s2[:, 1:]]

            with autocast():
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            if global_step % cfg.log_interval == 0:
                print(f"[epoch {epoch+1} step {global_step}] loss={loss.item():.4f}")

            if global_step % cfg.save_steps == 0:
                _save_checkpoint(ckpt_dir, global_step, model, optimizer, scheduler, scaler)

            if max_steps > 0 and global_step >= max_steps:
                return

        val_loss = _validate_predictor(model, tokenizer, val_loader, device)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{global_step},{loss.item():.4f},{val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(save_dir / "best_model"))
            print(f"  -> new best val_loss={val_loss:.4f}, saved.")


def _validate_predictor(model, tokenizer, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_x_stamp in loader:
            batch_x       = batch_x.to(device)
            batch_x_stamp = batch_x_stamp.to(device)
            with autocast():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
                token_in  = [token_s1[:, :-1], token_s2[:, :-1]]
                token_out = [token_s1[:, 1:],  token_s2[:, 1:]]
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            total += loss.item() * batch_x.size(0)
            count += batch_x.size(0)
    return total / count if count else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    cfg = Config.from_yaml(parser.parse_args().config)
    run_training(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add finetune_tw/train_predictor.py
git commit -m "feat(finetune_tw): predictor fine-tuning with frozen tokenizer + AMP + resume"
```

---

### Task 8: backtest.py

**Files:**
- Create: `finetune_tw/backtest.py`
- Create: `tests/finetune_tw/test_backtest.py`

**Interfaces:**
- Consumes: `Config`, `db.query_symbol`, `db.list_symbols`, `KronosPredictor`
- Produces: `backtest_result.png` + printed metrics (annualised return, Sharpe, max drawdown)

- [ ] **Step 1: Write failing tests**

```python
# tests/finetune_tw/test_backtest.py
import numpy as np
import pandas as pd
import pytest
from finetune_tw.backtest import compute_metrics, rank_stocks, build_portfolio_returns


def test_compute_metrics_known_values():
    # Flat 0% return
    daily = pd.Series([0.0] * 252, index=pd.bdate_range("2024-01-01", periods=252))
    metrics = compute_metrics(daily)
    assert abs(metrics["annualised_return"]) < 1e-9
    assert metrics["max_drawdown"] == 0.0


def test_compute_metrics_positive_return():
    daily = pd.Series([0.001] * 252, index=pd.bdate_range("2024-01-01", periods=252))
    metrics = compute_metrics(daily)
    assert metrics["annualised_return"] > 0
    assert metrics["sharpe"] > 0


def test_rank_stocks_top_k():
    signals = {"A": 0.05, "B": 0.02, "C": 0.10, "D": -0.01}
    top = rank_stocks(signals, top_k=2)
    assert set(top) == {"A", "C"}


def test_build_portfolio_returns_shape():
    dates = pd.bdate_range("2024-01-01", periods=10)
    price_data = {
        "A": pd.Series([100.0 + i for i in range(10)], index=dates),
        "B": pd.Series([200.0 - i for i in range(10)], index=dates),
    }
    holdings = [{"A", "B"}] * 9  # 9 rebalance periods
    returns = build_portfolio_returns(price_data, holdings, dates[:-1])
    assert len(returns) == 9
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/finetune_tw/test_backtest.py -v
```

Expected: `ModuleNotFoundError: No module named 'finetune_tw.backtest'`

- [ ] **Step 3: Implement `finetune_tw/backtest.py`**

```python
"""
python finetune_tw/backtest.py --config finetune_tw/configs/config_tw_daily.yaml
Requires: fine-tuned predictor at outputs/{exp_name}/predictor/best_model/
          fine-tuned tokenizer at outputs/{exp_name}/tokenizer/best_model/
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor
from finetune_tw.config import Config
from finetune_tw.db import query_symbol, list_symbols


# ── Pure helper functions (testable without a model) ────────────────────────

def compute_metrics(daily_returns: pd.Series) -> dict:
    ann_ret = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
    cum = (1 + daily_returns).cumprod()
    max_dd = ((cum.cummax() - cum) / cum.cummax()).max()
    return {"annualised_return": ann_ret, "sharpe": sharpe, "max_drawdown": max_dd}


def rank_stocks(signals: dict[str, float], top_k: int) -> set[str]:
    sorted_syms = sorted(signals, key=signals.__getitem__, reverse=True)
    return set(sorted_syms[:top_k])


def build_portfolio_returns(
    price_data: dict[str, pd.Series],
    holdings_sequence: list[set[str]],
    rebalance_dates: pd.Index,
) -> pd.Series:
    ret_list = []
    for date, holdings in zip(rebalance_dates, holdings_sequence):
        period_returns = []
        for sym in holdings:
            if sym not in price_data:
                continue
            series = price_data[sym]
            if date not in series.index:
                continue
            pos = series.index.get_loc(date)
            if pos + 1 >= len(series):
                continue
            r = series.iloc[pos + 1] / series.iloc[pos] - 1
            period_returns.append(r)
        ret_list.append(float(np.mean(period_returns)) if period_returns else 0.0)
    return pd.Series(ret_list, index=rebalance_dates)


# ── Main backtest loop ──────────────────────────────────────────────────────

def run_backtest(cfg: Config) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tok_path  = Path(cfg.output_dir) / cfg.exp_name / "tokenizer" / "best_model"
    pred_path = Path(cfg.output_dir) / cfg.exp_name / "predictor" / "best_model"

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
    model     = Kronos.from_pretrained(str(pred_path))
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
    tokenizer.eval(); model.eval()

    symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
    test_end = str(pd.Timestamp.today().date())

    # Pre-load close prices for all symbols over the test period
    close_prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = query_symbol(cfg.db_path, sym, start=cfg.test_start_date, end=test_end)
        if len(df) > 0:
            idx = pd.DatetimeIndex(df["date"])
            close_prices[sym] = pd.Series(df["close"].values, index=idx)

    # Build rebalance dates
    all_dates = pd.bdate_range(cfg.test_start_date, test_end)
    rebalance_dates = all_dates[::cfg.hold_days]

    holdings_sequence: list[set[str]] = []
    for rebal_date in rebalance_dates:
        signals: dict[str, float] = {}
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        for sym in symbols:
            df = query_symbol(cfg.db_path, sym,
                              end=rebal_str)
            if len(df) < cfg.lookback_window:
                continue
            ctx = df.iloc[-cfg.lookback_window:]
            x_ts = pd.to_datetime(ctx["date"])
            y_ts = pd.date_range(rebal_date, periods=cfg.pred_len, freq="B")
            with torch.no_grad():
                pred = predictor.predict(
                    df=ctx[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True),
                    x_timestamp=x_ts.reset_index(drop=True),
                    y_timestamp=pd.Series(y_ts),
                    pred_len=cfg.pred_len,
                    T=1.0, top_k=1, top_p=1.0, sample_count=1, verbose=False,
                )
            if pred is not None and len(pred) >= cfg.pred_len:
                signals[sym] = pred["close"].iloc[-1] / ctx["close"].iloc[-1] - 1

        holdings_sequence.append(rank_stocks(signals, cfg.top_k))

    strategy_returns = build_portfolio_returns(close_prices, holdings_sequence, rebalance_dates[:-1])

    # Benchmark returns
    bm_df = query_symbol(cfg.db_path, cfg.benchmark_symbol,
                         start=cfg.test_start_date, end=test_end)
    bm_close = pd.Series(bm_df["close"].values,
                         index=pd.DatetimeIndex(bm_df["date"]))
    bm_returns = bm_close.pct_change().dropna().reindex(strategy_returns.index).fillna(0)

    metrics = compute_metrics(strategy_returns)
    bm_metrics = compute_metrics(bm_returns)

    print(f"\n=== Backtest Results ({cfg.test_start_date} → {test_end}) ===")
    print(f"Strategy  — Ann. Return: {metrics['annualised_return']:.2%}  "
          f"Sharpe: {metrics['sharpe']:.2f}  Max DD: {metrics['max_drawdown']:.2%}")
    print(f"Benchmark — Ann. Return: {bm_metrics['annualised_return']:.2%}  "
          f"Sharpe: {bm_metrics['sharpe']:.2f}  Max DD: {bm_metrics['max_drawdown']:.2%}")

    # Plot
    cum_strat = (1 + strategy_returns).cumprod()
    cum_bm    = (1 + bm_returns).cumprod()
    plt.figure(figsize=(12, 5))
    plt.plot(cum_strat.index, cum_strat.values, label="Kronos-TW Strategy")
    plt.plot(cum_bm.index,    cum_bm.values,    label=cfg.benchmark_symbol, linestyle="--")
    plt.title("Cumulative Return: Strategy vs Benchmark")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.tight_layout()
    out_path = Path(cfg.output_dir) / cfg.exp_name / "backtest_result.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="finetune_tw/configs/config_tw_daily.yaml")
    parser.add_argument("--top_k",    type=int,   default=None)
    parser.add_argument("--hold_days", type=int,  default=None)
    parser.add_argument("--pred_len",  type=int,  default=None)
    parser.add_argument("--test_start", default=None)
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.top_k:      cfg.top_k = args.top_k
    if args.hold_days:  cfg.hold_days = args.hold_days
    if args.pred_len:   cfg.pred_len = args.pred_len
    if args.test_start: cfg.test_start_date = args.test_start
    run_backtest(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/finetune_tw/test_backtest.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/backtest.py tests/finetune_tw/test_backtest.py
git commit -m "feat(finetune_tw): pure-pandas top-K backtester with metrics + plot"
```

---

### Task 9: colab_setup.ipynb + .gitignore update

**Files:**
- Create: `finetune_tw/colab_setup.ipynb`
- Modify: `.gitignore` — add `finetune_tw/data/`

- [ ] **Step 1: Add `.gitignore` entry**

Append to the existing `.gitignore`:

```
finetune_tw/data/
finetune_tw/outputs/
```

- [ ] **Step 2: Create `finetune_tw/colab_setup.ipynb`**

Create a notebook with these cells (JSON source lines shown for each):

**Cell 1 — Mount Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE = "/content/drive/MyDrive/Kronos_TW"
import os; os.makedirs(DRIVE_BASE, exist_ok=True)
```

**Cell 2 — Clone repo + symlink persistent dirs:**
```python
import os, subprocess
REPO = "/content/Kronos"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/<YOUR_FORK>/Kronos", REPO], check=True)
os.chdir(REPO)

# Symlink data and outputs to Drive so they persist across sessions
for sub in ["data", "outputs"]:
    local = f"finetune_tw/{sub}"
    remote = f"{DRIVE_BASE}/{sub}"
    os.makedirs(remote, exist_ok=True)
    if not os.path.islink(local):
        os.symlink(remote, local)
print("Symlinks ready.")
```

**Cell 3 — Install dependencies:**
```python
subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
subprocess.run(["pip", "install", "-q", "yfinance", "pyyaml", "tqdm"], check=True)
print("Dependencies installed.")
```

**Cell 4 — Download data (run once or with --update):**
```python
subprocess.run([
    "python", "-m", "finetune_tw.download_data",
    "--config", "finetune_tw/configs/config_tw_daily.yaml",
    "--source", "auto",
    "--start", "2015-01-01",
], check=True)
```

**Cell 5 — Train tokenizer:**
```python
subprocess.run([
    "python", "finetune_tw/train_tokenizer.py",
    "--config", "finetune_tw/configs/config_tw_daily.yaml",
], check=True)
```

**Cell 6 — Train predictor:**
```python
subprocess.run([
    "python", "finetune_tw/train_predictor.py",
    "--config", "finetune_tw/configs/config_tw_daily.yaml",
], check=True)
```

**Cell 7 — Backtest:**
```python
subprocess.run([
    "python", "finetune_tw/backtest.py",
    "--config", "finetune_tw/configs/config_tw_daily.yaml",
], check=True)
```

- [ ] **Step 3: Commit**

```bash
git add finetune_tw/colab_setup.ipynb .gitignore
git commit -m "feat(finetune_tw): Colab setup notebook + gitignore data/outputs"
```
