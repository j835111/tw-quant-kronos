import pandas as pd
import pytest
import finetune_tw.db as db_module
from finetune_tw.db import (
    init_db,
    upsert_prices,
    query_symbol,
    query_symbols_window,
    list_symbols,
    get_last_date,
)

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

def test_query_symbols_window_chunks_large_symbol_lists(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2317.TW", _make_df(3))
    upsert_prices(db, "2330.TW", _make_df(3))
    upsert_prices(db, "2454.TW", _make_df(3))

    monkeypatch.setattr(db_module, "_QUERY_SYMBOLS_WINDOW_CHUNK_SIZE", 2, raising=False)

    original_read_sql = db_module.pd.read_sql
    calls = 0

    def counting_read_sql(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_read_sql(*args, **kwargs)

    monkeypatch.setattr(db_module.pd, "read_sql", counting_read_sql)

    df = query_symbols_window(
        db,
        ["2330.TW", "2454.TW", "2317.TW"],
        start="2024-01-01",
        end="2024-01-03",
    )

    assert calls == 2
    assert sorted(df["symbol"].unique().tolist()) == ["2317.TW", "2330.TW", "2454.TW"]
    assert list(df[["symbol", "date"]].itertuples(index=False, name=None)) == [
        ("2317.TW", "2024-01-01"),
        ("2317.TW", "2024-01-02"),
        ("2317.TW", "2024-01-03"),
        ("2330.TW", "2024-01-01"),
        ("2330.TW", "2024-01-02"),
        ("2330.TW", "2024-01-03"),
        ("2454.TW", "2024-01-01"),
        ("2454.TW", "2024-01-02"),
        ("2454.TW", "2024-01-03"),
    ]

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
    assert last == "2024-01-05"  # 5 business days from 2024-01-01 (Mon-Fri)

def test_get_last_date_missing_symbol(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    assert get_last_date(db, "9999.TW") is None
