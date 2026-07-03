import sqlite3

import numpy as np
import pandas as pd
import pytest

from finetune_tw.round6_diagnostics import (
    aggregate_period,
    per_day_metrics,
    twse_trading_days,
)


@pytest.fixture()
def calendar_db(tmp_path):
    """daily_prices with: 2 clean trading days, 1 typhoon day (stocks but no benchmark),
    1 dirty benchmark day (benchmark row but almost no stocks)."""
    db_path = tmp_path / "tw.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE daily_prices (symbol TEXT, date TEXT, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, amount REAL)"
    )
    rows = []
    clean_days = ["2024-01-02", "2024-01-03"]
    for date in clean_days:
        rows.append(("^TWII", date))
        rows.extend((f"S{i}", date) for i in range(20))
    # typhoon closure: provider emitted stock rows but the benchmark is absent
    rows.extend((f"S{i}", "2024-01-04") for i in range(20))
    # dirty benchmark row on a non-trading day with almost no stocks
    rows.append(("^TWII", "2024-01-06"))
    rows.append(("S0", "2024-01-06"))
    conn.executemany(
        "INSERT INTO daily_prices VALUES (?, ?, 1, 1, 1, 1, 1, 1)", rows
    )
    conn.commit()
    conn.close()
    return db_path


def test_twse_trading_days_intersects_benchmark_and_symbol_count(calendar_db):
    days = twse_trading_days(calendar_db, min_symbols=10)
    assert days == {"2024-01-02", "2024-01-03"}


def _scored_frame(n_symbols=30, dates=("2024-01-02", "2024-04-15")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for date in dates:
        labels = rng.normal(size=n_symbols)
        for s in range(n_symbols):
            rows.append(
                {"date": date, "symbol": f"S{s}", "label": labels[s], "score": labels[s]}
            )
    return pd.DataFrame(rows)


def test_per_day_metrics_perfect_score():
    df = _scored_frame()
    daily = per_day_metrics(df, score_col="score", top_k=5)
    assert len(daily) == 2
    assert np.allclose(daily["rank_ic"], 1.0)
    assert np.allclose(daily["overlap_topk"], 1.0)
    assert (daily["top_excess"] > 0).all()


def test_per_day_metrics_inverted_score():
    df = _scored_frame()
    df["score"] = -df["score"]
    daily = per_day_metrics(df, score_col="score", top_k=5)
    assert np.allclose(daily["rank_ic"], -1.0)
    assert (daily["top_excess"] < 0).all()


def test_aggregate_period_quarterly_groups():
    df = _scored_frame(dates=("2024-01-02", "2024-01-03", "2024-04-15"))
    daily = per_day_metrics(df, score_col="score", top_k=5)
    q = aggregate_period(daily, freq="Q")
    assert list(q["period"]) == ["2024Q1", "2024Q2"]
    assert list(q["days"]) == [2, 1]
    assert np.allclose(q["mean_ic"], 1.0)
