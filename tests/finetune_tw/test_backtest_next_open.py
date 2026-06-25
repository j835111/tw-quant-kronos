from __future__ import annotations

import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
import pytest


def _seed_calendar_db(tmp_path) -> str:
    db_path = str(tmp_path / "calendar.db")
    init_db(db_path)

    benchmark = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000.0] * 5,
            "amount": [100_500.0, 101_500.0, 102_500.0, 103_500.0, 104_500.0],
        }
    )
    upsert_prices(db_path, "^TWII", benchmark)
    return db_path


def test_load_trading_calendar_uses_benchmark_dates(tmp_path):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_calendar_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
    )

    dates = bo._load_trading_calendar(cfg, end="2024-01-31")

    assert list(dates.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]


def test_build_signal_and_execution_dates_drops_last_anchor_without_next_day():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )

    signal_dates, execution_dates = bo._build_signal_and_execution_dates(
        trading_dates,
        hold_days=2,
    )

    assert list(signal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-05"]
    assert list(execution_dates.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-08"]


def test_build_next_open_portfolio_returns_combines_gap_and_rebalance_intraday():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"])
    execution_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-05"])
    holdings = [{"A", "B"}, {"C"}]

    price_frames = {
        "A": pd.DataFrame(
            {
                "open": [100.0, 110.0, 121.0],
                "close": [110.0, 121.0, 133.1],
            },
            index=trading_dates,
        ),
        "B": pd.DataFrame(
            {
                "open": [200.0, 220.0, 198.0],
                "close": [220.0, 198.0, 217.8],
            },
            index=trading_dates,
        ),
        "C": pd.DataFrame(
            {
                "open": [50.0, 50.0, 50.0],
                "close": [50.0, 50.0, 55.0],
            },
            index=trading_dates,
        ),
    }

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=holdings,
        execution_dates=execution_dates,
        trading_dates=trading_dates,
    )

    assert list(daily_returns.index.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert daily_returns.iloc[0] == pytest.approx(0.10)
    assert daily_returns.iloc[1] == pytest.approx(0.0)
    assert daily_returns.iloc[2] == pytest.approx(0.10)
    assert len(period_returns) == 1
    assert period_returns.iloc[0] == pytest.approx(0.10)


def test_build_next_open_portfolio_returns_single_execution_emits_first_intraday():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(["2024-01-03"])
    execution_dates = pd.DatetimeIndex(["2024-01-03"])
    holdings = [{"A"}]

    price_frames = {
        "A": pd.DataFrame(
            {
                "open": [100.0],
                "close": [110.0],
            },
            index=trading_dates,
        ),
    }

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=holdings,
        execution_dates=execution_dates,
        trading_dates=trading_dates,
    )

    assert period_returns.empty
    assert list(daily_returns.index.strftime("%Y-%m-%d")) == ["2024-01-03"]
    assert daily_returns.iloc[0] == pytest.approx(0.10)


def test_build_next_open_portfolio_returns_preserves_zero_period_when_outgoing_missing():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-05"])
    execution_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-05"])
    holdings = [{"A"}, {"B"}]

    price_frames = {
        "A": pd.DataFrame(
            {
                "open": [100.0],
                "close": [110.0],
            },
            index=pd.DatetimeIndex(["2024-01-03"]),
        ),
        "B": pd.DataFrame(
            {
                "open": [50.0],
                "close": [55.0],
            },
            index=pd.DatetimeIndex(["2024-01-05"]),
        ),
    }

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames=price_frames,
        holdings_sequence=holdings,
        execution_dates=execution_dates,
        trading_dates=trading_dates,
    )

    assert list(daily_returns.index.strftime("%Y-%m-%d")) == ["2024-01-03"]
    assert daily_returns.iloc[0] == pytest.approx(0.10)
    assert list(period_returns.index.strftime("%Y-%m-%d")) == ["2024-01-03"]
    assert period_returns.iloc[0] == pytest.approx(0.0)


def test_build_next_open_portfolio_returns_returns_empty_series_for_zero_executions():
    import finetune_tw.backtest_next_open as bo

    period_returns, daily_returns = bo.build_next_open_portfolio_returns(
        price_frames={},
        holdings_sequence=[],
        execution_dates=pd.DatetimeIndex([]),
        trading_dates=pd.DatetimeIndex([]),
    )

    assert period_returns.empty
    assert daily_returns.empty
