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
    holdings = [{"A", "B"}] * 9  # one holdings set per hold period (len(dates) - 1)
    period_returns, daily_returns = build_portfolio_returns(price_data, holdings, dates)

    # one period return per rebalance interval, indexed by all but the last date
    assert isinstance(period_returns, pd.Series)
    assert len(period_returns) == 9
    assert list(period_returns.index) == list(dates[:-1])
    # equal-weight A(+1/day) and B(-1/day) over the first interval: (0.01 + -0.005)/2
    assert period_returns.iloc[0] == pytest.approx((0.01 + (-0.005)) / 2)

    # daily returns aggregated across hold periods, non-empty
    assert isinstance(daily_returns, pd.Series)
    assert len(daily_returns) > 0
