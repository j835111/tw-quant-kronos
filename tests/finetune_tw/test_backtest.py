import numpy as np
import pandas as pd
import pytest
from finetune_tw.backtest import (
    ModelSpec,
    build_portfolio_returns,
    compute_metrics,
    compute_raw_signals,
    rank_stocks,
    run_backtest,
)


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


def test_compute_raw_signals_bulk_loads_history_once_and_applies_per_rebalance_recency_window(monkeypatch):
    calls: list[tuple[str, tuple[str, ...], str, str]] = []

    history_frames = {
        "AAA": pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "high": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
                "low": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5],
                "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                "volume": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
                "amount": [1000.0, 1210.0, 1440.0, 1690.0, 1960.0, 2250.0],
            },
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09", "2024-01-10"]
            ),
        ),
        "SPARSE": pd.DataFrame(
            {
                "open": [20.0, 21.0, 22.0, 23.0],
                "high": [20.5, 21.5, 22.5, 23.5],
                "low": [19.5, 20.5, 21.5, 22.5],
                "close": [20.0, 21.0, 22.0, 23.0],
                "volume": [200.0, 210.0, 220.0, 230.0],
                "amount": [4000.0, 4410.0, 4840.0, 5290.0],
            },
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-10"]),
        ),
        "CCC": pd.DataFrame(
            {
                "open": [30.0, np.nan, 32.0, 33.0, 34.0, 35.0],
                "high": [30.5, 31.5, 32.5, 33.5, 34.5, 35.5],
                "low": [29.5, 30.5, 31.5, 32.5, 33.5, 34.5],
                "close": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
                "volume": [300.0, 310.0, 320.0, 330.0, 340.0, 350.0],
                "amount": [9000.0, 9610.0, 10240.0, 10890.0, 11560.0, 12250.0],
            },
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09", "2024-01-10"]
            ),
        ),
    }

    def fake_loader(db_path, symbols, start, end):
        calls.append((db_path, tuple(symbols), start, end))
        return history_frames

    monkeypatch.setattr("finetune_tw.backtest.load_symbol_history_frames", fake_loader)

    class FakePredictor:
        def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, **kwargs):
            preds = []
            for df in df_list:
                last_close = df["close"].iloc[-1]
                preds.append(
                    pd.DataFrame(
                        {"close": [last_close * 1.1, last_close * 1.25]},
                    )
                )
            return preds

    cfg = type(
        "Cfg",
        (),
        {"db_path": "fake.db", "lookback_window": 3},
    )()

    rebal_dates = pd.DatetimeIndex([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10")])
    raw_preds = compute_raw_signals(
        predictor=FakePredictor(),
        cfg=cfg,
        rebal_dates=rebal_dates,
        pred_len=2,
        symbols=["AAA", "SPARSE", "CCC"],
    )

    assert len(calls) == 1
    assert calls[0] == ("fake.db", ("AAA", "SPARSE", "CCC"), "2023-12-28", "2024-01-10")
    assert list(raw_preds) == ["2024-01-03", "2024-01-10"]
    assert list(raw_preds["2024-01-03"]) == ["AAA"]
    assert "AAA" in raw_preds["2024-01-10"]
    assert "SPARSE" not in raw_preds["2024-01-10"]
    assert raw_preds["2024-01-03"]["AAA"].tolist() == pytest.approx([0.1, 0.25])
    assert raw_preds["2024-01-10"]["AAA"].tolist() == pytest.approx([0.1, 0.25])


def test_run_backtest_preloads_close_prices_but_queries_benchmark_directly(monkeypatch, tmp_path):
    load_price_calls = []
    query_calls = []
    plot_calls = []

    def fake_load_price_field_series(db_path, symbols, start, end, field):
        load_price_calls.append((db_path, tuple(symbols), start, end, field))
        dates = pd.bdate_range("2024-01-01", periods=3)
        return {
            "AAA": pd.Series([10.0, 11.0, 12.0], index=dates),
            "BBB": pd.Series([20.0, 21.0, 22.0], index=dates),
        }

    def fake_query_symbol(db_path, symbol, start, end):
        query_calls.append((db_path, symbol, start, end))
        return pd.DataFrame(
            {"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "close": [100.0, 101.0, 102.0]}
        )

    def fake_build_model_specs(cfg):
        return {"round2": ModelSpec("Round 2", "tok", {}, "pred", {})}

    def fake_load_predictor_from_spec(spec, cfg):
        return object()

    def fake_compute_raw_signals(predictor, cfg, rebal_dates, pred_len, symbols):
        return {
            rebal_dates[0].strftime("%Y-%m-%d"): {
                "AAA": pd.Series([0.01]),
                "BBB": pd.Series([0.02]),
            }
        }

    def fake_plot_backtest_results(data, out_dir):
        plot_calls.append((data["model_key"], out_dir))
        path = out_dir / "plot.png"
        path.write_text("plot")
        return path

    monkeypatch.setattr("finetune_tw.backtest.load_price_field_series", fake_load_price_field_series)
    monkeypatch.setattr("finetune_tw.backtest.query_symbol", fake_query_symbol)
    monkeypatch.setattr("finetune_tw.backtest.list_symbols", lambda db_path: ["AAA", "BBB", "TWII"])
    monkeypatch.setattr("finetune_tw.backtest.build_model_specs", fake_build_model_specs)
    monkeypatch.setattr("finetune_tw.backtest.load_predictor_from_spec", fake_load_predictor_from_spec)
    monkeypatch.setattr("finetune_tw.backtest.compute_raw_signals", fake_compute_raw_signals)
    monkeypatch.setattr("finetune_tw.backtest.plot_backtest_results", fake_plot_backtest_results)
    monkeypatch.setattr("finetune_tw.backtest.torch.cuda.empty_cache", lambda: None)

    cfg = type(
        "Cfg",
        (),
        {
            "db_path": "fake.db",
            "benchmark_symbol": "TWII",
            "test_start_date": "2024-01-01",
            "top_k": 1,
            "min_signal_threshold": 0.0,
            "output_dir": str(tmp_path),
            "exp_name": "exp",
        },
    )()

    out_path = run_backtest(cfg, "round2", [1])

    assert load_price_calls == [("fake.db", ("AAA", "BBB"), "2024-01-01", str(pd.Timestamp.today().date()), "close")]
    assert query_calls == [("fake.db", "TWII", "2024-01-01", str(pd.Timestamp.today().date()))]
    assert plot_calls
    assert out_path.exists()
