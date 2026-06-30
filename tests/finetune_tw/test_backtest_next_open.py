from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
import pytest

def _make_history_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [10.0, 10.5, 11.0],
            "high": [10.2, 10.7, 11.2],
            "low": [9.8, 10.3, 10.8],
            "close": [10.1, 10.6, 11.1],
            "volume": [100.0, 110.0, 120.0],
            "amount": [1_010.0, 1_166.0, 1_332.0],
        }
    )


def _make_open_prediction_frame(
    base_open: float,
    future_opens: list[float],
    volume: float,
) -> pd.DataFrame:
    opens = [base_open, *future_opens]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [price + 0.5 for price in opens],
            "low": [price - 0.5 for price in opens],
            "close": opens,
            "volume": [volume] * len(opens),
        },
        dtype=float,
    )


class _FakeBatchPredictor:
    def __init__(self, preds: list[pd.DataFrame]):
        self._preds = preds
        self._cursor = 0
        self.batch_sizes: list[int] = []

    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        **kwargs,
    ):
        batch_size = len(df_list)
        self.batch_sizes.append(batch_size)
        start = self._cursor
        end = start + batch_size
        self._cursor = end
        return self._preds[start:end]


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


def _seed_runner_db(tmp_path) -> str:
    db_path = str(tmp_path / "runner.db")
    init_db(db_path)

    benchmark = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 102.0, 101.0, 103.0, 104.0],
            "volume": [1_000.0] * 5,
            "amount": [100_000.0, 102_000.0, 101_000.0, 103_000.0, 104_000.0],
        }
    )
    sym_1101 = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [100.0] * 5,
            "amount": [1_050.0, 1_150.0, 1_250.0, 1_350.0, 1_450.0],
        }
    )
    sym_1216 = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [20.0, 20.5, 21.0, 21.5, 22.0],
            "high": [21.0, 21.5, 22.0, 22.5, 23.0],
            "low": [19.0, 19.5, 20.0, 20.5, 21.0],
            "close": [20.4, 20.8, 21.4, 21.9, 22.4],
            "volume": [200.0] * 5,
            "amount": [4_080.0, 4_160.0, 4_280.0, 4_380.0, 4_480.0],
        }
    )

    upsert_prices(db_path, "^TWII", benchmark)
    upsert_prices(db_path, "1101.TW", sym_1101)
    upsert_prices(db_path, "1216.TW", sym_1216)
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


def test_load_trading_calendar_raises_for_missing_benchmark_rows(tmp_path):
    import finetune_tw.backtest_next_open as bo

    db_path = str(tmp_path / "empty_calendar.db")
    init_db(db_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
    )

    with pytest.raises(
        ValueError,
        match=r"No benchmark rows found for \^TWII between 2024-01-01 and 2024-01-31\.",
    ):
        bo._load_trading_calendar(cfg, end="2024-01-31")


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


def test_compute_raw_signals_open_bulk_preloads_once_and_returns_open_returns(monkeypatch):
    import finetune_tw.backtest_next_open as bo

    cfg = Config(
        db_path="unused.db",
        lookback_window=2,
    )
    rebal_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    symbols = ["1101.TW", "1216.TW"]
    preload_calls = []

    history_frames = {
        "1101.TW": pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "high": [10.2, 10.7, 11.2],
                "low": [9.8, 10.3, 10.8],
                "close": [10.1, 10.6, 11.1],
                "volume": [100.0, 110.0, 120.0],
                "amount": [1_010.0, 1_166.0, 1_332.0],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
        "1216.TW": pd.DataFrame(
            {
                "open": [20.0, 20.5, 21.0],
                "high": [20.2, 20.7, 21.2],
                "low": [19.8, 20.3, 20.8],
                "close": [20.1, 20.6, 21.1],
                "volume": [200.0, 210.0, 220.0],
                "amount": [4_020.0, 4_326.0, 4_642.0],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
    }

    def fake_load_symbol_history_frames(db_path, seen_symbols, start, end):
        preload_calls.append((db_path, tuple(seen_symbols), start, end))
        return history_frames

    monkeypatch.setattr(bo, "load_symbol_history_frames", fake_load_symbol_history_frames)

    predictor = _FakeBatchPredictor(
        [
            _make_open_prediction_frame(10.0, [10.3, 10.6], volume=100.0),
            _make_open_prediction_frame(20.0, [20.2, 20.5], volume=200.0),
            _make_open_prediction_frame(10.0, [10.4, 10.8], volume=100.0),
            _make_open_prediction_frame(20.0, [20.3, 20.7], volume=200.0),
        ]
    )

    raw_preds = bo.compute_raw_signals_open(
        predictor,
        cfg,
        rebal_dates,
        pred_len=3,
        symbols=symbols,
    )

    assert preload_calls == [("unused.db", tuple(symbols), "2023-12-30", "2024-01-04")]
    assert raw_preds["2024-01-03"]["1101.TW"].tolist() == pytest.approx([0.03, 0.06])
    assert "pred_frame" not in raw_preds["2024-01-03"]["1101.TW"].attrs


def test_compute_raw_signals_open_applies_per_rebalance_recency_cutoff(monkeypatch):
    import finetune_tw.backtest_next_open as bo

    cfg = Config(
        db_path="unused.db",
        lookback_window=2,
    )
    rebal_dates = pd.DatetimeIndex(["2024-01-03", "2024-01-08"])
    symbols = ["SPARSE.TW"]

    history_frames = {
        "SPARSE.TW": pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "high": [10.2, 10.7, 11.2],
                "low": [9.8, 10.3, 10.8],
                "close": [10.1, 10.6, 11.1],
                "volume": [100.0, 110.0, 120.0],
                "amount": [1_010.0, 1_166.0, 1_332.0],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-08"]),
        )
    }

    monkeypatch.setattr(bo, "load_symbol_history_frames", lambda *args, **kwargs: history_frames)

    predictor = _FakeBatchPredictor(
        [_make_open_prediction_frame(10.0, [10.2, 10.4], volume=100.0)]
    )

    raw_preds = bo.compute_raw_signals_open(
        predictor,
        cfg,
        rebal_dates,
        pred_len=3,
        symbols=symbols,
    )

    assert set(raw_preds) == {"2024-01-03", "2024-01-08"}
    assert set(raw_preds["2024-01-03"]) == {"SPARSE.TW"}
    assert raw_preds["2024-01-08"] == {}
    assert predictor.batch_sizes == [1]


def test_load_price_frames_uses_shared_price_loader(monkeypatch):
    import finetune_tw.backtest_next_open as bo

    cfg = Config(
        db_path="unused.db",
        test_start_date="2024-01-01",
    )
    seen = {}

    def fake_load_price_frame_fields(db_path, symbols, start, end, fields):
        seen["call"] = {
            "db_path": db_path,
            "symbols": list(symbols),
            "start": start,
            "end": end,
            "fields": list(fields),
        }
        return {
            "1101.TW": pd.DataFrame(
                {
                    "open": [10.0, 11.0],
                    "close": [10.5, 11.5],
                },
                index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
            )
        }

    monkeypatch.setattr(bo, "load_price_frame_fields", fake_load_price_frame_fields)
    monkeypatch.setattr(
        bo,
        "query_symbol",
        lambda *args, **kwargs: pytest.fail("_load_price_frames() should not query symbols one by one"),
    )

    frames = bo._load_price_frames(cfg, ["1101.TW", "1216.TW"], end="2024-01-31")

    assert seen["call"] == {
        "db_path": "unused.db",
        "symbols": ["1101.TW", "1216.TW"],
        "start": "2024-01-01",
        "end": "2024-01-31",
        "fields": ["open", "close"],
    }
    assert list(frames["1101.TW"].columns) == ["open", "close"]


def test_build_portfolio_returns_equal_weight():
    import finetune_tw.backtest_next_open as bo

    dates = pd.bdate_range("2024-01-01", periods=2)
    price_data = {
        "A": pd.Series([100.0, 110.0], index=dates),
        "B": pd.Series([200.0, 180.0], index=dates),
    }

    period_returns, daily_returns = bo.build_portfolio_returns(
        price_data,
        [{"A", "B"}],
        dates,
        weights=None,
    )

    assert period_returns.iloc[0] == pytest.approx(0.0)
    assert daily_returns.iloc[0] == pytest.approx(0.0)


def test_build_portfolio_returns_custom_weights():
    import finetune_tw.backtest_next_open as bo

    dates = pd.bdate_range("2024-01-01", periods=2)
    price_data = {
        "A": pd.Series([100.0, 110.0], index=dates),
        "B": pd.Series([200.0, 180.0], index=dates),
    }

    period_returns, daily_returns = bo.build_portfolio_returns(
        price_data,
        [{"A", "B"}],
        dates,
        weights={"A": 0.75, "B": 0.25},
    )

    assert period_returns.iloc[0] == pytest.approx(0.05)
    assert daily_returns.iloc[0] == pytest.approx(0.05)


def test_run_backtest_next_open_saves_suffix_outputs_and_schema(tmp_path, monkeypatch):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_runner_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
        output_dir=str(tmp_path / "outputs"),
        exp_name="next_open_case",
        top_k=1,
        min_signal_threshold=0.0,
    )

    monkeypatch.setattr(
        bo,
        "build_model_specs",
        lambda _cfg: {
            "round0": SimpleNamespace(label="Round 0"),
        },
    )
    monkeypatch.setattr(bo, "load_predictor_from_spec", lambda spec, _cfg: object())
    monkeypatch.setattr(bo, "_today", lambda: pd.Timestamp("2024-01-09"))

    def fake_compute_raw_signals(
        predictor,
        seen_cfg,
        signal_dates,
        pred_len,
        symbols,
    ):
        assert seen_cfg is cfg
        assert list(signal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-05"]
        assert pred_len == 3
        assert symbols == ["1101.TW", "1216.TW"]
        return {
            "2024-01-02": {
                "1101.TW": pd.Series([0.02, 0.03]),
                "1216.TW": pd.Series([0.01, 0.015]),
            },
            "2024-01-05": {
                "1101.TW": pd.Series([0.01, 0.02]),
                "1216.TW": pd.Series([0.03, 0.04]),
            },
        }

    monkeypatch.setattr(bo, "compute_raw_signals", fake_compute_raw_signals)

    out_path = bo.run_backtest_next_open(cfg, "round0", [2])

    assert out_path.name == "backtest_returns_round0_next_open.json"
    assert out_path.exists()
    assert (out_path.parent / "backtest_round0_next_open.png").exists()

    data = json.loads(out_path.read_text())
    assert set(data) == {
        "model_key",
        "model_label",
        "test_start",
        "test_end",
        "top_k",
        "hold_variants",
        "benchmark",
    }
    variant = data["hold_variants"]["2"]
    assert len(variant["dates"]) == len(variant["daily_returns"])


def test_run_backtest_next_open_uses_exact_variant_schedule_for_non_multiple_holds(
    tmp_path,
    monkeypatch,
):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_runner_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
        output_dir=str(tmp_path / "outputs"),
        exp_name="next_open_case_non_multiple",
        top_k=1,
        min_signal_threshold=0.0,
    )

    monkeypatch.setattr(
        bo,
        "build_model_specs",
        lambda _cfg: {
            "round0": SimpleNamespace(label="Round 0"),
        },
    )
    monkeypatch.setattr(bo, "load_predictor_from_spec", lambda spec, _cfg: object())
    monkeypatch.setattr(bo, "_today", lambda: pd.Timestamp("2024-01-09"))
    monkeypatch.setattr(
        bo,
        "compute_metrics",
        lambda dr: {"annualised_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
    )
    monkeypatch.setattr(
        bo,
        "plot_backtest_next_open_results",
        lambda data, out_dir: out_dir / "backtest_round0_next_open.png",
    )

    seen_signal_dates = {}
    seen_execution_dates = {}

    def fake_compute_raw_signals(
        predictor,
        seen_cfg,
        signal_dates,
        pred_len,
        symbols,
    ):
        assert seen_cfg is cfg
        assert list(signal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-08"]
        assert pred_len == 6
        assert symbols == ["1101.TW", "1216.TW"]
        return {
            d.strftime("%Y-%m-%d"): {
                "1101.TW": pd.Series([0.01] * pred_len),
                "1216.TW": pd.Series([0.02] * pred_len),
            }
            for d in signal_dates
        }

    def fake_signals_to_holdings(raw_preds, signal_dates, hold_days, top_k, threshold):
        seen_signal_dates[hold_days] = list(signal_dates.strftime("%Y-%m-%d"))
        return [{f"hold_{hold_days}"} for _ in signal_dates]

    def fake_build_next_open_portfolio_returns(
        price_frames,
        holdings_sequence,
        execution_dates,
        trading_dates,
    ):
        hold_days = int(next(iter(holdings_sequence[0])).split("_")[1])
        seen_execution_dates[hold_days] = list(execution_dates.strftime("%Y-%m-%d"))
        daily = pd.Series([0.0], index=pd.DatetimeIndex([execution_dates[0]]))
        return pd.Series(dtype=float), daily

    monkeypatch.setattr(bo, "compute_raw_signals", fake_compute_raw_signals)
    monkeypatch.setattr(bo, "signals_to_holdings", fake_signals_to_holdings)
    monkeypatch.setattr(
        bo,
        "build_next_open_portfolio_returns",
        fake_build_next_open_portfolio_returns,
    )

    bo.run_backtest_next_open(cfg, "round0", [3, 5])

    assert seen_signal_dates[3] == ["2024-01-02", "2024-01-08"]
    assert seen_execution_dates[3] == ["2024-01-03", "2024-01-09"]
    assert seen_signal_dates[5] == ["2024-01-02"]
    assert seen_execution_dates[5] == ["2024-01-03"]


def test_run_backtest_next_open_raises_when_variant_has_no_realized_daily_returns(
    tmp_path,
    monkeypatch,
):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_runner_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
        output_dir=str(tmp_path / "outputs"),
        exp_name="next_open_case_empty_daily",
        top_k=1,
        min_signal_threshold=0.0,
    )

    monkeypatch.setattr(
        bo,
        "build_model_specs",
        lambda _cfg: {
            "round0": SimpleNamespace(label="Round 0"),
        },
    )
    monkeypatch.setattr(bo, "load_predictor_from_spec", lambda spec, _cfg: object())
    monkeypatch.setattr(bo, "_today", lambda: pd.Timestamp("2024-01-09"))
    monkeypatch.setattr(
        bo,
        "compute_raw_signals",
        lambda predictor, seen_cfg, signal_dates, pred_len, symbols: {
            d.strftime("%Y-%m-%d"): {
                sym: pd.Series([0.01] * pred_len) for sym in symbols
            }
            for d in signal_dates
        },
    )
    monkeypatch.setattr(
        bo,
        "signals_to_holdings",
        lambda raw_preds, signal_dates, hold_days, top_k, threshold: [
            {"1101.TW"} for _ in signal_dates
        ],
    )
    monkeypatch.setattr(
        bo,
        "build_next_open_portfolio_returns",
        lambda price_frames, holdings_sequence, execution_dates, trading_dates: (
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        ),
    )
    monkeypatch.setattr(
        bo,
        "compute_metrics",
        lambda dr: pytest.fail("compute_metrics() should not be called for empty daily_returns"),
    )

    with pytest.raises(
        ValueError,
        match=r"No realized daily returns for top_k=1 hold_days=2",
    ):
        bo.run_backtest_next_open(cfg, "round0", [2])


def test_main_rejects_removed_atr_weights_flag(monkeypatch, capsys):
    import finetune_tw.backtest_next_open as bo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_next_open",
            "--model",
            "round0",
            "--atr-weights",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        bo.main()

    assert excinfo.value.code == 2
    assert "unrecognized arguments: --atr-weights" in capsys.readouterr().err
