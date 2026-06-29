import pandas as pd
import pytest

from finetune_tw.backtest_data import (
    build_rebalance_inputs,
    load_price_field_series,
    load_price_frame_fields,
    load_symbol_history_frames,
)
from finetune_tw.db import init_db, upsert_prices


def _make_history(
    start: str,
    periods: int,
    base_open: float,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    opens = [base_open + i for i in range(periods)]
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": opens,
            "high": [price + 1.0 for price in opens],
            "low": [price - 1.0 for price in opens],
            "close": [price + 0.5 for price in opens],
            "volume": [1_000.0 + i for i in range(periods)],
            "amount": [(price + 0.5) * (1_000.0 + i) for i, price in enumerate(opens)],
        }
    )


def test_load_symbol_history_frames_groups_by_symbol_and_keeps_ohlcva(tmp_path):
    db_path = str(tmp_path / "history.db")
    init_db(db_path)
    upsert_prices(db_path, "1101.TW", _make_history("2024-01-01", 4, 10.0))
    upsert_prices(db_path, "1216.TW", _make_history("2024-01-01", 4, 20.0))

    frames = load_symbol_history_frames(
        db_path,
        ["1216.TW", "1101.TW"],
        start="2024-01-01",
        end="2024-01-31",
    )

    assert list(frames.keys()) == ["1216.TW", "1101.TW"]
    assert list(frames["1101.TW"].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert isinstance(frames["1101.TW"].index, pd.DatetimeIndex)
    assert list(frames["1101.TW"].index.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]

    close_series = load_price_field_series(
        db_path,
        ["1101.TW", "1216.TW"],
        start="2024-01-01",
        end="2024-01-31",
        field="close",
    )
    assert close_series["1101.TW"].name == "close"
    assert close_series["1216.TW"].iloc[-1] == 23.5

    price_frames = load_price_frame_fields(
        db_path,
        ["1101.TW", "1216.TW"],
        start="2024-01-01",
        end="2024-01-31",
        fields=["open", "close"],
    )
    assert list(price_frames["1101.TW"].columns) == ["open", "close"]


def test_load_symbol_history_frames_returns_empty_dict_when_no_rows(tmp_path):
    db_path = str(tmp_path / "history.db")
    init_db(db_path)

    assert load_symbol_history_frames(
        db_path,
        ["1101.TW"],
        start="2024-01-01",
        end="2024-01-31",
    ) == {}


def test_load_price_helpers_raise_for_invalid_fields(tmp_path):
    db_path = str(tmp_path / "history.db")
    init_db(db_path)
    upsert_prices(db_path, "1101.TW", _make_history("2024-01-01", 4, 10.0))

    with pytest.raises(ValueError, match="Expected field to be one of"):
        load_price_field_series(
            db_path,
            ["1101.TW"],
            start="2024-01-01",
            end="2024-01-31",
            field="adj_close",
        )

    with pytest.raises(ValueError, match="Expected fields to be drawn from"):
        load_price_frame_fields(
            db_path,
            ["1101.TW"],
            start="2024-01-01",
            end="2024-01-31",
            fields=["open", "adj_close"],
        )


def test_build_rebalance_inputs_uses_preloaded_history_and_skips_invalid_symbols():
    dates = pd.bdate_range("2024-01-01", periods=6)

    good = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5],
            "volume": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "amount": [1_050.0, 1_161.5, 1_275.0, 1_390.5, 1_508.0, 1_627.5],
        },
        index=dates,
    )
    good.index.name = "date"

    short = good.iloc[:2].copy()
    short.index.name = "date"

    nan_frame = good.copy()
    nan_frame.loc[dates[3], "close"] = float("nan")
    nan_frame.index.name = "date"

    batch_syms, batch_dfs, batch_xts, batch_yts = build_rebalance_inputs(
        history_frames={
            "GOOD": good,
            "SHORT": short,
            "NAN": nan_frame,
        },
        symbols=["GOOD", "SHORT", "NAN"],
        rebal_date=pd.Timestamp("2024-01-08"),
        lookback_window=3,
        pred_len=2,
    )

    assert batch_syms == ["GOOD"]
    assert list(batch_dfs[0].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert isinstance(batch_dfs[0].index, pd.RangeIndex)
    assert list(batch_dfs[0]["open"]) == [13.0, 14.0, 15.0]
    assert list(batch_xts[0].dt.strftime("%Y-%m-%d")) == [
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
    ]
    assert list(batch_yts[0].dt.strftime("%Y-%m-%d")) == [
        "2024-01-08",
        "2024-01-09",
    ]
