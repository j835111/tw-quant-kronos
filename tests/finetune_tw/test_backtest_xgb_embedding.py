import pandas as pd

from finetune_tw.backtest import rank_stocks, signals_to_holdings
from finetune_tw.backtest_xgb_embedding import xgb_signals_to_raw_preds


def test_xgb_signals_to_raw_preds_plugs_into_signals_to_holdings():
    xgb_preds_by_date = {
        "2024-01-02": {"A": 0.05, "B": 0.02, "C": -0.01, "D": 0.10},
    }
    raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=5)

    # Every symbol's series must be long enough for iloc[hold_days - 1] and equal the raw score.
    for sym, series in raw_preds["2024-01-02"].items():
        assert len(series) == 5
        assert series.iloc[4] == xgb_preds_by_date["2024-01-02"][sym]

    holdings = signals_to_holdings(raw_preds, pd.DatetimeIndex(["2024-01-02"]), hold_days=5, top_k=2)
    assert holdings == [{"D", "A"}]  # top 2 by score: D=0.10, A=0.05


def test_xgb_signals_to_raw_preds_is_hold_days_invariant():
    xgb_preds_by_date = {"2024-01-02": {"A": 0.05, "B": 0.02}}
    for hold_days in (3, 5, 10):
        raw_preds = xgb_signals_to_raw_preds(xgb_preds_by_date, hold_days=hold_days)
        assert raw_preds["2024-01-02"]["A"].iloc[hold_days - 1] == 0.05
