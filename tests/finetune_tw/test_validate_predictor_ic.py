import numpy as np
import pandas as pd
from finetune_tw.ic_validation import validate_predictor_ic


class _Cfg:
    pred_len = 5
    val_ic_horizons = 5


def test_validate_predictor_ic_perfect_skill_returns_high_ic():
    # Fake predictor that returns the true future path -> IC should be ~1.0
    actual_paths = {
        "A": [101, 102, 103, 104, 105],
        "B": [100, 99, 98, 97, 96],
        "C": [100, 100.5, 101, 101.5, 102],
        "D": [100, 99.5, 99, 98.5, 98],
    }
    order = ["A", "B", "C", "D"]
    ctx_open = 100.0

    def build_ctx(sym, date):
        df = pd.DataFrame({"open": [100.0]*3, "high": [100.0]*3, "low": [100.0]*3,
                           "close": [100.0, 100.0, 100.0], "volume": [1.0]*3, "amount": [1.0]*3})
        return (df, pd.Series(pd.bdate_range("2024-01-01", periods=3)),
                pd.Series(pd.bdate_range(date, periods=5)), pd.Timestamp(date), ctx_open)

    def predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, _order=order, _ap=actual_paths):
        # df_list arrives in the order validate_predictor_ic enumerates val_universe
        res = []
        for i in range(len(df_list)):
            sym = _order[i % len(_order)]
            res.append(pd.DataFrame({"open": _ap[sym][:pred_len], "close": _ap[sym][:pred_len]}))
        return res

    def actual_lookup(sym, ctx_last_date, n):
        return np.array(actual_paths[sym][:n], dtype=float)

    ic = validate_predictor_ic(predict_batch, actual_lookup, order,
                               [pd.Timestamp("2024-03-01")], _Cfg(), build_ctx)
    assert ic > 0.9


def test_validate_predictor_ic_short_pred_open_filtered_out():
    # Symbols with pred_open too short to satisfy required_pred_len should be
    # silently skipped, not crash, and not pollute the IC result.
    class _Cfg:
        pred_len = 5
        val_ic_horizons = 3  # required_pred_len = min(3, 4) + 1 = 4

    # 4 good symbols + 1 short → only 4 pass guard (need ≥3 for cross-sectional IC)
    good_paths = {
        "A": [101.0, 102.0, 103.0, 104.0, 105.0],
        "B": [100.0, 99.0, 98.0, 97.0, 96.0],
        "C": [100.0, 100.5, 101.0, 101.5, 102.0],
        "D": [100.0, 99.5, 99.0, 98.5, 98.0],
    }
    short_opens = [100.0, 99.0]   # only 2 values → filtered (< required_pred_len=4)
    ctx_open = 100.0
    order = ["A", "B", "C", "D", "SHORT"]

    def build_ctx(sym, date):
        df = pd.DataFrame({"open": [100.0]*3, "high": [100.0]*3, "low": [100.0]*3,
                           "close": [100.0]*3, "volume": [1.0]*3, "amount": [1.0]*3})
        return (df, pd.Series(pd.bdate_range("2024-01-01", periods=3)),
                pd.Series(pd.bdate_range(date, periods=5)), pd.Timestamp(date), ctx_open)

    def predict_batch(df_list, x_ts, y_ts, pred_len):
        results = []
        for i in range(len(df_list)):
            sym = order[i % len(order)]
            opens = good_paths.get(sym, short_opens)[:pred_len]
            results.append(pd.DataFrame({"open": opens, "close": opens}))
        return results

    def actual_lookup(sym, ctx_last_date, n):
        path = good_paths.get(sym, short_opens)
        return np.array(path[:n], dtype=float)

    # SHORT is filtered out (too short); the 4 good symbols produce a valid IC
    ic = validate_predictor_ic(predict_batch, actual_lookup, order,
                               [pd.Timestamp("2024-03-01")], _Cfg(), build_ctx)
    assert np.isfinite(ic), f"Expected finite IC from 4 valid symbols, got {ic}"
