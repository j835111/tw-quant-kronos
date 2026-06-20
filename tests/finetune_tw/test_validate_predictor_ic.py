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
    ctx_close = 100.0

    def build_ctx(sym, date):
        df = pd.DataFrame({"open": [100.0]*3, "high": [100.0]*3, "low": [100.0]*3,
                           "close": [100.0, 100.0, ctx_close], "volume": [1.0]*3, "amount": [1.0]*3})
        return (df, pd.Series(pd.bdate_range("2024-01-01", periods=3)),
                pd.Series(pd.bdate_range(date, periods=5)), pd.Timestamp(date), ctx_close)

    def predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, _order=order, _ap=actual_paths):
        # df_list arrives in the order validate_predictor_ic enumerates val_universe
        res = []
        for i in range(len(df_list)):
            sym = _order[i % len(_order)]
            res.append(pd.DataFrame({"close": _ap[sym][:pred_len]}))
        return res

    def actual_lookup(sym, ctx_last_date, n):
        return np.array(actual_paths[sym][:n], dtype=float)

    ic = validate_predictor_ic(predict_batch, actual_lookup, order,
                               [pd.Timestamp("2024-03-01")], _Cfg(), build_ctx)
    assert ic > 0.9
