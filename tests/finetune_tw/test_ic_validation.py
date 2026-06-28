from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finetune_tw.ic_validation import (
    _collect_rows_for_date,
    EarlyStopper,
    collect_validation_rows_by_date,
    compute_validation_metrics_from_rows,
    mean_cross_sectional_ic,
    pick_val_dates,
    pick_val_universe,
    rank_ic,
    validate_predictor_ic,
    validate_predictor_ic_ir,
)


def test_rank_ic_perfect_positive():
    assert rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_rank_ic_perfect_negative():
    assert rank_ic([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_rank_ic_too_few_points_is_nan():
    assert np.isnan(rank_ic([1, 2], [3, 4]))


def test_rank_ic_zero_variance_is_nan():
    assert np.isnan(rank_ic([1, 1, 1, 1], [1, 2, 3, 4]))


def test_mean_cross_sectional_ic_averages_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),
        "d2": ([1, 2, 3, 4], [40, 30, 20, 10]),
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(0.0)


def test_mean_cross_sectional_ic_skips_nan_groups():
    per_group = {
        "d1": ([1, 2, 3, 4], [10, 20, 30, 40]),
        "d2": ([1, 1], [2, 3]),
    }
    assert mean_cross_sectional_ic(per_group) == pytest.approx(1.0)


def test_pick_val_universe_deterministic_and_sized():
    syms = [f"{i:04d}" for i in range(1000)]
    a = pick_val_universe(syms, 150, seed=42)
    b = pick_val_universe(syms, 150, seed=42)
    assert a == b
    assert len(a) == 150
    assert len(set(a)) == 150


def test_pick_val_universe_returns_all_if_small():
    syms = ["A", "B", "C"]
    assert pick_val_universe(syms, 150) == ["A", "B", "C"]


def test_pick_val_dates_count_and_bounds():
    dates = pick_val_dates("2024-01-01", "2024-06-30", 8)
    assert len(dates) <= 8
    assert dates == sorted(dates)
    assert dates[0] >= pd.Timestamp("2024-01-01")
    assert dates[-1] <= pd.Timestamp("2024-06-30")


def test_early_stopper_first_value_is_best():
    es = EarlyStopper(patience=2, mode="max")
    is_best, stop = es.update(0.1)
    assert is_best and not stop
    assert es.best == 0.1


def test_early_stopper_improvement_resets_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.1)
    is_best, stop = es.update(0.2)
    assert is_best and not stop


def test_early_stopper_stops_after_patience():
    es = EarlyStopper(patience=2, mode="max")
    es.update(0.3)
    assert es.update(0.2) == (False, False)
    assert es.update(0.1) == (False, False)
    assert es.update(0.1) == (False, True)


def test_early_stopper_nan_counts_as_no_improvement():
    es = EarlyStopper(patience=1, mode="max")
    es.update(0.3)
    assert es.update(float("nan")) == (False, False)
    assert es.update(float("nan")) == (False, True)


def _make_cfg(pred_len=3, val_ic_horizons=1):
    return SimpleNamespace(pred_len=pred_len, val_ic_horizons=val_ic_horizons)


def _make_ctx(last_date="2024-01-02", ctx_ref=99.0):
    ctx_df = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [1000.0, 1100.0],
            "amount": [10000.0, 11000.0],
        }
    )
    x_ts = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    y_ts = pd.Series(pd.bdate_range("2024-01-03", periods=3))
    return ctx_df, x_ts, y_ts, pd.Timestamp(last_date), ctx_ref


def test_collect_rows_returns_open():
    cfg = _make_cfg(pred_len=3, val_ic_horizons=2)
    pred_df = pd.DataFrame(
        {
            "open": [11.0, 12.0, 13.0],
            "close": [101.0, 102.0, 103.0],
        }
    )

    def build_ctx_fn(sym, date):
        assert sym == "2330"
        return _make_ctx(ctx_ref=88.0)

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        assert len(df_list) == len(x_timestamp_list) == len(y_timestamp_list) == 1
        assert pred_len == cfg.pred_len
        return [pred_df]

    rows = _collect_rows_for_date(
        predict_batch_fn,
        ["2330"],
        pd.Timestamp("2024-01-03"),
        cfg,
        build_ctx_fn,
    )

    assert len(rows) == 1
    sym, pred_open, pred_open_t1, last_date = rows[0]
    assert sym == "2330"
    np.testing.assert_allclose(pred_open, [11.0, 12.0, 13.0])
    assert pred_open_t1 == pytest.approx(11.0)
    assert last_date == pd.Timestamp("2024-01-02")


def test_collect_rows_by_date_batches_once_per_date():
    cfg = SimpleNamespace(pred_len=3, val_ic_horizons=2)
    val_dates = pd.to_datetime(["2024-01-03", "2024-01-04"])
    ctx_df_a, x_ts_a, y_ts_a, last_date_a, _ = _make_ctx(last_date="2024-01-02", ctx_ref=10.0)
    ctx_df_b, x_ts_b, y_ts_b, last_date_b, _ = _make_ctx(last_date="2024-01-02", ctx_ref=11.0)
    ctx_df_c, x_ts_c, y_ts_c, last_date_c, _ = _make_ctx(last_date="2024-01-03", ctx_ref=12.0)
    contexts_by_date = {
        pd.Timestamp("2024-01-03"): [
            ("AAA", ctx_df_a, x_ts_a, y_ts_a, last_date_a),
            ("BBB", ctx_df_b, x_ts_b, y_ts_b, last_date_b),
        ],
        pd.Timestamp("2024-01-04"): [
            ("CCC", ctx_df_c, x_ts_c, y_ts_c, last_date_c),
        ],
    }
    pred_df = pd.DataFrame({"open": [11.0, 12.0, 13.0], "close": [11.0, 12.0, 13.0]})
    calls = []

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        calls.append((len(df_list), pred_len))
        return [pred_df.copy() for _ in df_list]

    rows_by_date = collect_validation_rows_by_date(
        predict_batch_fn,
        contexts_by_date,
        cfg,
    )

    assert calls == [(2, cfg.pred_len), (1, cfg.pred_len)]
    assert {pd.Timestamp(date): len(rows) for date, rows in rows_by_date.items()} == {
        pd.Timestamp("2024-01-03"): 2,
        pd.Timestamp("2024-01-04"): 1,
    }


def test_collect_rows_by_date_uses_prepared_batch_callback_with_precomputed_stamps():
    cfg = SimpleNamespace(pred_len=3, val_ic_horizons=2)
    x_stamp = np.ones((2, 5), dtype=np.float32)
    y_stamp = np.full((3, 5), 2.0, dtype=np.float32)
    ctx_df_a, x_ts_a, y_ts_a, last_date_a, _ = _make_ctx(last_date="2024-01-02", ctx_ref=10.0)
    ctx_df_b, x_ts_b, y_ts_b, last_date_b, _ = _make_ctx(last_date="2024-01-02", ctx_ref=11.0)
    contexts_by_date = {
        pd.Timestamp("2024-01-03"): [
            ("AAA", ctx_df_a, x_ts_a, y_ts_a, last_date_a, x_stamp, y_stamp),
            ("BBB", ctx_df_b, x_ts_b, y_ts_b, last_date_b, x_stamp * 3, y_stamp),
        ],
    }
    pred_df = pd.DataFrame({"open": [11.0, 12.0, 13.0], "close": [11.0, 12.0, 13.0]})
    legacy_calls = []
    prepared_calls = []

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        legacy_calls.append((df_list, x_timestamp_list, y_timestamp_list, pred_len))
        return [pred_df.copy() for _ in df_list]

    def prepared_batch_predict_fn(
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        x_stamp_list,
        y_stamp_list,
    ):
        prepared_calls.append(
            {
                "batch_size": len(df_list),
                "pred_len": pred_len,
                "x_stamp_list": x_stamp_list,
                "y_stamp_list": y_stamp_list,
            }
        )
        return [pred_df.copy() for _ in df_list]

    rows_by_date = collect_validation_rows_by_date(
        predict_batch_fn,
        contexts_by_date,
        cfg,
        prepared_batch_predict_fn=prepared_batch_predict_fn,
    )

    assert legacy_calls == []
    assert len(prepared_calls) == 1
    assert prepared_calls[0]["batch_size"] == 2
    assert prepared_calls[0]["pred_len"] == cfg.pred_len
    assert prepared_calls[0]["x_stamp_list"][0] is x_stamp
    assert prepared_calls[0]["x_stamp_list"][1] is not x_stamp
    assert prepared_calls[0]["y_stamp_list"][0] is y_stamp
    assert prepared_calls[0]["y_stamp_list"][1] is y_stamp
    assert len(rows_by_date[pd.Timestamp("2024-01-03")]) == 2


def test_collect_rows_by_date_falls_back_to_legacy_predict_when_no_prepared_callback():
    cfg = SimpleNamespace(pred_len=3, val_ic_horizons=2)
    x_stamp = np.ones((2, 5), dtype=np.float32)
    y_stamp = np.full((3, 5), 2.0, dtype=np.float32)
    ctx_df_a, x_ts_a, y_ts_a, last_date_a, _ = _make_ctx(last_date="2024-01-02", ctx_ref=10.0)
    contexts_by_date = {
        pd.Timestamp("2024-01-03"): [
            ("AAA", ctx_df_a, x_ts_a, y_ts_a, last_date_a, x_stamp, y_stamp),
        ],
    }
    pred_df = pd.DataFrame({"open": [11.0, 12.0, 13.0], "close": [11.0, 12.0, 13.0]})
    legacy_calls = []

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        legacy_calls.append((df_list, x_timestamp_list, y_timestamp_list, pred_len))
        return [pred_df.copy() for _ in df_list]

    rows_by_date = collect_validation_rows_by_date(
        predict_batch_fn,
        contexts_by_date,
        cfg,
        prepared_batch_predict_fn=None,
    )

    assert len(legacy_calls) == 1
    assert legacy_calls[0][3] == cfg.pred_len
    assert len(rows_by_date[pd.Timestamp("2024-01-03")]) == 1


def test_collect_rows_by_date_rejects_unsupported_context_tuple_shape():
    cfg = SimpleNamespace(pred_len=3, val_ic_horizons=2)
    ctx_df, x_ts, y_ts, last_date, ctx_ref = _make_ctx(last_date="2024-01-02", ctx_ref=10.0)
    bad_contexts_by_date = {
        pd.Timestamp("2024-01-03"): [
            ("AAA", ctx_df, x_ts, y_ts, last_date, ctx_ref),
        ],
    }

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        raise AssertionError("predict_batch_fn should not be called for unsupported context tuple shape")

    with pytest.raises(ValueError, match="Expected validation context tuples with 5 or 7 fields"):
        collect_validation_rows_by_date(
            predict_batch_fn,
            bad_contexts_by_date,
            cfg,
        )


def test_compute_validation_metrics_reuses_rows_for_both_outputs():
    cfg = SimpleNamespace(pred_len=3, val_ic_horizons=2)
    val_dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])
    rows_by_date = {
        pd.Timestamp("2024-01-03"): [
            ("AAA", np.array([100.0, 101.0, 102.0]), 100.0, pd.Timestamp("2024-01-02")),
            ("BBB", np.array([100.0, 102.0, 104.0]), 100.0, pd.Timestamp("2024-01-02")),
            ("CCC", np.array([100.0, 103.0, 106.0]), 100.0, pd.Timestamp("2024-01-02")),
        ],
        pd.Timestamp("2024-01-04"): [
            ("AAA", np.array([100.0, 102.0, 104.0]), 100.0, pd.Timestamp("2024-01-03")),
            ("BBB", np.array([100.0, 103.0, 106.0]), 100.0, pd.Timestamp("2024-01-03")),
            ("CCC", np.array([100.0, 104.0, 108.0]), 100.0, pd.Timestamp("2024-01-03")),
        ],
        pd.Timestamp("2024-01-05"): [
            ("AAA", np.array([100.0, 103.0, 106.0]), 100.0, pd.Timestamp("2024-01-04")),
            ("BBB", np.array([100.0, 104.0, 108.0]), 100.0, pd.Timestamp("2024-01-04")),
            ("CCC", np.array([100.0, 105.0, 110.0]), 100.0, pd.Timestamp("2024-01-04")),
        ],
    }
    calls = []

    def actual_lookup(sym, last_date, n):
        calls.append((sym, pd.Timestamp(last_date), n))
        base = {"AAA": 100.0, "BBB": 101.0, "CCC": 102.0}[sym]
        day_bump = {
            pd.Timestamp("2024-01-02"): 0.0,
            pd.Timestamp("2024-01-03"): 1.0,
            pd.Timestamp("2024-01-04"): 2.0,
        }[pd.Timestamp(last_date)]
        return np.array([base, base + day_bump + 2.0, base + day_bump + 4.0], dtype=float)[:n]

    val_ic, ic_ir = compute_validation_metrics_from_rows(
        rows_by_date,
        actual_lookup,
        val_dates,
        cfg,
        target_horizon=1,
    )

    assert np.isfinite(val_ic)
    assert np.isfinite(ic_ir)
    assert len(calls) == len(val_dates) * 3


def test_validate_ic_open_to_open():
    cfg = _make_cfg(pred_len=3, val_ic_horizons=1)
    val_universe = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    val_dates = [pd.Timestamp("2024-01-03")]

    pred_by_sym = {
        sym: pd.DataFrame(
            {
                "open": [100.0, 101.0 + idx, 102.0 + idx],
                "close": [500.0 - idx, 400.0 - idx, 300.0 - idx],
            }
        )
        for idx, sym in enumerate(val_universe)
    }
    actual_by_sym = {
        sym: np.array([100.0, 201.0 + idx, 202.0 + idx], dtype=float)
        for idx, sym in enumerate(val_universe)
    }

    def build_ctx_fn(sym, date):
        return _make_ctx(ctx_ref=100.0)

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        assert pred_len == cfg.pred_len
        return [pred_by_sym[sym] for sym in val_universe[: len(df_list)]]

    def actual_lookup(sym, last_date, n):
        assert n == cfg.pred_len
        return actual_by_sym[sym][:n]

    val_ic = validate_predictor_ic(
        predict_batch_fn,
        actual_lookup,
        val_universe,
        val_dates,
        cfg,
        build_ctx_fn,
    )

    assert val_ic == pytest.approx(1.0)


def test_validate_ic_ir_open_to_open():
    cfg = _make_cfg(pred_len=3, val_ic_horizons=2)
    val_universe = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    val_dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])

    pred_by_sym = {
        sym: pd.DataFrame(
            {
                "open": [100.0, 101.0 + idx, 102.0 + idx],
                "close": [800.0 - idx, 700.0 - idx, 600.0 - idx],
            }
        )
        for idx, sym in enumerate(val_universe)
    }
    actual_by_sym = {
        sym: np.array([100.0, 201.0 + idx, 202.0 + idx], dtype=float)
        for idx, sym in enumerate(val_universe)
    }

    def build_ctx_fn(sym, date):
        return _make_ctx(ctx_ref=100.0)

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        assert pred_len == cfg.pred_len
        return [pred_by_sym[sym] for sym in val_universe[: len(df_list)]]

    def actual_lookup(sym, last_date, n):
        assert n == cfg.pred_len
        return actual_by_sym[sym][:n]

    ic_ir = validate_predictor_ic_ir(
        predict_batch_fn,
        actual_lookup,
        val_universe,
        val_dates,
        cfg,
        build_ctx_fn,
        target_horizon=1,
    )

    assert np.isfinite(ic_ir)
    assert ic_ir > 1e7


def test_validate_ic_ir_target_horizon_zero_short_circuits():
    cfg = _make_cfg(pred_len=3, val_ic_horizons=2)
    calls = {"predict": 0, "actual": 0}

    def build_ctx_fn(sym, date):
        raise AssertionError("build_ctx_fn should not be called when target_horizon <= 0")

    def predict_batch_fn(df_list, x_timestamp_list, y_timestamp_list, pred_len):
        calls["predict"] += 1
        raise AssertionError("predict_batch_fn should not be called when target_horizon <= 0")

    def actual_lookup(sym, last_date, n):
        calls["actual"] += 1
        raise AssertionError("actual_lookup should not be called when target_horizon <= 0")

    ic_ir = validate_predictor_ic_ir(
        predict_batch_fn,
        actual_lookup,
        ["AAA", "BBB", "CCC"],
        pd.to_datetime(["2024-01-03", "2024-01-04"]),
        cfg,
        build_ctx_fn,
        target_horizon=0,
    )

    assert np.isnan(ic_ir)
    assert calls == {"predict": 0, "actual": 0}
