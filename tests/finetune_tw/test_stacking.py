import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from finetune_tw.stacking import StackingModel, FEATURE_COLS, build_feature_row
from finetune_tw.signal import KronosSignal
from finetune_tw.analog import AnalogEngine, AnalogFeatures


def _make_feature_df(n_dates: int = 20, n_syms: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    syms = [f"{i:04d}.TW" for i in range(n_syms)]
    rows = []
    for d in dates:
        for s in syms:
            row = {col: float(rng.normal()) for col in FEATURE_COLS}
            row["date"] = d
            row["symbol"] = s
            row["fwd_return"] = float(rng.normal(0, 0.02))
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["date", "symbol"])
    return df


def test_stacking_model_fit_predict():
    df = _make_feature_df(10, 30)
    model = StackingModel(num_rounds=20)
    model.fit(df)
    scores = model.predict(df.drop(columns=["fwd_return"]))
    assert len(scores) == len(df)
    assert not scores.isna().any()


def test_stacking_model_scores_ranked():
    df = _make_feature_df(5, 40)
    model = StackingModel(num_rounds=20)
    model.fit(df)
    scores = model.predict(df.drop(columns=["fwd_return"]))
    assert scores.std() > 0


def test_stacking_model_save_load(tmp_path):
    df = _make_feature_df(5, 30)
    model = StackingModel(num_rounds=10)
    model.fit(df)
    path = str(tmp_path / "stacker.lgb")
    model.save(path)
    loaded = StackingModel.load(path)
    original_scores = model.predict(df.drop(columns=["fwd_return"]))
    loaded_scores = loaded.predict(df.drop(columns=["fwd_return"]))
    pd.testing.assert_series_equal(original_scores, loaded_scores, check_names=False)


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 23


def test_build_feature_row_with_kronos_signal():
    sig = KronosSignal(mean_return=0.02, q10=0.0, q50=0.02, q90=0.04, dispersion=0.01, dir_prob=0.7)
    sym_close = [100.0 + i for i in range(70)]
    bench_close = [1000.0 + i for i in range(70)]
    dates = pd.bdate_range("2022-01-03", periods=70).strftime("%Y-%m-%d")
    sym_df = pd.DataFrame({
        "date": dates, "open": sym_close, "high": [c * 1.01 for c in sym_close],
        "low": [c * 0.99 for c in sym_close], "close": sym_close,
        "volume": [1000.0] * 70, "amount": [c * 1000 for c in sym_close],
    })
    bench_df = sym_df.copy()
    bench_df["close"] = bench_close
    cfg = type("Cfg", (), {"lookback_window": 90, "pred_len": 10})()
    as_of = pd.Timestamp(dates[-1])

    row = build_feature_row("2330.TW", as_of, sig, sym_df, bench_df, None, cfg)
    assert row is not None
    assert "kronos_mean" in row
    assert row["kronos_mean"] == pytest.approx(0.02)
    assert "ma20_gap" in row
    assert "alpha_20d" in row
    assert row["analog_q50"] == pytest.approx(0.0)


def test_build_feature_row_with_real_analog_engine():
    """Real AnalogEngine.fit → query → build_feature_row integration."""
    n = 80
    close = [100.0 + i * 0.5 for i in range(n)]
    dates = pd.bdate_range("2020-01-02", periods=n).strftime("%Y-%m-%d")
    sym_df = pd.DataFrame({
        "date": dates, "open": close, "high": [c * 1.01 for c in close],
        "low": [c * 0.99 for c in close], "close": close,
        "volume": [1000.0] * n, "amount": [c * 1000 for c in close],
    })
    bench_df = sym_df.copy()

    fake_df = sym_df.copy()

    engine = AnalogEngine(n_neighbors=5, window=10)
    with patch("finetune_tw.analog.query_symbol", return_value=fake_df):
        engine.fit(":memory:", ["2330.TW"], cutoff_date="2020-05-01", pred_len=5)

    cfg = type("Cfg", (), {"lookback_window": 60, "pred_len": 5})()
    as_of = pd.Timestamp(dates[-1])

    row = build_feature_row("2330.TW", as_of, None, sym_df, bench_df, engine, cfg)
    assert row is not None
    # Analog features should be non-zero when engine has neighbors
    assert "analog_q50" in row
    # analog_up_prob must be in [0, 1]
    assert 0.0 <= row["analog_up_prob"] <= 1.0
