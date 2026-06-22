import pandas as pd
import pytest
from unittest.mock import MagicMock

from finetune_tw.signal import KronosSignal, KronosSignalExtractor


def _make_pred_df(close_val: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [close_val] * 10,
            "high": [close_val] * 10,
            "low": [close_val] * 10,
            "close": [close_val] * 10,
            "volume": [0.0] * 10,
            "amount": [0.0] * 10,
        }
    )


def _make_mock_predictor(close_values: list[float]) -> MagicMock:
    calls = iter(close_values)

    def predict_batch(
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        T,
        top_k,
        top_p,
        sample_count,
        verbose,
    ):
        assert sample_count == 1
        val = next(calls, 1.0)
        return [_make_pred_df(val) for _ in df_list]

    predictor = MagicMock()
    predictor.predict_batch.side_effect = predict_batch
    return predictor


def test_kronos_signal_fields():
    sig = KronosSignal(
        mean_return=0.05,
        q10=0.01,
        q50=0.05,
        q90=0.09,
        dispersion=0.02,
        dir_prob=0.8,
    )
    assert sig.mean_return == pytest.approx(0.05)
    assert sig.dir_prob == pytest.approx(0.8)


def test_extract_date_returns_signals():
    predictor = _make_mock_predictor([1.05] * 20)

    cfg = MagicMock()
    cfg.db_path = ":memory:"
    cfg.lookback_window = 3
    cfg.pred_len = 5

    extractor = KronosSignalExtractor(predictor, n_samples=3, top_k=40)

    import finetune_tw.signal as sig_mod

    original = sig_mod.KronosSignalExtractor._load_context

    def fake_load_context(self, sym, as_of, cfg):
        ctx_df = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [0.0, 0.0, 0.0],
                "amount": [0.0, 0.0, 0.0],
            }
        )
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
        y_ts = pd.Series(pd.date_range("2024-01-04", periods=5, freq="B"))
        return ctx_df, x_ts, y_ts

    sig_mod.KronosSignalExtractor._load_context = fake_load_context
    try:
        result = extractor.extract_date(
            pd.Timestamp("2024-01-04"), ["2330.TW"], cfg, horizon=4
        )
    finally:
        sig_mod.KronosSignalExtractor._load_context = original

    assert "2330.TW" in result
    sig = result["2330.TW"]
    assert isinstance(sig, KronosSignal)
    assert sig.dir_prob == pytest.approx(1.0)


def test_extract_date_range_returns_dataframe():
    predictor = _make_mock_predictor([1.05] * 100)

    cfg = MagicMock()
    cfg.db_path = ":memory:"
    cfg.lookback_window = 3
    cfg.pred_len = 5

    extractor = KronosSignalExtractor(predictor, n_samples=2, top_k=40)

    import finetune_tw.signal as sig_mod

    original = sig_mod.KronosSignalExtractor._load_context

    def fake_load_context(self, sym, as_of, cfg):
        ctx_df = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [0.0, 0.0, 0.0],
                "amount": [0.0, 0.0, 0.0],
            }
        )
        x_ts = pd.Series(pd.date_range("2024-01-01", periods=3, freq="B"))
        y_ts = pd.Series(pd.date_range("2024-01-04", periods=5, freq="B"))
        return ctx_df, x_ts, y_ts

    sig_mod.KronosSignalExtractor._load_context = fake_load_context
    try:
        df = extractor.extract_date_range(
            [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
            ["2330.TW", "2317.TW"],
            cfg,
            horizon=4,
        )
    finally:
        sig_mod.KronosSignalExtractor._load_context = original

    assert isinstance(df, pd.DataFrame)
    assert "kronos_mean" in df.columns
    assert "kronos_dir_prob" in df.columns
    assert df.index.names == ["date", "symbol"]
