import numpy as np
import pytest
from unittest.mock import patch
import pandas as pd
from finetune_tw.analog import AnalogEngine, AnalogFeatures


def _make_price_df(close_values: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    n = len(close_values)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates, "open": close_values,
        "high": [c * 1.01 for c in close_values],
        "low": [c * 0.99 for c in close_values],
        "close": close_values,
        "volume": [1000.0] * n, "amount": [c * 1000 for c in close_values],
    })


def test_analog_engine_fit_and_query():
    """Build index from synthetic data, query should return AnalogFeatures."""
    close = list(range(100, 200))  # 100 rows
    fake_df = _make_price_df(close)

    engine = AnalogEngine(n_neighbors=5, window=10)
    with patch("finetune_tw.analog.query_symbol", return_value=fake_df):
        engine.fit(":memory:", ["2330.TW"], cutoff_date="2020-06-01", pred_len=5)

    assert len(engine._keys) > 0
    assert len(engine._fwd_returns) > 0

    recent_close = np.array(list(range(150, 160)), dtype=float)
    recent_volume = np.full(10, 1000.0)
    result = engine.query(recent_close, recent_volume)

    assert result is not None
    assert isinstance(result, AnalogFeatures)
    assert 0.0 <= result.up_prob <= 1.0
    assert result.n_analogs <= 5


def test_analog_engine_empty_returns_none():
    engine = AnalogEngine(n_neighbors=5, window=10)
    # No fit() called → _keys is empty
    recent_close = np.ones(10)
    recent_volume = np.ones(10)
    result = engine.query(recent_close, recent_volume)
    assert result is None


def test_analog_features_fields():
    af = AnalogFeatures(
        fwd_q25=0.01, fwd_q50=0.03, fwd_q75=0.05,
        up_prob=0.7, max_gain=0.12, max_loss=-0.08,
        dispersion=0.03, n_analogs=20,
    )
    assert af.up_prob == pytest.approx(0.7)
    assert af.n_analogs == 20


def test_point_in_time_cutoff():
    """Verify that fit uses a strict cutoff before as_of."""
    calls = []

    def mock_query(db_path, symbol, start=None, end=None):
        calls.append(end)
        return _make_price_df([100.0] * 10)  # too short, will be skipped

    engine = AnalogEngine(n_neighbors=5, window=10)
    with patch("finetune_tw.analog.query_symbol", side_effect=mock_query):
        engine.fit(":memory:", ["2330.TW"], cutoff_date="2024-01-10", pred_len=10)

    # The end date passed to query_symbol must be BEFORE cutoff_date
    assert len(calls) > 0
    for end_date in calls:
        assert end_date < "2024-01-10"
